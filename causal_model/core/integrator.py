import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F

from causal_model.core.networks import SparseCodebookMoE
from encoder import CausalEncoder, CausalEncoder
from quantizer import DualVQQuantizer
from confounder_approx import ConfounderPosterior, ConfounderPrior
from predictors import MoETransitionHead, ImprovedRewardHead, TerminationPredictor, StateModulator
from networks import SparseCodebookMoE


class CausalModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.hidden_state_dim = params.Models.WorldModel.HiddenStateDim
        self.code_dim = params.Models.CausalModel.CodeDim
        self.num_codes_tr = params.Models.CausalModel.NumCodesTr
        self.num_codes_re = params.Models.CausalModel.NumCodesRe
        self.hidden_dim = params.Models.CausalModel.HiddenDim
        self.use_confounder = params.Models.CausalModel.UseConfounder
        self.device = device

        self.loss_weights = {
            # Primary Prediction Losses
            'transition': params.Models.CausalModel.Predictors.TransitionWeight,
            'reward': params.Models.CausalModel.Predictors.RewardWeight,
            'termination': params.Models.CausalModel.Predictors.TerminationWeight
        }

        """Encoder"""
        self.encoder_params = params.Models.CausalModel.Encoder
        self.causal_encoder = CausalEncoder(hidden_state_dim=self.hidden_state_dim,
                                     tr_proj_dim=self.encoder_params.TransProjDim,
                                     re_proj_dim=self.encoder_params.RewProjDim,
                                     hidden_dim=self.encoder_params.HiddenDim)

        """
        QUANTIZER
        Losses:
            codebook_loss_tr, codebook_loss_re
            commitment_loss_tr, commitment_loss_re
        Loss weights:
            Commitment Loss:
                beta_tr, beta_re
        """
        self.quantizer_params = params.Models.CausalModel.Quantizer
        self.quantizer = DualVQQuantizer(code_dim=self.code_dim,
                                         num_codes_tr=self.num_codes_tr,
                                         num_codes_re=self.num_codes_re,
                                         beta_tr=self.quantizer_params.BetaTransition,
                                         beta_re=self.quantizer_params.BetaReward,
                                         tr_temperature=self.quantizer_params.TransitionTemp,
                                         tr_min_temperature=self.quantizer_params.TransitionMinTemperature,
                                         tr_anneal_factor=self.quantizer_params.TransitionAnnealFactor,
                                         re_temperature=self.quantizer_params.RewardTemp,
                                         re_min_temperature=self.quantizer_params.RewardMinTemperature,
                                         re_anneal_factor=self.quantizer_params.RewardAnnealFactor,
                                         normalize=self.quantizer_params.NormalizedInputs,
                                         coupling=self.quantizer_params.Coupling,
                                         sparsity_weight=self.quantizer_params.SparsityWeight, # Coupling sparsity
                                         lambda_couple=self.quantizer_params.LambdaCouple,
                                         hidden_dim=self.hidden_dim,
                                         )

        """
        CONFOUNDER VARIATIONAL INFERENCE
            Prior Losses:
                Code_alignment_loss, prior_regularization()
            Posterior Losses:
                regularization_loss(): l2_reg, code_sparsity
                kl_loss
            Prior Loss weights:
                code_alignment_loss, reg_loss_weight
            Post Loss weights:
                reg_loss(): l2_reg_weight, code_sparsity_weight
                kl_loss_weights        
        """

        self.confounder_params = params.Models.CausalModel.Confounder
        self.loss_weights.update({
            'prior_reg_weight': self.confounder_params.PriorRegWeight,
            'prior_code_align': self.confounder_params.PriorCodeAlign,
            'post_reg': self.confounder_params.PostReg,
            'post_sparsity': self.confounder_params.PostCodeSparsity,
            'kl_loss_weight': self.confounder_params.PostKLWeight
        })
        self.confounder_prior_net = ConfounderPrior(num_codes=self.num_codes_tr,
                                                    code_dim=self.code_dim,
                                                    conf_dim=self.confounder_params.ConfDim,
                                                    hidden_state_proj_dim=self.encoder_params.TransProjDim,
                                                    momentum=self.confounder_params.PriorMomentum)
        self.confounder_post_net = ConfounderPosterior(code_dim=self.code_dim,
                                                       conf_dim=self.confounder_params.ConfDim,
                                                       num_codes=self.num_codes_tr,
                                                       hidden_dim=self.confounder_params.HiddenDim)

        """
        PREDICTOR HEADS
            Transition Head: aux_loss (SparseCodebookMoE), Sparsity Loss
            Loss weights: aux_loss, Sparsity_loss
        """
        self.predictor_params = self.params.Models.CausalModel.Predictors
        self.loss_weights.update({
            'tr_aux_loss': self.predictor_params.Transition.AuxiliaryWeight,
            'tr_sparsity_weight': self.predictor_params.Transition.MaskSparsityWeight
        })
        self.state_modulator = StateModulator(self.hidden_state_dim, self.predictor_params.Transition.HiddenDim,
                                              self.code_dim, self.confounder_params.ConfDim)

        self.tr_head = MoETransitionHead(hidden_state_dim=self.hidden_state_dim,
                                         hidden_dim=self.predictor_params.Transition.HiddenDim,
                                         code_dim=self.code_dim,
                                         conf_dim=self.confounder_params.ConfDim,
                                         state_modulator=self.state_modulator,
                                         use_importance_weighted_moe=self.predictor_params.Transition.UseImportanceWeightedMoE
                                         )

        self.re_head = ImprovedRewardHead(num_codes=self.num_codes_re,
                                          code_dim=self.code_dim,
                                          hidden_dim=self.predictor_params.Reward.HiddenDim,
                                          hidden_state_dim=self.hidden_state_dim,
                                          num_heads=self.predictor_params.Reward.NumHeads)

        self.terminator = TerminationPredictor(hidden_state_dim=self.hidden_state_dim,
                                               hidden_units=self.predictor_params.Termination.HiddenDim,
                                               act=self.predictor_params.Termination.Activation,
                                               layer_num=self.predictor_params.Termination.NumLayers,
                                               dropout=self.predictor_params.Termination.Dropout,
                                               dtype=params.Models.WorldModel.dtype,
                                               device=self.device)


    def forward(self, h, detach_encoder=True):
        if detach_encoder: # Blocks gradient flow from the causal model to world model.
            h_stable = h.detach()
        else:
            h_stable = h
        # Projection of raw hidden state into h_tr and h_re
        h_proj_tr, h_proj_re = self.causal_encoder(h_stable)

        # Projections go into Dual Codebook Quantizer
        quant_output_dict = self.quantizer(h_proj_tr, h_proj_re)
        code_emb_tr = quant_output_dict['quantized_tr']
        # Quantization Loss Total - loss weights in-code
        quant_loss = quant_output_dict['total_loss']

        # Confounder approximation with transition codebook
        # Prior uses EMA codebook entries ('code_emb_momentum') that require discrete indices for lookup.
        # hard_tr codes ensure prior stability during posterior training (no grad flow to codebook)
        mu_prior, logvar_prior = self.confounder_prior_net(h_proj_tr,
                                                           code_ids=quant_output_dict['hard_tr'].argmax(-1)) # Discrete indices [B, T]
        prior_code_alignment_loss = self.confounder_prior_net.code_alignment_loss() * self.loss_weights['prior_code_align']
        prior_reg_loss = self.confounder_prior_net.prior_regularization(mu_prior) * self.loss_weights['prior_reg_weight']

        # Confounder posterior
        u_post, kl_loss = self.confounder_post_net(h_proj_tr, code_emb_tr,
                                                   mu_prior, logvar_prior)
        post_l2_reg, post_sparsity = self.confounder_post_net.regularization_loss() * self.loss_weights['post_reg']
        confounder_loss = prior_code_alignment_loss + prior_reg_loss + \
                           (post_l2_reg * self.loss_weights['post_reg']) + \
                           (post_sparsity * self.loss_weights['post_sparsity']) + \
                           (kl_loss * self.loss_weights['kl_loss_weight'])
        causal_loss = quant_loss + confounder_loss

        return quant_output_dict, u_post, causal_loss