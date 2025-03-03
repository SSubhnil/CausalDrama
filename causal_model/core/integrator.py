import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F

from causal_model.core.networks import SparseCodebookMoE
from encoder import CausalEncoder, CausalEncoder
from quantizer import DualVQQuantizer
from confounder_approx import ConfounderPosterior, ConfounderPrior
from predictors import MoETransitionHead, ImprovedRewardHead, TerminationPredictor
from networks import SparseCodebookMoE


class CausalModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.hidden_state_dim = params.Models.WorldModel.HiddenStateDim
        self.code_dim = params.Models.CausalModel.CodeDim
        self.num_codes_tr = params.Models.CausalModel.NumCodesTr
        self.num_codes_re = params.Models.CausalModel.NumCodesRe
        self.hidden_dim = params.Models.CausalModel.HiddenDim
        self.kl_weight = params.Models.CausalModel.KLWeight
        self.quant_weight = params.Models.CausalModel.QuantWeight
        self.reg_weight = params.Models.CausalModel.RegWeight
        self.inv_weight = params.Models.CausalModel.InvarianceWeight
        self.sparsity_weight = params.Models.CausalModel.SparsityWeight
        self.use_confounder = params.Models.CausalModel.UseConfounder
        self.device = device


        """Encoder"""
        self.encoder_params = params.Models.CausalModel.Encoder
        self.causal_encoder = CausalEncoder(hidden_state_dim=self.hidden_state_dim,
                                     tr_proj_dim=self.encoder_params.TransProjDim,
                                     re_proj_dim=self.encoder_params.RewProjDim,
                                     hidden_dim=self.encoder_params.HiddenDim)

        """Quantizer"""
        self.quantizer_params = params.Models.CausalModel.Quantizer
        self.quantizer = DualVQQuantizer(code_dim=self.code_dim,
                                         num_codes_tr=self.num_codes_tr,
                                         num_codes_re=self.num_codes_re,
                                         beta=self.quantizer_params.Beta,
                                         tr_temperature=self.quantizer_params.TransitionTemp,
                                         tr_min_temperature=self.quantizer_params.TransitionMinTemperature,
                                         tr_anneal_factor=self.quantizer_params.TransitionAnnealFactor,
                                         re_temperature=self.quantizer_params.RewardTemp,
                                         re_min_temperature=self.quantizer_params.RewardMinTemperature,
                                         re_anneal_factor=self.quantizer_params.RewardAnnealFactor,
                                         normalize=self.quantizer_params.NormalizedInputs,
                                         coupling=self.quantizer_params.Coupling,
                                         lambda_couple=self.quantizer_params.LambdaCouple,
                                         hidden_dim=self.hidden_dim,
                                         )

        """Confounder Inference"""
        self.confounder_params = params.Models.CausalModel.Confounder
        self.confounder_prior_net = ConfounderPrior(num_codes=self.num_codes_tr,
                                                    code_dim=self.code_dim,
                                                    conf_dim=self.confounder_params.ConfDim,
                                                    hidden_state_proj_dim=self.encoder_params.TransProjDim,
                                                    momentum=self.confounder_params.PriorMomentum)
        self.confounder_post_net = ConfounderPosterior(code_dim=self.code_dim,
                                                       conf_dim=self.confounder_params.ConfDim,
                                                       num_codes=self.num_codes_tr,
                                                       hidden_dim=self.confounder_params.HiddenDim)
        """Predictor Heads"""
        self.predictor_params = self.params.Models.CausalModel.Predictors
        self.moe_net = SparseCodebookMoE(num_experts=self.predictor_params.Transition.NumOfExperts,
                                         hidden_dim=self.predictor_params.Transition.HiddenDim,
                                         code_dim=self.code_dim,
                                         quantizer=self.quantizer,
                                         top_k=self.predictor_params.Transition.TopK,
                                         )
        self.tr_head = MoETransitionHead(moe_net=self.moe_net,
                                         hidden_state_dim=self.hidden_state_dim,
                                         hidden_dim=self.predictor_params.Transition.HiddenDim,
                                         code_dim=self.code_dim,
                                         conf_dim=self.confounder_params.ConfDim,
                                         sparsity_weight=self.predictor_params.Transition.MaskSparsityWeight)

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



    def forward(self, h):
        # Projection of raw hidden state into h_tr and h_re
        h_proj_tr, h_proj_re = self.causal_encoder(h)

        # Projections go into Dual Codebook Quantizer
        quant_output_dict = self.quantizer(h_proj_tr, h_proj_re)
        code_emb_tr = quant_output_dict['quantized_tr']
        code_emb_re = quant_output_dict['quantized_re']

        # Confounder approximation with transition codebook
        # Prior uses EMA codebook entries ('code_emb_momentum') that require discrete indices for lookup.
        # hard_tr codes ensure prior stability during posterior training (no grad flow to codebook)
        mu_prior, logvar_prior = self.confounder_prior_net(h_proj_tr,
                                                           code_ids=quant_output_dict['hard_tr'].argmax(-1)) # Discrete indices [B, T]

        u_post, kl_loss = self.confounder_post_net(h_proj_tr, code_emb_tr,
                                                   mu_prior, logvar_prior)

        # Prediction Heads
        # Transition prediction
        next_state_logits, h_modulated, total_loss = self.tr_head(h, code_emb_tr, u_post)
        # Reward Prediction
        reward_pred = self.re_head(h_modulated, code_emb_re, quant_output_dict['q_re'])
        # Termination head
        termination = self.terminator(h_modulated)

        return next_state_logits, reward_pred, termination


