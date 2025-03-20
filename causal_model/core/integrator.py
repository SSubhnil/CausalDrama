import torch
import torch.nn as nn
import time
from line_profiler import profile
from .encoder import CausalEncoder
from .quantizer import DualVQQuantizer
from .confounder_approx import ConfounderPosterior, ConfounderPrior
from .predictors import MoETransitionHead, ImprovedRewardHead, TerminationPredictor, StateModulator


class CausalModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.hidden_state_dim = params.Models.WorldModel.HiddenStateDim
        self.code_dim_tr = params.Models.CausalModel.TrCodeDim
        self.code_dim_re = params.Models.CausalModel.ReCodeDim
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
        self.quantizer = DualVQQuantizer(code_dim_tr=self.code_dim_tr,
                                         code_dim_re=self.code_dim_re,
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
                                         sparsity_weight=self.quantizer_params.SparsityWeight,  # Coupling sparsity
                                         lambda_couple=self.quantizer_params.LambdaCouple,
                                         hidden_dim=self.hidden_dim,
                                         use_cdist=self.quantizer_params.UseCDist
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
                                                    code_dim=self.code_dim_tr,
                                                    conf_dim=self.confounder_params.ConfDim,
                                                    hidden_state_proj_dim=self.encoder_params.TransProjDim,
                                                    momentum=self.confounder_params.PriorMomentum)
        self.confounder_post_net = ConfounderPosterior(code_dim=self.code_dim_tr,
                                                       conf_dim=self.confounder_params.ConfDim,
                                                       num_codes=self.num_codes_tr,
                                                       hidden_dim=self.confounder_params.HiddenDim)

        """
        PREDICTOR HEADS
            Transition Head: aux_loss (SparseCodebookMoE), Sparsity Loss
            Loss weights: aux_loss, Sparsity_loss
        """
        self.predictor_params = params.Models.CausalModel.Predictors
        self.loss_weights.update({
            'tr_aux_loss': self.predictor_params.Transition.AuxiliaryWeight,
            'tr_sparsity_weight': self.predictor_params.Transition.MaskSparsityWeight
        })
        self.state_modulator = StateModulator(self.hidden_state_dim, self.predictor_params.Transition.HiddenDim,
                                              self.code_dim_tr, self.confounder_params.ConfDim,
                                              self.predictor_params.ComputeInvarianceLoss)

        self.tr_head = MoETransitionHead(hidden_state_dim=self.hidden_state_dim,
                                         hidden_dim=self.predictor_params.Transition.HiddenDim,
                                         code_dim=self.code_dim_tr,
                                         conf_dim=self.confounder_params.ConfDim,
                                         num_experts=self.predictor_params.Transition.NumOfExperts,
                                         top_k=self.predictor_params.Transition.TopK,
                                         state_modulator=self.state_modulator,
                                         quantizer=self.quantizer,
                                         use_importance_weighted_moe=self.predictor_params.Transition.UseImportanceWeightedMoE
                                         )

        self.re_head = ImprovedRewardHead(num_codes=self.num_codes_re,
                                          code_dim=self.code_dim_re,
                                          hidden_dim=self.predictor_params.Reward.HiddenDim,
                                          hidden_state_dim=self.hidden_state_dim,
                                          )

        self.terminator = TerminationPredictor(hidden_state_dim=self.hidden_state_dim,
                                               hidden_units=self.predictor_params.Termination.HiddenDim,
                                               act=self.predictor_params.Termination.Activation,
                                               layer_num=self.predictor_params.Termination.NumLayers,
                                               dropout=self.predictor_params.Termination.Dropout,
                                               dtype=params.Models.WorldModel.dtype,
                                               device=self.device)

    @profile
    def forward(self, h, global_step, training=False):

        # Common operations needed for both paths
        if global_step < 1033:
            h_stable = h.detach()
        else:
            h_stable = h

        # Projection of raw hidden state into h_tr and h_re
        h_proj_tr, h_proj_re = self.causal_encoder(h_stable)

        # Different paths for training and inference
        if training:
            # Full training path with all losses
            quant_output_dict = self.quantizer(h_proj_tr, h_proj_re, training=True)
            code_emb_tr = quant_output_dict['quantized_tr']
            quant_loss = quant_output_dict['loss']

            if global_step % 3 == 0:  # Prior phase
                # Freeze posterior gradients but compute values
                with torch.no_grad():
                    # Confounder prior network calculations with all losses
                    mu_prior, logvar_prior = self.confounder_prior_net(h_proj_tr,
                                                                       code_ids=quant_output_dict['hard_tr'].argmax(-1))
                    # Confounder posterior with KL loss
                    u_post, _, _ = self.confounder_post_net(h_proj_tr, code_emb_tr)

                prior_code_alignment_loss = self.confounder_prior_net.code_alignment_loss() * self.loss_weights[
                    'prior_code_align']
                prior_reg_loss = self.confounder_prior_net.prior_regularization(mu_prior) * self.loss_weights[
                    'prior_reg_weight']
                # For phased training, return only prior-related components
                confounder_loss = prior_code_alignment_loss + prior_reg_loss

            else:  # Posterior phase
                # Compute prior but detach gradients
                with torch.no_grad():
                    mu_prior, logvar_prior = self.confounder_prior_net(h_proj_tr.detach(),
                                                                       code_ids=quant_output_dict['hard_tr'].argmax(-1))
                    # Allow gradients for posterior
                    u_post, mu_post, logvar_post = self.confounder_post_net(h_proj_tr, code_emb_tr)
                    kl_loss = self.confounder_post_net.gaussian_KL(mu_post, logvar_post, mu_prior.detach(),
                                                                   logvar_prior.detach())
                    # Regularization losses
                    post_l2_reg, post_sparsity = self.confounder_post_net.regularization_loss()
                    post_l2_reg = post_l2_reg * self.loss_weights['post_reg']
                    post_sparsity = post_sparsity * self.loss_weights['post_sparsity']
                    kl_loss = kl_loss * self.loss_weights['kl_loss_weight']

                    # For phased training, return only posterior-related components
                    confounder_loss = kl_loss + post_l2_reg + post_sparsity

            # Combine all losses
            causal_loss = quant_loss + confounder_loss
            return quant_output_dict, u_post, causal_loss

        else:  # Inference
            # Optimized inference path - skip unnecessary computations
            quant_output_dict = self.quantizer(h_proj_tr, h_proj_re, training=False)
            code_emb_tr = quant_output_dict['quantized_tr']

            # Only calculate what's needed for inference
            mu_prior, logvar_prior = self.confounder_prior_net(h_proj_tr.detach(),
                                                               code_ids=quant_output_dict['hard_tr'].argmax(-1))

            # Skip KL and regularization loss calculations
            u_post = self._inference_posterior(h_proj_tr, code_emb_tr, mu_prior, logvar_prior)

            return quant_output_dict, u_post, None

    def _inference_posterior(self, h, code_emb, mu_prior, logvar_prior):
        """Fast inference-only posterior calculation"""
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            # Generate posterior parameters
            mu_post = torch.clamp(self.confounder_post_net.confounder_post_mu_net(h, code_emb), min=-10.0, max=10.0)
            logvar_post = self.confounder_post_net.confounder_post_logvar_net(h, code_emb)

            # Reparameterize
            return self.confounder_post_net.reparameterize(mu_post, logvar_post)
