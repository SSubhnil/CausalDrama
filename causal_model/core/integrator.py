import torch
import torch.nn as nn
import time
from line_profiler import profile
from .encoder import CausalEncoder
from .quantizer import DualVQQuantizer
from .confounder_approx import ConfounderPosterior, ConfounderPrior, MixtureConfounderPrior


class CausalModel(nn.Module):
    def __init__(self, params, device, action_dim, stoch_dim, d_model):
        super().__init__()
        self.hidden_state_dim = d_model
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.code_dim_tr = params.Models.CausalModel.TrCodeDim
        self.code_dim_re = params.Models.CausalModel.ReCodeDim
        self.num_codes_tr = params.Models.CausalModel.NumCodesTr
        self.num_codes_re = params.Models.CausalModel.NumCodesRe
        self.hidden_dim = params.Models.CausalModel.HiddenDim
        self.use_confounder = params.Models.CausalModel.UseConfounder
        self.device = device
        combined_input_dim = self.hidden_state_dim
        if action_dim is not None:
            combined_input_dim += 1  # For 1D action after unsqueeze
        if stoch_dim is not None:
            combined_input_dim += stoch_dim

        self.loss_weights = {
            # Primary Prediction Losses
            'transition': params.Models.CausalModel.Predictors.TransitionWeight,
            'reward': params.Models.CausalModel.Predictors.RewardWeight,
            'termination': params.Models.CausalModel.Predictors.TerminationWeight
        }

        """Encoder"""
        self.encoder_params = params.Models.CausalModel.Encoder
        self.causal_encoder = CausalEncoder(hidden_state_dim=self.hidden_state_dim,
                                            action_dim=self.action_dim,
                                            stoch_dim=self.stoch_dim,
                                            tr_proj_dim=self.encoder_params.TransProjDim,
                                            re_proj_dim=self.encoder_params.RewProjDim,
                                            hidden_dim=self.encoder_params.HiddenDim,
                                            combined_input_dim=combined_input_dim,
                                            embedding_mode=self.encoder_params.Embedding)

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
        self.use_mixture_prior = self.confounder_params.UseMixturePrior
        self.loss_weights.update({
            'prior_reg_weight': self.confounder_params.PriorRegWeight,
            'prior_code_align': self.confounder_params.PriorCodeAlign,
            'post_reg': self.confounder_params.PostReg,
            'post_sparsity': self.confounder_params.PostCodeSparsity,
            'kl_loss_weight': self.confounder_params.PostKLWeight
        })
        if not self.use_mixture_prior:
            self.confounder_prior_net = ConfounderPrior(num_codes=self.num_codes_tr,
                                                        code_dim=self.code_dim_tr,
                                                        conf_dim=self.confounder_params.ConfDim,
                                                        hidden_state_proj_dim=self.encoder_params.TransProjDim,
                                                        momentum=self.confounder_params.PriorMomentum)
        else:
            self.confounder_prior_net = MixtureConfounderPrior(num_codes=self.num_codes_tr,
                                                               code_dim=self.code_dim_tr,
                                                               conf_dim=self.confounder_params.ConfDim,
                                                               hidden_state_proj_dim=self.encoder_params.TransProjDim,
                                                               momentum=self.confounder_params.PriorMomentum)

        self.confounder_post_net = ConfounderPosterior(code_dim=self.code_dim_tr,
                                                       conf_dim=self.confounder_params.ConfDim,
                                                       num_codes=self.num_codes_tr,
                                                       hidden_dim=self.confounder_params.HiddenDim)

    @profile
    def forward(self, h, global_step, training=False):

        # Projection of raw hidden state into h_tr and h_re
        h_proj_tr, h_proj_re = self.causal_encoder(h)

        # Different paths for training and inference
        if training:
            self.anneal_quantizer_temperature(global_step)

            # Full training path with all losses
            quant_output_dict = self.quantizer(h_proj_tr, h_proj_re, training=True)
            code_emb_tr = quant_output_dict['quantized_tr']
            quant_loss = quant_output_dict['loss']

            if not self.use_mixture_prior:
                mu_prior, logvar_prior = self.confounder_prior_net(h_proj_tr,
                                                                   code_ids=quant_output_dict['hard_tr'].argmax(-1))
            else:
                mix_weights, mu_prior, logvar_prior = self.confounder_prior_net(h_proj_tr,
                                                                                code_ids=quant_output_dict[
                                                                                    'hard_tr'].argmax(
                                                                                    -1) if 'hard_tr' in quant_output_dict else None)

            # Prior losses
            prior_code_alignment_loss = self.confounder_prior_net.code_alignment_loss() * self.loss_weights[
                'prior_code_align']
            prior_reg_loss = self.confounder_prior_net.prior_regularization(mu_prior) * self.loss_weights[
                'prior_reg_weight']
            prior_loss = prior_code_alignment_loss + prior_reg_loss

            # Confounder posterior
            u_post, mu_post, logvar_post = self.confounder_post_net(h_proj_tr, code_emb_tr)

            # Posterior losses - (optional) detach prior parameters
            if not self.use_mixture_prior:
                kl_loss = self.confounder_post_net.gaussian_KL(mu_post, logvar_post, mu_prior.detach(),
                                                               logvar_prior.detach())
            else:
                kl_loss = self.confounder_post_net.gaussian_KL_to_mixture(mu_post, logvar_post,
                                                                          mix_weights, mu_prior.detach(),
                                                                          logvar_prior.detach())

            # Regularization losses
            post_l2_reg, post_sparsity = self.confounder_post_net.regularization_loss()
            post_l2_reg = post_l2_reg * self.loss_weights['post_reg']
            post_sparsity = post_sparsity * self.loss_weights['post_sparsity']
            kl_loss = kl_loss * self.loss_weights['kl_loss_weight']

            # Combine all losses
            posterior_loss = kl_loss + post_l2_reg + post_sparsity
            causal_loss = quant_loss + prior_loss + posterior_loss
            return quant_output_dict, u_post, causal_loss, quant_loss, prior_loss, posterior_loss

        else:  # Inference
            # Optimized inference path - skip unnecessary computations
            quant_output_dict = self.quantizer(h_proj_tr, h_proj_re, training=False)
            code_emb_tr = quant_output_dict['quantized_tr']

            # Only calculate what's needed for inference
            if not self.use_mixture_prior:
                mu_prior, logvar_prior = self.confounder_prior_net(h_proj_tr.detach(),
                                                                   code_ids=quant_output_dict['hard_tr'].argmax(-1))
            else:
                _, mu_prior, logvar_prior = self.confounder_prior_net(h_proj_tr.detach(),
                                                                      code_ids=quant_output_dict['hard_tr'].argmax(
                                                                          -1) if 'hard_tr' in quant_output_dict else None)
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

    def anneal_quantizer_temperature(self, global_step):
        """Anneals quantizer temperature based on training progress"""
        # Early warmup phase
        if global_step < 2000:
            return  # Keep initial temperatures

        # Main annealing phase - determine progress factor
        if global_step < 40000:
            # Early training: slow annealing
            effective_epoch = (global_step - 2000) / 8000
        else:
            # Later training: faster annealing
            effective_epoch = (global_step - 2000) / 4000

        # Apply temperature annealing
        self.quantizer.anneal_temperature(effective_epoch)

