import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F

from causal_model.core.networks import SparseCodebookMoE
from encoder import CausalEncoder, CausalEncoder
from quantizer import DualVQQuantizer
from confounder_approx import ConfounderPosterior, ConfounderPrior
from predictors import MoETransitionHead, RewardHead, ImprovedRewardHead
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
                                         temperature=self.quantizer_params.Temperature,
                                         normalize=self.quantizer_params.NormalizedInputs,
                                         coupling=self.quantizer_params.Coupling,
                                         lambda_couple=self.quantizer_params.LambdaCouple,
                                         hidden_dim=self.hidden_dim)

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
        self.tr_head = MoETransitionHead(hidden_state_dim=self.hidden_state_dim,
                                              hidden_dim=self.predictor_params.Transition.HiddenDim,
                                              code_dim=self.code_dim,
                                              num_codes=self.num_codes_tr,
                                              conf_dim=self.confounder_params.ConfDim,
                                              num_experts=self.predictor_params.Transition.NumOfExperts)

        self.re_head = ImprovedRewardHead(num_codes=self.num_codes_re,
                                          code_dim=self.code_dim,
                                          hidden_dim=self.predictor_params.Reward.HiddenDim,
                                          hidden_state_dim=self.hidden_state_dim,
                                          num_heads=self.predictor_params.Reward.NumHeads)



    def forward(self, h, q):
        enc_output = self.encoder(h)
        # Get soft code assignments
        code_ids = torch.argmax(q, dim=1)  # Differentiable via Gumbel-STG
        u, kl_loss = ConfounderApproximator(h, code_ids)  # [B, conf_dim]

        next_h =  self.decoder(enc_output['tr_features'],
                               enc_output['re_features'],
                               enc_output.get('confounder_sample')
                               )

        # Reward prediction integration
        # quant_out = self.quantizer(h_tr, h_re)
        # reward_pred = self.reward_head(quant_out['quantizedre'], quant_out['q_re'])

        return next_h, enc_output


