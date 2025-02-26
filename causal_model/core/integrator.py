import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F

from causal_model.core.networks import SparseCodebookMoE
from encoder import CausalEncoder, CausalEncoder_Confounder, CausalDecoder
from quantizer import DualVQQuantizer
from confounder_approx import ConfounderApproximator
from predictors import MoETransitionHead, RewardHead, ImprovedRewardHead
from networks import SparseCodebookMoE


class CausalModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.hidden_state_dim = params.Models.WorldModel.HiddenStateDim
        self.code_dim = params.Models.CausalModel.CodeDim
        self.conf_dim = params.Models.CausalModel.ConfDim
        self.num_codes_tr = params.Models.CausalModel.NumCodesTr
        self.num_codes_re = params.Models.CausalModel.NumCodesRe
        self.hidden_dim = params.Models.CausalModel.HiddenDim
        self.kl_weight = params.Models.CausalModel.KLWeight
        self.quant_weight = params.Models.CausalModel.QuantWeight
        self.reg_weight = params.Models.CausalModel.RegWeight
        self.inv_weight = params.Models.CausalModel.InvarianceWeight
        self.sparsity_weight = params.Models.CausalModel.SparsityWeight
        self.use_confounder = params.Models.CausalModel.UseConfounder
        self.device= device


        """Encoder"""
        self.encoder = self.create_causal_encoder(self.params)

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
        self.confounder_net = ConfounderApproximator(code_dim=params.code_dim,
                                                     conf_dim=params.conf_dim,
                                                     num_codes=params.num_codes_tr,
                                                     params=self.params)
        self.tr_predictor = MoETransitionHead

        # Simplified data flow
        # h_modulated = modulator(h_world, codes_u)  # Feature adaptation
        # moe_out = SparseCodebookMoE(h_modulated, code_emb)  # Mechanism selection

        # In MoETransitionHead:
        # h_modulated = h_modulated.detach()  # Stop gradient to world model

        # In Quantizer:
        # quantized_re = quantized_re.detach()  # Protect codebook

        # Hidden state modulation network

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

    def create_causal_encoder(self, params):
        if self.use_confounder:
            return CausalEncoder_Confounder(params)
        return CausalEncoder(params)

    def create_causal_decoder(self, params):
        return CausalDecoder(params, use_confounder=params.use_confounder)

