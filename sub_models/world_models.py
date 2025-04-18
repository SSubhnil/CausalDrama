import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch.distributions import OneHotCategorical, Normal
# import torchvision.transforms as T
import kornia.augmentation as K
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast
from sub_models.laprop import LaProp
from pytorch_warmup import LinearWarmup
# from nfnets import AGC
import time

from sub_models.functions_losses import SymLogTwoHotLoss, symlog
from sub_models.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from sub_models.transformer_model import StochasticTransformerKVCache
from mamba_ssm import MambaWrapperModel, MambaConfig, InferenceParams, update_graph_cache

from causal_model.core.integrator import CausalModel
from causal_model.core.predictors import MoETransitionHead, ImprovedRewardHead, TerminationPredictor
import agents
from line_profiler import profile
from torch.distributions.independent import Independent
import numpy as np
from tools import weight_init
import cv2


def track_grad_flow(named_parameters):
    '''Tracks gradients flowing through different components during training.'''
    component_grads = {}

    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            # Determine component based on parameter name
            if 'encoder' in n:
                component = 'encoder'
            elif 'sequence_model' in n:
                component = 'sequence_model'
            elif 'causal_model.tr_head' in n:
                component = 'transition_head'
            elif 'causal_model' in n:
                component = 'causal_model'
            elif 'dist_head' in n:
                component = 'dist_head'
            elif 'reward_decoder' in n:
                component = 'reward_decoder'
            elif 'termination_decoder' in n:
                component = 'termination_decoder'
            else:
                component = 'other'

            if component not in component_grads:
                component_grads[component] = []

            component_grads[component].append((n, p.grad.abs().mean().item(), p.grad.abs().max().item()))

    # Print gradient flow report
    print("\n=== Gradient Flow Report ===")
    total_params = 0
    for component, grads in component_grads.items():
        avg_of_avgs = sum(g[1] for g in grads) / len(grads) if grads else 0
        max_of_maxes = max(g[2] for g in grads) if grads else 0
        num_params = len(grads)
        total_params += num_params

        print(f"{component:20s} | avg_grad: {avg_of_avgs:.6f} | max_grad: {max_of_maxes:.6f} | params: {num_params}")

        # Print top 3 parameters with highest gradients
        top_grads = sorted(grads, key=lambda x: x[1], reverse=True)[:3]
        for name, avg, max_val in top_grads:
            print(f"  - {name:50s} | {avg:.6f} (avg) | {max_val:.6f} (max)")

    print(f"Total parameters: {total_params}")
    print("============================")


def profile_memory():
    '''Prints current GPU memory usage'''
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / (1024 ** 2):.1f}MB allocated, "
              f"{torch.cuda.memory_reserved() / (1024 ** 2):.1f}MB reserved")


class Encoder(nn.Module):
    def __init__(self, depth=128, mults=(1, 2, 4, 2), norm='rms', act='SiLU', kernel=4, padding='same',
                 first_stride=True, input_size=(3, 64, 64), dtype=None, device=None) -> None:
        super().__init__()
        act = getattr(nn, act)
        self.depths = [depth * mult for mult in mults]
        self.kernel = kernel
        self.stride = 2
        self.padding = (kernel - 1) // 2 if padding == 'same' else padding

        backbone = []
        current_channels, current_height, current_width = input_size

        # Define convolutional layers for image inputs
        for i, depth in enumerate(self.depths):
            stride = 1 if i == 0 and first_stride else self.stride
            conv = nn.Conv2d(in_channels=current_channels, out_channels=depth, kernel_size=kernel, stride=stride,
                             padding=self.padding, dtype=dtype, device=device)
            backbone.append(conv)
            backbone.append(nn.BatchNorm2d(depth, dtype=dtype, device=device))
            backbone.append(act())

            current_height, current_width = self._compute_output_dim(current_height, current_width, kernel, stride,
                                                                     self.padding)
            current_channels = depth

        self.backbone = nn.Sequential(*backbone)
        self.backbone.apply(weight_init)
        self.last_channels = self.depths[-1]
        self.output_dim = (self.last_channels, current_height, current_width)
        self.output_flatten_dim = self.last_channels * current_height * current_width

    def _compute_output_dim(self, height, width, kernel_size, stride, padding):
        new_height = (height - kernel_size + 2 * padding) // stride + 1
        new_width = (width - kernel_size + 2 * padding) // stride + 1
        return new_height, new_width

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L C H W -> (B L) C H W")
        x = self.backbone(x)
        x = rearrange(x, "(B L) C H W -> B L (C H W)", B=batch_size)
        return x


class Decoder(nn.Module):
    def __init__(self, stoch_dim, depth=128, mults=(1, 2, 4, 2), norm='rms', act='SiLU', kernel=4, padding='same',
                 first_stride=True, last_output_dim=(256, 4, 4), input_size=(3, 64, 64), cnn_sigmoid=False, dtype=None,
                 device=None) -> None:
        super().__init__()
        act = getattr(nn, act)
        self.depths = [depth * mult for mult in mults]
        self.kernel = kernel
        self.stride = 2
        self.padding = (kernel - 1) // 2 if padding == 'same' else padding
        self.output_padding = self.stride // 2 if padding == 'same' else 0
        self._cnn_sigmoid = cnn_sigmoid

        backbone = []
        # stem
        backbone.append(
            nn.Linear(stoch_dim, last_output_dim[0] * last_output_dim[1] * last_output_dim[2], bias=True, dtype=dtype,
                      device=device))
        backbone.append(Rearrange('B L (C H W) -> (B L) C H W', C=last_output_dim[0], H=last_output_dim[1]))
        backbone.append(nn.BatchNorm2d(last_output_dim[0], dtype=dtype, device=device))
        backbone.append(act())
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        current_channels, current_height, current_width = last_output_dim
        # Define convolutional layers for image inputs
        for i, depth in reversed(list(enumerate(self.depths[:-1]))):
            conv = nn.ConvTranspose2d(in_channels=current_channels, out_channels=depth, kernel_size=kernel,
                                      stride=self.stride, padding=self.padding, output_padding=self.output_padding,
                                      dtype=dtype, device=device)
            backbone.append(conv)
            backbone.append(nn.BatchNorm2d(depth, dtype=dtype, device=device))
            backbone.append(act())
            current_height, current_width = self._compute_transposed_output_dim(current_height, current_width, kernel,
                                                                                self.stride, self.padding,
                                                                                self.output_padding)
            current_channels = depth

        stride = 1 if i == 0 and first_stride else self.stride
        backbone.append(
            nn.ConvTranspose2d(
                in_channels=self.depths[0],
                out_channels=input_size[0],
                kernel_size=kernel,
                stride=stride,
                padding=self.padding,
                dtype=dtype, device=device
            )
        )
        current_height, current_width = self._compute_transposed_output_dim(current_height, current_width, kernel,
                                                                            stride, self.padding, 0)
        self.final_output_dim = (input_size[0], current_height, current_width)
        self.backbone = nn.Sequential(*backbone)
        self.backbone.apply(weight_init)

    def _compute_transposed_output_dim(self, height, width, kernel_size, stride, padding, output_padding):
        new_height = (height - 1) * stride - 2 * padding + kernel_size + output_padding
        new_width = (width - 1) * stride - 2 * padding + kernel_size + output_padding
        return new_height, new_width

    def forward(self, sample):
        batch_size = sample.shape[0]
        obs_hat = self.backbone(sample)
        obs_hat = rearrange(obs_hat, "(B L) C H W -> B L C H W", B=batch_size)
        if self._cnn_sigmoid:
            obs_hat = F.sigmoid(obs_hat)
        else:
            obs_hat += 0.5
        return obs_hat


class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''

    def __init__(self, image_feat_dim, hidden_state_dim, categorical_dim, class_dim, tr_predictor, unimix_ratio=0.01,
                 dtype=None, device=None) -> None:
        super().__init__()
        self.stoch_dim = categorical_dim
        self.post_head = nn.Linear(image_feat_dim, categorical_dim * class_dim, dtype=dtype, device=device)

        # Replacing the below head with MoE Transition head
        # self.prior_head = nn.Linear(hidden_state_dim, categorical_dim*class_dim, dtype=dtype, device=device)
        self.prior_head = tr_predictor
        self.unimix_ratio = unimix_ratio
        self.dtype = dtype
        self.device = device

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        if mixing_ratio > 0:
            probs = F.softmax(logits, dim=-1)
            # log_probs = F.log_softmax(logits, dim=-1) # Better numerical stability
            # Avoid direct lo of small values
            mixed_probs = mixing_ratio * torch.ones_like(probs, dtype=self.dtype,
                                                         device=self.device) / self.stoch_dim + (
                                  1 - mixing_ratio) * probs
            # log_mixed_probs = torch.log(mixing_ratio * torch.ones_like(log_probs.exp()) / self.stoch_dim +
            #                         (1 - mixing_ratio) * log_probs.exp())
            logits = torch.log(mixed_probs)
            # return log_mixed_probs
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        # Added to control exploding logs
        # logits = torch.clamp(logits, min=-15.0, max=15.0)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits, self.unimix_ratio)
        return logits

    def forward_prior(self, h, code_emb_tr=None):
        # next_state_logits, h_modulated, total_tr_loss, aux_loss, sparsity_loss = self.tr_head(h, code_emb_tr, u_post)
        logits, total_tr_loss, aux_loss, diversity_loss, sparsity_loss = self.prior_head(h, code_emb_tr)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits, self.unimix_ratio)
        return logits, total_tr_loss, aux_loss, diversity_loss, sparsity_loss

    def update_temperature(self):
        # Calls the update_temperature() in ImportanceWeightedMoE in networks.py
        self.prior_head.moe.update_temperature()


class RewardHead(nn.Module):
    def __init__(self, num_classes, inp_dim, hidden_units, act, layer_num, dtype=None, device=None) -> None:
        super().__init__()
        act = getattr(nn, act)

        # Create the backbone with dynamic number of layers
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Linear(inp_dim, hidden_units, bias=True, dtype=dtype, device=device))
            layers.append(RMSNorm(hidden_units, dtype=dtype, device=device))
            layers.append(act())

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_units, num_classes, dtype=dtype, device=device)

    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward


class TerminationHead(nn.Module):
    def __init__(self, inp_dim, hidden_units, act, layer_num, dtype=None, device=None) -> None:
        super().__init__()
        act = getattr(nn, act)

        # Create the backbone with dynamic number of layers
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Linear(inp_dim, hidden_units, bias=True, dtype=dtype, device=device))
            layers.append(RMSNorm(hidden_units, dtype=dtype, device=device))
            layers.append(act())

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_units, 1, dtype=dtype, device=device)

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        distance = (obs_hat - obs) ** 2
        loss = reduce(distance, "B L C H W -> B L", "sum")
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div) * self.free_bits, kl_div)
        return kl_div, real_kl_div


class WorldModel(nn.Module):
    def __init__(self, action_dim, config, device):
        super().__init__()
        self.hidden_state_dim = config.Models.WorldModel.HiddenStateDim
        self.final_feature_width = config.Models.WorldModel.Transformer.FinalFeatureWidth
        self.categorical_dim = config.Models.WorldModel.CategoricalDim
        self.class_dim = config.Models.WorldModel.ClassDim
        self.stoch_flattened_dim = self.categorical_dim * self.class_dim
        self.use_amp = config.BasicSettings.Use_amp
        self.use_cg = config.BasicSettings.Use_cg
        self.tensor_dtype = torch.bfloat16 if self.use_amp and not self.use_cg else config.Models.WorldModel.dtype
        self.save_every_steps = config.JointTrainAgent.SaveEverySteps
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.device = device  # Maybe it's not needed
        self.model = config.Models.WorldModel.Backbone
        self.use_causal_model = config.Models.UseCausal
        self.max_grad_norm = config.Models.WorldModel.Max_grad_norm
        max_seq_length = max(config.JointTrainAgent.BatchLength,
                             config.JointTrainAgent.ImagineContextLength + config.JointTrainAgent.ImagineBatchLength,
                             config.JointTrainAgent.RealityContextLength)

        # For new predictor heads
        self.code_dim_tr = config.Models.CausalModel.TrCodeDim
        self.code_dim_re = config.Models.CausalModel.ReCodeDim
        self.num_codes_tr = config.Models.CausalModel.NumCodesTr
        self.num_codes_re = config.Models.CausalModel.NumCodesRe
        self.hidden_dim = config.Models.CausalModel.HiddenDim
        self.use_confounder = config.Models.CausalModel.UseConfounder
        self.confounder_params = config.Models.CausalModel.Confounder

        # Image encoder
        self.encoder = Encoder(
            depth=config.Models.WorldModel.Encoder.Depth,
            mults=config.Models.WorldModel.Encoder.Mults,
            norm=config.Models.WorldModel.Encoder.Norm,
            act=config.Models.WorldModel.Act,
            kernel=config.Models.WorldModel.Encoder.Kernel,
            padding=config.Models.WorldModel.Encoder.Padding,
            input_size=config.Models.WorldModel.Encoder.InputSize,
            dtype=config.Models.WorldModel.dtype, device=device)

        if self.model == 'Transformer':
            self.sequence_model = StochasticTransformerKVCache(
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                feat_dim=self.hidden_state_dim,
                num_layers=config.Models.WorldModel.Transformer.NumLayers,
                num_heads=config.Models.WorldModel.Transformer.NumHeads,
                max_length=max_seq_length,
                dropout=config.Models.WorldModel.Dropout
            )
        elif self.model == 'Mamba':
            mamba_config = MambaConfig(
                d_model=self.hidden_state_dim,
                d_intermediate=config.Models.WorldModel.Mamba.d_intermediate,
                n_layer=config.Models.WorldModel.Mamba.n_layer,
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': config.Models.WorldModel.Mamba.ssm_cfg.d_state,
                }
            )
            self.sequence_model = MambaWrapperModel(mamba_config)
        elif self.model == 'Mamba2':
            mamba_config = MambaConfig(
                d_model=self.hidden_state_dim,
                d_intermediate=config.Models.WorldModel.Mamba.d_intermediate,
                n_layer=config.Models.WorldModel.Mamba.n_layer,
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': config.Models.WorldModel.Mamba.ssm_cfg.d_state,
                    'layer': 'Mamba2'}
            )
            self.sequence_model = MambaWrapperModel(mamba_config)
        else:
            raise ValueError(f"Unknown dynamics model: {self.model}")

        # Causal model definition
        self.causal_model = CausalModel(
            params=config,
            device=self.device,
            action_dim=action_dim,
            stoch_dim=self.stoch_flattened_dim,
            d_model=self.hidden_state_dim
        )
        # codebook_data = self.causal_model.quantizer.tr_quantizer.codebook.data.clone()
        codebook_data = self.causal_model.quantizer.vq.codebook.data.clone()
        """
        PREDICTOR HEADS
            Transition Head: aux_loss (SparseCodebookMoE), Sparsity Loss
            Loss weights: aux_loss, Sparsity_loss
        """
        self.world_causal_alignment_loss_weight = self.causal_model.quantizer_params.CausalWorldContrastiveLossWeight
        self.predictor_params = config.Models.CausalModel.Predictors
        # self.loss_weights.update({
        #     'tr_aux_loss': self.predictor_params.Transition.AuxiliaryWeight,
        #     'tr_sparsity_weight': self.predictor_params.Transition.MaskSparsityWeight
        # })

        self.tr_head = MoETransitionHead(hidden_state_dim=self.hidden_state_dim,
                                         hidden_dim=self.predictor_params.Transition.HiddenDim,
                                         code_dim=self.code_dim_tr,
                                         num_experts=self.predictor_params.Transition.NumOfExperts,
                                         top_k=self.predictor_params.Transition.TopK,
                                         codebook_data=codebook_data,
                                         use_importance_weighted_moe=self.predictor_params.Transition.UseImportanceWeightedMoE,
                                         slicing=self.predictor_params.Transition.Slicing,
                                         use_simple_mlp=self.predictor_params.Transition.UseSimpleMLP
                                         )

        # self.re_head = ImprovedRewardHead(num_codes=self.num_codes_re,
        #                                   code_dim=self.code_dim_re,
        #                                   hidden_dim=self.predictor_params.Reward.HiddenDim,
        #                                   hidden_state_dim=self.hidden_state_dim,
        #                                   )
        #
        # self.terminator = TerminationPredictor(hidden_state_dim=self.hidden_state_dim,
        #                                        hidden_units=self.predictor_params.Termination.HiddenDim,
        #                                        act=self.predictor_params.Termination.Activation,
        #                                        layer_num=self.predictor_params.Termination.NumLayers,
        #                                        dropout=self.predictor_params.Termination.Dropout,
        #                                        dtype=config.Models.WorldModel.dtype,
        #                                        device=self.device)

        self.dist_head = DistHead(
            image_feat_dim=self.encoder.output_flatten_dim,
            hidden_state_dim=self.hidden_state_dim,
            categorical_dim=self.categorical_dim,
            class_dim=self.class_dim,
            unimix_ratio=config.Models.WorldModel.Unimix_ratio,
            tr_predictor=self.tr_head,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        self.image_decoder = Decoder(
            stoch_dim=self.stoch_flattened_dim,
            depth=config.Models.WorldModel.Decoder.Depth,
            mults=config.Models.WorldModel.Decoder.Mults,
            norm=config.Models.WorldModel.Decoder.Norm,
            act=config.Models.WorldModel.Act,
            kernel=config.Models.WorldModel.Decoder.Kernel,
            padding=config.Models.WorldModel.Decoder.Padding,
            first_stride=config.Models.WorldModel.Decoder.FirstStrideOne,
            last_output_dim=self.encoder.output_dim,
            input_size=config.Models.WorldModel.Decoder.InputSize,
            cnn_sigmoid=config.Models.WorldModel.Decoder.FinalLayerSigmoid,
            dtype=config.Models.WorldModel.dtype, device=device
        )

        # self.reward_decoder = self.re_head
        # self.termination_decoder = self.terminator
        self.reward_decoder = RewardHead(
            num_classes=255,
            inp_dim=self.hidden_state_dim,
            hidden_units=config.Models.WorldModel.Reward.HiddenUnits,
            act=config.Models.WorldModel.Act,
            layer_num=config.Models.WorldModel.Reward.LayerNum,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        self.reward_decoder.apply(weight_init)
        self.termination_decoder = TerminationHead(
            inp_dim=self.hidden_state_dim,
            hidden_units=config.Models.WorldModel.Termination.HiddenUnits,
            act=config.Models.WorldModel.Act,
            layer_num=config.Models.WorldModel.Termination.LayerNum,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        self.termination_decoder.apply(weight_init)
        #
        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)

        if config.Separate.SeparateOptimizers:
            # World model parameters (excluding causal model)
            self.world_parameters = [p for n, p in self.named_parameters()
                                     if not n.startswith('causal_model')]
            # Causal model parameters
            self.causal_parameters = self.causal_model.parameters()

            # World model optimizer
            if config.Models.WorldModel.Optimiser == 'Laprop':
                self.world_optimizer = LaProp(self.world_parameters, lr=config.Models.WorldModel.Laprop.LearningRate,
                                              eps=config.Models.WorldModel.Laprop.Epsilon,
                                              weight_decay=config.Models.WorldModel.Weight_decay)

            elif config.Models.WorldModel.Optimiser == 'Adam':
                self.world_optimizer = torch.optim.AdamW(self.world_parameters,
                                                         lr=config.Models.WorldModel.Adam.LearningRate,
                                                         weight_decay=config.Models.WorldModel.Weight_decay)
            else:
                raise ValueError(f"Unknown optimiser: {config.Models.WorldModel.Optimiser}")

            # Causal model optimizer
            if config.Separate.CausalOptimizer == 'Laprop':
                self.causal_optimizer = LaProp(self.causal_parameters, lr=config.Separate.CausalModelLR,
                                               weight_decay=config.Models.WorldModel.Weight_decay)
            elif config.Separate.CausalOptimizer == 'Adam':
                self.causal_optimizer = torch.optim.AdamW(self.causal_parameters,
                                                          lr=config.Separate.CausalModelLR,
                                                          weight_decay=config.Models.WorldModel.Weight_decay)
            else:
                raise ValueError(f"Unknown optimiser: {config.Separate.CausalOptimizer}")

            # Separate schedulers
            self.lr_scheduler_world = torch.optim.lr_scheduler.LambdaLR(self.world_optimizer,
                                                                        lr_lambda=lambda step: 1.0)
            self.lr_scheduler_causal = torch.optim.lr_scheduler.LambdaLR(self.causal_optimizer,
                                                                         lr_lambda=lambda step: 1.0)
            # Separate warmup schedulers
            self.warmup_scheduler_world = LinearWarmup(self.world_optimizer,
                                                       warmup_period=config.Models.WorldModel.Warmup_steps)
            self.warmup_scheduler_causal = LinearWarmup(self.causal_optimizer,
                                                        warmup_period=config.Models.WorldModel.Warmup_steps)

        else:
            if config.Models.WorldModel.Optimiser == 'Laprop':
                self.world_optimizer = LaProp(self.world_parameters, lr=config.Models.WorldModel.Laprop.LearningRate,
                                              eps=config.Models.WorldModel.Laprop.Epsilon,
                                              weight_decay=config.Models.WorldModel.Weight_decay)
            elif config.Models.WorldModel.Optimiser == 'Adam':
                self.world_optimizer = torch.optim.AdamW(self.world_parameters,
                                                         lr=config.Models.WorldModel.Adam.LearningRate,
                                                         weight_decay=config.Models.WorldModel.Weight_decay)
            else:
                raise ValueError(f"Unknown optimiser: {config.Models.WorldModel.Optimiser}")
            # self.optimizer = AGC(self.parameters(), self.optimizer)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.world_optimizer, lr_lambda=lambda step: 1.0)
            self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=config.Models.WorldModel.Warmup_steps)

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and config.Models.WorldModel.dtype is not torch.bfloat16)

    def encode_obs(self, obs):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.straight_through_gradient(post_logits, sample_mode="random_sample", identifier="502")
            flattened_sample = self.flatten_sample(sample)
        return flattened_sample

    def calc_last_dist_feat(self, latent, action, inference_params=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            if self.model == 'Transformer':
                temporal_mask = get_subsequent_mask(latent)
                dist_feat = self.sequence_model(latent, action, temporal_mask)
            else:
                tokens = self.sequence_model.backbone.tokenizer(latent, action)
                quantizer_output, _ = self.causal_model(tokens, 1000, training=False)
                feature_mask = self.causal_model.mask_generator(quantizer_output['quantized'], tokens)
                tokens = tokens * feature_mask
                dist_feat = self.sequence_model(tokens, inference_params)

            last_dist_feat = dist_feat[:, -1:]
            # action_unsqueezed = action.unsqueeze(-1)
            # combined_input = torch.cat([latent, action_unsqueezed, dist_feat], dim=-1)
            # Add causal model processing
            # quant_output_dict, u_post, _ = self.causal_model(combined_input, 1000)
            # modulated_feat, _ = self.state_modulator(last_dist_feat, quant_output_dict['quantized_tr'],
            #                                                       u_post, 1000)
            # code_emb_tr = quant_output_dict['quantized'][:, -1:]  # Get transition codes

            prior_logits, _, _, _, _ = self.dist_head.forward_prior(last_dist_feat)
            prior_sample = self.straight_through_gradient(prior_logits, sample_mode="random_sample", identifier="522")
            prior_flattened_sample = self.flatten_sample(prior_sample)
            # Add this line to ensure consistent sequence dimensions
            prior_flattened_sample = prior_flattened_sample[:, -1:]
        return prior_flattened_sample, last_dist_feat

    def calc_last_post_feat(self, latent, action, current_obs, inference_params=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(current_obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.straight_through_gradient(post_logits, sample_mode="random_sample", identifier="532")
            flattened_sample = self.flatten_sample(sample)
            if self.model == 'Transformer':
                temporal_mask = get_subsequent_mask(latent)
                dist_feat = self.sequence_model(latent, action, temporal_mask)
            else:
                tokens = self.sequence_model.backbone.tokenizer(latent, action)
                dist_feat = self.sequence_model(tokens, inference_params)
            last_dist_feat = dist_feat[:, -1:]
            shifted_feat = last_dist_feat
            x = torch.cat((shifted_feat, flattened_sample), -1)
            post_feat = self._obs_out_layers(x)
            post_stat = self._obs_stat_layer(post_feat)
            post_logits = post_stat.reshape(list(post_stat.shape[:-1]) + [self.categorical_dim, self.categorical_dim])
            post_sample = self.straight_through_gradient(post_logits, sample_mode="random_sample", identifier="545")
            post_flattened_sample = self.flatten_sample(post_sample)

        return post_flattened_sample, post_feat

    """AVOID THIS, not necessary for Mamba"""

    # only called when using Transformer
    def predict_next(self, last_flattened_sample, action, log_video=True):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.sequence_model.forward_with_kv_cache(last_flattened_sample, action)
            prior_logits = self.dist_head.forward_prior(dist_feat)

            # decoding
            prior_sample = self.straight_through_gradient(prior_logits, sample_mode="random_sample", identifier="559")
            prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                obs_hat = self.image_decoder(prior_flattened_sample)
            else:
                obs_hat = None
            reward_hat = self.reward_decoder(dist_feat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, dist_feat

    def straight_through_gradient(self, logits, sample_mode="random_sample",
                                  global_step=None, logger=None, identifier="default"):
        # # Track logit statistics if logger is provided
        # if logger is not None and global_step is not None:
        #     # Basic statistics
        #     with torch.no_grad():
        #         if not torch.isnan(logits).any() and not torch.isinf(logits).any():
        #             logger.log(f"Logits/{identifier}/mean", logits.mean().item(), global_step=global_step)
        #             logger.log(f"Logits/{identifier}/std", logits.std().item(), global_step=global_step)
        #             logger.log(f"Logits/{identifier}/min", logits.min().item(), global_step=global_step)
        #             logger.log(f"Logits/{identifier}/max", logits.max().item(), global_step=global_step)

        #             # Log distribution histogram periodically (every 100 steps)
        #             if global_step % 2 == 0:
        #                 # Flatten and convert to numpy for histogram
        #                 flat_logits = logits.detach().cpu().flatten().numpy()
        #                 logger.log(f"Logits/{identifier}/histogram",
        #                            flat_logits, global_step=global_step)
        #         else:
        #             # Record when NaN/Inf occurs
        #             logger.log(f"Logits/{identifier}/nan_inf_detected", 1.0, global_step=global_step)

        # Save problematic tensor for analysis (careful with memory usage)
        # if global_step % 100 == 0:  # Limit frequency
        #     nan_mask = torch.isnan(logits)
        #     inf_mask = torch.isinf(logits)

        #     # Log counts of problematic values
        #     logger.log(f"Logits/{identifier}/nan_count", nan_mask.sum().item(), global_step=global_step)
        #     logger.log(f"Logits/{identifier}/inf_count", inf_mask.sum().item(), global_step=global_step)
        # Safety clamp on logits
        # logits = torch.clamp(logits, min=-50.0, max=50.0)

        # Check for NaN or Inf values for debugging
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("WARNING: NaN or Inf in logits from", identifier)
            # Replace problematic values
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits),
                                 torch.zeros_like(logits), logits)

        dist = OneHotCategorical(logits=logits)
        # dist = Independent(
        #     OneHotDist(logits), 1
        # )
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
            # sample = dist.sample()
        elif sample_mode == "mode":
            sample = dist.mode
            # sample = dist.mode()
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype, device):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            latent_size = (imagine_batch_size, imagine_batch_length + 1, self.stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length + 1, self.hidden_state_dim)
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.sample_buffer = torch.zeros(latent_size, dtype=dtype, device=device)
            self.dist_feat_buffer = torch.zeros(hidden_size, dtype=dtype, device=device)
            self.unmod_dist_feat_buffer = torch.zeros_like(self.dist_feat_buffer)
            self.action_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)

    """IGNORE THIS. Not necessary for Mamba."""

    def imagine_data(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video, logger, global_step):

        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=self.device)
        self.sequence_model.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)
        obs_hat_list = []

        # context
        context_latent = self.encode_obs(sample_obs)

        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i + 1],
                sample_action[:, i:i + 1],
                log_video=log_video
            )
        self.sample_buffer[:, 0:1] = last_latent
        self.dist_feat_buffer[:, 0:1] = last_dist_feat

        # imagine
        for i in range(imagine_batch_length):
            action, _ = agent.sample(
                torch.cat([self.sample_buffer[:, i:i + 1], self.dist_feat_buffer[:, i:i + 1]], dim=-1))
            self.action_buffer[:, i:i + 1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.sample_buffer[:, i:i + 1], self.action_buffer[:, i:i + 1], log_video=log_video)

            self.sample_buffer[:, i + 1:i + 2] = last_latent
            self.dist_feat_buffer[:, i + 1:i + 2] = last_dist_feat
            self.reward_hat_buffer[:, i:i + 1] = last_reward_hat
            self.termination_hat_buffer[:, i:i + 1] = last_termination_hat
            if log_video:
                obs_hat_list.append(last_obs_hat[::imagine_batch_size // 4] * 255)  # uniform sample vec_env

        if log_video:
            img_frames = torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 255)
            img_frames = img_frames.permute(1, 2, 3, 0, 4)
            img_frames = img_frames.reshape(imagine_batch_length, 3, 64, 64 * 4).cpu().float().detach().numpy().astype(
                np.uint8)
            logger.log("Imagine/predict_video", img_frames, global_step=global_step)

        return torch.cat([self.sample_buffer, self.dist_feat_buffer],
                         dim=-1), self.action_buffer, None, None, self.reward_hat_buffer, self.termination_hat_buffer

    def imagine_data2(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                      imagine_batch_size, imagine_batch_length, log_video, logger, global_step):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=self.device)
        # context
        context_latent = self.encode_obs(sample_obs)
        batch_size, seqlen_og, embedding_dim = context_latent.shape
        max_length = imagine_batch_length + seqlen_og

        if self.use_cg:
            if not hasattr(self.sequence_model, "_decoding_cache"):
                self.sequence_model._decoding_cache = None
            self.sequence_model._decoding_cache = update_graph_cache(
                self.sequence_model,
                self.sequence_model._decoding_cache,
                imagine_batch_size,
                seqlen_og,
                max_length,
                embedding_dim,
            )
            inference_params = self.sequence_model._decoding_cache.inference_params
            inference_params.reset(max_length, imagine_batch_size)
        else:
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=imagine_batch_size,
                                               key_value_dtype=torch.bfloat16 if self.use_amp else None)

        def get_hidden_state(samples, action, inference_params):
            decoding = inference_params.seqlen_offset > 0

            if not self.use_cg or not decoding:
                tokens = self.sequence_model.backbone.tokenizer(samples, action)
                quantizer_output, quant_loss = self.causal_model(tokens, global_step, training=False)
                feature_mask = self.causal_model.mask_generator(quantizer_output['quantized'], tokens)
                tokens = tokens * feature_mask
                hidden_state = self.sequence_model(tokens, inference_params=inference_params)
                # hidden_state = self.sequence_model(
                #     samples, action,
                #     inference_params=inference_params,
                    # num_last_tokens=1,
                    # ).logits.squeeze(dim=1)
                # )
            else:
                hidden_state = self.sequence_model._decoding_cache.run(
                    samples, action, inference_params.seqlen_offset
                )
            return hidden_state

        def should_stop(current_token, inference_params):
            if inference_params.seqlen_offset == 0:
                return False
            # if eos_token_id is not None and (current_token == eos_token_id).all():
            #     return True
            if inference_params.seqlen_offset >= max_length:
                return True
            return False

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp and not self.use_cg):
            context_dist_feat = get_hidden_state(context_latent, sample_action, inference_params)
            inference_params.seqlen_offset += context_dist_feat.shape[1]
            context_prior_logits, _, _, _, _ = self.dist_head.forward_prior(context_dist_feat)
            context_prior_sample = self.straight_through_gradient(context_prior_logits)
            context_flattened_sample = self.flatten_sample(context_prior_sample)

            dist_feat_list, sample_list = [context_dist_feat[:, -1:]], [context_flattened_sample[:, -1:]]
            self.sample_buffer[:, 0:1] = context_flattened_sample[:, -1:]
            self.dist_feat_buffer[:, 0:1] = context_dist_feat[:, -1:]
            action_list, old_logits_list = [], []
            i = 0
            while not should_stop(sample_list[-1], inference_params):
                action, logits = agent.sample(
                    torch.cat([self.sample_buffer[:, i:i + 1], self.dist_feat_buffer[:, i:i + 1]], dim=-1))
                action_list.append(action)
                self.action_buffer[:, i:i + 1] = action
                old_logits_list.append(logits)
                dist_feat = get_hidden_state(sample_list[-1], action_list[-1], inference_params)
                dist_feat_list.append(dist_feat)
                self.dist_feat_buffer[:, i + 1:i + 2] = dist_feat
                inference_params.seqlen_offset += sample_list[-1].shape[1]
                # if repetition_penalty == 1.0:
                #     sampled_tokens = sample_tokens(scores[-1], inference_params)
                # else:
                #     logits = modify_logit_for_repetition_penalty(
                #         scores[-1].clone(), sequences_cat, repetition_penalty
                #     )
                #     sampled_tokens = sample_tokens(logits, inference_params)
                #     sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
                prior_logits, _, _, _, _ = self.dist_head.forward_prior(dist_feat_list[-1])
                prior_sample = self.straight_through_gradient(prior_logits)
                prior_flattened_sample = self.flatten_sample(prior_sample)
                sample_list.append(prior_flattened_sample)
                self.sample_buffer[:, i + 1:i + 2] = prior_flattened_sample
                i += 1

            # sample_tensor = torch.cat(sample_list, dim=1)
            # dist_feat_tensor = torch.cat(dist_feat_list, dim=1)
            # action_tensor = torch.cat(action_list, dim=1)
            old_logits_tensor = torch.cat(old_logits_list, dim=1)

            reward_hat_tensor = self.reward_decoder(self.dist_feat_buffer[:, :-1])
            self.reward_hat_buffer = self.symlog_twohot_loss_func.decode(reward_hat_tensor)
            termination_hat_tensor = self.termination_decoder(self.dist_feat_buffer[:, :-1])
            self.termination_hat_buffer = termination_hat_tensor > 0

        if log_video:
            obs_hat = self.image_decoder(self.sample_buffer[::imagine_batch_size // 4]) * 255
            obs_hat = torch.clamp(obs_hat, 0, 255)
            img_frames = obs_hat.permute(1, 2, 3, 0, 4)
            img_frames = img_frames.reshape(imagine_batch_length + 1, 3, 64,
                                            64 * 4).cpu().float().detach().numpy().astype(np.uint8)
            logger.log("Imagine/predict_video", img_frames, global_step=global_step)

        return torch.cat([self.sample_buffer, self.dist_feat_buffer],
                         dim=-1), self.action_buffer, old_logits_tensor, torch.cat(
            [context_flattened_sample, context_dist_feat], dim=-1), self.reward_hat_buffer, self.termination_hat_buffer

    @profile
    def update(self, obs, action, reward, termination, global_step, epoch_step, logger=None):
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with (torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp)):
            # encoding
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            # post_logits = torch.clamp(post_logits, min=-15.0, max=15.0) # May remove this
            sample = self.straight_through_gradient(post_logits, sample_mode="random_sample",
                                                    global_step=global_step, logger=logger, identifier="post_logits")
            flattened_sample = self.flatten_sample(sample)
            # flattened_sample = self.encode_obs(obs)

            # decoding image
            obs_hat = self.image_decoder(flattened_sample)

            # dynamics core
            # if self.model == 'Transformer':
            #     temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
            #     dist_feat = self.sequence_model(flattened_sample, action, temporal_mask)
            # else:
            # dist_feat = self.sequence_model(flattened_sample, action)
            tokens = self.sequence_model.backbone.tokenizer(flattened_sample, action)
            quantizer_output, quant_loss = self.causal_model(tokens, global_step, training=True)
            feature_mask = self.causal_model.mask_generator(quantizer_output['quantized'], tokens)
            tokens = tokens * feature_mask
            dist_feat = self.sequence_model(tokens)

            # Contrastive losses
            code_contrastive_loss = quantizer_output['contrastive_loss']
            causal_world_contrastive_loss = self.world_causal_alignment_loss(dist_feat, quantizer_output['quantized'], temperature=0.1)

            causal_world_loss = self.world_causal_alignment_loss_weight * causal_world_contrastive_loss
            # PREDICTION HEADS (with directed gradient flow)
            # For transition prediction - aloow gradients to modulator but not to causal model internals
            prior_logits, total_tr_loss, aux_loss, diversity_loss, sparsity_loss = self.dist_head.forward_prior(
                dist_feat,
                #quantizer_output['quantized']
            )
            if self.tr_head.use_importance_weighted_moe:
                self.dist_head.update_temperature()  # Update MoE temperature
            # decoding reward and termination with dist_feat
            # reward_hat, re_head_loss = self.reward_decoder(dist_feat,
                                                           # quantizer_output['quantized_re'],
                                                           # quantizer_output['q_re']
                                                           # )

            pred_loss = 0.01 * aux_loss + 0.05 * sparsity_loss + 0.2 * diversity_loss

            # Common loss calculations
            # reward_decoded = self.symlog_twohot_loss_func.decode(reward_hat)  # Convert to scalar values
            # termination_hat = self.termination_decoder(dist_feat)

            # decoding reward and termination with dist_feat
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)
            reward_decoded = self.symlog_twohot_loss_func.decode(reward_hat)  # Convert to scalar values

            """Model losses from causal model"""
            # env loss
            reconstruction_loss = self.mse_loss_func(obs_hat[:batch_size], obs[:batch_size])
            reward_loss = F.mse_loss(reward_decoded, symlog(reward))
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            # dyn-rep loss
            # pred_loss = (self.loss_weights['tr_aux_loss'] * aux_loss) + (
            #             self.loss_weights['tr_sparsity_weight'] * sparsity_loss)
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(),
                                                                               prior_logits[:, :-1])
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:],
                                                                                           prior_logits[:,
                                                                                           :-1].detach())
            # Separate world model losses from causal model losses
            # Added a dynamics loss weight 0.8
            prediction_head_loss = pred_loss
            world_model_loss = reconstruction_loss + reward_loss + termination_loss + dynamics_loss + \
                               0.1 * representation_loss + prediction_head_loss + causal_world_loss
            causal_model_loss = quant_loss + code_contrastive_loss
            # total_loss = reconstruction_loss + reward_loss + termination_loss + dynamics_loss + 0.1 * representation_loss + causal_loss + pred_loss + re_head_loss

            # First backward for world model
            self.scaler.scale(world_model_loss).backward(retain_graph=True)  # Retain graph needs to be true

            # Update world model parameters
            self.scaler.unscale_(self.world_optimizer)
            torch.nn.utils.clip_grad_norm_([p for n, p in self.named_parameters()
                                            if not n.startswith('causal_model')],
                                           max_norm=self.max_grad_norm)
            self.scaler.step(self.world_optimizer)
            self.world_optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler_world.step()
            self.warmup_scheduler_world.dampen()

            # Backward for causal model
            self.scaler.scale(causal_model_loss).backward()
            self.scaler.unscale_(self.causal_optimizer)
            torch.nn.utils.clip_grad_norm_(self.causal_model.parameters(),
                                           max_norm=self.max_grad_norm)
            self.scaler.step(self.causal_optimizer)
            self.causal_optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler_causal.step()
            self.warmup_scheduler_causal.dampen()

            # Update scaler afterboth optimizers have been stepped
            self.scaler.update()

        if (global_step + epoch_step) % self.save_every_steps == 0:  # and global_step != 0:
            sample_obs = torch.clamp(obs[:3, 0, :] * 255, 0, 255).permute(0, 2, 3,
                                                                          1).cpu().detach().float().numpy().astype(
                np.uint8)
            sample_obs_hat = torch.clamp(obs_hat[:3, 0, :] * 255, 0, 255).permute(0, 2, 3,
                                                                                  1).cpu().detach().float().numpy().astype(
                np.uint8)

            concatenated_images = []
            for idx in range(3):
                concatenated_image = np.concatenate((sample_obs[idx], sample_obs_hat[idx]),
                                                    axis=0)  # Concatenate vertically
                concatenated_images.append(concatenated_image)

            # Combine selected images into one image
            final_image = np.concatenate(concatenated_images, axis=1)  # Concatenate horizontally
            height, width, _ = final_image.shape
            scale_factor = 6
            final_image_resized = cv2.resize(final_image, (width * scale_factor, height * scale_factor),
                                             interpolation=cv2.INTER_NEAREST)
            logger.log("Reconstruct/Reconstructed images", [final_image_resized], global_step=global_step)

        return reconstruction_loss.item(), reward_loss.item(), termination_loss.item(), \
            dynamics_loss.item(), dynamics_real_kl_div.item(), representation_loss.item(), \
            representation_real_kl_div.item(), quant_loss.item(), code_contrastive_loss.item(), causal_world_loss.item(), causal_model_loss.item(), world_model_loss.item()

    def world_causal_alignment_loss(self, dist_feat, code_emb, temperature=0.1):
        """
        Implements InfoNCE contrastive loss to align world model hidden states
        with causal model trajectory codes using windowed trajectories

        Args:
            dist_feat: Features from world model sequence [B, T, hidden_dim]
            code_emb: Code embeddings from causal model [B, M, code_dim]
                     (already arranged as batch x windows x features)
            temperature: Temperature parameter for softmax

        Returns:
            InfoNCE loss for alignment
        """
        code_emb = code_emb.detach()
        # 1. Sample windows from world model features using existing function
        B, T, D = dist_feat.shape
        M = code_emb.shape[1]  # Number of windows

        # Use the existing sample_windows function
        windowed_features = self.sample_windows(dist_feat)  # [B*M, W, D]

        # 2. For each window, use the final state as the representation
        window_features = windowed_features[:, -1, :]  # [B*M, D]

        # 3. Project to common space if dimensions differ
        if hasattr(self, 'alignment_projector'):
            window_features = self.alignment_projector(window_features)

        # 4. Reshape window features to match code_emb shape
        window_features = window_features.reshape(B, M, -1)  # [B, M, D]

        # 5. Compute InfoNCE loss for each window position
        total_loss = 0
        for m in range(M):
            # Get features and codes for this window position
            win_feat = window_features[:, m, :]  # [B, D]
            win_code = code_emb[:, m, :]  # [B, code_dim]

            # Normalize for cosine similarity
            win_feat_norm = F.normalize(win_feat, p=2, dim=1)
            win_code_norm = F.normalize(win_code, p=2, dim=1)

            # Compute similarity matrix
            similarity = torch.mm(win_feat_norm, win_code_norm.t()) / temperature  # [B, B]

            # InfoNCE loss - positives are along diagonal
            labels = torch.arange(B, device=dist_feat.device)
            loss = F.cross_entropy(similarity, labels)
            total_loss += loss

        return total_loss / M  # Average across windows

    def sample_windows(self, trajectory):
        """Sample fixed-size windows from a trajectory"""
        B, L, D = trajectory.shape

        # Calculate stride to evenly cover the trajectory
        # if self.stride is None:
        #     if L <= self.window_size:
        #         stride = 1
        #     else:
        #         stride = max(1, (L - self.window_size) // (self.num_windows - 1))
        # else:
        #     stride = self.stride
        stride = max(1, (L - self.causal_model.window_size) // (self.causal_model.num_windows - 1))

        windows = []
        for i in range(self.causal_model.num_windows):
            start_idx = min(i * stride, max(0, L - self.causal_model.window_size))
            windows.append(trajectory[:, start_idx:start_idx + self.causal_model.window_size])

        # Stack along batch dimension to process as independent trajectories
        return torch.cat(windows, dim=0)  # [B*M, W, D]