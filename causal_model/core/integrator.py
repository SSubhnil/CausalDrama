import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from line_profiler import profile
from .encoder import CausalEncoder, ConvolutionalTrajectoryEncoder
from .quantizer import DualVQQuantizer, TrajectoryQuantizer
from .networks import CausalMaskGenerator


class CausalModel(nn.Module):
    def __init__(self, params, device, action_dim, stoch_dim, d_model):
        super().__init__()
        self.device = device
        self.hidden_state_dim = d_model
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.code_dim_tr = params.Models.CausalModel.TrCodeDim
        self.code_dim_re = params.Models.CausalModel.ReCodeDim
        self.num_codes_tr = params.Models.CausalModel.NumCodesTr
        self.num_codes_re = params.Models.CausalModel.NumCodesRe
        self.hidden_dim = params.Models.CausalModel.HiddenDim
        self.use_confounder = params.Models.CausalModel.UseConfounder

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
        # self.causal_encoder = CausalEncoder(hidden_state_dim=self.hidden_state_dim,
        #                                     action_dim=self.action_dim,
        #                                     stoch_dim=self.stoch_dim,
        #                                     tr_proj_dim=self.encoder_params.TransProjDim,
        #                                     re_proj_dim=self.encoder_params.RewProjDim,
        #                                     hidden_dim=self.encoder_params.HiddenDim,
        #                                     combined_input_dim=combined_input_dim,
        #                                     embedding_mode=self.encoder_params.Embedding)
        self.causal_encoder = ConvolutionalTrajectoryEncoder(input_dim=combined_input_dim,
                                                             hidden_dim=self.encoder_params.HiddenDim,
                                                             projection_dim=self.encoder_params.TransProjDim,
                                                             embedding_mode=self.encoder_params.Embedding)
        # For mini-trajectories or windowed
        self.num_windows = self.encoder_params.NumWindows
        self.window_size = self.encoder_params.WindowSize
        self.stride = self.encoder_params.Stride
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
        # self.quantizer = DualVQQuantizer(code_dim_tr=self.code_dim_tr,
        #                                  code_dim_re=self.code_dim_re,
        #                                  num_codes_tr=self.num_codes_tr,
        #                                  num_codes_re=self.num_codes_re,
        #                                  beta_tr=self.quantizer_params.BetaTransition,
        #                                  beta_re=self.quantizer_params.BetaReward,
        #                                  tr_temperature=self.quantizer_params.TransitionTemp,
        #                                  tr_min_temperature=self.quantizer_params.TransitionMinTemperature,
        #                                  tr_anneal_factor=self.quantizer_params.TransitionAnnealFactor,
        #                                  re_temperature=self.quantizer_params.RewardTemp,
        #                                  re_min_temperature=self.quantizer_params.RewardMinTemperature,
        #                                  re_anneal_factor=self.quantizer_params.RewardAnnealFactor,
        #                                  normalize=self.quantizer_params.NormalizedInputs,
        #                                  coupling=self.quantizer_params.Coupling,
        #                                  sparsity_weight=self.quantizer_params.SparsityWeight,  # Coupling sparsity
        #                                  lambda_couple=self.quantizer_params.LambdaCouple,
        #                                  hidden_dim=self.hidden_dim,
        #                                  use_cdist=self.quantizer_params.UseCDist
        #                                  )

        self.quantizer = TrajectoryQuantizer(num_codes=self.num_codes_tr,
                                             code_dim=self.code_dim_tr,
                                             beta=self.quantizer_params.BetaTransition,
                                             initial_temp=self.quantizer_params.TransitionTemp,
                                             use_cdist=self.quantizer_params.UseCDist,
                                             normalize=self.quantizer_params.NormalizedInputs)

        self.mask_generator = CausalMaskGenerator(hidden_state_dim=self.hidden_state_dim,
                                                  code_dim=self.code_dim_tr  # For 1D action after unsqueeze
                                                  )

    @profile
    def forward(self, h, global_step, training=False):
        # Projection of trajectories into h [B, trajectory_projection]
        """Process trajectory into multiple windows, encode and quantize them"""
        # 1. Sample windows
        windowed_trajectories = self.sample_windows(h)  # [B*M, W, D]

        # 2. Encode each window as an independent trajectory
        encoded_windows = self.causal_encoder(windowed_trajectories)  # [B*M, code_dim]

        # 3. Quantize the encoded windows
        quantizer_output, quant_loss = self.quantizer(encoded_windows, global_step, training)

        # 4. Reshape outputs to separate batch and window dimensions
        B = h.shape[0]
        M = self.num_windows

        # Reshape quantized outputs to [B, M, code_dim]
        reshaped_output = {}
        for key, value in quantizer_output.items():
            if isinstance(value, torch.Tensor):
                # Skip reshaping scalar values or None values
                if value.dim() > 0:  # Check if it's not a scalar
                    new_shape = [B, M] + list(value.shape[1:])
                    reshaped_output[key] = value.reshape(new_shape)
                else:
                    reshaped_output[key] = value
            else:
                reshaped_output[key] = value

        # Different paths for training and inference
        if training:
            self.anneal_quantizer_temperature(global_step)
            return reshaped_output, quant_loss

        else:  # Inference
            return reshaped_output, None

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
        if self.stride is None:
            if L <= self.window_size:
                stride = 1
            else:
                stride = max(1, (L - self.window_size) // (self.num_windows - 1))
        else:
            stride = self.stride

        windows = []
        for i in range(self.num_windows):
            start_idx = min(i * stride, max(0, L - self.window_size))
            windows.append(trajectory[:, start_idx:start_idx + self.window_size])

        # Stack along batch dimension to process as independent trajectories
        return torch.cat(windows, dim=0)  # [B*M, W, D]

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