from .networks import ImportanceWeightedMoE, SparseCodebookMoE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from line_profiler import profile


class StateModulator(nn.Module):
    """Dedicated module for hidden state modulation based on causal factors."""

    def __init__(self, hidden_state_dim, hidden_dim, code_dim, conf_dim, compute_invariance_loss=False):
        super().__init__()
        self.compute_invariance_loss = compute_invariance_loss
        # Projection layer for H modulation
        self.proj_w = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.ReLU()
        )
        nn.init.orthogonal_(self.proj_w[0].weight)
        nn.init.normal_(self.proj_w[0].bias, std=1e-6)

        # Scale and Shift MLPs
        # h_world modulation components - scale and shift
        self.scale_mlp = nn.Sequential(
            nn.Linear(code_dim + conf_dim, hidden_state_dim),
            nn.SiLU(),
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.Sigmoid()  # [0,1] scaling
        )
        self.shift_mlp = nn.Sequential(nn.Linear(code_dim + conf_dim, hidden_state_dim),
                                       nn.SiLU(),
                                       nn.Linear(hidden_state_dim, hidden_state_dim)
                                       )

        # Orthogonal initialization
        nn.init.orthogonal_(self.scale_mlp[0].weight)
        nn.init.orthogonal_(self.shift_mlp[0].weight)
        # Mechanism invariance weight
        self.invariance_weight = nn.Parameter(torch.tensor(1.0))  # Learnable

        # Sigma annealing parameters
        self.sigma = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.sigma_min = 0.05
        self.sigma_decay = 0.9995
        self.register_buffer('step', torch.tensor(0))

    @profile
    def forward(self, h, code_emb, u, global_step):
        if h.shape[1] != code_emb.shape[1]:
            if h.shape[1] < code_emb.shape[1]:
                # Extract the last sequence element from code_emb and u
                if h.shape[1] == 1:
                    # Most common case: h has a single sequence element
                    code_emb = code_emb[:, -1:, :]  # Take last element and keep dim
                    u = u[:, -1:, :]  # Same for u
            # No need for else case as we're focusing on adapting code_emb
        # print(f"STATE MODULATOR INPUT: h={h.shape}, code_emb={code_emb.shape}, u={u.shape}")
        # Concatenate code and confounder
        if global_step < 1033:
            code_emb_stable = code_emb.detach()  # Blocks gradients to codebook
        else:
            code_emb_stable = code_emb
        modulation_input = torch.cat([code_emb_stable, u], dim=-1)

        # Generate scale/shift with gradient gates
        """May not be a good idea to detach here"""
        scale = torch.clamp(self.scale_mlp(modulation_input), min=0.1, max=10.0)
        shift = torch.clamp(self.shift_mlp(modulation_input), min=-10.0, max=10.0)

        # Verify dimensions before applying transformation
        assert h.shape[0] == scale.shape[0], f"Batch dim mismatch: h={h.shape[0]}, scale={scale.shape[0]}"
        assert h.shape[1] == scale.shape[1], f"Seq dim mismatch: h={h.shape[1]}, scale={scale.shape[1]}"

        # Modulated hidden state
        h_transformed = self.proj_w(h)  # W from doc

        # In StateModulator.forward method
        # print(f"h shape: {h.shape}, device: {h.device}")
        # print(f"code_emb shape: {code_emb.shape}, device: {code_emb.device}")
        # print(f"weight shape: {self.proj_w[0].weight.shape}, device: {self.proj_w[0].weight.device}")

        h_modulated = scale * h_transformed + shift

        # Compute invariance loss if requested (during training)
        inv_loss = 0.0
        if self.compute_invariance_loss:
            noise = torch.randn_like(h) * self.sigma
            h_transformed_noisy = self.proj_w(h + noise)
            h_modulated_noisy = scale * h_transformed_noisy + shift

            # Invariance loss: modulated output should be similar despite input noise
            # inv_loss = (h_modulated - h_modulated_noisy).pow(2).mean() / (h_modulated.detach().pow(2).mean()
            #   + 1e-7)
            inv_loss = F.smooth_l1_loss(h_modulated, h_modulated_noisy)  # Stable invariance loss
            # Update sigma for annealing
            self.step += 1
            self.sigma.data = torch.max(
                torch.tensor(self.sigma_min),
                self.sigma * self.sigma_decay ** (self.step / 100)
            )
        # print(f"STATE MODULATOR OUTPUT: h_modulated={h_modulated.shape}")

        return h_modulated, inv_loss


class MoETransitionHead(nn.Module):
    def __init__(self, hidden_state_dim, hidden_dim, code_dim, conf_dim, num_experts, top_k, codebook_data,
                 compute_inv_loss=False,
                 use_importance_weighted_moe=False, slicing=True, use_simple_mlp=False, importance_reg=0.01):
        super().__init__()

        # Choose between SparseCodeookMoE or FeatureImportanceWeightedMoE
        self.use_simple_mlp = use_simple_mlp
        self.use_importance_weighted_moe = use_importance_weighted_moe
        self.hidden_state_dim = hidden_state_dim
        self.compute_inv_loss = compute_inv_loss

        # Add confounder integration components
        # self.hidden_proj = nn.Linear(hidden_state_dim, hidden_state_dim)
        # self.conf_proj = nn.Linear(conf_dim, hidden_state_dim)

        # Integration layer
        # self.integration = nn.Sequential(
        #     nn.Linear(hidden_state_dim * 2, hidden_state_dim),
        #     nn.SiLU(),
        #     nn.Linear(hidden_state_dim, hidden_state_dim)
        # )

        if self.use_simple_mlp:
            # Simple MLP that takes concatenated h and u inputs
            self.simple_mlp = nn.Sequential(
                nn.Linear(hidden_state_dim + conf_dim, hidden_dim * 2),
                nn.SiLU(),
                # nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, 1024),
                # nn.Tanh()  # Bounded output for stability
            )

            # Initialize weights
            for m in self.simple_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            # Create ImportanceWeightedMoE with parameters from original MOE
            if self.use_importance_weighted_moe:
                self.moe = ImportanceWeightedMoE(
                    num_experts=num_experts,
                    hidden_state_dim=hidden_state_dim,
                    hidden_dim=hidden_dim,
                    code_dim=code_dim,
                    conf_dim=conf_dim,
                    codebook_data=codebook_data,
                    slicing=slicing,
                    top_k=top_k,
                    importance_reg=importance_reg
                )
            else:
                # Use the original MoE
                self.moe = SparseCodebookMoE(num_experts=self.num_experts,
                                             hidden_dim=self.hidden_dim,
                                             code_dim=self.code_dim,
                                             quantizer=self.quantizer,
                                             top_k=self.top_k,
                                             )

        # Confounding effect module
        # self.conf_mask = nn.Parameter(torch.zeros(hidden_dim))
        self.conf_mask = nn.Sequential(
            nn.Linear(conf_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1024),  # Changed from hidden_dim to 1024
            nn.Sigmoid()
        )
        self.f_conf = nn.Sequential(
            nn.Linear(conf_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1024)  # Changed from hidden_dim to 1024
        )

        # Final Projection layer to Next-state logits
        # Output from SparseCodebookMoE is a 1024 projection, but not logits
        self.final_proj = nn.Sequential(nn.Linear(1024, 1024 * 2),
                                        # or 2048 if output = torch.cat([moe_out, conf_effect], dim=-1)
                                        nn.Mish(),
                                        nn.Linear(1024 * 2, 1024),
                                        # nn.Tanh()  # Maintains bounded outputs while allowing density learning
                                        )

        """Replace with learnable sparsity mask"""
        # Initialize sparse mask - may not be as flexible as a fully learable mask
        # Initialize only the Linear layer's weights
        nn.init.normal_(self.conf_mask[0].weight, std=0.01)
        nn.init.zeros_(self.conf_mask[0].bias)  # Also initialize bias if needed
        # Register hook on the Linear layer's weight parameter specifically
        # self.conf_mask[0].weight.register_hook(lambda grad: grad * (torch.abs(grad) > 0.1).float())
        # self.conf_mask[0].weight.register_hook(lambda grad: grad * torch.sigmoid(10 * torch.abs(grad)))
        # More stable version lambda grad: grad * torch.clamp(torch.abs(grad), min=0.01, max=1.0)

    @profile
    def forward(self, h, code_emb, u):
        # Get stable code embeddings
        code_emb_stable = code_emb  # .detach()

        # Integrate confounder with hidden state
        # Concatenate along feature dimension
        combined = torch.cat([h, u], dim=-1)
        # integrated_h = self.integration(combined)
        if self.use_simple_mlp:
            # Pass through the simple MLP
            next_state_logits = self.simple_mlp(combined)

            # Return dummy values for losses to match MoE return signature
            # These will be ignored in the loss computation if MLP mode is active
            total_loss = 0.0
            aux_loss = 0.0
            diversity_loss = 0.0
            sparsity_loss = 0.0
        # Add residual connection to preserve original information
        # integrated_h = integrated_h + h
        else:
            # Process through MoE
            moe_out, aux_loss, diversity_loss, importance_stats = self.moe(combined, code_emb_stable)
            # moe_out = torch.clamp(moe_out, -100, 100)  # Add reasonable bounds

            # Create modulation input
            # modulation_input = torch.cat([code_emb_stable, u], dim=-1)

            # Calculate confounding effect
            # f_conf_out = self.f_conf(u)  # [10, 128, 512]
            # mask_output = self.conf_mask(u)  # [10, 128, 512]
            # conf_effect = f_conf_out * mask_output  # Element-wise multiplication

            # Compute sparsity loss
            # sparsity_loss = torch.abs(torch.mean(mask_output) - 0.3) # Added -0.3 to control sparsity

            # Combine outputs
            # output = moe_out * (1 - mask_output) + conf_effect
            # output = torch.cat([moe_out, conf_effect], dim=-1) # final_proj input is 2048 for this
            # output = moe_out + conf_effect # Change final_proj input dim to 1024 to use this

            # Final projection to next state logits
            next_state_logits = self.final_proj(moe_out)

            # Calculate regularization loss for integration network
            # Target ~30% activation of the integration weights
            # mask_values = torch.sigmoid(self.integration[2].weight.mean(dim=1))
            # sparsity_loss = torch.abs(torch.mean(mask_values) - 0.3)
            sparsity_loss = 0.0  # Remove this if using integration

            # Compute total loss
            total_loss = aux_loss + sparsity_loss + diversity_loss

        assert next_state_logits.shape[-1] == 1024, \
            f"Prediction head output dim {next_state_logits.shape[-1]} != 1024"

        return next_state_logits, total_loss, aux_loss, diversity_loss, sparsity_loss

    def get_importance_weights(self):
        """Return the current importance weights for analysis"""
        if self.use_importance_weighted_moe:
            temperature = torch.clamp(self.moe.importance_temperature, min=0.1, max=5.0)
            weights = [F.softmax(self.moe.feature_importance[i] / temperature, dim=0).detach().cpu()
                       for i in range(self.moe.num_experts)]
            return weights
        return None


class RewardHead(nn.Module):
    def __init__(self, num_codes, code_dim, hidden_dim=128):
        super().__init__()
        # Shared across all reward codes
        self.mlp = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 255)  # SymLog discretization
        )

    def forward(self, h_modulated, code_emb, q_re):
        """
        code_emb: [B, T, code_dim] from DualVQQuantizer.quantized_re
        q_re: [B, T, KRe] soft code weights
        """
        # Base reward prediction
        base_reward = self.mlp(code_emb)  # [B, T, 255]

        # Code-specific scaling
        code_weights = self.scale_proj(q_re.transpose(-1, -2))  # [B, T, 1]
        scaled_reward = base_reward * code_weights.sigmoid()

        # From causal model
        # reward_loss = self.symlog_loss(reward_pred, target_reward) # in world model
        # code_reg = 0.01 * quant_out['q_re'].pow(2).mean()  # Encourage sparse codes
        # total_loss += reward_loss + code_reg

        return scaled_reward


class ImprovedRewardHead(RewardHead):
    def __init__(self, num_codes, code_dim, hidden_state_dim, hidden_dim=256,
                 align_weight=0.05, code_reg_weight=0.01):
        super().__init__(num_codes, code_dim, hidden_dim)
        self.align_weight = align_weight
        self.code_reg_weight = code_reg_weight

        # Project modulated state to code space
        self.state_proj = nn.Sequential(
            nn.Linear(hidden_state_dim, code_dim),
            nn.SiLU()
        )

        # Code alignment (preserved from original)
        self.code_align = nn.Linear(code_dim, code_dim, bias=False)
        nn.init.orthogonal_(self.code_align.weight)

        # Feature fusion MLP (replaces attention mechanism)
        self.feature_fusion = nn.Sequential(
            nn.Linear(code_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, code_dim)
        )

        # Bin scaling (preserved from original)
        self.bin_centers = nn.Parameter(torch.linspace(-10, 10, 255))
        self.bin_scaler = nn.Sequential(
            nn.Linear(hidden_state_dim, 255),
            nn.Sigmoid()
        )

        # Final prediction MLP
        self.mlp = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 255)
        )

    def forward(self, h_modulated, code_emb, q_re):
        """
        h_modulated: [B,T,D_w] - modulated state from transition head
        code_emb: [B,T,D_c] - quantized reward codes
        q_re: [B,T,K] - code weights
        """
        # Project modulated state to code space
        state_proj = self.state_proj(h_modulated)

        # Apply code alignment
        aligned_code_emb = self.code_align(code_emb)

        # Calculate alignment loss (same as original)
        align_loss = F.mse_loss(state_proj, aligned_code_emb)

        # Code sparsity regularization (same as original)
        code_reg = q_re.pow(2).mean() * self.code_reg_weight

        # Fuse features with simple concatenation + MLP instead of attention
        fused_features = torch.cat([state_proj, aligned_code_emb], dim=-1)
        fused_features = self.feature_fusion(fused_features)

        # Generate bin scaling (same as original)
        scale = self.bin_scaler(h_modulated)
        scaled_bins = self.bin_centers * scale

        # Final prediction
        logits = self.mlp(fused_features)

        # Total head loss (same as original)
        total_head_loss = (align_loss * self.align_weight) + code_reg

        return logits, total_head_loss


class TerminationPredictor(nn.Module):
    def __init__(self, hidden_state_dim, hidden_units, act='SiLU', layer_num=2,
                 dropout=0.1, dtype=None, device=None):
        super().__init__()
        act_fn = getattr(nn, act)

        # Optional hidden state projection (if dimensions need adjustment)
        self.h_proj = nn.Sequential(nn.Linear(hidden_state_dim,
                                              hidden_units),
                                    nn.LayerNorm(hidden_units)  # Normalization for stability
                                    )

        # Create a backbone with dynamic number of layers
        layers = []
        for i in range(layer_num):
            inp_dim = hidden_units if i > 0 else hidden_units
            layers.append(nn.Linear(inp_dim, hidden_units, bias=True, dtype=dtype,
                                    device=device))
            # Using LayerNorm instead of RMSNorm for compatibility
            layers.append(nn.LayerNorm(hidden_units, dtype=dtype))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))  # Add dropout for regularization

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_units, 1, dtype=dtype, device=device)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_modulated):
        """
        Args:
            h_modulated: Modulated hidden state from transition predictor [B, T, Hidden_state_dim]

        Returns:
            termination: Termination probabilities [B, T]
        """
        # Project the modulated hidden state
        h_proj = self.h_proj(h_modulated)

        # Process h_modulated through  backbone
        feat = self.backbone(h_proj)

        # Final prediction
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # Remove last dimension

        return termination
