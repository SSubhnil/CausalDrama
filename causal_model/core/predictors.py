import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class MoETransitionHead(nn.Module):
    def __init__(self, moe_net, hidden_state_dim, hidden_dim, code_dim, conf_dim):
        super().__init__()
        self.moe = moe_net
        self.conf_mask = nn.Parameter(torch.zeros(hidden_dim))
        self.hidden_state_dim = hidden_state_dim
        self.f_conf = nn.Sequential(
            nn.Linear(conf_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Mechanism invariance weight
        self.invariance_weight = nn.Parameter(torch.tensor(1.0)) # Learnable

        # h_world modulation components - scale and shift
        self.scale_mlp = nn.Sequential(
            nn.Linear(code_dim + conf_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid() # [0,1] scaling
        )
        self.shift_mlp = nn.Sequential(nn.Linear(code_dim + conf_dim, hidden_dim),
                                       nn.SiLU(),
                                       nn.Linear(hidden_dim, hidden_dim)
                                       )

        # Orthogonal initialization
        nn.init.orthogonal_(self.scale_mlp[0].weight)
        nn.init.orthogonal_(self.shift_mlp[0].weight)

        # Projection layer for H modulation
        self.proj_w = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_state_dim),
        nn.ReLU()    # Matches F.relu in forward
        )
        nn.init.orthogonal_(self.proj_w[0].weight)
        nn.init.normal_(self.proj_w[0].bias, std=1e-6)

        # Final Projection layer to Next-state logits
        # Output from SparseCodebookMoE is a 1024 projection, but not logits
        self.final_proj = nn.Sequential(nn.Linear(1024, 1024),
                                        nn.LogSoftmax(dim=-1) # For categorical latent
                                        )

        # Sigma annealing parameters
        self.sigma = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.sigma_min = 0.05
        self.sigma_decay = 0.9995
        self.register_buffer('step', torch.tensor(0))

        """Replace with learnable sparsity mask"""
        # Initialize sparse mask - may not be as flexible as a fully learable mask
        nn.init.normal_(self.conf_mask, std=0.01)
        self.conf_mask.register_hook(lambda grad: grad * (torch.abs(grad) > 0.1).float())

    def forward(self, h, code_emb, u):
        """
        total_loss = pred_loss + 0.01*aux_loss + 0.1*conf_sparsity
        """
        clean_out, h_modulated, aux_loss, sparsity_loss = self._forward_impl(h, code_emb, u)

        total_loss = aux_loss + sparsity_loss

        # Annealed noise injection
        if self.training:
            noise = torch.randn_like(h) * self.sigma
            perturbed_out, _, _, _ = self._forward_impl(h + noise, code_emb, u)
            inv_loss = (clean_out - perturbed_out).pow(2).mean() / (clean_out.deatch().pow(2).mean() + 1e-7)
            total_loss += inv_loss

            # Update sigma
            self.step += 1
            self.sigma.data = torch.max(
                torch.tensor(self.sigma_min),
                self.sigma * self.sigma_decay ** (self.step/100)
            )
            # Alternative Linear Decay
            # self.sigma.data = torch.max(
            #     torch.tensor(self.sigma_min),
            #     self.sigma - (0.3-self.sigma_min)/total_steps
            # )

        next_state_logits = self.final_proj(clean_out)  # [B, 1024]

        assert next_state_logits.shape[-1] == 1024, \
            f"Prediction head output dim {next_state_logits.shape[-1]} != 1024"

        return next_state_logits, h_modulated, total_loss, aux_loss, sparsity_loss

    def _forward_impl(self, h, code_emb, u):
        # Concatenate code and confounder
        modulation_input = torch.cat([code_emb, u], dim=-1)

        # Generate scale/shift with gradient gates
        """May not be a good idea to detach here"""
        scale = self.scale_mlp(modulation_input) # Detach world model
        shift = self.shift_mlp(modulation_input)

        # Modulated hidden state
        h_transformed = self.proj_w(h) # W from doc
        h_modulated = scale * h_transformed + shift

        # Shared computation
        moe_out, aux_loss = self.moe(h_modulated, code_emb)

        conf_effect = self.f_conf(u) * torch.sigmoid(self.conf_mask)

        sparsity_loss = torch.mean(torch.abs(self.conf_mask))

        output = moe_out * (1 - self.conf_mask.sigmoid()) + conf_effect

        return output, h_modulated, aux_loss, sparsity_loss

    # def training_step(self, batch):
    #     h, codes, u = batch
    #     pred, (aux_loss, inv_loss) = model(h, codes, u)
    #
    #     pred_loss = F.mse_loss(pred, target)
    #     conf_sparsity = model.conf_mask.abs().mean()
    #     total_loss = pred_loss + 0.01 * aux_loss + 0.1 * inv_loss + 0.05 * conf_sparsity
    #
    #     total_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #     optimizer.step()


class RewardHead(nn.Module):
    def __init__(self, num_codes, code_dim, hidden_dim=128):
        super().__init__()
        # Shared across all reward codes
        self.mlp = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 255) # SymLog discretization
        )
        self.scale_proj = nn.Linear(num_codes, 1, bias=False) # [B, T, KRe] -> [B, T, 1]

    def forward(self, code_emb, q_re):
        """
        code_emb: [B, T, code_dim] from DualVQQuantizer.quantized_re
        q_re: [B, T, KRe] soft code weights
        """
        # Base reward prediction
        base_reward = self.mlp(code_emb) # [B, T, 255]

        # Code-specific scaling
        code_weights = self.scale_proj(q_re.transpose(-1, -2)) # [B, T, 1]
        scaled_reward = base_reward * code_weights.sigmoid()

        # From causal model
        # reward_loss = self.symlog_loss(reward_pred, target_reward) # in world model
        # code_reg = 0.01 * quant_out['q_re'].pow(2).mean()  # Encourage sparse codes
        # total_loss += reward_loss + code_reg

        return scaled_reward

class ImprovedRewardHead(RewardHead):
    def __init__(self, num_codes, code_dim, hidden_state_dim, num_heads=4, hidden_dim=256):
        super().__init__(num_codes, code_dim, hidden_dim)

        # For the h_modulated
        self.state_proj = nn.Sequential(nn.Linear(hidden_state_dim, code_dim),
                                        nn.SiLU())
        # Cross-codebook attention
        self.attn = nn.MultiheadAttention(embed_dim=code_dim*2, num_heads=num_heads,
                                          kdim=code_dim, vdim=code_dim)

        self.code_align = nn.Linear(code_dim, code_dim, bias=False)
        nn.init.orthogonal_(self.code_align.weight)

        # Learnable bins with modulated scaling
        self.bin_centers = nn.Parameter(torch.linspace(-10,10,255)) # Learnable
        self.bin_scaler = nn.Sequential(nn.Linear(hidden_state_dim, 255),
                                        nn.Sigmoid())

    def forward(self, h_modulated, code_emb, q_re):
        """
        h_modulated: [B,T,D_w] - modulated state from transition head
        code_emb: [B,T,D_c] - quantized reward codes
        q_re: [B,T,K] - code weights
        """
        # Project modulated state to code space
        # h_modulated = h_modulated.detach() # Avoids gradient flow into world model,
                                             # but the world model may fail to learn.
        state_proj = self.state_proj(h_modulated) # [B, T, D_c]

        # Create attention query from state-code fusion
        query = torch.cat([state_proj, code_emb], -1) # [B, T, 2D_c]

        # Attend to aligned codebook entries
        attn_out, _ = self.attn(query.transpose(0,1),
                                key=self.code_align(code_emb).transpose(0,1),
                                value=code_emb.transpose(0,1))

        attn_out = attn_out.transpose(0,1)  # [B, T, D_c]

        # State-adaptive bin scaling
        scale = self.bin_scalar(h_modulated) # [B, T, 255]
        scaled_bins = self.bin_centers * scale



        # Code-dependent logits
        logits = self.mlp(attn_out) # [B, T, 255]
        return (logits.softmax(-1) * scaled_bins).sum(-1) # Differentiable bins

        # Add to loss function
        #
        # total_loss += 0.05 * align_loss

class TerminationPredictor(nn.Module):
    def __init__(self, hidden_state_dim, hidden_units, act='SiLU', layer_num=2,
                 dropout=0.1, dtype=None, device=None):
        super().__init__()
        act_fn = getattr(nn, act)

        # Optional hidden state projection (if dimensions need adjustment)
        self.h_proj = nn.Sequential(nn.Linear(hidden_state_dim,
                                              hidden_units),
                                    nn.LayerNorm(hidden_units) # Normalization for stability
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
            layers.append(nn.Dropout(dropout)) # Add dropout for regularization

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_units, 1, dtype=dtype, device=device)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogoanl_(m.weight, gain=1.0)
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
        termination = termination.squeeze(-1) #  Remove last dimension

        return termination
