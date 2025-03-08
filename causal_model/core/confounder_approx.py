import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.set_float32_matmul_precision('high')  # Enable TF32 on Ampere

"""
Posterior computation only.
See end_dec.py for regularized confounder prior.
ALternately update the prior and posterior
Freeze prior when training posterior
Unfreeze prior every 3 steps
# First 1000 steps: 
    # optimizer.zero_grad()
    # code_loss = F.mse_loss(post_code_emb, prior_code_emb.detach())
    # code_loss.backward()
"""

class PosteriorHypernet(nn.Module):
    """Generates posterior network weights conditioned on the code IDs"""
    def __init__(self, code_dim, hidden_dim, out_dim):
        super().__init__()
        # Factorized weight generation
        self.w1_proj = nn.Linear(code_dim, 2*hidden_dim)
        self.w2_proj = nn.Linear(code_dim, 2*out_dim)

        # Kaiming initialization optimized for SiLU
        nn.init.kaiming_normal_(self.w1_proj.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.w2_proj.weight, nonlinearity='linear')

    def forward(self, x, code_emb):
        # Split weights for parallel computation
        w1 = self.w1_proj(code_emb).unflatten(-1, (2, self.hidden_dim))
        w2 = self.w2_proj(code_emb).unflatten(-1, (2, self.out_dim))

        # Fused operations
        x = F.silu(F.linear(x, w1[0], None) + w1[1])
        return F.linear(x, w2[0], None) + w2[1]

class ConfounderPosterior(nn.Module):
    """
    We implement Part 2B of report
    Components:
        - Code-conditioned variational posterior Q(u | h, c)
        - Hyper-network for parameter generation
        - Affine transformations code specific \phi_c(u)
    """

    def __init__(self, code_dim: int, conf_dim: int, num_codes: int,
                 embed_dim: int = None, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.conf_dim = conf_dim
        self.num_codes = num_codes

        self.embed_dim = code_dim  # Should match code embedding dimension
        if embed_dim is not None:
            self.embed_dim = embed_dim

        # Code embedding layer
        self.h_proj = nn.Linear(self.code_dim, self.hidden_dim)
        self.code_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.code_embed = nn.Embedding(self.num_codes, self.code_dim) # nn.Embedding(self.num_codes, self.embed_dim)
        nn.init.orthogonal_(self.code_embed.weight) # Preserve code distances

        # Posterior networks
        self.confounder_post_mu_net = PosteriorHypernet(self.code_dim, self.hidden_dim, self.conf_dim)

        self.confounder_post_logvar_net = PosteriorHypernet(self.code_dim, self.hidden_dim, self.conf_dim)

        # Helps pre-compute frequent calculations for reparameterization
        self._sqrt_2pi = torch.sqrt(torch.tensor(2 * torch.pi))
        self._eps = torch.finfo(torch.float32).eps


    def forward(self, h: torch.Tensor, quantized_tr, mu_prior, logvar_prior):
        """
        Args:
            h: Hidden state from world model [B, D]
            code_ids: Discrete code indices [B]

        Returns:
            u_transformer: Confounder after code-specific affine [B, conf_dim]
            kl_loss: D_KL(Q(u|h,c) || P(u|h)) + regularization
        """
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Call confounder prior network

            # Hypernetwork based posterior
            #code_emb = self.code_embed(code_ids)  # [B, embed_dim]
            # Maintains end-to-end gradient flow from posterior -> codebook (with actual code vectors)
            code_emb = quantized_tr # [B, T, D] from quantizer
            mu_post = self.confounder_post_mu_net(h, code_emb)
            logvar_post = self.confounder_post_logvar_net(h, code_emb)

            # Reparameterization trick
            u_post = self.reparameterize(mu_post, logvar_post)  # ~15% faster on CUDA

            kl_loss = self.gaussian_KL(mu_post, logvar_post, mu_prior, logvar_prior)

        # Preserve precision for embeddings
        self.code_embed = self.code_embed.to(torch.float32)
        return u_post, kl_loss

    # REDUNDANT due to non-use of scale and shift
    def regularization_loss_2(self, lamdba_weight=0.01):
        # Encourage sparse affine transformations
        scale_reg = torch.mean(torch.abs(self.affine_scale.weight))
        shift_reg = torch.mean(torch.abs(self.affine_shift.weight))
        return lamdba_weight * (scale_reg + shift_reg)

    # Main regularizer
    def regularization_loss(self):
        # Hypernetwork parameter regularization
        hyper_params = list(self.confounder_post_mu_net.parameters()) + \
                       list(self.confounder_post_logvar_net.parameters())

        l2_reg = torch.sum(torch.stack([torch.norm(p) for p in hyper_params]))

        # Code embedding sparsity
        code_sparsity = torch.mean(torch.abs(self.code_embed.weight))

        return l2_reg, code_sparsity

    def gaussian_KL(self, mu_post: torch.Tensor, logvar_post: torch.Tensor,
                    mu_prior: torch.Tensor, logvar_prior: torch.Tensor):
        return 0.5 * (
                (logvar_prior - logvar_post) +
                (torch.exp(logvar_post) + (mu_post - mu_prior) ** 2) /
                torch.exp(logvar_prior) - 1
        ).sum(1).mean()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar).clamp_min(self._eps)
        return mu + std * torch.randn_like(std)

# Initialized for confounder prior
class ConfounderPrior(nn.Module):
    def __init__(self, num_codes, code_dim, conf_dim, hidden_state_proj_dim, momentum=0.95):
        super().__init__()

        self.code_embed = nn.Embedding(num_codes, code_dim) # Shared embeddings
        self.register_buffer('code_emb_momentum', torch.empty_like(self.code_embed.weight))
        nn.init.orthogonal_(self.code_emb_momentum)  # Independent initialization
        self.momentum = momentum
        self.h_proj_dim = hidden_state_proj_dim

        # Physics-inspired parameter bounds
        self.mu_net = nn.Sequential(nn.Linear(self.h_proj_dim + code_dim, conf_dim),
                                    nn.LayerNorm(conf_dim),
                                    nn.Linear(conf_dim, conf_dim),
                                    nn.Tanh() # Constrained output
                                    )
        nn.init.kaiming_normal_(self.mu_net[0].weight, nonlinearity='linear')  # First linear layer
        # Initialize final layer for natural parameter bounds
        with torch.no_grad():
            self.mu_net[2].weight.data.uniform_(-0.01, 0.01)  # Small magnitude
            self.mu_net[2].bias.data.zero_()  # Center outputs
        self.register_buffer('mu_bounds', torch.tensor([-3.0, 3.0]))

        self.logvar_net = nn.Sequential(
            nn.Linear(self.h_proj_dim + code_dim, conf_dim),
            nn.Hardtanh(min_val=math.log(0.1), max_val=math.log(2.0))
        )


    def forward(self, h: torch.Tensor, code_ids: torch.Tensor = None):
        # Update momentum codebook
        with torch.no_grad():
            self.code_emb_momentum = (self.momentum * self.code_emb_momentum +
                                      (1 - self.momentum) * self.code_embed.weight.detach())

        # Handle 3D input (batch, sequence, features)
        if h.dim() == 3 and code_ids is not None:
            batch_size, seq_len = h.shape[0], h.shape[1]

            # Reshape code_ids to match sequence length
            if code_ids.dim() == 2 and code_ids.shape[1] != seq_len:
                # Take only the first seq_len indices
                code_ids = code_ids[:, :seq_len]

        # Ensure indices are in bounds
        max_idx = self.code_emb_momentum.size(0) - 1
        code_ids = torch.clamp(code_ids, 0, max_idx)

        # Lookup embeddings
        code_feat = self.code_emb_momentum[code_ids]

        # Concatenate along feature dimension
        combined = torch.cat([h, code_feat], dim=-1)

        # Generate outputs
        mu = self.mu_net(combined).clamp(*self.mu_bounds)
        logvar = self.logvar_net(combined)

        return mu, logvar

    def prior_regularization(self, mu):
        """Kinetic energy constraint"""
        energy = torch.norm(mu, dim=1).mean()
        return torch.relu(energy - 3.0)  # [1][3]

    def add_new_code(self, prototype_emb):
        """Initialization via next existing code"""
        with torch.no_grad():
            sim = F.cosine_similarity(prototype_emb, self.code_emb.weight)
            new_emb = 0.5*(self.code_emb.weight[sim.argmax()] + prototype_emb)
            self.code_emb.weight.data[-1] = new_emb

    # Slightly better than before
    def add_new_code_2(self, prototype_emb, prior_std=0.1):
        with torch.no_grad():
            # Sample perturbed prototypes
            proto_noise = prior_std * torch.randn_like(prototype_emb)
            perturbed_proto = prototype_emb + proto_noise

            # Find nearest in noise-robust space
            sim = F.cosine_similarity(perturbed_proto, self.code_emb.weight)
            new_emb = 0.7 * self.code_emb.weight[sim.argmax()] + 0.3 * perturbed_proto

    def code_alignment_loss(self):
        """Tries to match momentum vs actual embeddings"""
        return F.mse_loss(self.code_emb_momentum, self.code_embed.weight.detach())

"""
Few-shot phase: detecting new confounder

for batch in dataloader:
    # 1. Existing codes forward
    u_post, kl_loss = posterior(batch.h, batch.code_ids)

    # 2. Regularization terms
    prior_reg = prior.prior_regularization(mu_prior)
    post_reg = posterior.regularization_loss()

    # 3. Combined loss
    total_loss = kl_loss + 0.1 * prior_reg + 0.01 * post_reg

    # 4. New code injection (few-shot phase)
    if detect_new_confounder(batch):
        prototype = compute_prototype_emb(batch)
        prior.add_new_code(prototype)
        align_code_embeddings(prior, posterior)
"""


class ConfounderDetector(nn.Module):
    def __init__(self, causal_model, conf_dim):
        # Shared with causal model
        self.codebook = causal_model.code_emb
        self.register_buffer('residual_mean', torch.zeros(conf_dim))
        self.register_buffer('residual_var', torch.ones(conf_dim))

    def update_baseline(self, residuals):
        # EWMA for residual statistics
        self.residual_mean = 0.9 * self.residual_mean + 0.1 * residuals.mean(0)
        self.residual_var = 0.9 * self.residual_var + 0.1 * residuals.var(0)

    def __call__(self, residuals, kl_div):
        # Mahalanobis distance for residual anomalies
        standardized = (residuals - self.residual_mean) / self.residual_var.sqrt()
        anomaly_score = standardized.pow(2).sum(-1).sqrt()

        # KL-based mechanism breakdown detection
        kl_zscore = (kl_div - kl_div.mean()) / kl_div.std()

        # Combined detection criteria [Design Doc Eq.38]
        return (anomaly_score > 3.0) & (kl_zscore > 2.0)


class ConfounderMonitor:
    def __enter__(self):
        self.active = True

    def __exit__(self, *args):
        self.active = False
        # Finalize new code integration
        self.align_codebooks()


"""
detector = ConfounderDetector(causal_model)

for batch in dataloader:
    # Existing forward pass
    u_post, kl_loss = posterior(batch.h, batch.code_ids, prior)

    # Residual computation [Design Doc §9.2.5]
    pred_z = world_model(u_post)
    residuals = F.mse_loss(pred_z, batch.z, reduction='none')

    # Update baseline statistics
    detector.update_baseline(residuals)

    # Detect new confounders
    is_new_confounder = detector(residuals, kl_loss)

    if is_new_confounder.any():
        # Prototype extraction
        prototype_emb = causal_model.encode_prototype(
            batch.h[is_new_confounder],
            batch.z[is_new_confounder]
        )

        # Add new code to both prior/posterior
        prior.add_new_code(prototype_emb)
        posterior.code_embed.weight.data[-1] = prior.code_emb.weight[-1].detach()

"""


