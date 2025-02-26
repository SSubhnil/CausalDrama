import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from encoder import CausalEncoder_Confounder
torch.set_float32_matmul_precision('high')  # Enable TF32 on Ampere

"""
Posterior computation only.
See end_dec.py for regularized confounder prior.
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

    def __init__(self, code_dim: int, conf_dim: int, num_codes: int, params,
                 embed_dim: int = None, hidden_dim: int = 256):
        super().__init__()
        self.encoder = CausalEncoder_Confounder(params)

        self.embed_dim = embed_dim or code_dim // 2 # Default to code_dim/2
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.conf_dim = conf_dim
        self.num_codes = num_codes

        # Code embedding layer
        self.h_proj = nn.Linear(self.code_dim, self.hidden_dim)
        self.code_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.code_embed = nn.Embedding(self.num_codes, self.embed_dim)
        nn.init.orthogonal_(self.code_embed.weight) # Preserve code distances

        # Posterior networks
        self.confounder_post_mu_net = PosteriorHypernet(self.code_dim, self.hidden_dim, self.conf_dim)

        self.confounder_post_logvar_net = PosteriorHypernet(self.code_dim, self.hidden_dim, self.conf_dim)

        # Helps pre-compute frequent calculations for reparameterization
        self._sqrt_2pi = torch.sqrt(torch.tensor(2 * torch.pi))
        self._eps = torch.finfo(torch.float32).eps


    def forward(self, h: torch.Tensor, code_ids: torch.Tensor):
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
            from_confounder_prior = self.encoder(h, code_ids)
            mu_prior = from_confounder_prior['confounder_mu']
            logvar_prior = from_confounder_prior['confounder_logvar']

            # Hypernetwork based posterior
            code_emb = self.code_embed(code_ids)  # [B, embed_dim]
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

        return 0.001*l2_reg + 0.01*code_sparsity

    def gaussian_KL(self, mu_post: torch.Tensor, logvar_post: torch.Tensor,
                    mu_prior: torch.Tensor, logvar_prior: torch.Tensor):
        return 0.5 * (
                (logvar_prior - logvar_post) +
                (torch.exp(logvar_post) + (mu_post - mu_prior) ** 2) /
                torch.exp(logvar_prior) - 1
        ).sum(1).mean()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar).clamp_min(self._eps)
        return mu + std * torch.randn_like(std) * self._sqrt_2pi

# Initialized for confounder prior
class ConfounderPrior(nn.Module):
    def __init__(self, num_codes, code_dim, conf_dim, hidden_dim, momentum=0.95):
        super().__init__()

        self.code_emb = nn.Embedding(num_codes, code_dim) # Shared embeddings
        self.register_buffer('code_emb_momentum', torch.empty_like(self.code_emb.weight))
        nn.init.orthogonal_(self.code_emb_momentum)  # Independent initialization
        self.momentum = momentum

        # Physics-inspired parameter bounds
        self.mu_net = nn.Sequential(nn.Linear(code_dim, conf_dim),
                                    nn.LayerNorm(conf_dim),
                                    nn.Tanh() # Constrained output
                                    )
        nn.init.kaiming_normal_(self.mu_net[0].weight, nonlinearity='linear')  # First linear layer
        # Initialize final layer for natural parameter bounds
        with torch.no_grad():
            self.mu_net[-1].weight.data.uniform_(-0.01, 0.01)  # Small magnitude
            self.mu_net[-1].bias.data.zero_()  # Center outputs

        self.logvar = nn.Parameter(torch.zeros(conf_dim))
        # Initialize logvar near physical constraints (design doc Eq.27)
        self.logvar.data.normal_(math.log(0.5), 0.1)  # Start mid-range


    def forward(self, h: torch.Tensor, code_ids: torch.Tensor = None):
        # Update momentum codebook
        with torch.no_grad():
            self.code_emb_momentum = (self.momentum * self.code_emb_momentum +
                                      (1 - self.momentum) * self.code_emb.weight)

        code_feat = self.code_emb_momentum[code_ids]

        combined = torch.cat([h, code_feat], -1)

        mu = self.mu_net(combined).clamp(*self.mu_bounds) # [-3, 3] physical constraint
        logvar = self.logvar.expand_as(mu).clamp(math.log(0.1), math.log(2.0)) # Variance bounds

        return mu, logvar



