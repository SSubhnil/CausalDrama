import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from line_profiler import profile

from torch.cuda.amp import autocast

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
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.out_dim = out_dim

        # Factorized weight generation
        self.w1_proj = nn.Linear(code_dim, 2 * hidden_dim)
        self.w2_proj = nn.Linear(code_dim, 2 * out_dim)

        self.seq_proj = nn.Linear(code_dim, hidden_dim)  # For sequence dimension processing

        # Initialization (using smaller values for stability)
        nn.init.normal_(self.w1_proj.weight, std=0.01)
        nn.init.normal_(self.w2_proj.weight, std=0.01)
        nn.init.zeros_(self.w1_proj.bias)
        nn.init.zeros_(self.w2_proj.bias)

    def forward(self, x, code_emb):
        # Retain existing input reshaping code
        orig_x_shape = x.shape
        x = x.reshape(orig_x_shape[0], -1)  # [B, D]

        # Handle 3D code_emb and 2D x
        if code_emb.dim() == 3 and x.dim() == 2:
            B, T, C = code_emb.shape
            D = x.shape[1]  # Input feature dimension

            # Process code embeddings
            code_emb_flat = code_emb.reshape(B * T, C)
            w1 = self.w1_proj(code_emb_flat).unflatten(-1, (2, self.hidden_dim))
            w2 = self.w2_proj(code_emb_flat).unflatten(-1, (2, self.out_dim))

            # Expand x to match sequence dimension
            x_expanded = x.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)  # [B*T, D]

            # Process each batch element separately
            # h = torch.zeros(B * T, self.hidden_dim, device=x.device)
            # for i in range(B * T):
            #     weight = w1[i, 0]  # [hidden_dim]
            #     bias = w1[i, 1]  # [hidden_dim]

            #     # Apply the weight as a simple scaled sum of inputs
            #     h[i] = weight * x_expanded[i].sum() + bias

            weight_1 = w1[:, 0]  # [B*T, hidden_dim]
            bias_1 = w1[:, 1]  # [B*T, hidden_dim]

            # Sum across input featues (more efficient)
            x_sum = x_expanded.sum(dim=1, keepdim=True)  # [B*T, 1]

            h = F.silu(weight_1 * x_sum + bias_1)

            # Similarly for the second layer
            # output = torch.zeros(B * T, self.out_dim, device=x.device)
            # for i in range(B * T):
            #     weight = w2[i, 0]  # [out_dim]
            #     bias = w2[i, 1]  # [out_dim]

            #     output[i] = weight * h[i].sum() + bias

            weight_2 = w2[:, 0]  # [B*T, out_dim]
            bias_2 = w2[:, 1]  # [B*T, out_dim]
            h_sum = h.sum(dim=1, keepdim=True)  # [B*T, 1]
            output = weight_2 * h_sum + bias_2

            # Reshape to expected output dimensions
            output = output.reshape(B, T, self.out_dim)
            return output

        # Handle 2D case (original code unchanged)
        else:
            # Original 2D handling
            w1 = self.w1_proj(code_emb).unflatten(-1, (2, self.hidden_dim))
            w2 = self.w2_proj(code_emb).unflatten(-1, (2, self.out_dim))

            h = F.silu(F.linear(x, w1[:, 0].view(code_emb.size(0), -1).transpose(0, 1)[:, :x.size(1)]) + w1[:, 1])
            return F.linear(h, w2[:, 0].view(code_emb.size(0), -1).transpose(0, 1)[:, :self.hidden_dim]) + w2[:, 1]


class ConfounderPosterior(nn.Module):
    """
    We implement Part 2B of report
    Components:
        - Code-conditioned variational posterior Q(u | h, c)
        - Hyper-network for parameter generation
        - Affine transformations code specific phi_c(u)
    """

    def __init__(self, code_dim: int, conf_dim: int, num_codes: int,
                 embed_dim: int = None, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.conf_dim = conf_dim
        self.num_codes = num_codes
        self.embed_dim = code_dim

        if embed_dim is not None:
            self.embed_dim = embed_dim

        # Check dimensions match config
        config_hidden_dim = 512
        config_conf_dim = 128
        config_code_dim = 256

        if hidden_dim != config_hidden_dim or conf_dim != config_conf_dim or code_dim != config_code_dim:
            print(f"WARNING: Dimensions may not match config.yaml values. "
                  f"Expected: hidden_dim={config_hidden_dim}, conf_dim={config_conf_dim}, code_dim={config_code_dim}")

        # Code embedding layer
        self.h_proj = nn.Linear(self.code_dim, self.hidden_dim)
        self.code_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.code_embed = nn.Embedding(self.num_codes, self.code_dim)  # nn.Embedding(self.num_codes, self.embed_dim)
        nn.init.orthogonal_(self.code_embed.weight)  # Preserve code distances

        # Posterior networks
        self.confounder_post_mu_net = PosteriorHypernet(self.code_dim, self.hidden_dim, self.conf_dim)

        self.confounder_post_logvar_net = PosteriorHypernet(self.code_dim, self.hidden_dim, self.conf_dim)

        # Helps pre-compute frequent calculations for reparameterization
        self._sqrt_2pi = torch.sqrt(torch.tensor(2 * torch.pi))
        self._eps = torch.finfo(torch.float32).eps

    def check_dimensions(self, tensor, expected_shape, name="tensor", tolerance=0):
        """
        Verify tensor dimensions match expected shape.

        Args:
            tensor: The tensor to check
            expected_shape: Tuple of expected dimensions
            name: Name of tensor for error message
            tolerance: Number of dimensions that can differ

        Returns:
            bool: Whether dimensions are valid
        """
        if len(tensor.shape) != len(expected_shape):
            print(f"WARNING: {name} has {len(tensor.shape)} dimensions, expected {len(expected_shape)}")
            return False

        mismatches = sum(1 for a, b in zip(tensor.shape, expected_shape) if a != b)
        if mismatches > tolerance:
            print(f"WARNING: {name} shape {tensor.shape} doesn't match expected {expected_shape}")
            return False

        return True

    @profile
    def forward(self, h: torch.Tensor, code_emb):
        """
        Args:
            h: Hidden state projection from encoder
            code_emb: Code embeddings from quantizer
            mu_prior: Prior mean
            logvar_prior: Prior log variance
        """
        # Check dimensions
        # self.check_dimensions(h, (h.size(0), h.size(1), self.code_dim), "h input")
        # self.check_dimensions(code_emb, (code_emb.size(0), code_emb.size(1), self.code_dim), "code_emb")

        with (torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True)):
            # Generate posterior parameters
            # mu_post = self.confounder_post_mu_net(h, code_emb)
            mu_post = torch.clamp(self.confounder_post_mu_net(h, code_emb), min=-10.0, max=10.0)
            # self.check_dimensions(mu_post, (mu_post.size(0), mu_post.size(1), self.conf_dim), "mu_post")

            logvar_post = self.confounder_post_logvar_net(h, code_emb)
            # self.check_dimensions(logvar_post, (logvar_post.size(0), logvar_post.size(1), self.conf_dim), "logvar_post")

            # Reparameterization trick
            u_post = self.reparameterize(mu_post, logvar_post)

            return u_post, mu_post, logvar_post

    # REDUNDANT due to non-use of scale and shift
    def regularization_loss_2(self, lamdba_weight=0.01):
        # Encourage sparse affine transformations
        scale_reg = torch.mean(torch.abs(self.affine_scale.weight))
        shift_reg = torch.mean(torch.abs(self.affine_shift.weight))
        return lamdba_weight * (scale_reg + shift_reg)

    # Main regularizer
    def regularization_loss(self):
        # Hypernetwork parameter regularization
        with (torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True)):
            hyper_params = list(self.confounder_post_mu_net.parameters()) + \
                           list(self.confounder_post_logvar_net.parameters())

            l2_reg = torch.sum(torch.stack([torch.norm(p) for p in hyper_params]))

            # Code embedding sparsity
            code_sparsity = torch.mean(torch.abs(self.code_embed.weight))

            return l2_reg, code_sparsity

    def gaussian_KL(self, mu_post, logvar_post, mu_prior, logvar_prior):
        with (torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True)):
            # More stable KL computation
            var_post = torch.exp(logvar_post).clamp(min=1e-5)
            var_prior = torch.exp(logvar_prior).clamp(min=1e-5)

            # Logarithm form is more stable
            kl_div = 0.5 * (
                    logvar_prior - logvar_post +
                    var_post / var_prior +
                    ((mu_post - mu_prior) ** 2) / var_prior - 1
            )

            # Clip extreme values
            kl_div = torch.clamp(kl_div, min=-100, max=100)
            return kl_div.sum(1).mean()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).clamp_min(self._eps)
        return mu + std * torch.randn_like(std)

    def gaussian_KL_to_mixture(self, mu_post, logvar_post, mix_weights, prior_mus, prior_logvars):
        """KL divergence from Gaussian posterior to Gaussian mixture prior"""
        var_post = torch.exp(logvar_post)

        # Expand posterior parameters to compare with each mixture component
        # [B, T, 1, D] x [B, T, K, D] -> [B, T, K, D]
        mu_post_exp = mu_post.unsqueeze(2).expand_as(prior_mus)
        var_post_exp = var_post.unsqueeze(2).expand_as(prior_mus)
        logvar_post_exp = logvar_post.unsqueeze(2).expand_as(prior_mus)

        # Compute KL for each component: KL(q||p_i)
        var_prior = torch.exp(prior_logvars)
        kl_per_component = 0.5 * (
                prior_logvars - logvar_post_exp +
                (var_post_exp + (mu_post_exp - prior_mus) ** 2) / var_prior - 1
        ).sum(dim=-1)  # [B, T, K]

        # Use variational upper bound: KL(q||mix) <= Σ_i π_i KL(q||p_i)
        weighted_kl = (mix_weights * kl_per_component).sum(dim=-1)  # [B, T]

        return weighted_kl.mean()


# Initialized for confounder prior
class ConfounderPrior(nn.Module):
    def __init__(self, num_codes, code_dim, conf_dim, hidden_state_proj_dim, momentum=0.95):
        super().__init__()

        self.code_embed = nn.Embedding(num_codes, code_dim)  # Shared embeddings
        self.register_buffer('code_emb_momentum', torch.empty_like(self.code_embed.weight))
        nn.init.orthogonal_(self.code_emb_momentum)  # Independent initialization
        self.momentum = momentum
        self.h_proj_dim = hidden_state_proj_dim

        # Physics-inspired parameter bounds
        self.mu_net = nn.Sequential(nn.Linear(self.h_proj_dim + code_dim, conf_dim),
                                    nn.LayerNorm(conf_dim),
                                    nn.Linear(conf_dim, conf_dim),
                                    nn.Tanh()  # Constrained output
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
        with (torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True)):
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
        # energy = torch.norm(mu, dim=1).mean()
        # return torch.relu(energy - 3.0)  # [1][3]
        target_energy = 3.0
        energy = torch.norm(mu, dim=1).mean()
        return (energy - target_energy) ** 2  # Quadratic penalty

    def add_new_code(self, prototype_emb):
        """Initialization via next existing code"""
        with torch.no_grad():
            sim = F.cosine_similarity(prototype_emb, self.code_emb.weight)
            new_emb = 0.5 * (self.code_emb.weight[sim.argmax()] + prototype_emb)
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


class MixtureConfounderPrior(nn.Module):
    def __init__(self, num_codes, code_dim, conf_dim, hidden_state_proj_dim, momentum=0.90):
        super().__init__()
        self.code_embed = nn.Embedding(num_codes, code_dim)
        self.register_buffer('code_emb_momentum', torch.empty_like(self.code_embed.weight))
        nn.init.orthogonal_(self.code_emb_momentum)
        self.momentum = momentum
        self.h_proj_dim = hidden_state_proj_dim
        self.num_codes = num_codes
        self.conf_dim = conf_dim

        # Parameter networks (same as before)
        self.mu_net = nn.Sequential(
            nn.Linear(self.h_proj_dim + code_dim, conf_dim),
            nn.LayerNorm(conf_dim),
            nn.Linear(conf_dim, conf_dim),
            nn.Tanh()
        )

        self.logvar_net = nn.Sequential(
            nn.Linear(self.h_proj_dim + code_dim, conf_dim),
            nn.Hardtanh(min_val=math.log(0.1), max_val=math.log(2.0))
        )

        # NEW: Mixture weight network
        self.mix_weight_net = nn.Sequential(
            nn.Linear(self.h_proj_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_codes)
        )

        self.register_buffer('mu_bounds', torch.tensor([-3.0, 3.0]))

    def forward(self, h: torch.Tensor, code_ids: torch.Tensor = None):
        """Returns mixture parameters for p(u|h)"""
        # Handle dimensionality
        h_shape = h.shape
        if len(h_shape) == 3:  # [B, T, D]
            is_3d = True
            B, T, D = h_shape
        else:  # [B, D]
            is_3d = False
            B, D = h_shape
            h = h.unsqueeze(1)  # Add a time dimension for consistent processing

        # Update momentum codebook (selectively if code_ids provided)
        with torch.no_grad():
            # if code_ids is not None and self.training:
            #     # Update only for codes that were used (like original implementation)
            #     unique_indices = torch.unique(code_ids)
            #     self.code_emb_momentum[unique_indices] = (
            #         self.momentum * self.code_emb_momentum[unique_indices] +
            #         (1 - self.momentum) * self.code_embed.weight[unique_indices].detach()
            #     )
            # else:
            # Full update (can be inefficient with large codebooks)
            self.code_emb_momentum = (self.momentum * self.code_emb_momentum +
                                      (1 - self.momentum) * self.code_embed.weight.detach())

        with (torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True)):
            # Compute mixture weights: π(c_i^Tr|h)
            mix_logits = self.mix_weight_net(h)  # [B, T, num_codes]
            mix_weights = F.softmax(mix_logits, dim=-1)

            # Expand h to combine with each code
            h_expanded = h.unsqueeze(2).expand(-1, -1, self.num_codes, -1)  # [B, T, K, D]

            # Get all code embeddings
            all_codes = self.code_emb_momentum.unsqueeze(0).unsqueeze(0).expand(
                B, h.shape[1], -1, -1)  # [B, T, K, code_dim]

            # Combine h with each code embedding
            combined = torch.cat([h_expanded, all_codes], dim=-1)  # [B, T, K, D+code_dim]

            # Reshape for batch processing through networks
            flat_combined = combined.view(-1, combined.shape[-1])

            # Generate parameters for all components
            flat_mu = self.mu_net(flat_combined).clamp(*self.mu_bounds)
            flat_logvar = self.logvar_net(flat_combined)

            # Reshape back to original dimensionality
            mus = flat_mu.view(B, h.shape[1], self.num_codes, self.conf_dim)
            logvars = flat_logvar.view(B, h.shape[1], self.num_codes, self.conf_dim)

            # Restore original dimensions if input was 2D
            if not is_3d:
                mix_weights = mix_weights.squeeze(1)  # [B, num_codes]
                mus = mus.squeeze(1)  # [B, num_codes, conf_dim]
                logvars = logvars.squeeze(1)  # [B, num_codes, conf_dim]

            return mix_weights, mus, logvars

    def sample(self, h, code_ids=None):
        """Sample from the mixture distribution"""
        # Shape handling happens in forward
        mix_weights, mus, logvars = self.forward(h, code_ids)

        # Handle dimensionality for sampling
        is_3d = len(h.shape) == 3
        if is_3d:
            B, T = h.shape[0], h.shape[1]
        else:
            B = h.shape[0]
            T = 1
            mix_weights = mix_weights.unsqueeze(1)
            mus = mus.unsqueeze(1)
            logvars = logvars.unsqueeze(1)

        # Sample component indices based on mixture weights
        component_dist = torch.distributions.Categorical(probs=mix_weights)
        component_indices = component_dist.sample()  # [B, T]

        # Gather selected means and logvars
        batch_indices = torch.arange(B, device=h.device).view(-1, 1).expand(-1, T)
        seq_indices = torch.arange(T, device=h.device).view(1, -1).expand(B, -1)

        selected_mu = mus[batch_indices, seq_indices, component_indices]
        selected_logvar = logvars[batch_indices, seq_indices, component_indices]

        # Sample from the selected Gaussian
        std = torch.exp(0.5 * selected_logvar)
        eps = torch.randn_like(std)
        samples = selected_mu + eps * std

        # Restore original dimensions
        if not is_3d:
            samples = samples.squeeze(1)

        return samples

    def prior_regularization(self, mu):
        """Kinetic energy constraint"""
        # energy = torch.norm(mu, dim=1).mean()
        # return torch.relu(energy - 3.0)  # [1][3]
        target_energy = 3.0
        energy = torch.norm(mu, dim=1).mean()
        return (energy - target_energy) ** 2  # Quadratic penalty

    def add_new_code(self, prototype_emb):
        """Initialization via next existing code"""
        with torch.no_grad():
            sim = F.cosine_similarity(prototype_emb, self.code_emb.weight)
            new_emb = 0.5 * (self.code_emb.weight[sim.argmax()] + prototype_emb)
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


