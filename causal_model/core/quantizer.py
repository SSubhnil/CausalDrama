import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.cuda.amp import autocast
from line_profiler import profile


class VQQuantizer(nn.Module):
    """
    Quantizes embeddings into discrete codes via a codebook

    For a batch of hidden_states h (shape [B, D]), compute:
    - Soft assignment probabilities via temperature-scaled softmax over -L2 distances
    - A differentiable soft code (c_tilde) as the weighted sum over codebook entries
    - A hard code (c_hard) computed by taking the argmax over the assignments
    - A straight-through output (c_quantized) that uses hard code in forward pass
      but lets gradients flow from the soft code.
    - A quantization loss to update the codebook
    """

    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25,
                 initial_temp: float = 1.0, use_cdist: bool = True, normalize: bool = True):
        super().__init__()
        self.initial_temp = initial_temp  # Track base temperature
        self.temperature = initial_temp  # Active temperature
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.use_cdist = use_cdist
        # When True, codebook vectors reside on the unit sphere
        # Ensures cosine similarity calculation are equivalent to L2 distance when inputs are normalized
        self.normalize = normalize

        if self.normalize:
            self.codebook = nn.Parameter(torch.randn(self.num_codes, self.code_dim))
            with torch.no_grad():
                self.codebook.data = F.normalize(self.codebook.data, p=2, dim=1)  # Unit sphere

        else:
            # Initialize the codebook using Kaiming Uniform
            self.codebook = nn.Parameter(torch.empty(self.num_codes, self.code_dim))
            kaiming_uniform_(self.codebook, a=math.sqrt(5))

    @profile
    def forward(self, h: torch.Tensor, training=False):
        # Store original shape for reshaping later
        with (torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True)):
            original_shape = h.shape
            is_3d = len(original_shape) == 3

            # Flatten 3D input to 2D for processing
            if is_3d:
                h = h.reshape(-1, h.shape[-1])  # [B, T, D] -> [B*T, D]

            # Apply normalization if needed
            if self.normalize:
                h = F.normalize(h, p=2, dim=1, eps=1e-6)
                codebook = F.normalize(self.codebook, p=2, dim=1, eps=1e-6)
            else:
                codebook = self.codebook

            # Compute distances - optimized for either mode
            if self.normalize:
                # For normalized vectors, cosine similarity is proportional to squared distance
                similarities = torch.matmul(h, codebook.t())
                distances = 2 - 2 * similarities
            else:
                if self.use_cdist:
                    distances = torch.cdist(h, codebook, p=2) ** 2
                else:
                    h_sq = torch.sum(h ** 2, dim=1, keepdim=True)
                    codebook_sq = torch.sum(codebook ** 2, dim=1).view(1, -1)
                    distances = h_sq + codebook_sq - 2 * torch.matmul(h, codebook.t())
            # Compute soft assignments
            # Optimization: For inference, replace with deterministic argmin
            # Branch for training vs. inference
            if training:
                # Training path - stochastic with Gumbel-softmax
                q_flat = F.gumbel_softmax(-distances, tau=self.temperature, hard=False, dim=1)
                indices_flat = torch.argmax(q_flat, dim=1)

                # Soft codebook lookup
                c_tilde_flat = torch.einsum('bk,kd->bd', q_flat, codebook)
                c_hard_flat = codebook[indices_flat]

                # Loss calculations
                # diff = h_flat_norm - c_tilde_flat
                # codebook_loss = (diff.detach() ** 2).mean()
                # commitment_loss = (diff ** 2).mean()
                codebook_loss = F.mse_loss(c_tilde_flat, h.detach())
                commitment_loss = F.mse_loss(h, c_tilde_flat.detach())
                loss = codebook_loss + self.beta * commitment_loss

                # Straight-through estimator for training
                c_quantized = c_tilde_flat + (c_hard_flat - c_tilde_flat.detach())
            else:
                # Inference path - deterministic with argmin
                indices_flat = torch.argmin(distances, dim=1)

                # Create one-hot vectors for compatibility (more efficient than Gumbel-softmax)
                q_flat = torch.zeros_like(distances)
                q_flat.scatter_(1, indices_flat.unsqueeze(1), 1.0)

                # Hard codebook lookup only
                c_hard_flat = codebook[indices_flat]
                c_tilde_flat = c_hard_flat  # In inference, no need for soft codes
                c_quantized = c_hard_flat  # No straight-through needed

                # Skip loss computation
                loss = None

            # Reshape outputs based on input dimensions
            if is_3d:
                q = q_flat.reshape(original_shape[0], original_shape[1], -1)
                c_tilde = c_tilde_flat.reshape(original_shape[0], original_shape[1], -1)
                c_hard = c_hard_flat.reshape(original_shape[0], original_shape[1], -1)
                c_quantized = c_quantized.reshape(original_shape[0], original_shape[1], -1)
                indices = indices_flat.reshape(original_shape[0], original_shape[1])
            else:
                q = q_flat
                c_tilde = c_tilde_flat
                c_hard = c_hard_flat
                c_quantized = c_quantized
                indices = indices_flat

            return q, c_tilde, c_hard, c_quantized, loss, indices

    def set_temperature(self, new_temp: float):
        """
        Sets temperature to a new value
        """
        self.temperature = new_temp


class DualVQQuantizer(nn.Module):
    """
    Implements two separate VQQuantizers for the transition and reward branches

    Maintains two codebooks:
        - One for transition branch
        - One for reward branch
        - (Optional) Coupling loss between two branches
    """

    def __init__(self, code_dim_tr: int,
                 code_dim_re: int,
                 num_codes_tr: int,
                 num_codes_re: int,
                 beta_tr: float = 0.25,
                 beta_re: float = 0.25,
                 tr_temperature: float = 2.0, tr_min_temperature: float = 0.05, tr_anneal_factor: float = 0.97,
                 re_temperature: float = 1.5, re_min_temperature: float = 0.1, re_anneal_factor: float = 0.95,
                 use_cdist: bool = False, normalize: bool = False,
                 coupling: bool = False, lambda_couple: float = 0.1, sparsity_weight: float = 0.1,
                 hidden_dim: int = 512):
        super().__init__()
        # Tracking temperature annealing for codebooks
        self.tr_temperature = tr_temperature
        self.tr_min_temperature = tr_min_temperature
        self.re_temperature = re_temperature
        self.re_min_temperature = re_min_temperature
        self.tr_factor = tr_anneal_factor
        self.re_factor = re_anneal_factor

        self.tr_quantizer = VQQuantizer(num_codes=num_codes_tr, code_dim=code_dim_tr, beta=beta_tr,
                                        initial_temp=self.tr_temperature, use_cdist=use_cdist, normalize=normalize)
        self.re_quantizer = VQQuantizer(num_codes=num_codes_re, code_dim=code_dim_re, beta=beta_re,
                                        initial_temp=self.re_temperature, use_cdist=use_cdist, normalize=normalize)

        self.coupling = coupling

        if self.coupling:  # Only create coupling components when enabled
            self.coupling_mlp = nn.Sequential(nn.Linear(num_codes_re, hidden_dim), nn.SiLU(),
                                              nn.Linear(hidden_dim, num_codes_tr))
            self.coupling_mlp.register_parameter('sparsity_mask',
                                                 nn.Parameter(torch.ones_like(self.coupling_mlp[0].weight)))
            self.lambda_couple = lambda_couple
            self.sparsity_weight = sparsity_weight

    def forward(self, h_tr: torch.Tensor, h_re: torch.Tensor, training=False):
        """
        Args:
            h_tr (Tensor): Transition projections. shape [B, D]
            h_re (Tensor): Reward projections. shape [B, D]
        """
        q_tr, soft_tr, hard_tr, quantized_tr, quant_loss_tr, indices_tr = self.tr_quantizer(h_tr, training)
        q_re, soft_re, hard_re, quantized_re, quant_loss_re, indices_re = self.re_quantizer(h_re, training)

        output_dict = {
            'q_tr': q_tr,  # Soft assignments [B, T, KTr]
            'soft_tr': soft_tr,  # Soft codes [B, T, D]
            'hard_tr': hard_tr,  # Hard codes (indices) [B, T]
            'quantized_tr': quantized_tr,  # Straight-through codes [B, T, D]
            'indices_tr': indices_tr,  # Add this new field
            'q_re': q_re,  # Same for reward branch
            'soft_re': soft_re,
            'hard_re': hard_re,
            'quantized_re': quantized_re,
            'indices_re': indices_re,  # Add this new field
        }

        if training:
            # Only calculate losses during training
            total_loss = quant_loss_tr + quant_loss_re
            if self.coupling:  # Conditional coupling mechanism
                logits_tr_from_re = self.coupling_mlp(q_re.detach())  # Stop reward -> transition gradient
                p_tr_cond_re = F.log_softmax(logits_tr_from_re /
                                             self.tr_quantizer.temperature, dim=-1)

                kl_loss = F.kl_div(p_tr_cond_re, q_tr, reduction='batchmean',
                                   log_target=False)
                # Questionable reverse KL -> might remove
                # reverse_kl = F.kl_div(q_tr.log(), p_tr_cond_re.detach().exp(), reduction='batchmean')

                # May keep KL asymmetrical?
                coupling_loss = kl_loss * self.lambda_couple
                sparsity_loss = self.coupling_mlp.sparsity_mask.abs().mean()
                total_loss += coupling_loss + self.sparsity_weight * sparsity_loss

                output_dict['coupling_loss'] = coupling_loss

            output_dict['loss'] = total_loss
        else:
            # Skip loss calculation for inference
            output_dict['loss'] = None

        return output_dict

    def anneal_temperature(self, epoch):
        """
        Anneals the temperature for both the transition and reward quantizers.

        Args:
            epoch (float): Multiplicative factor (< 1 reduces temperature).
        """
        tr_temp = max(self.tr_min_temperature, self.tr_temperature * (self.tr_factor ** epoch))
        re_temp = max(self.re_min_temperature, self.re_temperature * (self.re_factor ** epoch))

        # update quantizer temperatures
        self.tr_quantizer.temperature = tr_temp
        self.re_quantizer.temperature = re_temp

    def set_temperature(self, new_temp: float, which: str = 'both'):
        """
        Sets the temperature for both quantizers to a new value.

        Args:
            new_temp (float): New temperature.
        """
        if which == 'both' or which == 'tr':
            self.tr_quantizer.temperature = new_temp
        if which == 'both' or which == 're':
            self.re_quantizer.temperature = new_temp


class TrajectoryQuantizer(nn.Module):
    """
    Quantizes trajectory projections into discrete codes using a single VQQuantizer
    Processes 2D input [B, proj_dim] where proj_dim is the projection dimension
    """

    def __init__(self, num_codes: int, code_dim: int,
                 beta: float = 0.25, initial_temp: float = 1.0,
                 use_cdist: bool = True, normalize: bool = True):
        super().__init__()
        self.vq = VQQuantizer(
            num_codes=num_codes,
            code_dim=code_dim,
            beta=beta,
            initial_temp=initial_temp,
            use_cdist=use_cdist,
            normalize=normalize
        )

    def forward(self, trajectories: torch.Tensor, training=False):
        """
        Input: trajectories [B, L, D]
        Output: Quantized trajectory codes [B, code_dim]
        """

        # Process through original VQQuantizer
        q, c_tilde, c_hard, c_quantized, loss, indices = self.vq(trajectories, training)

        return {
            'quantized': c_quantized,  # [B, code_dim]
            'indices': indices,  # [B]
            'loss': loss,
            'q': q,  # [B, num_codes]
            'c_tilde': c_tilde,  # [B, code_dim]
            'c_hard': c_hard  # [B, code_dim]
        }

    def info_nce_contrastive_loss(self, c_quantized, temperature=0.1, windowed=True):
        """
        Implements InfoNCE loss as in DOMINO paper

        Args:
            c_quantized: Quantized codes [B, code_dim]
        """
        if windowed:
            B, M, D = c_quantized.shape
            # Flatten batch and window dimensions
            c_quantized = c_quantized.reshape(-1, D)  # [B*M, D]
            batch_size = B * M
        else:
            batch_size = c_quantized.shape[0]

        # Normalize codes for cosine similarity
        c_norm = F.normalize(c_quantized, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.mm(c_norm, c_norm.t()) / temperature

        # InfoNCE uses positives (same context) and negatives (different contexts)
        # For each example, "positive" is itself, "negatives" are all others
        labels = torch.arange(batch_size, device=c_quantized.device)

        # Cross entropy loss (equivalent to InfoNCE)
        loss = F.cross_entropy(similarity, labels)

        return loss

    def anneal_temperature(self, epoch):
        """Delegate temperature annealing to underlying VQQuantizer"""
        self.vq.set_temperature(max(
            self.vq.initial_temp * (0.97 ** epoch),
            self.vq.temperature
        ))

    def set_temperature(self, new_temp: float):
        """Direct temperature control"""
        self.vq.set_temperature(new_temp)

