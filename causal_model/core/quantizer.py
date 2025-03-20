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
                h_flat = h.reshape(-1, h.shape[-1])  # [B, T, D] -> [B*T, D]
            else:
                h_flat = h

            # Apply normalization if needed
            if self.normalize:
                h_flat_norm = F.normalize(h_flat, p=2, dim=1, eps=1e-6)
                codebook = F.normalize(self.codebook, p=2, dim=1, eps=1e-6)
            else:
                h_flat_norm = h_flat
                codebook = self.codebook

            # Compute distances
            if self.use_cdist:
                distances = torch.cdist(h_flat_norm, codebook, p=2) ** 2
            else:
                # h_sq = torch.sum(h_flat_norm ** 2, dim=1, keepdim=True)
                # codebook_sq = torch.sum(codebook ** 2, dim=1).view(1, -1)

                # # Add batch dimension to create 3D tensors
                # h_flat_norm_3d = h_flat_norm.unsqueeze(0)  # [1, B, D]
                # codebook_t_3d = codebook.t().unsqueeze(0)  # [1, D, K]

                # # Now use baddbmm with 3D tensors
                # bmm_result = torch.baddbmm(
                #     torch.zeros(1, h_flat_norm.size(0), codebook.size(0), device=h_flat_norm.device),
                #     h_flat_norm_3d,
                #     codebook_t_3d,
                #     beta=0.0, alpha=-2.0
                # )

                # distances = h_sq + codebook_sq + bmm_result.squeeze(0)

                # Original implmentation
                h_sq = torch.sum(h_flat_norm ** 2, dim=1, keepdim=True)
                codebook_sq = torch.sum(codebook ** 2, dim=1).view(1, -1)  # Reshape for broadcasting
                distances = h_sq + codebook_sq - 2 * torch.matmul(h_flat_norm, codebook.t())
            # Compute soft assignments
            # Optimization: For inference, replace with deterministic argmin
            if not training:
                indices_flat = torch.argmin(distances, dim=1)
                q_flat = torch.zeros_like(distances)
                q_flat.scatter_(1, indices_flat.unsqueeze(1), 1.0)
            else:
                q_flat = F.gumbel_softmax(-distances, tau=self.temperature, hard=False, dim=1)
                indices_flat = torch.argmax(q_flat, dim=1)
            # Compute soft codes
            c_tilde_flat = torch.einsum('bk,kd->bd', q_flat, codebook)

            # Hard codebook lookup
            c_hard_flat = codebook[indices_flat]

            # Calculate losses using flat tensors (matching shapes)
            diff = h_flat_norm - c_tilde_flat
            codebook_loss = (diff.detach() ** 2).mean()
            commitment_loss = (diff ** 2).mean()

            # Reshape everything back if input was 3D
            if is_3d:
                q = q_flat.reshape(original_shape[0], original_shape[1], -1)
                c_tilde = c_tilde_flat.reshape(original_shape[0], original_shape[1], -1)
                c_hard = c_hard_flat.reshape(original_shape[0], original_shape[1], -1)
            else:
                # Unsqueeze to add sequence dimension for 2D inputs
                q = q_flat
                c_tilde = c_tilde_flat
                c_hard = c_hard_flat

            # Straight-through estimator
            c_quantized = c_tilde + (c_hard - c_tilde.detach())

            loss = codebook_loss + self.beta * commitment_loss

            return q, c_tilde, c_hard, c_quantized, loss, indices_flat

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

    def forward(self, h_tr: torch.Tensor, h_re: torch.Tensor):
        """
        Args:
            h_tr (Tensor): Transition projections. shape [B, D]
            h_re (Tensor): Reward projections. shape [B, D]
        """
        q_tr, soft_tr, hard_tr, quantized_tr, quant_loss_tr, indices_tr = self.tr_quantizer(h_tr)
        q_re, soft_re, hard_re, quantized_re, quant_loss_re, indices_re = self.re_quantizer(h_re)
        total_loss = quant_loss_tr + quant_loss_re

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
            'loss': total_loss
        }

        if self.coupling:  # Conditional coupling mechanism
            logits_tr_from_re = self.coupling_mlp(q_re.detach())  # Stop reward -> transition gradient
            p_tr_cond_re = F.log_softmax(logits_tr_from_re /
                                         self.tr_quantizer.temperature, dim=-1)

            kl_loss = F.kl_div(p_tr_cond_re, q_tr.detach(), reduction='batchmean',
                               log_target=False)
            # Questionable reverse KL -> might remove
            # reverse_kl = F.kl_div(q_tr.log(), p_tr_cond_re.detach().exp(), reduction='batchmean')

            # May keep KL asymmetrical?
            coupling_loss = kl_loss * self.lambda_couple
            sparsity_loss = self.coupling_mlp.sparsity_mask.abs().mean()
            total_loss += coupling_loss + self.sparsity_weight * sparsity_loss

            output_dict.update({
                'coupling_loss': coupling_loss,
                'loss': total_loss
            })

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
