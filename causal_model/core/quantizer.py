import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_

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
                 initial_temp: float = 1.0, use_cdist: bool = False, normalize: bool = True):
        super().__init__()
        self.initial_temp = initial_temp # Track base temperature
        self.temperature = initial_temp # Active temperature
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
                self.codebook.data = F.normalize(self.codebook.data, p=2, dim=1) # Unit sphere

        else:
            # Initialize the codebook using Kaiming Uniform
            self.codebook = nn.Parameter(torch.empty(self.num_codes, self.code_dim))
            kaiming_uniform_(self.codebook, a=math.sqrt(5))

    def forward(self, h: torch.Tensor):
        """
        Args:
            h (Tensor): Input latent embeddings of shape [B, D]
        """
        # Unit sphere normalization with numerical stability
        if self.normalize:
            h = F.normalize(h, p=2, dim=1, eps=1e-6)
            codebook = F.normalize(self.codebook, p=2, dim=1, eps=1e-6)
        else:
            codebook = self.codebook

        # Compute L2
        if self.use_cdist:
            distances = torch.cdist(h, codebook, p=2) ** 2  # Shape: [B, num_codes]
        else:
            h_sq = torch.sum(h ** 2, dim=1, keepdim=True)  # [B, 1]
            codebook_sq = torch.sum(codebook ** 2, dim=1)  # [num_codes]
            distances = h_sq + codebook_sq - 2 * torch.matmul(h, codebook.t())

        # Compute Gumbel-soft assignments with temperature scaling
        q = F.gumbel_softmax(-distances, tau=self.temperature,hard=False, dim=1) # [B, num_codes]

        # Gumbel Soft quantized code as a weighted sum
        c_tilde = torch.bmm(q.unsqueeze(1), codebook.expand(q.size(0), -1, -1)).squeeze(1) # more efficient
        # c_tilde = torch.matmul(q, codebook) # [B, D]

        # Hard assignments via argmax
        indices = torch.argmax(q, dim=1)  # [B]  Non-differentiable
        c_hard = codebook[indices]        # [B, D]

        # Straight-through estimator: use hard code in forward pass but gradients
        # flow through c_tilde
        """Double check"""
        c_quantized = c_tilde + (c_hard - c_tilde.detach())

        # Stabilized losses
        codebook_loss = F.mse_loss(h.detach(), c_tilde) # Pulls codebook to embedded h
        commitment_loss = F.mse_loss(h, c_tilde.detach()) # Push h to codebook

        # Quantization loss: push codebook vectors toward h (with h detached) and commitment loss
        loss = codebook_loss + self.beta * commitment_loss

        return q, c_tilde, c_hard, c_quantized, loss

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
    def __init__(self, code_dim: int,
                 num_codes_tr: int,
                 num_codes_re: int,
                 beta_tr: float = 0.25,
                 beta_re: float = 0.25,
                 tr_temperature: float = 2.0, tr_min_temperature: float = 0.05, tr_anneal_factor: float = 0.97,
                 re_temperature:float = 1.5, re_min_temperature: float = 0.1, re_anneal_factor: float = 0.95,
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

        self.tr_quantizer = VQQuantizer(num_codes=num_codes_tr, code_dim=code_dim, beta=beta_tr,
                                        initial_temp=self.tr_temperature, use_cdist=use_cdist, normalize=normalize)
        self.re_quantizer = VQQuantizer(num_codes=num_codes_re, code_dim=code_dim, beta=beta_re,
                                        initial_temp=self.re_temperature, use_cdist=use_cdist, normalize=normalize)

        self.coupling = coupling


        if self.coupling: # Only create coupling components when enabled
            self.coupling_mlp = nn.Sequential(nn.Linear(num_codes_re, hidden_dim), nn.SiLU(),
                                              nn.Linear(hidden_dim, num_codes_tr))
            self.coupling_mlp.register_parameter('sparsity_mask', nn.Parameter(torch.ones_like(self.coupling_mlp[0].weight)))
            self.lambda_couple = lambda_couple
            self.sparsity_weight = sparsity_weight

    def forward(self, h_tr: torch.Tensor, h_re: torch.Tensor):
        """
        Args:
            h_tr (Tensor): Transition projections. shape [B, D]
            h_re (Tensor): Reward projections. shape [B, D]
        """
        q_tr, soft_tr, hard_tr, quantized_tr, quant_loss_tr = self.tr_quantizer(h_tr)
        q_re, soft_re, hard_re, quantized_re, quant_loss_re = self.re_quantizer(h_re)
        total_loss = quant_loss_tr + quant_loss_re

        output_dict = {
            'q_tr': q_tr,                       # Soft assignments [B, T, KTr]
            'soft_tr': soft_tr,                 # Soft codes [B, T, D]
            'hard_tr': hard_tr,                 # Hard codes (indices) [B, T]
            'quantized_tr': quantized_tr,       # Straight-through codes [B, T, D]
            'q_re': q_re,                       # Same for reward branch
            'soft_re': soft_re,
            'hard_re': hard_re,
            'quantized_re': quantized_re,
            'loss': total_loss
        }

        if self.coupling: # Conditional coupling mechanism
            logits_tr_from_re = self.coupling_mlp(q_re.detach()) # Stop reward -> transition gradient
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
