import torch
import torch.nn as nn
from sentry_sdk.utils import epoch
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.ReLU, init_type='kaiming'):
        super().__init__()
        layers = []

        # Ensure hidden_dims is a list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Create a list of dimensions for all layers
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))

            if i < len(dims) - 2:  # NO activation after final layer
                layers.append(activation())

        self.net = nn.Sequential(*layers)
        self.init_type = init_type  # Store init_type as an instance variable
        self._init_weights(self.net)  # Apply weights initialization to the network
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in hidden_dims])

    def forward(self, x):
        layer_idx = 0
        norm_idx = 0

        for module in self.net:
            x = module(x)

            # Apply layer normalization after linear layers (but before activation)
            if isinstance(module, nn.Linear) and norm_idx < len(self.norms):
                x = self.norms[norm_idx](x)
                norm_idx += 1

        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.normal_(module.bias, std=1e-6)
            if self.init_type == 'kaiming':
                kaiming_normal_(module.weight)


# Switch transformers [Fedus et al. 2021] logic
class SparseCodebookMoE(nn.Module):
    def __init__(self, num_experts, hidden_dim, code_dim, quantizer, top_k=2):
        super().__init__()
        # Independent initialization
        # self.code_anchor = nn.Parameter(torch.num_experts, code_dim)
        # Codebook aligned initialization
        """Handle quantized_Tr in integrator.py"""
        self.register_buffer('code_anchor',
                             quantizer.tr_quantizer.codebook.data[:num_experts].clone())
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + code_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, 1024 // num_experts)  # Each expert handles a different part of output space
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(code_dim, num_experts)
        self.top_k = top_k
        self.aux_loss = nn.CrossEntropyLoss()

        for expert, anchor in zip(self.experts, self.code_anchor):
            nn.init.normal_(expert[0].weight, mean=anchor[0].item(), std=0.01)
            nn.init.zeros_(expert[-1].weight)

        # Use this for router, if cosine similarity is unstable
        self.router = nn.Sequential(
            nn.Linear(code_dim, num_experts, bias=False),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, h, code_emb):
        # h: [B, D], code_emb: [B, C]
        router_logits = F.cosine_similarity(
            code_emb.unsqueeze(1),  # [B, 1, C]
            self.code_anchor.unsqueeze(0),  # [1,E,C]
            dim=-1
        ) * 0.125  # Scale for stability

        # The following is cosine similarity is unstable
        # router_logits = self.router(code_emb)  # [B, E]
        expert_weights = F.gumbel_softmax(router_logits, tau=max(0.1, 0.5 * (0.95 ** epoch)),
                                          hard=False)  # Should be True for soft routing

        # Sparse expert selection
        topk_weights, topk_indices = torch.topk(expert_weights, self.top_k)
        expert_mask = torch.zeros_like(expert_weights).scatter(1, topk_indices, 1.0)
        expert_weights = expert_weights * expert_mask

        # Expert computation
        expert_input = torch.cat([h, code_emb], dim=-1)
        expert_outputs = [expert(expert_input) for expert in self.experts]  # Combine

        # Concatenate according to output partitions
        batch_size = h.shape[0]
        full_output = torch.zeros(batch_size, 1024, device=h.device)

        for i, (output, weight) in enumerate(zip(expert_outputs, expert_weights.transpose(0, 1))):
            start_idx = i * (1024 // self.num_experts)
            end_idx = (i + 1) * (1024 // self.num_experts)
            full_output[:, start_idx:end_idx] = weight.unsqueeze(1) * output

        assert full_output.shape[-1] == (1024 // self.num_experts), \
            "Expert output dim mismatch"

        ### Load balancing loss ###
        expert_counts = expert_weights.sum(0)

        # Google's soft-moe loss
        # expert_probs = expert_counts / expert_counts.sum()
        # aux_loss = (expert_probs * torch.log(expert_probs + 1e-7)).sum()

        # Better approach: minimize max-min expert count difference
        expert_load = expert_counts / expert_counts.sum()
        aux_loss = 0.5 * (expert_counts.std() + expert_load.entropy())

        return full_output, aux_loss


class ImportanceWeightedMoE(nn.Module):
    def __init__(self, num_experts, hidden_state_dim, hidden_dim, code_dim, quantizer, top_k=2, importance_reg=0.01):
        super().__init__()
        # Check codebook size first
        codebook_size = quantizer.tr_quantizer.codebook.data.size(0)

        # Ensure num_experts doesn't exceed codebook size
        safe_num_experts = min(num_experts, codebook_size)
        if safe_num_experts != num_experts:
            print(f"WARNING: Adjusted num_experts from {num_experts} to {safe_num_experts} to match codebook size")

        # Store the actual value
        self.num_experts = safe_num_experts

        # Initialize with existing code anchors up to safe_num_experts
        self.register_buffer('code_anchor',
                             quantizer.tr_quantizer.codebook.data[:safe_num_experts].clone())
        # Feature importance weights for each expert
        self.feature_importance = nn.Parameter(torch.zeros(num_experts, hidden_state_dim))
        # Initialize with slight randomness to break symmetry
        nn.init.normal_(self.feature_importance, mean=0.0, std=0.01)

        # Keep your existing expert structure
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_state_dim + code_dim, 2 * hidden_state_dim),
                nn.GELU(),
                nn.Linear(2 * hidden_state_dim, 1024 // num_experts)
            ) for _ in range(num_experts)
        ])

        # Router and other components from your original implementation
        self.router = nn.Sequential(
            nn.Linear(code_dim, num_experts, bias=False),
            nn.LogSoftmax(dim=-1)
        )
        self.top_k = top_k
        self.num_experts = num_experts

        # Importance regulation parameters
        self.importance_reg = importance_reg
        self.importance_temperature = nn.Parameter(torch.tensor(1.0))

        # Expert initialization (keeping your method)
        for expert, anchor in zip(self.experts, self.code_anchor):
            nn.init.normal_(expert[0].weight, mean=anchor[0].item(), std=0.01)
            nn.init.zeros_(expert[-1].weight)

    def forward(self, h, code_emb):
        """
        Forward pass for ImportanceWeightedMoE using all experts.

        Args:
            h: Hidden state tensor, shape [B, D] or [B, T, D]
            code_emb: Code embedding tensor, shape [B, D] or [B, T, D]
        """
        h_shape_orig = h.shape
        code_emb_shape_orig = code_emb.shape

        # 1. Ensure consistent dimensions between inputs
        if h.dim() != code_emb.dim():
            if h.dim() < code_emb.dim():
                h = h.unsqueeze(1).expand(-1, code_emb.shape[1], -1)
            else:
                code_emb = code_emb.reshape(code_emb.shape[0], -1)

        # 2. Verify batch dimensions match after reshaping
        if h.shape[0] != code_emb.shape[0]:
            raise ValueError(f"Batch dimensions don't match after reshaping: {h.shape[0]} vs {code_emb.shape[0]}")

        # 3. Compute router logits (cosine similarity)
        is_3d = h.dim() == 3
        code_anchor_norm = F.normalize(self.code_anchor, p=2, dim=1)

        if is_3d:
            B, T, D = code_emb.shape
            code_emb_flat = code_emb.reshape(-1, D)
            router_logits = torch.mm(
                F.normalize(code_emb_flat, p=2, dim=1),
                code_anchor_norm.t()
            ).reshape(B, T, -1) * 0.125
        else:
            router_logits = torch.mm(
                F.normalize(code_emb, p=2, dim=1),
                code_anchor_norm.t()
            ) * 0.125

        # 4. Get FULL expert weight distribution (no top-k filtering)
        expert_weights = F.gumbel_softmax(router_logits, tau=0.1, hard=False)

        # 5. Initialize output tensor
        batch_size = h.shape[0]
        seq_len = h.shape[1] if is_3d else 1

        # 6. Process through ALL experts with importance weighting
        temperature = torch.clamp(self.importance_temperature, min=0.1, max=5.0)

        # Compute importance for all experts at once
        importance = F.softmax(self.feature_importance / temperature, dim=1)  # [num_experts, hidden_dim]

        if is_3d:
            importance = importance.unsqueeze(1).unsqueeze(0)  # [1, num_experts, 1, hidden_dim]
            h = h.unsqueeze(1)  # [batch_size, 1, seq_len, hidden_dim]
        else:
            importance = importance.unsqueeze(0)  # [1, num_experts, hidden_dim]
            h = h.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Apply importance weighting to all experts simultaneously
        weighted_h = h * importance  # [batch_size, num_experts, seq_len, hidden_dim] or [batch_size, num_experts, hidden_dim]

        # Prepare inputs for all experts
        expert_inputs = torch.cat([weighted_h, code_emb.unsqueeze(1).expand(-1, self.num_experts, -1, -1)], dim=-1)

        # Process through all experts simultaneously
        expert_outputs = torch.stack([expert(expert_inputs[:, i]) for i, expert in enumerate(self.experts)], dim=1)

        # Combine expert outputs
        slice_size = 1024 // self.num_experts
        full_output = torch.zeros(batch_size, seq_len, 1024, device=h.device)
        for i in range(self.num_experts):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            full_output[..., start_idx:end_idx] = expert_outputs[:, i, ..., :slice_size] * expert_weights[..., i:i + 1]

        # 7. Calculate auxiliary losses
        expert_counts = expert_weights.sum(0)
        expert_load = expert_counts / (expert_counts.sum() + 1e-8)
        routing_loss = 0.5 * (expert_counts.std() + (-expert_load * torch.log(expert_load + 1e-8)).sum())

        # Calculate importance entropies
        importance_entropies = -(importance * torch.log(importance + 1e-8)).sum(dim=-1).mean(dim=0)
        avg_entropy = importance_entropies.mean()
        aux_loss = routing_loss - self.importance_reg * avg_entropy

        # Create stats dictionary
        importance_stats = {
            'entropy': importance_entropies.tolist(),
            'mean': importance.mean(dim=-1).squeeze().tolist(),
            'std': importance.std(dim=-1).squeeze().tolist()
        }

        return full_output, aux_loss, importance_stats
