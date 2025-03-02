import torch
import torch.nn as nn
from sentry_sdk.utils import epoch
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.ReLU, init_type='kaiming'):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(input_dim, output_dim))

            if i < len(dims)-2: # NO activation after final layer
                layers.append(activation())

        self.net = nn.Sequential(*layers)
        layers.apply(self._init_weights)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in hidden_dims])

    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i % 2 == 0 and i//2 < len(self.norms): # Apply norm after linear layer
                x = self.norms[i//2](x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.normal_(module.bias, std=1e-6)
        if self.ini_type == 'kaiming':
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
                             quantizer.tr_quantizer.weight.data[:num_experts].clone())
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + code_dim, 4*hidden_dim),
                nn.GELU(),
                nn.Linear(4*hidden_dim, hidden_dim)
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
            self.code_anchor.unsqueeze(0), # [1,E,C]
            dim=-1
        ) * 0.125 # Scale for stability

        # The following is cosine similarity is unstable
        # router_logits = self.router(code_emb)  # [B, E]
        expert_weights = F.gumbel_softmax(router_logits, tau=max(0.1, 0.5*(0.95**epoch)), hard=False) # Should be True for soft routing

        # Sparse expert selection
        topk_weights, topk_indices = torch.topk(expert_weights, self.top_k)
        expert_mask = torch.zeros_like(expert_weights).scatter(1, topk_indices, 1.0)
        expert_weights = expert_weights * expert_mask

        # Expert computation
        expert_input = torch.cat([h, code_emb], dim=-1)
        moe_out = torch.stack([expert(expert_input) for expert in self.experts], dim=1) # Combine

        # Weighted sum
        output = torch.einsum('be,bed->bd', expert_weights, moe_out)

        # Load balancing loss
        expert_counts = expert_weights.sum(0)

        # Google's soft-moe loss
        # expert_probs = expert_counts / expert_counts.sum()
        # aux_loss = (expert_probs * torch.log(expert_probs + 1e-7)).sum()

        # Better approach: minimize max-min expert count difference
        expert_load = expert_counts / expert_counts.sum()
        aux_loss = 0.5*(expert_counts.std() + expert_load.entropy())

        return output, aux_loss


