import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import MLP

class CausalEncoder(nn.Module):
    def __init__(self, hidden_state_dim: int, tr_proj_dim: int, re_proj_dim: int, hidden_dim: int, embedding_mode: str='continuous'):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        self.tr_proj_dim = tr_proj_dim
        self.re_proj_dim = re_proj_dim
        self.hidden_dim = hidden_dim
        self.embedding_mode = embedding_mode

        # Feature Projection Layers
        self.tr_proj = MLP(self.hidden_state_dim, self.tr_proj_dim, self.hidden_dim,
                           activation=nn.SiLU)
        self.re_proj = MLP(self.hidden_state_dim, self.re_proj_dim, self.hidden_dim,
                           activation=nn.SiLU)

        # Embedding normalization for continuous embedding mode
        self.tr_norm = nn.LayerNorm(self.tr_proj_dim)
        self.re_norm = nn.LayerNorm(self.re_proj_dim)

        # Learnable embedding tables for fully discrete mode
        self.tr_embed_table = nn.Embedding(256, tr_proj_dim) # Fixed vocabulary size
        self.re_embed_table = nn.Embedding(128, re_proj_dim) # Could be config parameters

    def forward(self, h, use_discrete_table=False):
        if self.embedding_mode == 'projection' or self.embedding_mode == 'continuous':
            tr_proj = self.tr_proj(h)
            re_proj = self.re_proj(h)

            if self.embedding_mode == 'continuous':
                # Apply normalization for embedding-like behavior
                tr_proj = self.tr_norm(tr_proj)
                re_proj = self.re_norm(re_proj)

            return tr_proj, re_proj