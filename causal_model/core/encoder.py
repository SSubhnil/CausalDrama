import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import MLP

class CausalEncoder(nn.Module):
    def __init__(self, hidden_state_dim: int, action_dim: int, stoch_dim: int, tr_proj_dim: int, re_proj_dim: int, hidden_dim: int,
                 combined_input_dim: int, embedding_mode: str='continuous'):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        self.tr_proj_dim = tr_proj_dim
        self.re_proj_dim = re_proj_dim
        self.hidden_dim = hidden_dim
        self.embedding_mode = embedding_mode
        input_dim = combined_input_dim if combined_input_dim is not None else (hidden_state_dim + (action_dim or 0) + (stoch_dim or 0))

        # Feature Projection Layers
        self.tr_proj = MLP(input_dim, self.tr_proj_dim, self.hidden_dim,
                           activation=nn.SiLU)
        self.re_proj = MLP(input_dim, self.re_proj_dim, self.hidden_dim,
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

class ConvolutionalTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, projection_dim, embedding_mode='projection'):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=8, stride=4, padding=2),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(8)
        # Final projection
        if embedding_mode == 'continuous':
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim * 8, projection_dim),
                nn.Tanh()  # Continuous embedding with bounded range
            )
        else:
            self.projection = nn.Linear(hidden_dim * 8, projection_dim)

    def forward(self, x):
        # x shape: [B, L, D]
        # Transpose for 1D convolution [B, D, L]
        x = x.transpose(1, 2)
        # Apply convolutions to reduce sequence length
        x = self.temporal_conv(x)  # [B, hidden_dim, L/16]
        # Pool to fixed length
        x = self.pool(x)  # [B, hidden_dim, 8]
        # Flatten and project
        x = x.reshape(x.size(0), -1)
        return self.projection(x)