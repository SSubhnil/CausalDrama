import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import MLP

class CausalEncoder(nn.Module):
    def __init__(self, hidden_state_dim: int, tr_proj_dim: int, re_proj_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        self.tr_proj_dim = tr_proj_dim
        self.re_proj_dim = re_proj_dim
        self.hidden_dim = hidden_dim

        # Feature Projection Layers
        self.tr_proj = MLP(self.hidden_state_dim, self.tr_proj_dim, self.hidden_dim,
                           activation=nn.SiLU)
        self.re_proj = MLP(self.hidden_state_dim, self.re_proj_dim, self.hidden_dim,
                           activation=nn.SiLU)

    def forward(self, h):
        # Extract Features
        return{'tr_features': self.tr_proj(h), 're_features': self.re_proj(h)}
