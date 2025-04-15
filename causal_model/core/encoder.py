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
    def __init__(self, input_dim, hidden_dim, projection_dim, device, embedding_mode='projection'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_mode = embedding_mode
        self.device = device

        # Define convolutional layers with default kernel sizes
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)

        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(4)

        # Final projection
        if embedding_mode == 'continuous':
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim * 4, projection_dim),
                nn.Tanh()  # Continuous embedding with bounded range
            )
        else:
            self.projection = nn.Linear(hidden_dim * 4, projection_dim)

    def adjust_conv_layer(self, conv_layer, input_length, device):
        """Adjusts the convolutional layer's kernel size if necessary."""
        if input_length < conv_layer.kernel_size[0]:
            new_kernel_size = input_length
            new_padding = max(0, (new_kernel_size - 1) // 2)  # Adjust padding to maintain output dimensions

            # Create a new Conv1d layer with adjusted kernel size
            adjusted_conv = nn.Conv1d(
                in_channels=conv_layer.in_channels,
                out_channels=conv_layer.out_channels,
                kernel_size=new_kernel_size,
                stride=conv_layer.stride,
                padding=new_padding,
                bias=conv_layer.bias is not None
            )
            adjusted_conv.weight.data[:, :, :new_kernel_size] = conv_layer.weight.data[:, :, :new_kernel_size]
            if conv_layer.bias is not None:
                adjusted_conv.bias.data = conv_layer.bias.data

            adjusted_conv = adjusted_conv.to(device)

            return adjusted_conv

        return conv_layer

    def forward(self, x):
        # x shape: [B, L, D]
        x = x.transpose(1, 2)  # [B, D, L]

        device = x.device

        # Dynamically adjust convolutional layers based on input length
        x_length = x.shape[-1]
        self.conv1 = self.adjust_conv_layer(self.conv1, x_length, device)
        x = self.conv1(x)

        x_length = x.shape[-1]
        self.conv2 = self.adjust_conv_layer(self.conv2, x_length, device)
        x = self.conv2(x)

        x_length = x.shape[-1]
        self.conv3 = self.adjust_conv_layer(self.conv3, x_length, device)
        x = self.conv3(x)

        # Pooling and projection
        x = self.pool(x)  # [B, hidden_dim, fixed_length]
        x = x.reshape(x.size(0), -1)
        return self.projection(x)
