"""Projection heads and modality-specific residual adapters."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.relu(self.dense(features)))


class ResidualBlockStack(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, d_model),
                    nn.Dropout(dropout),
                )
                for _ in range(layers)
            ]
        )
        self.pre_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for block, pre_norm in zip(self.blocks, self.pre_norms):
            features = features + block(pre_norm(features))
        return features


class DeepProjectionAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        d_model: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU())
        self.residual_stack = ResidualBlockStack(d_model, hidden_dim, layers, dropout)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, output_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(features)
        hidden = self.residual_stack(hidden)
        return self.output_projection(hidden)
