from __future__ import annotations

import torch
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
