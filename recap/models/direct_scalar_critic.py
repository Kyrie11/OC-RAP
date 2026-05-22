from __future__ import annotations

import torch
import torch.nn as nn


class DirectScalarCritic(nn.Module):
    """Ablation-only R=f(b,a) critic. Never use for --method ours."""

    def __init__(self, D_action: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_action, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        B, K = actions.shape[:2]
        return self.net(actions.reshape(B, K, -1)).squeeze(-1)
