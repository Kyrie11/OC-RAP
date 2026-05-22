from __future__ import annotations

import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    def __init__(self, D_state: int = 6, H_p1: int = 11, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_state * H_p1, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        B, K = actions.shape[:2]
        return self.net(actions.reshape(B, K, -1).float())


class OptionEncoder(nn.Module):
    def __init__(self, D_state: int = 6, H_r1: int = 26, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_state * H_r1, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

    def forward(self, options: torch.Tensor) -> torch.Tensor:
        B, K, L = options.shape[:3]
        return self.net(options.reshape(B, K, L, -1).float())
