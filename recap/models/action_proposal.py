from __future__ import annotations

import torch
import torch.nn as nn


class NeuralActionProposal(nn.Module):
    """Final-mode action proposal head.

    It predicts compact route-anchor/lateral/speed/curvature/brake parameters. A
    deterministic projection step in `proposals/action_projection.py` must be used
    before candidates enter CARE/MERO.
    """

    def __init__(self, feature_dim: int = 128, K_raw: int = 64, D_param: int = 7):
        super().__init__()
        self.K_raw = K_raw
        self.D_param = D_param
        self.head = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True), nn.Linear(feature_dim, K_raw * D_param))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        return self.head(features).reshape(B, self.K_raw, self.D_param)
