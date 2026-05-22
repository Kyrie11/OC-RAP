from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBEVEncoder(nn.Module):
    def __init__(self, in_channels: int = 24, history_steps: int = 10, hidden: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.history_steps = history_steps
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * history_steps, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, hidden), nn.ReLU(inplace=True),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        # bev [B,Hh,C,H,W]
        if bev.dim() != 5:
            raise ValueError("bev must be [B,Hh,C,H,W]")
        B, Hh, C, H, W = bev.shape
        x = bev.reshape(B, Hh * C, H, W).float()
        # Training smoke tests often use CPU; downsample large BEV crops before
        # the compact encoder. Full-resolution models can replace this encoder.
        if H > 64 or W > 64:
            stride_h = max(1, H // 64)
            stride_w = max(1, W // 64)
            x = x[:, :, ::stride_h, ::stride_w]
        return self.net(x)
