from __future__ import annotations

import torch
from torch import nn

from .encoders import MLPEncoder


class OCRAPModel(nn.Module):
    """Compact neural OC-RAP model.

    This is still deliberately lightweight compared with the full paper model,
    but it follows the paper's factorization: scene-prefix encoding -> root
    probabilities, root-option recovery margins, and post-prefix observation
    embeddings whose distances induce the observation-compatibility kernel.
    """

    def __init__(
        self,
        input_dim: int,
        num_roots: int = 8,
        num_options: int = 24,
        d_model: int = 128,
        d_obs: int = 64,
        tau_obs: float = 1.0,
    ):
        super().__init__()
        self.num_roots = int(num_roots)
        self.num_options = int(num_options)
        self.d_obs = int(d_obs)
        self.tau_obs = float(max(tau_obs, 1e-6))
        self.encoder = MLPEncoder(input_dim, d_model)
        self.root_logits = nn.Linear(d_model, self.num_roots)
        self.margin_head = nn.Linear(d_model, self.num_roots * self.num_options)
        self.obs_embed_head = nn.Linear(d_model, self.num_roots * self.d_obs)
        self.utility_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        root_logits = self.root_logits(h)
        margins = self.margin_head(h).reshape(x.shape[0], self.num_roots, self.num_options)
        obs_embeddings = self.obs_embed_head(h).reshape(x.shape[0], self.num_roots, self.d_obs)
        diff = obs_embeddings.unsqueeze(2) - obs_embeddings.unsqueeze(1)
        dist2 = (diff * diff).mean(dim=-1)
        C = torch.exp(-dist2 / self.tau_obs).clamp(0.0, 1.0)
        eye = torch.eye(self.num_roots, dtype=C.dtype, device=C.device).unsqueeze(0)
        C = C * (1 - eye) + eye
        return {
            "root_logits": root_logits,
            "margins": margins,
            "obs_embeddings": obs_embeddings,
            "c_star": C,
            "utility": self.utility_head(h).squeeze(-1),
        }
