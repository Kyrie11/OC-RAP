from __future__ import annotations

import torch
from torch import nn

from .encoders import MLPEncoder


class OCRAPModel(nn.Module):
    def __init__(self, input_dim: int, num_roots: int = 8, num_options: int = 24, d_model: int = 128):
        super().__init__()
        self.num_roots = num_roots
        self.num_options = num_options
        self.encoder = MLPEncoder(input_dim, d_model)
        self.root_logits = nn.Linear(d_model, num_roots)
        self.margin_head = nn.Linear(d_model, num_roots * num_options)
        self.compat_head = nn.Linear(d_model, num_roots * num_roots)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        root_logits = self.root_logits(h)
        margins = self.margin_head(h).reshape(x.shape[0], self.num_roots, self.num_options)
        C = torch.sigmoid(self.compat_head(h).reshape(x.shape[0], self.num_roots, self.num_roots))
        eye = torch.eye(self.num_roots, dtype=C.dtype, device=C.device).unsqueeze(0)
        C = 0.5 * (C + C.transpose(1, 2))
        C = C * (1 - eye) + eye
        return {"root_logits": root_logits, "margins": margins, "c_star": C}
