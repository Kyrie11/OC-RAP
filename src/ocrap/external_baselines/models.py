from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ExternalBaselineBatch:
    x: torch.Tensor
    mask: torch.Tensor
    target_index: torch.Tensor
    utility: torch.Tensor
    hard: torch.Tensor
    harm: torch.Tensor
    r_orc: torch.Tensor
    r_dep: torch.Tensor
    feasible: torch.Tensor


class CandidateSetTransformer(nn.Module):
    """Lightweight candidate-set Transformer used by learned external baselines.

    The original external papers use richer scene encoders.  This adapter keeps
    the same high-level decision structure while consuming OC-RAP's already
    vectorized ego/agent/map/route/prefix features.  Each candidate prefix is a
    token, candidate tokens attend to one another, and heads predict policy
    scores plus utility/risk/recoverability quantities used by the baseline
    selector.
    """

    def __init__(
        self,
        input_dim: int,
        max_candidates: int = 32,
        d_model: int = 192,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_candidates = int(max_candidates)
        self.d_model = int(d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.pos = nn.Parameter(torch.zeros(1, self.max_candidates, self.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=4 * self.d_model,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.d_model)
        self.policy_head = nn.Linear(self.d_model, 1)
        self.utility_head = nn.Linear(self.d_model, 1)
        self.hard_head = nn.Linear(self.d_model, 1)
        self.harm_head = nn.Linear(self.d_model, 1)
        self.oracle_rec_head = nn.Linear(self.d_model, 1)
        self.deploy_rec_head = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B,N,D], got {tuple(x.shape)}")
        B, N, _ = x.shape
        h = self.input_proj(x)
        if N > self.pos.shape[1]:
            extra = self.pos[:, -1:, :].expand(1, N - self.pos.shape[1], -1)
            pos = torch.cat([self.pos, extra], dim=1)[:, :N]
        else:
            pos = self.pos[:, :N]
        h = h + pos
        key_padding_mask = None if mask is None else ~mask.bool()
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        logits = self.policy_head(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1.0e4)
        return {
            "logits": logits,
            "utility": self.utility_head(h).squeeze(-1),
            "hard": self.hard_head(h).squeeze(-1),
            "harm": self.harm_head(h).squeeze(-1),
            "r_orc": self.oracle_rec_head(h).squeeze(-1),
            "r_dep": self.deploy_rec_head(h).squeeze(-1),
        }


def build_model_from_cfg(input_dim: int, cfg: dict[str, Any]) -> CandidateSetTransformer:
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    mcfg = bcfg.get("model", {}) if isinstance(bcfg.get("model", {}), dict) else {}
    return CandidateSetTransformer(
        input_dim=int(input_dim),
        max_candidates=int(mcfg.get("max_candidates", bcfg.get("max_candidates", 32))),
        d_model=int(mcfg.get("d_model", 192)),
        num_layers=int(mcfg.get("num_layers", 2)),
        num_heads=int(mcfg.get("num_heads", 4)),
        dropout=float(mcfg.get("dropout", 0.15)),
    )
