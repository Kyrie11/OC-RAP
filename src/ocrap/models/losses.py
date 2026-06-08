from __future__ import annotations

import torch
import torch.nn.functional as F


def margin_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = (pred - target) ** 2
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def anti_oracle_loss(pred_r_orc: torch.Tensor, pred_r_dep: torch.Tensor, teacher_artifact: torch.Tensor) -> torch.Tensor:
    gap = pred_r_orc - pred_r_dep
    return F.relu(gap) .mul(teacher_artifact.float()).mean()
