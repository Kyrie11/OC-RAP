from __future__ import annotations

import torch
import torch.nn.functional as F


def margin_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = (pred - target) ** 2
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def anti_oracle_loss(pred_r_orc: torch.Tensor, pred_r_dep: torch.Tensor, teacher_artifact: torch.Tensor, delta_neg: float = 0.0) -> torch.Tensor:
    """Anti-oracle loss from Eq. (18): I_art * [R_dep - delta_neg]_+.

    The loss must push predicted deployable recoverability down on oracle
    artifacts.  Penalizing the oracle-deployability gap itself would instead
    encourage the model to close the gap, which is the opposite of the paper's
    anti-oracle objective.
    """
    del pred_r_orc  # retained in the signature for backwards compatibility
    return F.relu(pred_r_dep - float(delta_neg)).mul(teacher_artifact.float()).mean()


def artifact_gap_loss(pred_gap: torch.Tensor, teacher_gap: torch.Tensor, teacher_artifact: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Encourage a positive oracle-to-deployable gap on oracle artifacts.

    The main SmoothL1 losses regress R_orc and R_dep separately.  In practice,
    rare artifact cases can be underweighted, so this auxiliary ranking-style
    term directly teaches the model that artifacts should preserve an oracle
    headroom/deployable-headroom separation.
    """
    mask = teacher_artifact.float() > 0.5
    if not bool(mask.any()):
        return pred_gap.sum() * 0.0
    target = torch.clamp(teacher_gap.float(), min=float(margin))
    return F.relu(target[mask] - pred_gap[mask]).mean()


def deployability_classification_loss(pred_r_dep: torch.Tensor, teacher_r_dep: torch.Tensor, gamma: float = 0.0) -> torch.Tensor:
    """Balanced binary supervision for calibrated deployability admission.

    Regression alone is weak near the admission threshold.  This term makes the
    sign/threshold of predicted deployable recovery learnable while keeping the
    numerical margin regression intact.
    """
    target = (teacher_r_dep.float() >= float(gamma)).float()
    logits = pred_r_dep.float() - float(gamma)
    pos = target > 0.5
    neg = ~pos
    if not bool(pos.any()) or not bool(neg.any()):
        return F.binary_cross_entropy_with_logits(logits, target)
    weight = torch.where(pos, 0.5 / pos.float().mean().clamp_min(1e-6), 0.5 / neg.float().mean().clamp_min(1e-6))
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)
