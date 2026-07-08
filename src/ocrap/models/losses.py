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


def shared_option_q_regression_loss(pred_q: torch.Tensor, teacher_q: torch.Tensor, root_valid: torch.Tensor, option_valid: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 supervision for the shared recovery-option value Q(i,l).

    Earlier training supervised only aggregated R_dep/R_orc and raw margins.
    That lets the model learn that a prefix is less oracle-artifact-like, but it
    can still pick the wrong shared recovery option at execution time.  This
    auxiliary target directly teaches the option table used by the selector and
    by DRS evaluation.
    """
    mask = root_valid.bool().unsqueeze(-1) & option_valid.bool().unsqueeze(1) & torch.isfinite(teacher_q)
    if not bool(mask.any()):
        return pred_q.sum() * 0.0
    target = torch.clamp(torch.nan_to_num(teacher_q.detach(), nan=0.0, posinf=5.0, neginf=-5.0), min=-5.0, max=5.0)
    pred = torch.clamp(torch.nan_to_num(pred_q, nan=0.0, posinf=5.0, neginf=-5.0), min=-5.0, max=5.0)
    return F.smooth_l1_loss(pred[mask], target[mask])


def shared_option_admission_loss(
    pred_q: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Balanced BCE for whether each shared option is deployably admissible.

    This is the supervision missing when offline results show low ODG/FRA_cand
    but poor executed DRS: the model knows which prefix is less artifact-prone,
    but not which shared option is actually executable across compatible roots.
    """
    mask = root_valid.bool().unsqueeze(-1) & option_valid.bool().unsqueeze(1) & torch.isfinite(teacher_q)
    if not bool(mask.any()):
        return pred_q.sum() * 0.0
    target = (teacher_q.detach() >= float(gamma)).float()
    logits = torch.nan_to_num(pred_q - float(gamma), nan=-20.0, posinf=20.0, neginf=-20.0)
    weights = torch.clamp(root_probs.float(), min=0.0).unsqueeze(-1).expand_as(logits)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    # Balance positives and negatives so rare deployable options are not washed out.
    pos = (target > 0.5) & mask
    neg = (target <= 0.5) & mask
    if bool(pos.any()) and bool(neg.any()):
        pos_scale = 0.5 / pos.float().mean().clamp_min(1e-6)
        neg_scale = 0.5 / neg.float().mean().clamp_min(1e-6)
        weights = weights * torch.where(pos, pos_scale, torch.where(neg, neg_scale, torch.zeros_like(weights)))
    denom = weights.sum().clamp_min(1.0)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (loss * weights).sum() / denom


def _root_weights(root_probs: torch.Tensor, root_valid: torch.Tensor) -> torch.Tensor:
    w = torch.clamp(root_probs.float(), min=0.0) * root_valid.bool().float()
    return w / w.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def shared_option_success_regression_loss(
    pred_q: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    gamma: float = 0.0,
    temperature: float = 0.35,
) -> torch.Tensor:
    """Regress the probability that a *single shared* option succeeds.

    Executed DRS is not a per-root best-action metric.  At runtime the planner
    must pick one recovery option shared across all compatible roots.  This loss
    aggregates root mass first and supervises the option-level success vector
    that the selector actually needs.
    """
    root_w = _root_weights(root_probs, root_valid).unsqueeze(-1)
    finite = torch.isfinite(teacher_q)
    opt_mask = option_valid.bool()
    mask = root_valid.bool().unsqueeze(-1) & opt_mask.unsqueeze(1) & finite
    if not bool(mask.any()):
        return pred_q.sum() * 0.0
    tau = max(float(temperature), 1.0e-3)
    pred_prob = torch.sigmoid((torch.nan_to_num(pred_q, nan=-20.0, posinf=20.0, neginf=-20.0) - float(gamma)) / tau)
    teacher_prob = (teacher_q.detach() >= float(gamma)).float()
    pred_success = (pred_prob * root_w * mask.float()).sum(dim=1)
    teacher_success = (teacher_prob * root_w * mask.float()).sum(dim=1)
    valid_opt = opt_mask & mask.any(dim=1)
    if not bool(valid_opt.any()):
        return pred_q.sum() * 0.0
    return F.smooth_l1_loss(pred_success[valid_opt], teacher_success[valid_opt])


def shared_option_success_bce_loss(
    pred_q: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    gamma: float = 0.0,
    temperature: float = 0.35,
) -> torch.Tensor:
    """Stronger BCE supervision for shared-option success probability.

    Smooth-L1 on success probability is often numerically small even when the
    ranking is poor.  This BCE target gives larger gradients when the model is
    over-confident about low-success shared recovery options, which was the
    failure mode observed in v8: predicted DRS proxy was high while executed DRS
    remained low.
    """
    root_w = _root_weights(root_probs, root_valid).unsqueeze(-1)
    finite = torch.isfinite(teacher_q)
    opt_mask = option_valid.bool()
    mask = root_valid.bool().unsqueeze(-1) & opt_mask.unsqueeze(1) & finite
    if not bool(mask.any()):
        return pred_q.sum() * 0.0
    tau = max(float(temperature), 1.0e-3)
    pred_prob = torch.sigmoid((torch.nan_to_num(pred_q, nan=-20.0, posinf=20.0, neginf=-20.0) - float(gamma)) / tau)
    teacher_prob = (teacher_q.detach() >= float(gamma)).float()
    pred_success = (pred_prob * root_w * mask.float()).sum(dim=1).clamp(1.0e-4, 1.0 - 1.0e-4)
    teacher_success = (teacher_prob * root_w * mask.float()).sum(dim=1).clamp(0.0, 1.0)
    valid_opt = opt_mask & mask.any(dim=1)
    if not bool(valid_opt.any()):
        return pred_q.sum() * 0.0
    # Upweight ambiguous/intervention-critical options where success is neither
    # trivially 0 nor trivially 1.
    t = teacher_success[valid_opt]
    p = pred_success[valid_opt]
    w = 1.0 + 2.0 * torch.minimum(t, 1.0 - t)
    return (F.binary_cross_entropy(p, t, reduction="none") * w).mean()


def best_shared_option_loss(
    pred_q: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    gamma: float = 0.0,
    temperature: float = 0.35,
) -> torch.Tensor:
    """CE for the best globally shared recovery option.

    The old target used argmax_l Q(i,l) independently for each root i.  That is
    misaligned with deployability because one executed option must work across
    the compatible root set.  The teacher target is now the option maximizing
    root-mass deployable success, with a small value tie-breaker.
    """
    root_w = _root_weights(root_probs, root_valid).unsqueeze(-1)
    finite = torch.isfinite(teacher_q)
    opt_mask = option_valid.bool()
    mask = root_valid.bool().unsqueeze(-1) & opt_mask.unsqueeze(1) & finite
    if not bool(mask.any()) or not bool(opt_mask.any()):
        return pred_q.sum() * 0.0
    tau = max(float(temperature), 1.0e-3)
    teacher_success = ((teacher_q.detach() >= float(gamma)).float() * root_w * mask.float()).sum(dim=1)
    teacher_value = (torch.clamp(torch.nan_to_num(teacher_q.detach(), nan=-5.0, posinf=5.0, neginf=-5.0), -5.0, 5.0) * root_w * mask.float()).sum(dim=1)
    teacher_score = teacher_success + 0.05 * teacher_value
    teacher_score = torch.where(opt_mask, teacher_score, torch.full_like(teacher_score, -1.0e9))
    target = torch.argmax(teacher_score, dim=-1)

    pred_prob = torch.sigmoid((torch.nan_to_num(pred_q, nan=-20.0, posinf=20.0, neginf=-20.0) - float(gamma)) / tau)
    pred_success = (pred_prob * root_w * mask.float()).sum(dim=1)
    pred_value = (torch.clamp(torch.nan_to_num(pred_q, nan=-5.0, posinf=5.0, neginf=-5.0), -5.0, 5.0) * root_w * mask.float()).sum(dim=1)
    logits = pred_success + 0.05 * pred_value
    logits = torch.where(opt_mask, logits, torch.full_like(logits, -1.0e4))
    valid_row = opt_mask.any(dim=-1) & mask.any(dim=(1, 2))
    if not bool(valid_row.any()):
        return pred_q.sum() * 0.0
    return F.cross_entropy(logits[valid_row], target[valid_row])
