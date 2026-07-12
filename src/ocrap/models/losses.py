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



def groupwise_candidate_ranking_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    teacher_artifact: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    candidate_index: torch.Tensor,
    *,
    margin: float = 0.25,
    gap_weight: float = 0.25,
    teacher_gap_weight: float = 0.25,
    artifact_only: bool = True,
) -> torch.Tensor:
    """Rank candidates within the same scene-time group.

    The selector must choose a prefix in a scene-time candidate set, but the
    pointwise losses only teach absolute margins.  This loss pushes the learned
    score of the teacher-best deployable candidate above false-recoverable /
    oracle-artifact candidates from the same scene-time group.
    """
    if pred_r_dep.numel() <= 1:
        return pred_r_dep.sum() * 0.0
    pr = pred_r_dep.float().reshape(-1)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=5.0, neginf=0.0), min=0.0)
    trd = teacher_r_dep.float().reshape(-1)
    tro = teacher_r_orc.float().reshape(-1)
    ta = teacher_artifact.float().reshape(-1) > 0.5
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    ci = candidate_index.reshape(-1)
    n = min(pr.numel(), pg.numel(), trd.numel(), tro.numel(), ta.numel(), sh.numel(), ti.numel(), ci.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, trd, tro, ta, sh, ti, ci = pr[:n], pg[:n], trd[:n], tro[:n], ta[:n], sh[:n], ti[:n], ci[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(trd) & torch.isfinite(tro)
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_score = trd - float(teacher_gap_weight) * teacher_gap
    pred_score = pr - float(gap_weight) * pg
    losses: list[torch.Tensor] = []
    # Batch sizes are small; explicit grouping keeps the logic readable and CPU/GPU safe.
    keys = torch.stack([sh, ti], dim=1)
    unique = torch.unique(keys[finite], dim=0)
    for key in unique:
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < 2:
            continue
        idx = torch.where(mask)[0]
        # Prefer deployable teacher candidates; if none exists, rank by least-bad teacher score.
        deployable = idx[trd[idx] >= 0.0]
        pool = deployable if deployable.numel() > 0 else idx
        best_local = pool[torch.argmax(teacher_score[pool])]
        neg = idx[(tro[idx] >= 0.0) & (trd[idx] < 0.0)]
        if bool(artifact_only):
            neg = idx[ta[idx] | ((tro[idx] >= 0.0) & (trd[idx] < 0.0))]
        else:
            worse = teacher_score[idx] < teacher_score[best_local] - 1.0e-6
            neg = idx[worse | ta[idx] | ((tro[idx] >= 0.0) & (trd[idx] < 0.0))]
        neg = neg[neg != best_local]
        if neg.numel() == 0:
            continue
        diff = pred_score[best_local] - pred_score[neg]
        losses.append(F.relu(float(margin) - diff).mean())
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()

def groupwise_candidate_ce_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    utility: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    *,
    temperature: float = 0.35,
    pred_gap_weight: float = 0.35,
    teacher_gap_weight: float = 0.35,
    utility_weight: float = 0.03,
    require_deployable_target: bool = True,
) -> torch.Tensor:
    """Cross-entropy over candidates in each scene-time group.

    Pairwise artifact ranking is not enough for contact recovery: the selected
    teacher action can be a non-artifact candidate whose advantage is higher
    post-contact deployability, not merely avoiding an oracle artifact.  This
    loss treats the scene-time candidate set as the decision space and directly
    teaches the learned candidate score to put its probability mass on the
    teacher-best observation-consistent deployable candidate.
    """
    if pred_r_dep.numel() <= 1:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    u = torch.nan_to_num(utility.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    trd = teacher_r_dep.float().reshape(-1)
    tro = teacher_r_orc.float().reshape(-1)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    n = min(pr.numel(), pg.numel(), u.numel(), trd.numel(), tro.numel(), sh.numel(), ti.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, u, trd, tro, sh, ti = pr[:n], pg[:n], u[:n], trd[:n], tro[:n], sh[:n], ti[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(trd) & torch.isfinite(tro)
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    # Utility is deliberately weak here.  The teacher decision should primarily
    # reflect deployable recovery, but small utility ties keep benign scenes
    # nominal-preserving.
    teacher_score = trd - float(teacher_gap_weight) * teacher_gap + float(utility_weight) * u
    pred_score = pr - float(pred_gap_weight) * pg + float(utility_weight) * u
    tau = max(float(temperature), 1.0e-3)
    losses: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=1)
    unique = torch.unique(keys[finite], dim=0)
    for key in unique:
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < 2:
            continue
        idx = torch.where(mask)[0]
        target_pool = idx[trd[idx] >= 0.0] if bool(require_deployable_target) else idx
        if target_pool.numel() == 0:
            target_pool = idx
        best = target_pool[torch.argmax(teacher_score[target_pool])]
        local_target = int((idx == best).nonzero(as_tuple=False).reshape(-1)[0].item())
        logits = pred_score[idx].reshape(1, -1) / tau
        target = torch.tensor([local_target], device=pred_r_dep.device, dtype=torch.long)
        losses.append(F.cross_entropy(logits, target))
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()



def groupwise_score_distillation_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    utility: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    *,
    pred_gap_weight: float = 0.45,
    teacher_gap_weight: float = 0.45,
    utility_weight: float = 0.02,
    teacher_temperature: float = 0.20,
    pred_temperature: float = 0.30,
    min_group_size: int = 2,
) -> torch.Tensor:
    """Dense distillation of the teacher candidate ranking within a scene-time.

    v12 used a hard one-hot target for the single teacher-best candidate.  That
    gives sparse gradients and ignores the useful ordering among non-artifact
    deployable alternatives.  This KL loss transfers the full teacher ranking
    distribution over the candidate group, making contact recovery less dependent
    on artifact-only negatives.
    """
    if pred_r_dep.numel() <= 1:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    u = torch.nan_to_num(utility.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    trd = torch.nan_to_num(teacher_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    tro = torch.nan_to_num(teacher_r_orc.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    n = min(pr.numel(), pg.numel(), u.numel(), trd.numel(), tro.numel(), sh.numel(), ti.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, u, trd, tro, sh, ti = pr[:n], pg[:n], u[:n], trd[:n], tro[:n], sh[:n], ti[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(trd) & torch.isfinite(tro)
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_score = trd - float(teacher_gap_weight) * teacher_gap + float(utility_weight) * u
    pred_score = pr - float(pred_gap_weight) * pg + float(utility_weight) * u
    tt = max(float(teacher_temperature), 1.0e-3)
    pt = max(float(pred_temperature), 1.0e-3)
    losses: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=1)
    unique = torch.unique(keys[finite], dim=0)
    for key in unique:
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < int(min_group_size):
            continue
        idx = torch.where(mask)[0]
        target = torch.softmax(teacher_score[idx] / tt, dim=0).detach()
        logp = torch.log_softmax(pred_score[idx] / pt, dim=0)
        losses.append(F.kl_div(logp, target, reduction="batchmean"))
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()


def _differentiable_shared_success(
    pred_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    *,
    gamma: float = 0.0,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Differentiable proxy for globally shared option success per candidate."""
    if pred_q.ndim != 3:
        return pred_q.reshape(pred_q.shape[0], -1).mean(dim=-1) * 0.0
    w = torch.clamp(root_probs.float(), min=0.0) * root_valid.bool().float()
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    valid = option_valid.bool().unsqueeze(1)
    logits = (pred_q.float() - float(gamma)) / max(float(temperature), 1.0e-3)
    succ = torch.sigmoid(torch.nan_to_num(logits, nan=-20.0, posinf=20.0, neginf=-20.0))
    succ = torch.where(valid, succ, torch.zeros_like(succ))
    shared = (w.unsqueeze(-1) * succ).sum(dim=1)
    shared = torch.where(option_valid.bool(), shared, torch.full_like(shared, -1.0))
    return torch.clamp(shared.max(dim=-1).values, min=0.0, max=1.0)





def macro_shared_success_calibration_loss(
    pred_q: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    macro_type_id: torch.Tensor,
    bucket_id: torch.Tensor,
    *,
    macro_ids: tuple[int, ...] = (2, 3, 5, 7),
    bucket_ids: tuple[int, ...] = (1, 2),
    gamma: float = 0.0,
    temperature: float = 0.25,
    pos_threshold: float = 0.80,
    neg_threshold: float = 0.05,
    pos_weight: float = 4.0,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    """Macro-conditioned calibration for the shared recovery-option success head.

    v21 could rank a few protective macros, but the selector still missed true
    brake cases because their predicted shared-option success sometimes stayed
    near zero even when the teacher shared option succeeded.  This loss supervises
    the candidate-level DRS proxy used by the selector, only for semantic recovery
    macros in near/contact buckets.  Ambiguous middle-success cases are ignored so
    the gradients focus on high-confidence deployable and non-deployable macros.
    """
    if pred_q.ndim != 3 or pred_q.shape[0] == 0 or not macro_ids or not bucket_ids:
        return pred_q.reshape(pred_q.shape[0], -1).sum() * 0.0
    pred_drs = _differentiable_shared_success(
        pred_q, root_probs, root_valid, option_valid, gamma=gamma, temperature=temperature
    ).reshape(-1)
    with torch.no_grad():
        teacher_drs = _differentiable_shared_success(
            teacher_q, root_probs, root_valid, option_valid, gamma=gamma, temperature=max(0.08, temperature * 0.5)
        ).reshape(-1)
    mac = macro_type_id.reshape(-1)
    bid = bucket_id.reshape(-1)
    n = min(pred_drs.numel(), teacher_drs.numel(), mac.numel(), bid.numel())
    if n <= 0:
        return pred_q.sum() * 0.0
    pred_drs, teacher_drs, mac, bid = pred_drs[:n], teacher_drs[:n], mac[:n], bid[:n]
    finite = torch.isfinite(pred_drs) & torch.isfinite(teacher_drs)
    macro_mask = torch.zeros_like(finite)
    for m in tuple(int(x) for x in macro_ids):
        macro_mask |= mac == int(m)
    bucket_mask = torch.zeros_like(finite)
    for b in tuple(int(x) for x in bucket_ids):
        bucket_mask |= bid == int(b)
    high = teacher_drs >= float(pos_threshold)
    low = teacher_drs <= float(neg_threshold)
    mask = finite & macro_mask & bucket_mask & (high | low)
    if not bool(mask.any()):
        return pred_q.sum() * 0.0
    target = high.float()
    p = pred_drs.clamp(1.0e-4, 1.0 - 1.0e-4)
    weights = torch.where(high, torch.full_like(p, float(pos_weight)), torch.full_like(p, float(neg_weight)))
    loss = F.binary_cross_entropy(p[mask], target[mask], weight=weights[mask], reduction="mean")
    return loss

def _torch_pcd_score(drs: torch.Tensor, r_dep: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
    """Torch analogue of evaluation.metrics.post_contact_deployability_score."""
    return torch.clamp(drs, 0.0, 1.0) * torch.sigmoid(r_dep.float()) * torch.exp(-torch.clamp(gap.float(), min=0.0, max=20.0))


def protective_macro_recovery_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    utility: torch.Tensor,
    pred_q: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    macro_type_id: torch.Tensor,
    is_nominal: torch.Tensor,
    bucket_id: torch.Tensor,
    *,
    macro_ids: tuple[int, ...] = (2, 7),
    bucket_ids: tuple[int, ...] = (2,),
    margin: float = 0.14,
    min_teacher_r_dep: float = 0.0,
    min_teacher_drs: float = 0.50,
    min_teacher_pcd_gain: float = 0.02,
    max_nominal_teacher_pcd: float = 0.90,
    pred_gap_weight: float = 0.18,
    pred_drs_weight: float = 0.65,
    utility_weight: float = 0.02,
    teacher_gap_weight: float = 0.10,
    teacher_drs_weight: float = 0.70,
    success_gamma: float = 0.0,
    success_temperature: float = 0.25,
    target_min_pred_drs: float = 0.62,
) -> torch.Tensor:
    """Macro-conditioned supervision for low-headroom protective recovery.

    The selector in contact regimes is supposed to admit brake/stabilize only
    when they are semantically protective *and* deployably recoverable.  Pointwise
    R_dep regression under-trains this case because the useful brake/stabilize
    candidate is rare inside each scene-time candidate set.  This loss activates
    only when the teacher-labelled group contains a protective macro whose
    post-contact deployability (DRS * margin gate * oracle-gap discount) is better
    than nominal.  It then ranks that macro above nominal and non-protective
    alternatives using the same deployability coordinates used by the selector.
    """
    if pred_r_dep.numel() <= 1 or not macro_ids or not bucket_ids:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    u = torch.nan_to_num(utility.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    trd = torch.nan_to_num(teacher_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    tro = torch.nan_to_num(teacher_r_orc.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pred_drs = _differentiable_shared_success(
        pred_q, root_probs, root_valid, option_valid, gamma=success_gamma, temperature=success_temperature
    ).reshape(-1)
    with torch.no_grad():
        teacher_drs = _differentiable_shared_success(
            teacher_q, root_probs, root_valid, option_valid, gamma=success_gamma, temperature=max(0.08, success_temperature * 0.5)
        ).reshape(-1)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    mac = macro_type_id.reshape(-1)
    isn = is_nominal.float().reshape(-1) > 0.5
    bid = bucket_id.reshape(-1)
    n = min(pr.numel(), pg.numel(), u.numel(), trd.numel(), tro.numel(), pred_drs.numel(), teacher_drs.numel(), sh.numel(), ti.numel(), mac.numel(), isn.numel(), bid.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, u, trd, tro, pred_drs, teacher_drs = pr[:n], pg[:n], u[:n], trd[:n], tro[:n], pred_drs[:n], teacher_drs[:n]
    sh, ti, mac, isn, bid = sh[:n], ti[:n], mac[:n], isn[:n], bid[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(trd) & torch.isfinite(tro) & torch.isfinite(pred_drs) & torch.isfinite(teacher_drs)
    bucket_mask = torch.zeros_like(finite)
    for b in tuple(int(x) for x in bucket_ids):
        bucket_mask |= bid == int(b)
    macro_mask_all = torch.zeros_like(finite)
    for m in tuple(int(x) for x in macro_ids):
        macro_mask_all |= mac == int(m)
    finite = finite & bucket_mask
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_pcd = _torch_pcd_score(teacher_drs, trd, teacher_gap)
    pred_pcd = _torch_pcd_score(pred_drs, pr, pg)
    teacher_score = trd - float(teacher_gap_weight) * teacher_gap + float(teacher_drs_weight) * teacher_drs + teacher_pcd
    pred_score = pr - float(pred_gap_weight) * pg + float(pred_drs_weight) * pred_drs + pred_pcd + float(utility_weight) * u
    losses: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=1)
    unique = torch.unique(keys[finite], dim=0)
    for key in unique:
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < 2:
            continue
        idx = torch.where(mask)[0]
        nom_idx = idx[isn[idx]]
        if nom_idx.numel() == 0:
            continue
        nom = nom_idx[0]
        prot_idx = idx[macro_mask_all[idx]]
        if prot_idx.numel() == 0:
            continue
        teacher_ok = (
            (trd[prot_idx] >= float(min_teacher_r_dep))
            & (teacher_drs[prot_idx] >= float(min_teacher_drs))
            & (teacher_pcd[prot_idx] >= teacher_pcd[nom] + float(min_teacher_pcd_gain))
            & (teacher_pcd[nom] <= float(max_nominal_teacher_pcd))
        )
        if not bool(teacher_ok.any()):
            continue
        cand = prot_idx[teacher_ok]
        target = cand[torch.argmax(teacher_score[cand])]
        # Rank the teacher-certified protective macro above nominal.
        losses.append(F.relu(float(margin) - (pred_score[target] - pred_score[nom])))
        # Also separate it from non-protective alternatives in the same group,
        # but do not punish other protective candidates (they may be ties).
        nonprot = idx[(idx != target) & (~macro_mask_all[idx])]
        if nonprot.numel() > 0:
            losses.append(F.relu(float(margin) - (pred_score[target] - pred_score[nonprot])).mean())
        # The target should look deployable to the selector in both aggregate
        # margin and shared-option success coordinates.
        losses.append(F.relu(float(min_teacher_r_dep) - pr[target]))
        losses.append(F.relu(float(target_min_pred_drs) - pred_drs[target]))
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()


def safe_nominal_preservation_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    utility: torch.Tensor,
    pred_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    is_nominal: torch.Tensor,
    bucket_id: torch.Tensor,
    *,
    margin: float = 0.18,
    pred_gap_weight: float = 0.35,
    utility_weight: float = 0.03,
    drs_weight: float = 0.30,
    min_nominal_success: float = 0.90,
    success_gamma: float = 0.0,
    success_temperature: float = 0.25,
) -> torch.Tensor:
    """Learn a no-recovery certificate on safe/background groups.

    This is stronger than the v12 hard nominal lock: on safe groups, the model is
    explicitly trained to score the nominal prefix above recovery prefixes and to
    keep the nominal shared-option success proxy high.  At inference, the
    selector can then abstain from intervention when no deployable recovery is
    admitted, instead of blindly falling back to a high-utility recovery action.
    """
    if pred_r_dep.numel() <= 1:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    u = torch.nan_to_num(utility.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    drs = _differentiable_shared_success(
        pred_q, root_probs, root_valid, option_valid, gamma=success_gamma, temperature=success_temperature
    ).reshape(-1)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    isn = is_nominal.float().reshape(-1) > 0.5
    bid = bucket_id.reshape(-1)
    n = min(pr.numel(), pg.numel(), u.numel(), drs.numel(), sh.numel(), ti.numel(), isn.numel(), bid.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, u, drs, sh, ti, isn, bid = pr[:n], pg[:n], u[:n], drs[:n], sh[:n], ti[:n], isn[:n], bid[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(drs) & (bid == 0)
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0
    pred_score = pr - float(pred_gap_weight) * pg + float(utility_weight) * u + float(drs_weight) * drs
    losses: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=1)
    unique = torch.unique(keys[finite], dim=0)
    for key in unique:
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < 2:
            continue
        idx = torch.where(mask)[0]
        nom_local = idx[isn[idx]]
        if nom_local.numel() == 0:
            continue
        nom = nom_local[0]
        others = idx[idx != nom]
        if others.numel() > 0:
            losses.append(F.relu(float(margin) - (pred_score[nom] - pred_score[others])).mean())
        losses.append(F.relu(float(min_nominal_success) - drs[nom]))
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()


def nominal_switch_consistency_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    utility: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    is_nominal: torch.Tensor,
    *,
    margin: float = 0.12,
    pred_gap_weight: float = 0.30,
    teacher_gap_weight: float = 0.35,
    utility_weight: float = 0.03,
    teacher_gain_margin: float = 0.06,
    nominal_deployable_gamma: float = 0.0,
    nominal_gap_max: float = 0.30,
) -> torch.Tensor:
    """Teach the model when *not* to leave the logged nominal prefix.

    This is a learned nominal-preservation objective, not a hard rule.  If the
    nominal prefix is teacher-deployable and no alternative has a material
    teacher advantage, its learned score should dominate.  If an alternative has
    a material deployability/gap advantage, it should outrank nominal.
    """
    if pred_r_dep.numel() <= 1:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    u = torch.nan_to_num(utility.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    trd = teacher_r_dep.float().reshape(-1)
    tro = teacher_r_orc.float().reshape(-1)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    isn = is_nominal.float().reshape(-1) > 0.5
    n = min(pr.numel(), pg.numel(), u.numel(), trd.numel(), tro.numel(), sh.numel(), ti.numel(), isn.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, u, trd, tro, sh, ti, isn = pr[:n], pg[:n], u[:n], trd[:n], tro[:n], sh[:n], ti[:n], isn[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(trd) & torch.isfinite(tro)
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_score = trd - float(teacher_gap_weight) * teacher_gap + float(utility_weight) * u
    pred_score = pr - float(pred_gap_weight) * pg + float(utility_weight) * u
    losses: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=1)
    unique = torch.unique(keys[finite], dim=0)
    for key in unique:
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < 2:
            continue
        idx = torch.where(mask)[0]
        nom_local = idx[isn[idx]]
        if nom_local.numel() == 0:
            continue
        nom = nom_local[0]
        nom_good = (trd[nom] >= float(nominal_deployable_gamma)) and (teacher_gap[nom] <= float(nominal_gap_max))
        if not bool(nom_good):
            continue
        others = idx[idx != nom]
        if others.numel() == 0:
            continue
        adv = teacher_score[others] - teacher_score[nom]
        should_switch = (adv >= float(teacher_gain_margin)) & (trd[others] >= float(nominal_deployable_gamma))
        should_stay = ~should_switch
        if bool(should_stay.any()):
            diff_stay = pred_score[nom] - pred_score[others[should_stay]]
            losses.append(F.relu(float(margin) - diff_stay).mean())
        if bool(should_switch.any()):
            diff_switch = pred_score[others[should_switch]] - pred_score[nom]
            losses.append(F.relu(float(margin) - diff_switch).mean())
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()

