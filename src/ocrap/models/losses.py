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







def _exact_teacher_shared_success(
    teacher_q: torch.Tensor,
    teacher_m_star: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    *,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Exact hard shared-option DRS used by calibration, in batched Torch form.

    The selected option maximizes root-probability success mass under teacher-q,
    with the same 0.01 mean-value tie-break as ``best_shared_option_index``.
    Success is then evaluated on teacher ``m_star`` exactly as deployment
    calibration does.  The result is a detached supervision target.
    """
    if teacher_q.ndim != 3 or teacher_m_star.ndim != 3:
        return teacher_q.reshape(teacher_q.shape[0], -1).mean(dim=-1) * 0.0
    q = torch.nan_to_num(teacher_q.float(), nan=-1.0e9, posinf=5.0, neginf=-5.0)
    m = torch.nan_to_num(teacher_m_star.float(), nan=-1.0e9, posinf=5.0, neginf=-5.0)
    b, k, l = q.shape
    if m.shape[:2] != (b, k):
        raise ValueError(f"teacher_m_star shape {tuple(m.shape)} incompatible with teacher_q {tuple(q.shape)}")
    if m.shape[2] != l:
        ll = min(l, m.shape[2])
        q = q[:, :, :ll]
        m = m[:, :, :ll]
        l = ll
    w = torch.clamp(root_probs.float()[:, :k], min=0.0)
    rv = root_valid.bool()[:, :k]
    w = w * rv.float()
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    ov = option_valid.bool()[:, :l]
    finite = rv.unsqueeze(-1) & ov.unsqueeze(1) & torch.isfinite(q)
    success_mass = (((q >= float(gamma)) & finite).float() * w.unsqueeze(-1)).sum(dim=1)
    value_score = (torch.where(finite, q.clamp(-5.0, 5.0), torch.zeros_like(q)) * w.unsqueeze(-1)).sum(dim=1)
    option_score = success_mass + 0.01 * value_score
    option_score = torch.where(ov, option_score, torch.full_like(option_score, -1.0e9))
    opt = option_score.argmax(dim=-1)
    gather_idx = opt.view(b, 1, 1).expand(-1, k, 1)
    selected_margin = torch.gather(m, dim=2, index=gather_idx).squeeze(-1)
    selected_valid = rv & torch.gather(ov.unsqueeze(1).expand(-1, k, -1), 2, gather_idx).squeeze(-1)
    success = selected_valid & torch.isfinite(selected_margin) & (selected_margin >= 0.0)
    return (w * success.float()).sum(dim=-1).clamp(0.0, 1.0).detach()

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




def deployability_dominance_calibration_loss(
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
    macro_ids: tuple[int, ...] = (2, 3, 5, 7),
    bucket_ids: tuple[int, ...] = (2,),
    margin: float = 0.12,
    min_teacher_pcd_gain: float = 0.04,
    min_teacher_best_pcd: float = 0.50,
    max_nominal_teacher_pcd: float = 0.62,
    pred_gap_weight: float = 0.20,
    pred_drs_weight: float = 0.35,
    utility_weight: float = 0.00,
    success_gamma: float = 0.0,
    success_temperature: float = 0.25,
    target_min_pred_pcd: float = 0.45,
    nominal_max_pred_pcd: float = 0.55,
) -> torch.Tensor:
    """Regime-conditioned deployability-dominance calibration.

    Selector-only rescue can only use learned deployability heads.  The v26
    audits show that those heads still rank a high-confidence nominal prefix
    above teacher-deployable recovery macros in some post-contact states, while
    in near-contact they over-admit brake when merge/yield/nominal is the true
    deployable choice.  This loss distills the *group-wise* teacher PCD ordering
    directly into the model: when a paper-eligible recovery macro has materially
    higher teacher PCD than nominal, its predicted PCD score must outrank nominal;
    when nominal teacher PCD is poor, the nominal predicted PCD is also capped.
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
    pr, pg, u, trd, tro = pr[:n], pg[:n], u[:n], trd[:n], tro[:n]
    pred_drs, teacher_drs = pred_drs[:n], teacher_drs[:n]
    sh, ti, mac, isn, bid = sh[:n], ti[:n], mac[:n], isn[:n], bid[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(trd) & torch.isfinite(tro) & torch.isfinite(pred_drs) & torch.isfinite(teacher_drs)
    bucket_mask = torch.zeros_like(finite)
    for b in tuple(int(x) for x in bucket_ids):
        bucket_mask |= bid == int(b)
    macro_mask = torch.zeros_like(finite)
    for m in tuple(int(x) for x in macro_ids):
        macro_mask |= mac == int(m)
    finite = finite & bucket_mask
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_pcd = _torch_pcd_score(teacher_drs, trd, teacher_gap)
    pred_pcd = _torch_pcd_score(pred_drs, pr, pg)
    # PCD is the primary target; the auxiliary score retains margin/gap/utility
    # signals so gradients do not collapse when PCD values tie near 0 or 0.62.
    pred_score = pred_pcd + pred_drs_weight * pred_drs + pr - pred_gap_weight * pg + utility_weight * u
    teacher_score = teacher_pcd + 0.25 * teacher_drs + trd - 0.10 * teacher_gap
    losses: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=1)
    for key in torch.unique(keys[finite], dim=0):
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < 2:
            continue
        idx = torch.where(mask)[0]
        nom_idx = idx[isn[idx]]
        if nom_idx.numel() == 0:
            continue
        nom = nom_idx[0]
        rec_idx = idx[macro_mask[idx]]
        if rec_idx.numel() == 0:
            continue
        good = rec_idx[
            (teacher_pcd[rec_idx] >= float(min_teacher_best_pcd))
            & (teacher_pcd[rec_idx] >= teacher_pcd[nom] + float(min_teacher_pcd_gain))
            & (teacher_pcd[nom] <= float(max_nominal_teacher_pcd))
        ]
        if good.numel() == 0:
            continue
        target = good[torch.argmax(teacher_score[good])]
        losses.append(F.relu(float(margin) - (pred_score[target] - pred_score[nom])))
        losses.append(F.relu(float(target_min_pred_pcd) - pred_pcd[target]))
        losses.append(F.relu(pred_pcd[nom] - float(nominal_max_pred_pcd)))
        # Separate from other non-target candidates in the group, but with a
        # smaller margin so legitimate ties among recovery families are allowed.
        others = idx[idx != target]
        if others.numel() > 0:
            losses.append(F.relu(0.5 * float(margin) - (pred_score[target] - pred_score[others])).mean())
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()


def direct_teacher_pcd_loss(
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
    macro_ids: tuple[int, ...] = (2, 3, 5, 7),
    positive_macro_ids: tuple[int, ...] | None = None,
    bucket_ids: tuple[int, ...] = (2,),
    success_gamma: float = 0.0,
    success_temperature: float = 0.25,
    regression_weight: float = 1.0,
    ranking_weight: float = 2.5,
    nominal_penalty_weight: float = 1.0,
    false_positive_weight: float = 1.5,
    margin: float = 0.18,
    min_teacher_pcd_gain: float = 0.015,
    min_teacher_best_pcd: float = 0.50,
    max_nominal_teacher_pcd: float = 0.68,
    target_min_pred_pcd: float = 0.52,
    nominal_max_pred_pcd: float = 0.50,
    focus_non_nominal_weight: float = 2.0,
    false_positive_margin: float = 0.03,
    component_weight: float = 0.0,
    positive_component_weight: float = 0.0,
    nominal_cap_weight: float = 1.0,
    positive_rank_all_weight: float = 0.0,
    positive_floor_weight: float = 0.0,
    positive_min_pred_r_dep: float = -1.0e9,
    positive_max_pred_gap: float = -1.0,
    positive_min_pred_drs: float = -1.0,
) -> torch.Tensor:
    """Directly distill teacher post-contact deployability into learned PCD.

    v30 audits showed a selector-side failure that cannot be fixed by loosening
    thresholds alone: in contact groups, paper-best brake/yield candidates can
    have higher teacher PCD while the learned heads still rank nominal or an
    unrelated merge/stabilize candidate higher.  DDC only trains a sparse
    dominance event.  This loss supervises the *absolute* teacher PCD for all
    candidates in the requested regimes and adds group-wise ranking/anti-false-
    positive terms around nominal.

    Macro ids follow prefix_generation.MACROS:
    nominal=0, brake=2, yield=3, merge=5, stabilize=7.

    v32 adds a separate positive_macro_ids argument.  macro_ids still controls
    dense regression and anti-false-positive suppression, while positive_macro_ids
    controls which recovery families are allowed to be promoted above nominal.
    This lets us keep merge/stabilize calibrated low without letting them compete
    with the contact brake/yield frontier.
    """
    if pred_r_dep.numel() <= 1 or not bucket_ids:
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
    pr, pg, u, trd, tro = pr[:n], pg[:n], u[:n], trd[:n], tro[:n]
    pred_drs, teacher_drs = pred_drs[:n], teacher_drs[:n]
    sh, ti, mac, isn, bid = sh[:n], ti[:n], mac[:n], isn[:n], bid[:n]

    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(trd) & torch.isfinite(tro) & torch.isfinite(pred_drs) & torch.isfinite(teacher_drs)
    bucket_mask = torch.zeros_like(finite)
    for b in tuple(int(x) for x in bucket_ids):
        bucket_mask |= bid == int(b)
    macro_mask = torch.zeros_like(finite)
    for m in tuple(int(x) for x in macro_ids):
        macro_mask |= mac == int(m)
    pos_ids = macro_ids if positive_macro_ids is None else positive_macro_ids
    positive_macro_mask = torch.zeros_like(finite)
    for m in tuple(int(x) for x in pos_ids):
        positive_macro_mask |= mac == int(m)
    finite = finite & bucket_mask
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0

    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_pcd = _torch_pcd_score(teacher_drs, trd, teacher_gap).detach()
    pred_pcd = _torch_pcd_score(pred_drs, pr, pg)

    # Absolute regression is what v30 was missing: bad merge/stabilize rescue
    # candidates should be explicitly regressed to low teacher PCD, not only
    # filtered at selector time.
    reg_mask = finite
    reg_w = torch.ones_like(pred_pcd)
    reg_w = torch.where(macro_mask & finite, reg_w * float(focus_non_nominal_weight), reg_w)
    reg = F.smooth_l1_loss(pred_pcd[reg_mask], teacher_pcd[reg_mask], reduction="none")
    reg_loss = (reg * reg_w[reg_mask]).sum() / reg_w[reg_mask].sum().clamp_min(1.0e-6)

    # Component-level supervision keeps gradients alive when the multiplicative
    # PCD proxy is wrong for several reasons at once (the v31 contact misses had
    # brake DRS high but learned R_dep too low and gap too high).
    comp_mask = finite & (macro_mask | isn)
    if bool(comp_mask.any()):
        comp = F.smooth_l1_loss(pr[comp_mask], trd[comp_mask], reduction="none")
        comp = comp + 0.5 * F.smooth_l1_loss(pg[comp_mask], teacher_gap[comp_mask], reduction="none")
        comp = comp + F.smooth_l1_loss(pred_drs[comp_mask], teacher_drs[comp_mask], reduction="none")
        comp_loss = comp.mean()
    else:
        comp_loss = pred_r_dep.sum() * 0.0

    rank_losses: list[torch.Tensor] = []
    nominal_losses: list[torch.Tensor] = []
    false_pos_losses: list[torch.Tensor] = []
    pred_score = pred_pcd + 0.20 * pred_drs + pr - 0.10 * pg + 0.01 * u
    keys = torch.stack([sh, ti], dim=1)
    for key in torch.unique(keys[finite], dim=0):
        group = finite & (sh == key[0]) & (ti == key[1])
        if int(group.sum().item()) < 2:
            continue
        idx = torch.where(group)[0]
        nom_idx = idx[isn[idx]]
        if nom_idx.numel() == 0:
            continue
        nom = nom_idx[0]
        rec_idx = idx[macro_mask[idx]]
        pos_rec_idx = idx[positive_macro_mask[idx]]
        if rec_idx.numel() == 0 or pos_rec_idx.numel() == 0:
            continue

        good = pos_rec_idx[
            (teacher_pcd[pos_rec_idx] >= float(min_teacher_best_pcd))
            & (teacher_pcd[pos_rec_idx] >= teacher_pcd[nom] + float(min_teacher_pcd_gain))
            & (teacher_pcd[nom] <= float(max_nominal_teacher_pcd))
        ]
        if good.numel() > 0:
            target = good[torch.argmax(teacher_pcd[good])]
            rank_losses.append(F.relu(float(margin) - (pred_score[target] - pred_score[nom])))
            rank_losses.append(F.relu(float(target_min_pred_pcd) - pred_pcd[target]))
            if float(positive_rank_all_weight) > 0.0:
                rank_losses.append(float(positive_rank_all_weight) * F.relu(float(margin) - (pred_score[good] - pred_score[nom])).mean())
            if float(positive_floor_weight) > 0.0:
                floor_terms = [F.relu(float(target_min_pred_pcd) - pred_pcd[good])]
                if float(positive_min_pred_r_dep) > -1.0e8:
                    floor_terms.append(F.relu(float(positive_min_pred_r_dep) - pr[good]))
                if float(positive_max_pred_gap) >= 0.0:
                    floor_terms.append(F.relu(pg[good] - float(positive_max_pred_gap)))
                if float(positive_min_pred_drs) >= 0.0:
                    floor_terms.append(F.relu(float(positive_min_pred_drs) - pred_drs[good]))
                rank_losses.append(float(positive_floor_weight) * torch.stack([t.mean() for t in floor_terms]).mean())
            if float(positive_component_weight) > 0.0:
                rank_losses.append(float(positive_component_weight) * F.smooth_l1_loss(pr[target], trd[target]))
                rank_losses.append(float(positive_component_weight) * 0.5 * F.smooth_l1_loss(pg[target], teacher_gap[target]))
                rank_losses.append(float(positive_component_weight) * F.smooth_l1_loss(pred_drs[target], teacher_drs[target]))
            nominal_losses.append(float(nominal_cap_weight) * F.relu(pred_pcd[nom] - float(nominal_max_pred_pcd)))
            worse = idx[(idx != target) & (teacher_pcd[idx] <= teacher_pcd[target] - float(min_teacher_pcd_gain))]
            if worse.numel() > 0:
                rank_losses.append(F.relu(0.5 * float(margin) - (pred_score[target] - pred_score[worse])).mean())

        # Anti false-positive: if nominal is better than a recovery macro under
        # teacher PCD, do not let learned PCD fabricate a rescue.  This targets
        # the v30 bad merge/stabilize rows where paper-best was nominal.
        bad_rec = rec_idx[teacher_pcd[nom] >= teacher_pcd[rec_idx] + float(false_positive_margin)]
        if bad_rec.numel() > 0:
            false_pos_losses.append(F.relu(float(margin) - (pred_score[nom] - pred_score[bad_rec])).mean())

    zero = pred_r_dep.sum() * 0.0
    rank_loss = torch.stack(rank_losses).mean() if rank_losses else zero
    nominal_loss = torch.stack(nominal_losses).mean() if nominal_losses else zero
    false_pos_loss = torch.stack(false_pos_losses).mean() if false_pos_losses else zero
    return (
        float(regression_weight) * reg_loss
        + float(ranking_weight) * rank_loss
        + float(nominal_penalty_weight) * nominal_loss
        + float(false_positive_weight) * false_pos_loss
        + float(component_weight) * comp_loss
    )

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




def observation_consistent_recovery_advantage_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
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
    macro_ids: tuple[int, ...] = (2, 3, 5, 7),
    bucket_ids: tuple[int, ...] = (1, 2),
    positive_gain: float = 0.03,
    negative_gain: float = 0.03,
    advantage_margin: float = 0.10,
    regression_weight: float = 1.0,
    ranking_weight: float = 1.0,
    component_inversion_weight: float = 0.5,
    false_positive_weight: float = 0.75,
    nominal_failure_pcd_max: float = 0.20,
    target_min_pred_pcd: float = 0.50,
    nominal_max_pred_pcd: float = 0.48,
    near_weight: float = 1.5,
    contact_weight: float = 1.0,
    success_gamma: float = 0.0,
    success_temperature: float = 0.25,
) -> torch.Tensor:
    """Observation-consistent counterfactual recovery-advantage calibration.

    Absolute PCD regression is not enough when the learned heads make a *paired
    inversion*: nominal is predicted highly deployable although its teacher PCD
    is near zero, while a shared recovery macro is predicted worse despite being
    the teacher-best action.  This loss operates on complete scene-time candidate
    groups and directly calibrates the counterfactual advantage

        Delta_PCD = PCD(best shared recovery) - PCD(nominal).

    Positive rescue groups receive advantage regression, pairwise ranking, a
    target floor, and component-wise anti-inversion supervision.  Negative rescue
    groups receive an anti-false-positive ranking term.  Bucket-specific weights
    make near-contact calibration explicit instead of inheriting a contact-only
    brake-tail objective.
    """
    if pred_r_dep.numel() <= 1 or not macro_ids or not bucket_ids:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    trd = torch.nan_to_num(teacher_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    tro = torch.nan_to_num(teacher_r_orc.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pred_drs = _differentiable_shared_success(
        pred_q, root_probs, root_valid, option_valid,
        gamma=success_gamma, temperature=success_temperature,
    ).reshape(-1)
    with torch.no_grad():
        teacher_drs = _differentiable_shared_success(
            teacher_q, root_probs, root_valid, option_valid,
            gamma=success_gamma, temperature=max(0.08, success_temperature * 0.5),
        ).reshape(-1)
    sh, ti = scene_hash.reshape(-1), time_index.reshape(-1)
    mac = macro_type_id.reshape(-1)
    isn = is_nominal.float().reshape(-1) > 0.5
    bid = bucket_id.reshape(-1)
    n = min(pr.numel(), pg.numel(), trd.numel(), tro.numel(), pred_drs.numel(), teacher_drs.numel(), sh.numel(), ti.numel(), mac.numel(), isn.numel(), bid.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, trd, tro = pr[:n], pg[:n], trd[:n], tro[:n]
    pred_drs, teacher_drs = pred_drs[:n], teacher_drs[:n]
    sh, ti, mac, isn, bid = sh[:n], ti[:n], mac[:n], isn[:n], bid[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(trd) & torch.isfinite(tro) & torch.isfinite(pred_drs) & torch.isfinite(teacher_drs)
    bucket_mask = torch.zeros_like(finite)
    for b in tuple(int(x) for x in bucket_ids):
        bucket_mask |= bid == b
    macro_mask = torch.zeros_like(finite)
    for m in tuple(int(x) for x in macro_ids):
        macro_mask |= mac == m
    finite &= bucket_mask
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0

    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_pcd = _torch_pcd_score(teacher_drs, trd, teacher_gap).detach()
    pred_pcd = _torch_pcd_score(pred_drs, pr, pg)
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    keys = torch.stack([sh, ti], dim=1)
    for key in torch.unique(keys[finite], dim=0):
        group = finite & (sh == key[0]) & (ti == key[1])
        idx = torch.where(group)[0]
        if idx.numel() < 2:
            continue
        noms = idx[isn[idx]]
        recs = idx[macro_mask[idx]]
        if noms.numel() == 0 or recs.numel() == 0:
            continue
        nom = noms[0]
        target = recs[torch.argmax(teacher_pcd[recs])]
        teacher_adv = teacher_pcd[target] - teacher_pcd[nom]
        pred_adv = pred_pcd[target] - pred_pcd[nom]
        bw = float(near_weight if int(bid[nom].item()) == 1 else contact_weight)
        group_terms: list[torch.Tensor] = []
        if float(teacher_adv.item()) >= float(positive_gain):
            group_terms.append(float(regression_weight) * F.smooth_l1_loss(pred_adv, teacher_adv))
            group_terms.append(float(ranking_weight) * F.relu(float(advantage_margin) - pred_adv))
            group_terms.append(float(ranking_weight) * 0.5 * F.relu(float(target_min_pred_pcd) - pred_pcd[target]))
            if float(teacher_pcd[nom].item()) <= float(nominal_failure_pcd_max):
                group_terms.append(float(ranking_weight) * 0.5 * F.relu(pred_pcd[nom] - float(nominal_max_pred_pcd)))
            # Anti-inversion on the three deployability components.  The teacher
            # deltas, rather than fixed semantic assumptions, define the sign.
            component = F.smooth_l1_loss(pred_drs[target] - pred_drs[nom], teacher_drs[target] - teacher_drs[nom])
            component = component + F.smooth_l1_loss(pr[target] - pr[nom], trd[target] - trd[nom])
            component = component + 0.5 * F.smooth_l1_loss(pg[nom] - pg[target], teacher_gap[nom] - teacher_gap[target])
            group_terms.append(float(component_inversion_weight) * component)
        elif float(teacher_adv.item()) <= -float(negative_gain):
            group_terms.append(float(false_positive_weight) * F.relu(float(advantage_margin) + pred_adv))
        else:
            # Ambiguous ties still receive gentle advantage regression; this
            # stabilizes the decision boundary without forcing intervention.
            group_terms.append(0.25 * float(regression_weight) * F.smooth_l1_loss(pred_adv, teacher_adv))
        losses.append(torch.stack(group_terms).mean())
        weights.append(bw)
    if not losses:
        return pred_r_dep.sum() * 0.0
    w = torch.as_tensor(weights, dtype=losses[0].dtype, device=losses[0].device)
    stacked = torch.stack(losses)
    return (stacked * w).sum() / w.sum().clamp_min(1.0e-6)



def direct_uncertainty_recovery_value_loss(
    pred_logit: torch.Tensor,
    pred_logvar: torch.Tensor,
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
    macro_ids: tuple[int, ...] = (2, 3, 5, 7),
    bucket_ids: tuple[int, ...] = (1, 2),
    temperature: float = 0.12,
    positive_gain: float = 0.03,
    negative_gain: float = 0.02,
    rank_margin: float = 0.04,
    point_weight: float = 0.15,
    listwise_weight: float = 0.35,
    advantage_weight: float = 1.0,
    centered_weight: float = 1.0,
    positive_group_weight: float = 4.0,
    negative_group_weight: float = 1.0,
    ambiguous_group_weight: float = 0.25,
    near_weight: float = 1.5,
    contact_weight: float = 1.0,
    min_group_range: float = 0.01,
    false_positive_weight: float = 1.0,
    variance_floor: float = 0.0025,
    output_mode: str = "probability",
    pairwise_weight: float = 0.0,
    top_rank_weight: float = 0.0,
    success_gamma: float = 0.0,
    success_temperature: float = 0.25,
    pred_opportunity_logit: torch.Tensor | None = None,
    opportunity_weight: float = 0.0,
    opportunity_pos_weight: float = 6.0,
    pred_harm_logit: torch.Tensor | None = None,
    harm_weight: float = 0.0,
    harm_pos_weight: float = 4.0,
    setwise_admission_weight: float = 0.0,
    opportunity_admission_weight: float = 0.35,
    harm_admission_weight: float = 0.75,
    selective_risk_weight: float = 0.0,
    selective_harm_budget: float = 0.05,
    selective_coverage_weight: float = 0.0,
    selective_coverage_target: float = 0.65,
    policy_distill_weight: float = 0.0,
    policy_teacher_temperature: float = 0.08,
    policy_regret_weight: float = 0.0,
    policy_regret_margin: float = 0.0,
    opportunity_soft_label_temperature: float = 0.0,
    harm_soft_label_temperature: float = 0.0,
    policy_decouple_admission: bool = True,
    policy_admission_distill_weight: float = 0.0,
    group_dro_weight: float = 0.0,
    group_dro_temperature: float = 0.35,
    group_dro_severity_thresholds: tuple[float, ...] = (0.25, 0.55),
    teacher_m_star: torch.Tensor | None = None,
    exact_teacher_pcd: bool = False,
    pred_rank_logit: torch.Tensor | None = None,
    preference_weight: float = 0.0,
    preference_temperature: float = 0.08,
    preference_min_gap: float = 0.01,
    preference_margin: float = 0.02,
    preference_confidence_scale: float = 0.05,
    preference_regret_weight: float = 0.0,
    preference_listwise_weight: float = 0.0,
    preference_gap_weight: float = 0.0,
    delta_nll_weight: float = 0.0,
    pred_delta_mean: torch.Tensor | None = None,
    pred_delta_logvar: torch.Tensor | None = None,
) -> torch.Tensor:
    """Learn a tri-state, nominal-relative, policy-level recovery certificate.

    v40 used an absolute heteroscedastic PCD target.  Positive recovery groups
    are rare, so that objective is dominated by easy negative/tied groups and a
    frozen scene CLS can converge to an almost constant prediction.  v41 keeps a
    small absolute anchor but makes the primary target the scene-time centred
    advantage ``PCD(candidate)-PCD(nominal)``.  Positive and negative groups are
    balanced explicitly, and listwise supervision is skipped for true ties.

    The variance output remains available for diagnostics/backward compatibility;
    selection uses a validation-calibrated additive residual bound instead of
    trusting the model's self-reported standard deviation.
    """
    if pred_logit.numel() <= 1:
        return pred_logit.sum() * 0.0
    raw_score = pred_logit.float().reshape(-1)
    mode = str(output_mode or "probability").strip().lower()
    if mode not in {"probability", "score"}:
        raise ValueError(f"Unsupported direct value output mode: {output_mode!r}")
    # v42 OCSAVA predicts an unbounded preference score. Only the optional
    # absolute anchor is mapped through sigmoid; pairwise/listwise objectives
    # operate on the raw score so candidate advantages are not compressed.
    score = raw_score if mode == "score" else torch.sigmoid(raw_score)
    rank_score = score if pred_rank_logit is None else pred_rank_logit.float().reshape(-1)
    direct_delta = None if pred_delta_mean is None else pred_delta_mean.float().reshape(-1)
    direct_delta_logvar = None if pred_delta_logvar is None else pred_delta_logvar.float().reshape(-1).clamp(-7.0, 2.0)
    point_mean = torch.sigmoid(raw_score)
    logvar = pred_logvar.float().reshape(-1).clamp(-7.0, 2.0)
    trd = torch.nan_to_num(teacher_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    tro = torch.nan_to_num(teacher_r_orc.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    with torch.no_grad():
        if bool(exact_teacher_pcd):
            if teacher_m_star is None:
                raise ValueError("exact_teacher_pcd=true requires teacher_m_star")
            teacher_drs = _exact_teacher_shared_success(
                teacher_q, teacher_m_star, root_probs, root_valid, option_valid, gamma=success_gamma
            ).reshape(-1)
        else:
            teacher_drs = _differentiable_shared_success(
                teacher_q, root_probs, root_valid, option_valid,
                gamma=success_gamma, temperature=max(0.08, success_temperature * 0.5),
            ).reshape(-1)
    sh, ti = scene_hash.reshape(-1), time_index.reshape(-1)
    mac = macro_type_id.reshape(-1)
    isn = is_nominal.float().reshape(-1) > 0.5
    bid = bucket_id.reshape(-1)
    sizes = [score.numel(), rank_score.numel(), point_mean.numel(), logvar.numel(), trd.numel(), tro.numel(), teacher_drs.numel(), sh.numel(), ti.numel(), mac.numel(), isn.numel(), bid.numel()]
    if direct_delta is not None:
        sizes.append(direct_delta.numel())
    if direct_delta_logvar is not None:
        sizes.append(direct_delta_logvar.numel())
    n = min(sizes)
    score, rank_score, point_mean, logvar, trd, tro, teacher_drs = score[:n], rank_score[:n], point_mean[:n], logvar[:n], trd[:n], tro[:n], teacher_drs[:n]
    sh, ti, mac, isn, bid = sh[:n], ti[:n], mac[:n], isn[:n], bid[:n]
    if direct_delta is not None:
        direct_delta = direct_delta[:n]
    if direct_delta_logvar is not None:
        direct_delta_logvar = direct_delta_logvar[:n]
    bucket_mask = torch.zeros((n,), dtype=torch.bool, device=score.device)
    for b in tuple(int(x) for x in bucket_ids):
        bucket_mask |= bid == b
    finite = bucket_mask & torch.isfinite(score) & torch.isfinite(rank_score) & torch.isfinite(point_mean) & torch.isfinite(logvar) & torch.isfinite(trd) & torch.isfinite(tro) & torch.isfinite(teacher_drs)
    if not bool(finite.any()):
        return pred_logit.sum() * 0.0
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    target = _torch_pcd_score(teacher_drs, trd, teacher_gap).detach().clamp(0.0, 1.0)
    variance = torch.exp(logvar).clamp_min(float(variance_floor))
    point = 0.5 * ((point_mean - target) ** 2 / variance + torch.log(variance))
    point_loss = point[finite].mean()

    macro_mask = torch.zeros((n,), dtype=torch.bool, device=score.device)
    for m in tuple(int(x) for x in macro_ids):
        macro_mask |= mac == m
    # v44: include regime in the group key; Near and Contact may share the
    # same WOMD scene-time but have different pressure-future teacher targets.
    keys = torch.stack([bid, sh, ti], dim=1)
    tau = max(float(temperature), 1.0e-3)
    opportunity_logits = None
    if pred_opportunity_logit is not None:
        opportunity_logits = pred_opportunity_logit.float().reshape(-1)[:n]
    harm_logits = None
    if pred_harm_logit is not None:
        harm_logits = pred_harm_logit.float().reshape(-1)[:n]
    group_losses: list[torch.Tensor] = []
    group_weights: list[float] = []
    group_domains: list[tuple[int, int, int, int]] = []
    for key in torch.unique(keys[finite], dim=0):
        idx = torch.where(finite & (bid == key[0]) & (sh == key[1]) & (ti == key[2]))[0]
        noms = idx[isn[idx]]
        recs = idx[macro_mask[idx] & (~isn[idx])]
        if noms.numel() == 0 or recs.numel() == 0:
            continue
        nom = noms[0]
        t_delta = target[recs] - target[nom]
        p_delta = score[recs] - score[nom]
        if direct_delta is not None:
            p_delta = direct_delta[recs]
        r_delta = rank_score[recs] - rank_score[nom]
        # Emphasise decision-relevant deltas while retaining all candidates.
        mag_w = 1.0 + torch.clamp(t_delta.abs() / max(float(positive_gain), 1.0e-3), max=4.0)
        centered = (mag_w * F.smooth_l1_loss(p_delta, t_delta, reduction='none')).sum() / mag_w.sum().clamp_min(1.0e-6)
        terms = [float(centered_weight) * centered]
        if float(delta_nll_weight) > 0.0:
            if direct_delta_logvar is not None:
                delta_var = torch.exp(direct_delta_logvar[recs]).clamp_min(float(variance_floor))
            else:
                delta_var = (variance[recs] + variance[nom]).clamp_min(float(variance_floor))
            delta_nll = 0.5 * (((p_delta - t_delta).square() / delta_var) + torch.log(delta_var))
            terms.append(float(delta_nll_weight) * (mag_w * delta_nll).sum() / mag_w.sum().clamp_min(1.0e-6))
        group_range = float((target[idx].max() - target[idx].min()).item())
        if group_range >= float(min_group_range) and float(listwise_weight) > 0.0:
            teacher_prob = torch.softmax((target[idx] - target[idx].mean()) / tau, dim=0)
            pred_log_prob = torch.log_softmax((score[idx] - score[idx].mean()) / tau, dim=0)
            terms.append(float(listwise_weight) * F.kl_div(pred_log_prob, teacher_prob, reduction='sum'))
        best_j = int(torch.argmax(t_delta).item())
        t_adv = t_delta[best_j]
        p_adv = p_delta[best_j]
        pos_mask = t_delta >= float(positive_gain)
        harmful_mask = t_delta <= -float(negative_gain)
        # Only deltas beyond the negative margin are harmful negatives. Exact
        # and near ties form the tri-state dead zone and should be regressed
        # gently toward their teacher delta rather than forced below a margin.
        neg_mask = harmful_mask
        tie_mask = (~pos_mask) & (~harmful_mask)
        # v48.5 ECPR: confidence-paced exact best-vs-rest preference.  It is
        # applied only when the exact teacher has a material recovery opportunity,
        # and downweights near-ties whose ordering is not stable across splits.
        if bool(pos_mask.any()) and (float(preference_weight) > 0.0 or float(preference_regret_weight) > 0.0):
            best_rank_j = int(torch.argmax(t_delta).item())
            best_teacher = t_delta[best_rank_j]
            gaps = (best_teacher - t_delta).detach()
            competitor = torch.arange(recs.numel(), device=recs.device) != best_rank_j
            competitor &= gaps >= float(preference_min_gap)
            pref_tau = max(float(preference_temperature), 1.0e-3)
            if bool(competitor.any()) and float(preference_weight) > 0.0:
                confidence = torch.clamp(gaps[competitor] / max(float(preference_confidence_scale), 1.0e-4), 0.0, 1.0)
                pred_gap = r_delta[best_rank_j] - r_delta[competitor]
                required = float(preference_margin) + torch.clamp(gaps[competitor], max=0.25)
                pref = F.softplus((required - pred_gap) / pref_tau) * pref_tau
                terms.append(float(preference_weight) * (confidence * pref).sum() / confidence.sum().clamp_min(1.0e-6))
            # Exact expected-regret term over nominal plus recovery candidates.
            if float(preference_regret_weight) > 0.0:
                pref_logits = torch.cat([r_delta.new_zeros(1), r_delta], dim=0) / pref_tau
                pref_prob = torch.softmax(pref_logits, dim=0)
                teacher_util_pref = torch.cat([t_delta.new_zeros(1), t_delta], dim=0)
                expected = (pref_prob * teacher_util_pref).sum()
                regret_pref = torch.clamp(best_teacher - expected, min=0.0)
                confidence_group = torch.clamp((best_teacher - float(positive_gain)) / max(float(preference_confidence_scale), 1.0e-4), 0.0, 1.0)
                terms.append(float(preference_regret_weight) * confidence_group * regret_pref.square())
            if float(preference_listwise_weight) > 0.0:
                teacher_pref = torch.softmax(t_delta.detach() / pref_tau, dim=0)
                pred_pref_log = torch.log_softmax(r_delta / pref_tau, dim=0)
                confidence_group = torch.clamp((best_teacher - float(positive_gain)) / max(float(preference_confidence_scale), 1.0e-4), 0.0, 1.0)
                terms.append(
                    float(preference_listwise_weight)
                    * confidence_group
                    * F.kl_div(pred_pref_log, teacher_pref, reduction="sum")
                )
            if float(preference_gap_weight) > 0.0 and recs.numel() > 1:
                teacher_others = t_delta[torch.arange(recs.numel(), device=recs.device) != best_rank_j]
                teacher_second = teacher_others.max()
                teacher_best_gap = (best_teacher - teacher_second).detach().clamp(min=0.0, max=0.25)
                pred_others = r_delta[torch.arange(recs.numel(), device=recs.device) != best_rank_j]
                pred_alt = torch.maximum(pred_others.max(), r_delta.new_zeros(()))
                pred_best_gap = r_delta[best_rank_j] - pred_alt
                confidence_gap = torch.clamp(teacher_best_gap / max(float(preference_confidence_scale), 1.0e-4), 0.0, 1.0)
                terms.append(
                    float(preference_gap_weight)
                    * confidence_gap
                    * F.smooth_l1_loss(pred_best_gap, teacher_best_gap, reduction="mean")
                )
        opp_delta_logits = None
        if opportunity_logits is not None:
            opp_delta_logits = opportunity_logits[recs] - opportunity_logits[nom]
        harm_delta_logits = None
        if harm_logits is not None:
            harm_delta_logits = harm_logits[recs] - harm_logits[nom]
        if opp_delta_logits is not None and float(opportunity_weight) > 0.0:
            opp_temp = float(opportunity_soft_label_temperature)
            labels = (
                torch.sigmoid((t_delta - float(positive_gain)) / max(opp_temp, 1.0e-4))
                if opp_temp > 0.0 else pos_mask.to(dtype=opp_delta_logits.dtype)
            ).to(dtype=opp_delta_logits.dtype)
            # Opportunity is explicitly candidate-vs-nominal, matching the
            # deployed admission decision rather than an absolute candidate tag.
            # Soft labels avoid treating tiny cross-split teacher perturbations at
            # the gain margin as contradictory supervision.
            pos_w = torch.as_tensor(max(float(opportunity_pos_weight), 1.0), dtype=opp_delta_logits.dtype, device=opp_delta_logits.device)
            opp = F.binary_cross_entropy_with_logits(opp_delta_logits, labels, pos_weight=pos_w)
            terms.append(float(opportunity_weight) * opp)
        if harm_delta_logits is not None and float(harm_weight) > 0.0:
            harm_temp = float(harm_soft_label_temperature)
            harm_labels = (
                torch.sigmoid((-t_delta - float(negative_gain)) / max(harm_temp, 1.0e-4))
                if harm_temp > 0.0 else harmful_mask.to(dtype=harm_delta_logits.dtype)
            ).to(dtype=harm_delta_logits.dtype)
            harm_pos_w = torch.as_tensor(max(float(harm_pos_weight), 1.0), dtype=harm_delta_logits.dtype, device=harm_delta_logits.device)
            harm_loss = F.binary_cross_entropy_with_logits(harm_delta_logits, harm_labels, pos_weight=harm_pos_w)
            terms.append(float(harm_weight) * harm_loss)
        # v48.4 DRA-RCD: separate candidate ranking from action admission.
        # Candidate ordering must not be trained through a harm head whose labels
        # are currently the least transferable part of the dataset.  The value-only
        # distribution learns *which* recovery is best; opportunity/harm remain in
        # the admission distribution that decides whether to leave nominal.
        recovery_admission_logits = p_delta
        if opp_delta_logits is not None:
            recovery_admission_logits = recovery_admission_logits + float(opportunity_admission_weight) * F.logsigmoid(opp_delta_logits)
        if harm_delta_logits is not None:
            recovery_admission_logits = recovery_admission_logits + float(harm_admission_weight) * F.logsigmoid(-harm_delta_logits)
        admission_class_logits = torch.cat([score[nom:nom + 1] * 0.0, recovery_admission_logits], dim=0) / tau
        rank_class_logits = torch.cat([rank_score[nom:nom + 1] * 0.0, r_delta], dim=0) / tau
        if bool(pos_mask.any()):
            target_class = 1 + int(torch.argmax(t_delta).item())
        else:
            target_class = 0
        target_tensor = torch.tensor([target_class], dtype=torch.long, device=score.device)
        if float(setwise_admission_weight) > 0.0:
            terms.append(float(setwise_admission_weight) * F.cross_entropy(admission_class_logits.unsqueeze(0), target_tensor))

        if float(selective_risk_weight) > 0.0 or float(selective_coverage_weight) > 0.0:
            policy_prob = torch.softmax(admission_class_logits, dim=0)
            recovery_prob = policy_prob[1:]
            harmful_mass = (recovery_prob * harmful_mask.to(recovery_prob.dtype)).sum()
            risk_excess = F.relu(harmful_mass - float(selective_harm_budget))
            if float(selective_risk_weight) > 0.0:
                terms.append(float(selective_risk_weight) * risk_excess.square())
            if float(selective_coverage_weight) > 0.0 and bool(pos_mask.any()):
                positive_mass = (recovery_prob * pos_mask.to(recovery_prob.dtype)).sum()
                coverage_shortfall = F.relu(float(selective_coverage_target) - positive_mass)
                terms.append(float(selective_coverage_weight) * coverage_shortfall.square())

        if float(policy_distill_weight) > 0.0 or float(policy_regret_weight) > 0.0 or float(policy_admission_distill_weight) > 0.0:
            teacher_util = torch.cat([
                torch.zeros((1,), dtype=t_delta.dtype, device=t_delta.device),
                t_delta,
            ], dim=0)
            if bool(pos_mask.any()):
                teacher_prob = torch.softmax(
                    teacher_util / max(float(policy_teacher_temperature), 1.0e-3), dim=0
                ).detach()
            else:
                teacher_prob = torch.zeros_like(teacher_util)
                teacher_prob[0] = 1.0
            ranking_logits = rank_class_logits if bool(policy_decouple_admission) else admission_class_logits
            ranking_prob = torch.softmax(ranking_logits, dim=0)
            if float(policy_distill_weight) > 0.0:
                terms.append(
                    float(policy_distill_weight)
                    * F.kl_div(torch.log(ranking_prob.clamp_min(1.0e-8)), teacher_prob, reduction="sum")
                )
            if float(policy_admission_distill_weight) > 0.0:
                admission_prob = torch.softmax(admission_class_logits, dim=0)
                terms.append(
                    float(policy_admission_distill_weight)
                    * F.kl_div(torch.log(admission_prob.clamp_min(1.0e-8)), teacher_prob, reduction="sum")
                )
            if float(policy_regret_weight) > 0.0:
                expected_teacher_adv = (ranking_prob[1:] * t_delta).sum()
                oracle_teacher_adv = torch.maximum(
                    torch.zeros((), dtype=t_delta.dtype, device=t_delta.device), t_delta.max()
                )
                regret = F.relu(oracle_teacher_adv - expected_teacher_adv - float(policy_regret_margin))
                terms.append(float(policy_regret_weight) * regret.square())
        if float(pairwise_weight) > 0.0:
            pair_terms = []
            if bool(pos_mask.any()):
                pair_terms.append(F.softplus((float(rank_margin) - r_delta[pos_mask]) / tau).mean() * tau)
            if bool(neg_mask.any()):
                pair_terms.append(float(false_positive_weight) * F.softplus((r_delta[neg_mask] + float(rank_margin)) / tau).mean() * tau)
            if bool(tie_mask.any()):
                pair_terms.append(
                    float(ambiguous_group_weight)
                    * F.smooth_l1_loss(r_delta[tie_mask], t_delta[tie_mask])
                )
            if pair_terms:
                terms.append(float(pairwise_weight) * torch.stack(pair_terms).sum())
        if float(top_rank_weight) > 0.0 and bool(pos_mask.any()) and bool(neg_mask.any()):
            best_pos = r_delta[best_j]
            strongest_neg = torch.max(r_delta[neg_mask])
            terms.append(float(top_rank_weight) * F.softplus((float(rank_margin) - (best_pos - strongest_neg)) / tau) * tau)
        if float(t_adv.item()) >= float(positive_gain):
            cls_w = float(positive_group_weight)
            terms.append(float(advantage_weight) * (F.smooth_l1_loss(p_adv, t_adv) + F.softplus((float(rank_margin) - p_adv) / tau) * tau))
        elif float(t_adv.item()) <= -float(negative_gain):
            cls_w = float(negative_group_weight)
            max_pred = torch.max(p_delta)
            terms.append(float(advantage_weight) * float(false_positive_weight) * F.softplus((max_pred + float(rank_margin)) / tau) * tau)
        else:
            cls_w = float(ambiguous_group_weight)
            terms.append(0.25 * float(advantage_weight) * F.smooth_l1_loss(p_adv, t_adv))
        b = int(bid[nom].item())
        bucket_w = float(near_weight) if b == 1 else (float(contact_weight) if b == 2 else 1.0)
        group_losses.append(torch.stack(terms).sum())
        group_weights.append(max(1.0e-6, cls_w * bucket_w))
        # Pseudo-environments use only training-observable grouping plus teacher
        # state for supervision: regime, nominal severity, opportunity state, and
        # teacher-best macro.  This prevents a single train-specific severity/macro
        # pocket from dominating without peeking at calibration or test data.
        severity_value = float(target[nom].detach().item())
        severity_bin = sum(severity_value > float(x) for x in tuple(group_dro_severity_thresholds))
        state_bin = 2 if bool(pos_mask.any()) else (0 if bool(harmful_mask.all()) else 1)
        best_macro = int(mac[recs[best_j]].detach().item()) if recs.numel() else -1
        group_domains.append((b, int(severity_bin), int(state_bin), best_macro))
    if not group_losses:
        return float(point_weight) * point_loss
    gl = torch.stack(group_losses)
    gw = torch.as_tensor(group_weights, dtype=gl.dtype, device=gl.device)
    erm_grouped = (gl * gw).sum() / gw.sum().clamp_min(1.0e-6)
    dro_mix = min(max(float(group_dro_weight), 0.0), 1.0)
    grouped = erm_grouped
    if dro_mix > 0.0 and group_domains:
        domain_means: list[torch.Tensor] = []
        for domain in sorted(set(group_domains)):
            mask = torch.as_tensor([d == domain for d in group_domains], dtype=torch.bool, device=gl.device)
            dw = gw[mask]
            domain_means.append((gl[mask] * dw).sum() / dw.sum().clamp_min(1.0e-6))
        if domain_means:
            dm = torch.stack(domain_means)
            dro_tau = max(float(group_dro_temperature), 1.0e-3)
            robust_grouped = dro_tau * (torch.logsumexp(dm / dro_tau, dim=0) - torch.log(torch.as_tensor(float(dm.numel()), dtype=dm.dtype, device=dm.device)))
            grouped = (1.0 - dro_mix) * erm_grouped + dro_mix * robust_grouped
    return float(point_weight) * point_loss + grouped
