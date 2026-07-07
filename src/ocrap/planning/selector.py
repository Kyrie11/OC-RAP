from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionResult:
    selected_index: int
    reason: str
    admitted: np.ndarray


def crisp_select(utility: np.ndarray, r_dep: np.ndarray, hard: np.ndarray, harm: np.ndarray, feasible: np.ndarray, gamma_rec: float = 0.0, gamma_H: float = 0.0, gamma_D: float = 0.0, nominal_index: int = 0) -> SelectionResult:
    utility = np.asarray(utility, dtype=float)
    r_dep = np.asarray(r_dep, dtype=float)
    hard = np.asarray(hard, dtype=float)
    harm = np.asarray(harm, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    admitted = feasible & (r_dep >= gamma_rec) & (hard <= gamma_H) & (harm <= gamma_D)
    if 0 <= nominal_index < len(utility) and admitted[nominal_index]:
        return SelectionResult(int(nominal_index), "nominal_admitted", admitted)
    if admitted.any():
        idxs = np.where(admitted)[0]
        best = int(idxs[np.argmax(utility[idxs])])
        return SelectionResult(best, "best_admitted_utility", admitted)
    # Lexicographic fallback: minimize hard violation, then harm, then maximize r_dep, then utility.
    order = sorted(range(len(utility)), key=lambda i: (not feasible[i], hard[i], harm[i], -r_dep[i], -utility[i]))
    return SelectionResult(int(order[0]), "lexicographic_fallback", admitted)

def _as_1d_float(x: np.ndarray | None, n: int, default: float = 0.0) -> np.ndarray:
    if x is None:
        return np.full((n,), float(default), dtype=float)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size), constant_values=float(default))
    return np.where(np.isfinite(arr[:n]), arr[:n], float(default))


def constrained_lcb_select(
    utility: np.ndarray,
    r_dep: np.ndarray,
    hard: np.ndarray,
    harm: np.ndarray,
    feasible: np.ndarray,
    gamma_rec: float = 0.0,
    gamma_H: float = 0.0,
    gamma_D: float = 0.0,
    nominal_index: int = 0,
    *,
    pred_gap: np.ndarray | None = None,
    nominal_deviation: np.ndarray | None = None,
    lcb_beta: float = 0.0,
    nominal_slack: float = 0.0,
    nominal_slack_gap_limit: float = 0.5,
    intervention_penalty: float = 0.03,
    deviation_penalty: float = 0.15,
    recovery_bonus: float = 0.02,
    fallback_rec_weight: float = 0.10,
    fallback_lcb_margin: float = 0.05,
    fallback_gap_margin: float = 0.25,
    nominal_fallback_lcb_slack: float = 0.05,
) -> SelectionResult:
    """Calibrated constrained selector used by OC-RAP in closed loop.

    It uses a conservative recovery lower-bound proxy
    ``r_lcb = r_dep - beta * max(0, oracle_deployable_gap)`` and adds
    explicit intervention/deviation costs so OC-RAP does not over-intervene when
    the nominal prefix is only slightly below the calibrated threshold.
    """
    utility = np.asarray(utility, dtype=float).reshape(-1)
    n = int(utility.size)
    r_dep = _as_1d_float(r_dep, n, default=-np.inf)
    hard = _as_1d_float(hard, n, default=np.inf)
    harm = _as_1d_float(harm, n, default=np.inf)
    feasible = np.asarray(feasible, dtype=bool).reshape(-1)
    if feasible.size < n:
        feasible = np.pad(feasible, (0, n - feasible.size), constant_values=False)
    feasible = feasible[:n]
    gap = np.maximum(0.0, _as_1d_float(pred_gap, n, default=0.0))
    dev = np.maximum(0.0, _as_1d_float(nominal_deviation, n, default=0.0))
    rec_lcb = r_dep - float(lcb_beta) * gap
    safe = feasible & (hard <= float(gamma_H)) & (harm <= float(gamma_D))
    admitted = safe & (rec_lcb >= float(gamma_rec))
    idxs = np.arange(n)
    intervention = (idxs != int(nominal_index)).astype(float)
    score = utility - float(intervention_penalty) * intervention - float(deviation_penalty) * dev + float(recovery_bonus) * (rec_lcb - float(gamma_rec))
    if 0 <= nominal_index < n and safe[nominal_index]:
        nom_margin = rec_lcb[nominal_index] - float(gamma_rec)
        nom_gap = gap[nominal_index]
        if admitted[nominal_index]:
            return SelectionResult(int(nominal_index), "nominal_lcb_admitted", admitted)
        if nom_margin >= -float(nominal_slack) and nom_gap <= float(nominal_slack_gap_limit):
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_slack_lcb_admitted", admitted)
    if admitted.any():
        cand = np.where(admitted)[0]
        return SelectionResult(int(cand[np.argmax(score[cand])]), "best_admitted_lcb_score", admitted)
    # No candidate satisfies the calibrated recovery constraint.  This branch is
    # deliberately recovery-first.  Earlier versions used a utility-dominated
    # soft fallback, which could pick high-utility prefixes with lower predicted
    # deployable recoverability and larger oracle--deployable gap.  That defeats
    # the purpose of OC-RAP exactly in low-headroom regimes.  We therefore:
    #   1. keep hard/harm safety as the first gate;
    #   2. restrict fallback candidates to the near-best recovery LCB set;
    #   3. within that set, reject candidates with much larger predicted gap;
    #   4. use utility/intervention/deviation only as a final tie-breaker.
    if safe.any():
        fallback_pool = safe.copy()
    elif feasible.any():
        cand = np.where(feasible)[0]
        min_hard = float(np.min(hard[cand]))
        near_hard = feasible & (hard <= min_hard + 1e-6)
        cand2 = np.where(near_hard)[0]
        min_harm = float(np.min(harm[cand2])) if cand2.size else float(np.min(harm[cand]))
        fallback_pool = near_hard & (harm <= min_harm + 1e-6)
    else:
        fallback_pool = np.ones((n,), dtype=bool)

    cand = np.where(fallback_pool)[0]
    if cand.size == 0:
        cand = np.arange(n)
    best_lcb = float(np.max(rec_lcb[cand])) if cand.size else -np.inf

    if 0 <= nominal_index < n and fallback_pool[nominal_index]:
        nom_lcb = float(rec_lcb[nominal_index])
        nom_gap = float(gap[nominal_index])
        best_gap = float(np.min(gap[cand])) if cand.size else nom_gap
        if nom_lcb >= best_lcb - float(nominal_fallback_lcb_slack) and nom_gap <= best_gap + float(fallback_gap_margin):
            return SelectionResult(int(nominal_index), "nominal_recovery_guarded_fallback", admitted)

    near_best_lcb = fallback_pool & (rec_lcb >= best_lcb - float(fallback_lcb_margin))
    cand = np.where(near_best_lcb)[0]
    if cand.size:
        best_gap = float(np.min(gap[cand]))
        gap_guarded = near_best_lcb & (gap <= best_gap + float(fallback_gap_margin))
        if gap_guarded.any():
            cand = np.where(gap_guarded)[0]
    else:
        cand = np.where(fallback_pool)[0]

    rec_shortfall = np.maximum(0.0, float(gamma_rec) - rec_lcb)
    fallback_score = score - float(fallback_rec_weight) * rec_shortfall - 10.0 * np.maximum(0.0, hard - float(gamma_H)) - 2.0 * np.maximum(0.0, harm - float(gamma_D))
    return SelectionResult(int(cand[np.argmax(fallback_score[cand])]) if cand.size else 0, "recovery_guarded_fallback", admitted)


def calibrated_constrained_select(
    utility: np.ndarray,
    r_dep: np.ndarray,
    hard: np.ndarray,
    harm: np.ndarray,
    feasible: np.ndarray,
    gamma_rec: float = 0.0,
    gamma_H: float = 0.0,
    gamma_D: float = 0.0,
    nominal_index: int = 0,
    *,
    pred_gap: np.ndarray | None = None,
    nominal_deviation: np.ndarray | None = None,
    lcb_beta: float = 0.10,
    shortfall_penalty: float = 1.0,
    gap_penalty: float = 0.05,
    intervention_penalty: float = 0.03,
    deviation_penalty: float = 0.15,
    recovery_bonus: float = 0.02,
    admission_bonus: float = 0.02,
    nominal_slack: float = 0.03,
    nominal_slack_gap_limit: float = 0.50,
    nominal_utility_slack: float = 0.05,
    safe_nominal_slack: float = 0.12,
    regime_name: str | None = None,
    intervention_budget_rate: float | None = None,
    intervention_budget_used: float | None = None,
    intervention_budget_steps: float | None = None,
    intervention_budget_penalty: float = 0.25,
    prefer_admitted: bool = False,
    switch_score_margin: float = 0.0,
    safe_switch_score_margin: float = 0.10,
    safe_min_rec_lcb_gain: float = 0.05,
    safe_min_gap_reduction: float = 0.15,
    budget_preserve_nominal: bool = True,
    budget_nominal_slack: float = 0.08,
    pred_drs: np.ndarray | None = None,
    deployability_bonus: float = 0.0,
    contact_deployability_bonus: float = 0.0,
    contact_gap_penalty: float = 0.0,
    safe_hard_nominal_guard: bool = True,
    safe_nominal_max_gap: float = 0.20,
    safe_override_require_both: bool = True,
    safe_min_drs_gain: float = 0.10,
) -> SelectionResult:
    """Soft calibrated OC-RAP selector.

    ``constrained_lcb_select`` is deliberately hard-gated: candidates below the
    calibrated recovery threshold all fall into a recovery-first fallback.  That
    is useful for stress tests, but in natural/safe scenes it can over-intervene
    because a candidate that misses gamma by a tiny amount is treated like one
    that misses it by a large amount.  This selector keeps hard/harm feasibility
    as a constraint, then turns recovery-threshold shortfall into a continuous
    penalty.  It also has an optional intervention budget and a stronger
    nominal-preservation rule for safe/background buckets.
    """
    utility = np.asarray(utility, dtype=float).reshape(-1)
    n = int(utility.size)
    if n <= 0:
        return SelectionResult(0, "empty_calibrated_selector", np.zeros((0,), dtype=bool))
    r_dep = _as_1d_float(r_dep, n, default=-np.inf)
    hard = _as_1d_float(hard, n, default=np.inf)
    harm = _as_1d_float(harm, n, default=np.inf)
    feasible = np.asarray(feasible, dtype=bool).reshape(-1)
    if feasible.size < n:
        feasible = np.pad(feasible, (0, n - feasible.size), constant_values=False)
    feasible = feasible[:n]
    gap = np.maximum(0.0, _as_1d_float(pred_gap, n, default=0.0))
    dev = np.maximum(0.0, _as_1d_float(nominal_deviation, n, default=0.0))
    drs_proxy = np.clip(_as_1d_float(pred_drs, n, default=0.0), 0.0, 1.0)

    rec_lcb = r_dep - float(lcb_beta) * gap
    safe = feasible & (hard <= float(gamma_H)) & (harm <= float(gamma_D))
    admitted = safe & (rec_lcb >= float(gamma_rec))
    idxs = np.arange(n)
    intervention = (idxs != int(nominal_index)).astype(float)
    regime = (regime_name or "").lower()
    is_safe_regime = "safe" in regime or "normal" in regime or "background" in regime
    is_contact_regime = "contact" in regime

    # Continuous recovery shortfall; finite guard prevents NaNs from dominating.
    rec_shortfall = np.maximum(0.0, float(gamma_rec) - rec_lcb)
    rec_shortfall = np.where(np.isfinite(rec_shortfall), rec_shortfall, 1.0e6)
    score = (
        utility
        - float(shortfall_penalty) * rec_shortfall
        - float(gap_penalty) * gap
        - float(intervention_penalty) * intervention
        - float(deviation_penalty) * dev
        + float(recovery_bonus) * (rec_lcb - float(gamma_rec))
        + float(admission_bonus) * admitted.astype(float)
        + float(deployability_bonus) * drs_proxy
    )
    if is_contact_regime:
        score = score + float(contact_deployability_bonus) * drs_proxy - float(contact_gap_penalty) * gap

    # If the current rollout already exceeded the intervention budget, increase
    # the marginal cost of deviating from nominal.  This is intentionally a soft
    # budget because emergency recovery should still be selectable.
    if intervention_budget_rate is not None and intervention_budget_steps not in {None, 0}:
        try:
            used = float(intervention_budget_used or 0.0)
            steps = max(1.0, float(intervention_budget_steps or 1.0))
            target = float(intervention_budget_rate)
            over = max(0.0, (used / steps) - target)
            if over > 0.0:
                score = score - float(intervention_budget_penalty) * over * intervention
        except Exception:
            pass

    if safe.any():
        pool = safe.copy()
    elif feasible.any():
        # Preserve the original hard/harm safety priority when no fully safe
        # candidate exists.
        cand = np.where(feasible)[0]
        min_hard = float(np.min(hard[cand]))
        near_hard = feasible & (hard <= min_hard + 1.0e-6)
        cand2 = np.where(near_hard)[0]
        min_harm = float(np.min(harm[cand2])) if cand2.size else float(np.min(harm[cand]))
        pool = near_hard & (harm <= min_harm + 1.0e-6)
    else:
        pool = np.ones((n,), dtype=bool)

    # Stress-regime option: when calibrated-admitted candidates exist, rank only
    # within them.  The default remains soft-constrained for backward
    # compatibility, but near-contact/contact experiments should set
    # prefer_admitted_by_bucket=true so DRS is not sacrificed to high utility.
    rank_pool = pool & admitted if bool(prefer_admitted) and bool((pool & admitted).any()) else pool

    cand = np.where(rank_pool)[0]
    if cand.size == 0:
        cand = np.arange(n)
    best_idx = int(cand[np.argmax(score[cand])])

    # Nominal-preserving calibration.  In safe/background regimes, recovery
    # shortfall by itself should not trigger large behavior changes unless the
    # alternative is clearly better in calibrated score and materially improves
    # deployable-recovery LCB or oracle-deployable gap.
    nominal_extra_slack = float(safe_nominal_slack if is_safe_regime else nominal_slack)
    budget_exceeded = False
    if intervention_budget_rate is not None and intervention_budget_steps not in {None, 0}:
        try:
            budget_exceeded = (float(intervention_budget_used or 0.0) / max(1.0, float(intervention_budget_steps or 1.0))) >= float(intervention_budget_rate)
        except Exception:
            budget_exceeded = False

    if 0 <= nominal_index < n and pool[nominal_index]:
        nom_gap_ok = gap[nominal_index] <= float(nominal_slack_gap_limit)
        nom_near_gamma = rec_lcb[nominal_index] >= float(gamma_rec) - nominal_extra_slack
        nom_near_score = score[nominal_index] >= score[best_idx] - float(nominal_utility_slack)
        if admitted[nominal_index]:
            return SelectionResult(int(nominal_index), "nominal_calibrated_admitted", admitted)
        if bool(budget_preserve_nominal) and budget_exceeded and nom_gap_ok and rec_lcb[nominal_index] >= float(gamma_rec) - float(budget_nominal_slack):
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_budget_preserved", admitted)
        if is_safe_regime and nom_gap_ok and nom_near_gamma:
            score_gain = float(score[best_idx] - score[nominal_index])
            rec_gain = float(rec_lcb[best_idx] - rec_lcb[nominal_index])
            gap_reduction = float(gap[nominal_index] - gap[best_idx])
            drs_gain = float(drs_proxy[best_idx] - drs_proxy[nominal_index])
            material_recovery_gain = rec_gain >= float(safe_min_rec_lcb_gain)
            material_gap_gain = gap_reduction >= float(safe_min_gap_reduction)
            material_drs_gain = drs_gain >= float(safe_min_drs_gain)
            if bool(safe_hard_nominal_guard) and gap[nominal_index] <= float(safe_nominal_max_gap):
                # Safe_v2 is the nominal-preservation regime.  Do not switch for
                # a single noisy headroom advantage; require a score improvement
                # plus consistent recovery/gap evidence.
                enough_evidence = (material_recovery_gain and material_gap_gain) if bool(safe_override_require_both) else (material_recovery_gain or material_gap_gain or material_drs_gain)
                if (not enough_evidence) or score_gain < float(safe_switch_score_margin):
                    admitted = admitted.copy()
                    admitted[nominal_index] = True
                    return SelectionResult(int(nominal_index), "nominal_safe_switch_guard", admitted)
            # Safe/background is a nominal-preserving operating region.  Switch
            # away from nominal only when the alternative is not just higher
            # scoring, but meaningfully improves calibrated recoverability or
            # reduces oracle-deployable gap.
            if (not material_recovery_gain and not material_gap_gain and not material_drs_gain) or score_gain < float(safe_switch_score_margin):
                admitted = admitted.copy()
                admitted[nominal_index] = True
                return SelectionResult(int(nominal_index), "nominal_safe_switch_guard", admitted)
        if nom_gap_ok and nom_near_gamma and nom_near_score:
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_calibrated_preserved", admitted)
        if (not is_safe_regime) and float(score[best_idx] - score[nominal_index]) < float(switch_score_margin) and nom_gap_ok and nom_near_gamma:
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_switch_margin_preserved", admitted)

    reason = "best_calibrated_admitted_score" if admitted[best_idx] else "best_calibrated_soft_constraint"
    if bool(prefer_admitted) and bool((pool & admitted).any()) and admitted[best_idx]:
        reason = "best_calibrated_prefer_admitted"
    return SelectionResult(best_idx, reason, admitted)

