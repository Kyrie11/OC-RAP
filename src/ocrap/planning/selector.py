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

