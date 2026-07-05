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
    rec_shortfall = np.maximum(0.0, float(gamma_rec) - rec_lcb)
    fallback_score = score - float(fallback_rec_weight) * rec_shortfall - 10.0 * np.maximum(0.0, hard - float(gamma_H)) - 2.0 * np.maximum(0.0, harm - float(gamma_D))
    if feasible.any():
        cand = np.where(feasible)[0]
        return SelectionResult(int(cand[np.argmax(fallback_score[cand])]), "soft_constrained_fallback", admitted)
    return SelectionResult(int(np.argmax(fallback_score)) if n else 0, "soft_constrained_no_feasible_fallback", admitted)

