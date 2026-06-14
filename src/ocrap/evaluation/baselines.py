from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocrap.planning.selector import SelectionResult, crisp_select


BASELINES = ["nominal", "risk_aware", "backup_filter", "contingency", "oracle_filter", "ocrap", "ocrap_teacher"]


@dataclass
class BaselineSelection:
    selected_index: int
    reason: str
    admitted: np.ndarray
    score: np.ndarray


def _best_by_score(score: np.ndarray, feasible: np.ndarray) -> int:
    feasible = np.asarray(feasible, dtype=bool)
    score = np.asarray(score, dtype=float)
    if feasible.any():
        idxs = np.where(feasible)[0]
        return int(idxs[np.argmax(score[idxs])])
    return int(np.argmax(score)) if score.size else 0


def _admit_then_utility(admitted: np.ndarray, utility: np.ndarray, nominal_index: int = 0) -> int:
    admitted = np.asarray(admitted, dtype=bool)
    utility = np.asarray(utility, dtype=float)
    if 0 <= nominal_index < len(utility) and admitted[nominal_index]:
        return int(nominal_index)
    if admitted.any():
        idxs = np.where(admitted)[0]
        return int(idxs[np.argmax(utility[idxs])])
    return _best_by_score(utility, np.ones_like(admitted, dtype=bool))


def select_baseline(
    method: str,
    utility: np.ndarray,
    pred_r_dep: np.ndarray,
    teacher_r_dep: np.ndarray,
    teacher_r_orc: np.ndarray,
    hard: np.ndarray,
    harm: np.ndarray,
    feasible: np.ndarray,
    gamma_rec: float,
    gamma_H: float,
    gamma_D: float,
    cfg: dict | None = None,
) -> BaselineSelection:
    """Select one candidate with a paper-baseline-style rule.

    These baselines intentionally operate on existing dataset labels/features so
    they can be evaluated before adding a separate neural model for each
    baseline.  ``ocrap`` is the only method that uses predicted deployable
    recoverability; ``ocrap_teacher`` is an upper-bound diagnostic that uses the
    teacher deployable label directly.
    """
    cfg = cfg or {}
    method = str(method).lower()
    utility = np.asarray(utility, dtype=float)
    pred_r_dep = np.asarray(pred_r_dep, dtype=float)
    teacher_r_dep = np.asarray(teacher_r_dep, dtype=float)
    teacher_r_orc = np.asarray(teacher_r_orc, dtype=float)
    hard = np.asarray(hard, dtype=float)
    harm = np.asarray(harm, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    safe_mask = feasible & (hard <= gamma_H) & (harm <= gamma_D)

    if method == "ocrap":
        sel: SelectionResult = crisp_select(utility, pred_r_dep, hard, harm, feasible, gamma_rec=gamma_rec, gamma_H=gamma_H, gamma_D=gamma_D)
        return BaselineSelection(sel.selected_index, sel.reason, sel.admitted, pred_r_dep)

    if method == "ocrap_teacher":
        sel = crisp_select(utility, teacher_r_dep, hard, harm, feasible, gamma_rec=gamma_rec, gamma_H=gamma_H, gamma_D=gamma_D)
        return BaselineSelection(sel.selected_index, "teacher_deployable_upper_bound", sel.admitted, teacher_r_dep)

    if method == "nominal":
        admitted = np.zeros_like(feasible, dtype=bool)
        idx = 0 if len(feasible) and feasible[0] else _best_by_score(utility, feasible)
        admitted[idx] = True
        return BaselineSelection(idx, "nominal_prefix", admitted, utility)

    if method == "risk_aware":
        bcfg = cfg.get("baselines", {}) if isinstance(cfg.get("baselines", {}), dict) else {}
        lam_harm = float(bcfg.get("risk_lambda", 1.0))
        lam_hard = float(bcfg.get("hard_lambda", 10.0))
        score = utility - lam_harm * harm - lam_hard * hard
        idx = _best_by_score(score, feasible)
        admitted = feasible.copy()
        return BaselineSelection(idx, "utility_minus_risk", admitted, score)

    if method == "backup_filter":
        # A conventional safety backup admits actions when some branch-wise
        # recovery appears feasible, without enforcing observation consistency.
        admitted = safe_mask & (teacher_r_orc >= gamma_rec)
        idx = _admit_then_utility(admitted, utility)
        return BaselineSelection(idx, "branchwise_backup_filter", admitted, teacher_r_orc)

    if method == "oracle_filter":
        # Strong oracle-recoverability baseline: same admission as branch-wise
        # recovery, but scored by oracle recoverability before utility.
        admitted = safe_mask & (teacher_r_orc >= gamma_rec)
        if admitted.any():
            idxs = np.where(admitted)[0]
            score = teacher_r_orc + 1.0e-3 * utility
            idx = int(idxs[np.argmax(score[idxs])])
        else:
            idx = _best_by_score(teacher_r_orc + 1.0e-3 * utility, feasible)
        return BaselineSelection(idx, "oracle_recoverability_filter", admitted, teacher_r_orc)

    if method == "contingency":
        # Branch-specific contingency planner: maximize oracle recovery headroom,
        # then nominal utility.  It is expected to fail on oracle artifacts.
        score = teacher_r_orc + 1.0e-3 * utility
        idx = _best_by_score(score, feasible)
        admitted = safe_mask & (teacher_r_orc >= gamma_rec)
        return BaselineSelection(idx, "branch_specific_contingency", admitted, score)

    raise ValueError(f"Unknown evaluation method {method!r}; valid methods: {BASELINES}")
