from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocrap.planning.selector import SelectionResult, constrained_lcb_select, crisp_select


BASELINES = [
    "nominal",
    "log_replay",
    "idm_proxy",
    "mpc_proxy",
    "risk_aware",
    "backup_filter",
    "contingency",
    "oracle_filter",
    "ocrap",
    "ocrap_teacher",
]


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


def _finite_or_proxy(primary: np.ndarray, proxy: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    primary = np.asarray(primary, dtype=float)
    out = primary.copy()
    bad = ~np.isfinite(out)
    if proxy is not None:
        p = np.asarray(proxy, dtype=float)
        if p.shape != out.shape:
            p = np.resize(p, out.shape)
        out[bad] = p[bad]
    bad = ~np.isfinite(out)
    if bad.any():
        fb = np.asarray(fallback, dtype=float)
        if fb.shape != out.shape:
            fb = np.resize(fb, out.shape)
        out[bad] = fb[bad]
    return out


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
    *,
    pred_r_orc: np.ndarray | None = None,
    pred_gap: np.ndarray | None = None,
    nominal_deviation: np.ndarray | None = None,
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
    oracle_signal = _finite_or_proxy(teacher_r_orc, pred_r_orc, pred_r_dep)

    if method == "ocrap":
        scfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
        selector_name = str(scfg.get("ocrap_selector", scfg.get("selector", "lcb_constrained"))).lower()
        if selector_name in {"crisp", "hard", "hard_threshold"}:
            sel: SelectionResult = crisp_select(utility, pred_r_dep, hard, harm, feasible, gamma_rec=gamma_rec, gamma_H=gamma_H, gamma_D=gamma_D)
            return BaselineSelection(sel.selected_index, sel.reason, sel.admitted, pred_r_dep)
        beta = float(scfg.get("lcb_beta", 0.10))
        sel = constrained_lcb_select(
            utility, pred_r_dep, hard, harm, feasible,
            gamma_rec=gamma_rec, gamma_H=gamma_H, gamma_D=gamma_D,
            pred_gap=pred_gap, nominal_deviation=nominal_deviation,
            lcb_beta=beta,
            nominal_slack=float(scfg.get("nominal_slack", 0.03)),
            nominal_slack_gap_limit=float(scfg.get("nominal_slack_gap_limit", 0.50)),
            intervention_penalty=float(scfg.get("intervention_penalty", 0.03)),
            deviation_penalty=float(scfg.get("deviation_penalty", 0.15)),
            recovery_bonus=float(scfg.get("recovery_bonus", 0.02)),
            fallback_rec_weight=float(scfg.get("fallback_rec_weight", 0.10)),
            fallback_lcb_margin=float(scfg.get("fallback_lcb_margin", 0.05)),
            fallback_gap_margin=float(scfg.get("fallback_gap_margin", 0.25)),
            nominal_fallback_lcb_slack=float(scfg.get("nominal_fallback_lcb_slack", 0.05)),
        )
        gap_arr = np.asarray(pred_gap if pred_gap is not None else np.zeros_like(pred_r_dep), dtype=float)
        score = pred_r_dep - beta * np.maximum(0.0, gap_arr)
        return BaselineSelection(sel.selected_index, sel.reason, sel.admitted, score)

    if method == "ocrap_teacher":
        sel = crisp_select(utility, teacher_r_dep, hard, harm, feasible, gamma_rec=gamma_rec, gamma_H=gamma_H, gamma_D=gamma_D)
        return BaselineSelection(sel.selected_index, "teacher_deployable_upper_bound", sel.admitted, teacher_r_dep)

    if method == "nominal":
        admitted = np.zeros_like(feasible, dtype=bool)
        idx = 0 if len(feasible) and feasible[0] else _best_by_score(utility, feasible)
        admitted[idx] = True
        return BaselineSelection(idx, "nominal_prefix", admitted, utility)

    if method == "log_replay":
        # Explicit logged/nominal rollout baseline. Candidate generation writes
        # the nominal/log-following prefix as candidate 0 when available; if it is
        # infeasible, fall back to the best feasible utility candidate so the
        # closed-loop runner can continue on degenerate frames.
        admitted = np.zeros_like(feasible, dtype=bool)
        idx = 0 if len(feasible) and feasible[0] else _best_by_score(utility, feasible)
        admitted[idx] = True
        return BaselineSelection(idx, "log_replay_prefix", admitted, utility)

    if method == "idm_proxy":
        # Lightweight IDM-style heuristic over OC-RAP candidates. It does not
        # reimplement a full car-following simulator; it prefers feasible,
        # low-hard-violation and low-harm candidates while keeping utility.
        bcfg = cfg.get("baselines", {}) if isinstance(cfg.get("baselines", {}), dict) else {}
        lam_harm = float(bcfg.get("idm_harm_lambda", 2.0))
        lam_hard = float(bcfg.get("idm_hard_lambda", 15.0))
        score = utility - lam_harm * harm - lam_hard * hard
        idx = _best_by_score(score, feasible)
        admitted = feasible.copy()
        return BaselineSelection(idx, "idm_proxy_utility_safety", admitted, score)

    if method == "mpc_proxy":
        # Lightweight constrained-MPC proxy over the same candidate lattice. The
        # hard safety mask is applied first; inside it the controller maximizes
        # nominal utility. This is a sanity baseline, not a reproduction of a
        # specific published MPC implementation.
        admitted = safe_mask.copy()
        if admitted.any():
            idx = _admit_then_utility(admitted, utility)
        else:
            score = utility - 25.0 * hard - 5.0 * harm
            idx = _best_by_score(score, feasible)
        return BaselineSelection(idx, "mpc_proxy_constrained_lattice", admitted, utility)

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
        admitted = safe_mask & (oracle_signal >= gamma_rec)
        idx = _admit_then_utility(admitted, utility)
        return BaselineSelection(idx, "branchwise_backup_filter", admitted, oracle_signal)

    if method == "oracle_filter":
        # Strong oracle-recoverability baseline: same admission as branch-wise
        # recovery, but scored by oracle recoverability before utility.
        admitted = safe_mask & (oracle_signal >= gamma_rec)
        score = oracle_signal + 1.0e-3 * utility
        if admitted.any():
            idxs = np.where(admitted)[0]
            idx = int(idxs[np.argmax(score[idxs])])
        else:
            idx = _best_by_score(score, feasible)
        return BaselineSelection(idx, "oracle_recoverability_filter", admitted, oracle_signal)

    if method == "contingency":
        # Branch-specific contingency planner: maximize oracle recovery headroom,
        # then nominal utility.  It is expected to fail on oracle artifacts.
        score = oracle_signal + 1.0e-3 * utility
        idx = _best_by_score(score, feasible)
        admitted = safe_mask & (oracle_signal >= gamma_rec)
        return BaselineSelection(idx, "branch_specific_contingency", admitted, score)

    raise ValueError(f"Unknown evaluation method {method!r}; valid methods: {BASELINES}")
