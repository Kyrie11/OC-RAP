from __future__ import annotations

import numpy as np


def false_recoverability_admission(admitted: np.ndarray, teacher_r_dep: np.ndarray) -> float:
    admitted = np.asarray(admitted, dtype=bool)
    r = np.asarray(teacher_r_dep, dtype=float)
    if admitted.sum() == 0:
        return 0.0
    return float(np.mean(r[admitted] < 0.0))


def executed_false_admission(selected_idx: int, teacher_r_dep: np.ndarray) -> float:
    r = np.asarray(teacher_r_dep, dtype=float)
    return float(r[int(selected_idx)] < 0.0)


def deployable_recovery_success(
    m_star: np.ndarray,
    root_probs: np.ndarray,
    selected_options: np.ndarray | int,
    root_valid: np.ndarray | None = None,
) -> float:
    """Teacher deployable-recovery success for an executed candidate.

    ``m_star[k,l]`` is the teacher margin of recovery option ``l`` under root
    ``k``.  Earlier versions normalized all root probabilities, including
    padded/invalid roots when a sample had fewer than ``num_roots`` valid roots.
    That can under- or over-estimate DRS on heterogeneous datasets.  The metric
    now masks invalid roots before normalization while remaining backward
    compatible when ``root_valid`` is absent.
    """
    M = np.asarray(m_star, dtype=float)
    if M.ndim != 2 or M.size == 0:
        return 0.0
    p = np.asarray(root_probs, dtype=float).reshape(-1)[: M.shape[0]]
    if p.size < M.shape[0]:
        p = np.pad(p, (0, M.shape[0] - p.size), constant_values=0.0)
    if root_valid is not None:
        valid = np.asarray(root_valid, dtype=float).reshape(-1)[: M.shape[0]] > 0.5
        if valid.size < M.shape[0]:
            valid = np.pad(valid, (0, M.shape[0] - valid.size), constant_values=False)
        p = np.where(valid, p, 0.0)
    if isinstance(selected_options, (int, np.integer)):
        opt = np.full(M.shape[0], int(selected_options), dtype=int)
    else:
        opt = np.asarray(selected_options, dtype=int).reshape(-1)
        if opt.size == 0:
            opt = np.zeros(M.shape[0], dtype=int)
    opt = np.clip(opt, 0, M.shape[1] - 1)
    vals = np.array([M[k, opt[min(k, len(opt) - 1)]] >= 0.0 for k in range(M.shape[0])], dtype=float)
    denom = float(p.sum())
    if denom <= 1e-8:
        # If all valid probability mass disappeared due to a malformed sample,
        # fall back to a uniform distribution over valid rows instead of all rows.
        if root_valid is not None:
            valid = np.asarray(root_valid, dtype=float).reshape(-1)[: M.shape[0]] > 0.5
            if valid.any():
                p = valid.astype(float) / float(valid.sum())
            else:
                p = np.ones(M.shape[0], dtype=float) / float(M.shape[0])
        else:
            p = np.ones(M.shape[0], dtype=float) / float(M.shape[0])
    else:
        p = p / denom
    return float(np.sum(p * vals))


def nominal_utility_preservation(nominal_u: float, selected_u: float, sigma_u: float = 1.0) -> dict[str, float]:
    regret = float(nominal_u - selected_u)
    return {"nominal_regret": regret, "bounded_NUP": float(np.exp(-max(0.0, regret) / max(float(sigma_u), 1e-6)))}


def summarize_selection_metrics(records: list[dict], sigma_u: float = 1.0) -> dict:
    if not records:
        return {}
    fra_cand = np.mean([r["fra_cand"] for r in records])
    fra_exec = np.mean([r["fra_exec"] for r in records])
    drs = np.mean([r["drs"] for r in records])
    odg = np.mean([r["odg"] for r in records])
    odg_pos = np.mean([max(0.0, r["odg"]) for r in records])
    nup = np.mean([r["nup"] for r in records])
    art = [r for r in records if r.get("artifact", False)]
    out = {"FRA_cand": float(fra_cand), "FRA_exec": float(fra_exec), "DRS": float(drs), "ODG": float(odg), "ODG_pos": float(odg_pos), "bounded_NUP": float(nup), "artifact_selection_rate": float(np.mean([r.get("selected_artifact", False) for r in records]))}
    if art:
        out.update({"ODG_artifact": float(np.mean([r["odg"] for r in art])), "FRA_exec_artifact": float(np.mean([r["fra_exec"] for r in art])), "DRS_artifact": float(np.mean([r["drs"] for r in art]))})
    else:
        out.update({"ODG_artifact": None, "FRA_exec_artifact": None, "DRS_artifact": None})
    return out
