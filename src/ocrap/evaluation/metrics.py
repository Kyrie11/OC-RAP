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


def deployable_recovery_success(m_star: np.ndarray, root_probs: np.ndarray, selected_options: np.ndarray | int) -> float:
    M = np.asarray(m_star, dtype=float)
    p = np.asarray(root_probs, dtype=float).reshape(-1)
    if isinstance(selected_options, (int, np.integer)):
        opt = np.full(M.shape[0], int(selected_options), dtype=int)
    else:
        opt = np.asarray(selected_options, dtype=int).reshape(-1)
    vals = np.array([M[k, opt[min(k, len(opt)-1)]] >= 0.0 for k in range(M.shape[0])], dtype=float)
    p = p / max(float(p.sum()), 1e-8)
    return float(np.sum(p[: len(vals)] * vals))


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
