from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ocmero import oc_mero
from .selector import crisp_select


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def bounded_nup(u_nom: float, u_sel: float, sigma_u: float = 1.0) -> float:
    return float(math.exp(-max(0.0, u_nom - u_sel) / max(sigma_u, 1e-8)))


def sample_drs(m_star: np.ndarray, root_probs: np.ndarray, best_options: np.ndarray) -> float:
    K = len(root_probs)
    vals = []
    weights = []
    for k in range(K):
        l = int(best_options[k])
        vals.append(float(m_star[k, l] >= 0.0))
        weights.append(float(root_probs[k]))
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-8)
    return float(np.sum(w * np.asarray(vals, dtype=np.float64)))


def evaluate_group_teacher_outcome(group: list[dict[str, Any]], pred_r_dep: np.ndarray, pred_r_orc: np.ndarray, pred_margin: np.ndarray, pred_prob: np.ndarray, pred_C: np.ndarray, gamma: float, cfg: dict, method: str = "ocrap") -> dict[str, Any]:
    n = len(group)
    utility = np.asarray([float(g["utility"]) for g in group])
    hard = np.asarray([float(g["hard_violation"]) for g in group])
    harm = np.asarray([float(g["harm_proxy"]) for g in group])
    feasible = np.asarray([bool(int(g["feasible"])) for g in group])
    teacher_dep = np.asarray([float(g["r_dep_star"]) for g in group])
    teacher_orc = np.asarray([float(g["r_orc_star"]) for g in group])
    if method == "nominal":
        selected = 0
        admitted = [0]
    elif method == "oracle_filter":
        sel = crisp_select(utility, pred_r_orc, hard, harm, feasible, gamma, gamma_H=float(cfg.get("selection", {}).get("gamma_H", 0.0)), gamma_D=float(cfg.get("selection", {}).get("gamma_D", 5.0)))
        selected, admitted = sel.selected_index, sel.admitted_indices
    elif method == "risk_aware":
        score = utility - float(cfg.get("baselines", {}).get("risk_lambda", 1.0)) * harm - 5.0 * hard
        selected = int(np.argmax(score))
        admitted = np.where((harm <= float(cfg.get("selection", {}).get("gamma_D", 5.0))) & feasible)[0].astype(int).tolist()
    elif method == "backup_filter":
        # Stop/brake_lane options are first 6 by construction.
        backup_score = np.max(pred_margin[:, :, :6], axis=(1, 2))
        sel = crisp_select(utility, backup_score, hard, harm, feasible, 0.0, gamma_H=float(cfg.get("selection", {}).get("gamma_H", 0.0)), gamma_D=float(cfg.get("selection", {}).get("gamma_D", 5.0)))
        selected, admitted = sel.selected_index, sel.admitted_indices
    elif method == "contingency":
        # Branch-wise contingency behaves like oracle recovery admission but selects by utility.
        sel = crisp_select(utility, pred_r_orc, hard, harm, feasible, gamma, gamma_H=float(cfg.get("selection", {}).get("gamma_H", 0.0)), gamma_D=float(cfg.get("selection", {}).get("gamma_D", 5.0)))
        selected, admitted = sel.selected_index, sel.admitted_indices
    else:
        sel = crisp_select(utility, pred_r_dep, hard, harm, feasible, gamma, gamma_H=float(cfg.get("selection", {}).get("gamma_H", 0.0)), gamma_D=float(cfg.get("selection", {}).get("gamma_D", 5.0)))
        selected, admitted = sel.selected_index, sel.admitted_indices

    # DRS uses predicted policy choice but teacher margins for selected action.
    s = group[selected]
    oc = oc_mero(pred_margin[selected], pred_prob[selected], pred_C[selected], alpha=float(cfg.get("ocmero", {}).get("alpha", 0.2)), beta=float(cfg.get("ocmero", {}).get("beta", 0.2)), option_valid=np.asarray(s["option_valid"]).astype(bool))
    drs = sample_drs(np.asarray(s["m_star"], dtype=np.float64), np.asarray(s["root_probs"], dtype=np.float64), oc.best_option)
    admitted_dep = teacher_dep[admitted] if admitted else np.asarray([])
    return {
        "selected": selected,
        "admitted": admitted,
        "fra_cand_num": int(np.sum(admitted_dep < 0.0)) if admitted else 0,
        "fra_cand_den": int(len(admitted)),
        "fra_exec": float(teacher_dep[selected] < 0.0),
        "drs": drs,
        "odg": float(teacher_orc[selected] - teacher_dep[selected]),
        "odg_pos": float(max(0.0, teacher_orc[selected] - teacher_dep[selected])),
        "nominal_regret": float(utility[0] - utility[selected]),
        "nup": bounded_nup(float(utility[0]), float(utility[selected]), sigma_u=float(cfg.get("metrics", {}).get("sigma_u", 1.0))),
        "intervention": float(selected != 0),
        "collision": float(hard[selected] > 0.0),
        "hard_violation": float(hard[selected]),
        "harm_proxy": float(harm[selected]),
        "artifact_selected": float(bool(int(s["i_art_star"]))),
        "regime_label": s.get("regime_label", {}),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    out = {
        "FRA_cand": safe_div(sum(r["fra_cand_num"] for r in records), sum(r["fra_cand_den"] for r in records)),
        "FRA_exec": float(np.mean([r["fra_exec"] for r in records])),
        "DRS": float(np.mean([r["drs"] for r in records])),
        "ODG": float(np.mean([r["odg"] for r in records])),
        "ODG_pos": float(np.mean([r["odg_pos"] for r in records])),
        "nominal_regret": float(np.mean([r["nominal_regret"] for r in records])),
        "NUP": float(np.mean([r["nup"] for r in records])),
        "intervention_rate": float(np.mean([r["intervention"] for r in records])),
        "collision_rate": float(np.mean([r["collision"] for r in records])),
        "hard_violation": float(np.mean([r["hard_violation"] for r in records])),
        "harm_proxy": float(np.mean([r["harm_proxy"] for r in records])),
        "artifact_selection_rate": float(np.mean([r["artifact_selected"] for r in records])),
        "num_groups": float(len(records)),
    }
    art = [r for r in records if r["artifact_selected"] > 0.0]
    if art:
        out["ODG_artifact_selected"] = float(np.mean([r["odg"] for r in art]))
    return out
