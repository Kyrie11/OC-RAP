from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionResult:
    selected_index: int
    admitted_indices: list[int]
    reason: str


def crisp_select(
    utility: np.ndarray,
    r_dep: np.ndarray,
    hard_violation: np.ndarray,
    harm_proxy: np.ndarray,
    feasible: np.ndarray,
    gamma_rec: float,
    gamma_H: float = 0.0,
    gamma_D: float = 5.0,
) -> SelectionResult:
    utility = np.asarray(utility, dtype=np.float64)
    r_dep = np.asarray(r_dep, dtype=np.float64)
    hard = np.asarray(hard_violation, dtype=np.float64)
    harm = np.asarray(harm_proxy, dtype=np.float64)
    feas = np.asarray(feasible).astype(bool)
    adm_mask = (r_dep >= gamma_rec) & (hard <= gamma_H) & (harm <= gamma_D) & feas
    admitted = np.where(adm_mask)[0].astype(int).tolist()
    if 0 in admitted:
        return SelectionResult(0, admitted, "nominal_admitted")
    if admitted:
        best = int(admitted[int(np.argmax(utility[admitted]))])
        return SelectionResult(best, admitted, "best_utility_admitted")
    remaining = np.where(feas)[0]
    if remaining.size == 0:
        remaining = np.arange(len(utility))
    # Lexicographic fallback: feasible -> min hard violation -> min harm -> max R_dep -> max U.
    best = sorted(remaining.astype(int).tolist(), key=lambda i: (hard[i], harm[i], -r_dep[i], -utility[i]))[0]
    return SelectionResult(int(best), admitted, "lexicographic_fallback")
