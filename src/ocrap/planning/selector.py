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
