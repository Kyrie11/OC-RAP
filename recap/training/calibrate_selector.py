from __future__ import annotations

import numpy as np
try:
    from scipy.stats import beta
except Exception:
    beta = None


def cp_ucb(s: int, n: int, xi: float) -> float:
    if n <= 0:
        return 1.0
    if s == n:
        return 1.0
    if beta is not None:
        return float(beta.ppf(1 - xi, s + 1, n - s))
    # Conservative Hoeffding fallback.
    phat = s / n
    return float(min(1.0, phat + np.sqrt(np.log(1 / max(xi, 1e-12)) / (2 * n))))


def calibrate_q(R_pred: np.ndarray, R_star: np.ndarray, dH_pred: np.ndarray, dH_star: np.ndarray, action_mask: np.ndarray, q_R_grid=None, q_H_grid=None, eta_R: float = 0.70, epsilon_H: float = 0.05, delta_R: float = 0.05, delta_H: float = 0.05, xi: float = 0.05) -> dict:
    q_R_grid = np.arange(0.0, 0.501, 0.01) if q_R_grid is None else q_R_grid
    q_H_grid = np.arange(0.0, 0.301, 0.005) if q_H_grid is None else q_H_grid
    n = R_pred.shape[0]
    best_R = (float(q_R_grid[-1]), True, n, 1.0)
    for q in q_R_grid:
        losses = []
        for i in range(n):
            S = action_mask[i] & (R_pred[i] - q >= eta_R)
            losses.append(bool(np.any(R_star[i, S] < eta_R)) if np.any(S) else False)
        s = int(np.sum(losses)); u = cp_ucb(s, n, xi)
        if u <= delta_R:
            best_R = (float(q), False, s, u); break
    best_H = (float(q_H_grid[-1]), True, n, 1.0)
    for q in q_H_grid:
        losses = []
        for i in range(n):
            S = action_mask[i] & (dH_pred[i] + q <= epsilon_H)
            losses.append(bool(np.any(dH_star[i, S] > epsilon_H)) if np.any(S) else False)
        s = int(np.sum(losses)); u = cp_ucb(s, n, xi)
        if u <= delta_H:
            best_H = (float(q), False, s, u); break
    return {
        "q_R": best_R[0], "q_H": best_H[0], "eta_R": eta_R, "epsilon_H": epsilon_H,
        "delta_R": delta_R, "delta_H": delta_H, "xi": xi, "n_calib": int(n),
        "s_R": int(best_R[2]), "s_H": int(best_H[2]),
        "cp_ucb_R": float(best_R[3]), "cp_ucb_H": float(best_H[3]),
        "calibration_failed_R": bool(best_R[1]), "calibration_failed_H": bool(best_H[1]),
        "dataset_version": "", "split": "calib", "mode_alignment": "fixed_semantic_index",
    }
