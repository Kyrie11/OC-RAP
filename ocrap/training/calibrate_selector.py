from __future__ import annotations

import numpy as np
from ocrap.models.selector import select_action, SelectorParams


def cp_ucb(s: int, n: int, xi: float) -> float:
    if n <= 0:
        return 1.0
    if s == n:
        return 1.0
    # Distribution-free upper confidence bound.  This avoids the very heavy
    # scipy.stats import in CLI runs while keeping conformal calibration
    # conservative for small calibration sets.
    phat = s / n
    return float(min(1.0, phat + np.sqrt(np.log(1 / max(xi, 1e-12)) / (2 * n))))


def calibrate_q(
    R_pred: np.ndarray,
    R_star: np.ndarray,
    dH_pred: np.ndarray,
    dH_star: np.ndarray,
    action_mask: np.ndarray,
    q_R_grid=None,
    q_H_grid=None,
    eta_R: float = 0.70,
    epsilon_H: float = 0.05,
    delta_R: float = 0.05,
    delta_H: float = 0.05,
    xi: float = 0.05,
    H_pred=None,
    H_star=None,
    C_pred=None,
    C_star=None,
    U_drv=None,
    q_delta_grid=None,
    q_C_grid=None,
    eta_H: float = 0.50,
    delta_delta: float | None = None,
    delta_C: float | None = None,
) -> dict:
    """Selected-action CRISP calibration over all four offsets.

    The old implementation only searched q_R/q_delta and left q_H/q_C at zero.
    This version calibrates q_R, q_H, q_delta and q_C jointly with a monotone
    coordinate search.  Exhaustive Cartesian search over 26x26x31x26 grid points
    is prohibitively slow for realistic calibration sets; the coordinate search
    increases the offset for whichever constraint currently has the largest UCB
    violation ratio and stops at the first feasible tuple.
    """
    q_R_grid = np.arange(0.0, 0.501, 0.02) if q_R_grid is None else np.asarray(q_R_grid, dtype=float)
    q_H_grid = np.arange(0.0, 0.501, 0.02) if q_H_grid is None else np.asarray(q_H_grid, dtype=float)
    q_delta_grid = np.arange(0.0, 0.301, 0.01) if q_delta_grid is None else np.asarray(q_delta_grid, dtype=float)
    q_C_grid = np.arange(0.0, 0.501, 0.02) if q_C_grid is None else np.asarray(q_C_grid, dtype=float)
    delta_delta = float(delta_H if delta_delta is None else delta_delta)
    delta_C = float(delta_H if delta_C is None else delta_C)
    n = int(R_pred.shape[0])
    if H_pred is None:
        H_pred = np.zeros_like(R_pred)
    if H_star is None:
        H_star = np.zeros_like(R_star)
    if C_pred is None:
        C_pred = np.zeros_like(R_pred)
    if C_star is None:
        C_star = np.zeros_like(R_star)
    if U_drv is None:
        U_drv = np.zeros_like(R_pred)

    # If even zero empirical violations cannot satisfy the requested confidence
    # at this calibration-set size, exhaustive grid search cannot certify the
    # deltas.  Return the most conservative offsets immediately and mark the
    # result rather than spending minutes/hours looping over impossible tuples.
    min_zero_loss_ucb = cp_ucb(0, max(n, 1), xi)
    if min_zero_loss_ucb > min(float(delta_R), float(delta_H), float(delta_delta), float(delta_C)):
        return {
            "q_R": float(q_R_grid[-1]),
            "q_H": float(q_H_grid[-1]),
            "q_delta": float(q_delta_grid[-1]),
            "q_C": float(q_C_grid[-1]),
            "eta_R": eta_R,
            "eta_H": eta_H,
            "epsilon_H": epsilon_H,
            "delta_R": delta_R,
            "delta_H": delta_H,
            "delta_delta": delta_delta,
            "delta_C": delta_C,
            "xi": xi,
            "n_calib": int(n),
            "cp_ucb_R": float(min_zero_loss_ucb),
            "cp_ucb_H": float(min_zero_loss_ucb),
            "cp_ucb_delta": float(min_zero_loss_ucb),
            "cp_ucb_C": float(min_zero_loss_ucb),
            "calibration_infeasible_sample_size": True,
            "split": "calib",
            "mode_alignment": "fixed_semantic_index",
        }

    params = SelectorParams(eta_R=eta_R, eta_H=eta_H, epsilon_H=epsilon_H)

    def evaluate_tuple(qR: float, qH: float, qd: float, qC: float) -> tuple[float, float, float, float, int]:
        losses_R = []
        losses_H = []
        losses_d = []
        losses_C = []
        q = {"q_R": float(qR), "q_H": float(qH), "q_delta": float(qd), "q_C": float(qC)}
        for i in range(n):
            prof = {"R": R_pred[i], "H": H_pred[i], "dH": dH_pred[i], "C": C_pred[i], "B": np.zeros_like(R_pred[i]), "K_post": np.zeros_like(R_pred[i])}
            sel = select_action(list(range(R_pred.shape[1])), prof, U_drv[i], q=q, masks={"action_mask": action_mask[i]}, params=params)
            a = int(sel["action_index"])
            if a < 0:
                continue
            losses_R.append(bool(R_star[i, a] < eta_R))
            losses_H.append(bool(H_star[i, a] > eta_H))
            losses_d.append(bool(dH_star[i, a] > epsilon_H))
            losses_C.append(bool(C_star[i, a] > 0.0))
        nr = max(len(losses_R), 1)
        return (
            cp_ucb(int(np.sum(losses_R)), nr, xi),
            cp_ucb(int(np.sum(losses_H)), nr, xi),
            cp_ucb(int(np.sum(losses_d)), nr, xi),
            cp_ucb(int(np.sum(losses_C)), nr, xi),
            int(len(losses_R)),
        )

    grids = [q_R_grid, q_H_grid, q_delta_grid, q_C_grid]
    idx = [0, 0, 0, 0]
    best = None
    max_steps = sum(len(g) for g in grids) + 4
    targets = np.asarray([delta_R, delta_H, delta_delta, delta_C], dtype=float)
    for _ in range(max_steps):
        qR, qH, qd, qC = (grids[j][idx[j]] for j in range(4))
        uR, uH, ud, uC, n_selected = evaluate_tuple(qR, qH, qd, qC)
        u = np.asarray([uR, uH, ud, uC], dtype=float)
        best = (qR, qH, qd, qC, uR, uH, ud, uC, n_selected)
        if np.all(u <= targets):
            break
        ratios = u / np.maximum(targets, 1e-12)
        advanced = False
        for j in np.argsort(-ratios):
            if idx[int(j)] + 1 < len(grids[int(j)]):
                idx[int(j)] += 1
                advanced = True
                break
        if not advanced:
            break
    assert best is not None
    feasible = bool(np.all(np.asarray(best[4:8]) <= targets))
    return {
        "q_R": float(best[0]),
        "q_H": float(best[1]),
        "q_delta": float(best[2]),
        "q_C": float(best[3]),
        "eta_R": eta_R,
        "eta_H": eta_H,
        "epsilon_H": epsilon_H,
        "delta_R": delta_R,
        "delta_H": delta_H,
        "delta_delta": delta_delta,
        "delta_C": delta_C,
        "xi": xi,
        "n_calib": int(n),
        "n_selected_for_calibration": int(best[8]),
        "cp_ucb_R": float(best[4]),
        "cp_ucb_H": float(best[5]),
        "cp_ucb_delta": float(best[6]),
        "cp_ucb_C": float(best[7]),
        "calibration_feasible": feasible,
        "calibration_search": "monotone_coordinate_grid",
        "split": "calib",
        "mode_alignment": "fixed_semantic_index",
    }
