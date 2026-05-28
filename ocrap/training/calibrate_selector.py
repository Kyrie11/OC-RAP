from __future__ import annotations

import itertools
import numpy as np
from ocrap.models.selector import select_action, SelectorParams


def cp_ucb(s: int, n: int, xi: float) -> float:
    if n <= 0: return 1.0
    if s == n: return 1.0
    # Distribution-free upper confidence bound.  This avoids the very heavy
    # scipy.stats import in CLI runs while keeping conformal calibration
    # conservative for small calibration sets.
    phat=s/n
    return float(min(1.0, phat + np.sqrt(np.log(1/max(xi,1e-12))/(2*n))))


def calibrate_q(R_pred: np.ndarray, R_star: np.ndarray, dH_pred: np.ndarray, dH_star: np.ndarray, action_mask: np.ndarray, q_R_grid=None, q_H_grid=None, eta_R: float = 0.70, epsilon_H: float = 0.05, delta_R: float = 0.05, delta_H: float = 0.05, xi: float = 0.05, H_pred=None, H_star=None, C_pred=None, C_star=None, U_drv=None, q_delta_grid=None, q_C_grid=None, eta_H: float = 0.50) -> dict:
    """Selected-action CRISP calibration over all four offsets.

    Backward compatible with the old q_R/q_H signature; when H/C/U are absent,
    it calibrates q_R and q_delta on selected actions and returns q_H=q_C=0.
    """
    q_R_grid=np.arange(0.0,0.501,0.02) if q_R_grid is None else q_R_grid
    q_H_grid=np.array([0.0]) if q_H_grid is None else q_H_grid
    q_delta_grid=np.arange(0.0,0.301,0.01) if q_delta_grid is None else q_delta_grid
    q_C_grid=np.array([0.0]) if q_C_grid is None else q_C_grid
    n=R_pred.shape[0]
    if H_pred is None: H_pred=np.zeros_like(R_pred)
    if H_star is None: H_star=np.zeros_like(R_star)
    if C_pred is None: C_pred=np.zeros_like(R_pred)
    if C_star is None: C_star=np.zeros_like(R_star)
    if U_drv is None: U_drv=np.zeros_like(R_pred)
    best=None
    for qR,qH,qd,qC in itertools.product(q_R_grid, q_H_grid, q_delta_grid, q_C_grid):
        losses_R=[]; losses_H=[]; losses_d=[]; losses_C=[]
        for i in range(n):
            prof={"R":R_pred[i],"H":H_pred[i],"dH":dH_pred[i],"C":C_pred[i],"B":np.zeros_like(R_pred[i]),"K_post":np.zeros_like(R_pred[i])}
            sel=select_action(list(range(R_pred.shape[1])), prof, U_drv[i], q={"q_R":qR,"q_H":qH,"q_delta":qd,"q_C":qC}, masks={"action_mask":action_mask[i]}, params=SelectorParams(eta_R=eta_R, eta_H=eta_H, epsilon_H=epsilon_H))
            a=sel["action_index"]
            if a < 0: continue
            losses_R.append(R_star[i,a] < eta_R)
            losses_H.append(H_star[i,a] > eta_H)
            losses_d.append(dH_star[i,a] > epsilon_H)
            losses_C.append(C_star[i,a] > 0.0)
        nr=max(len(losses_R),1)
        uR=cp_ucb(int(np.sum(losses_R)),nr,xi); uH=cp_ucb(int(np.sum(losses_H)),nr,xi); ud=cp_ucb(int(np.sum(losses_d)),nr,xi); uC=cp_ucb(int(np.sum(losses_C)),nr,xi)
        if uR <= delta_R and uH <= delta_H and ud <= delta_H and uC <= delta_H:
            best=(qR,qH,qd,qC,uR,uH,ud,uC); break
    if best is None:
        best=(float(q_R_grid[-1]), float(q_H_grid[-1]), float(q_delta_grid[-1]), float(q_C_grid[-1]), 1.0,1.0,1.0,1.0)
    return {"q_R":float(best[0]),"q_H":float(best[1]),"q_delta":float(best[2]),"q_C":float(best[3]),"eta_R":eta_R,"eta_H":eta_H,"epsilon_H":epsilon_H,"delta_R":delta_R,"delta_H":delta_H,"delta_delta":delta_H,"delta_C":delta_H,"xi":xi,"n_calib":int(n),"cp_ucb_R":float(best[4]),"cp_ucb_H":float(best[5]),"cp_ucb_delta":float(best[6]),"cp_ucb_C":float(best[7]),"split":"calib","mode_alignment":"fixed_semantic_index"}
