from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np
from recap.utils.masks import masked_argmax


@dataclass
class SelectorParams:
    eta_R: float = 0.70
    eta_H: float = 0.50
    epsilon_H: float = 0.05
    lambda_B: float = 0.2
    lambda_U: float = 0.0
    lambda_U_selector: float = 0.0
    lambda_K: float = 0.2
    lambda_H: float = 1.0
    lambda_C: float = 1.0
    lambda_R: float = 1.0
    method: str = "ocrap"
    no_harm_constraint: bool = False
    no_rule_constraint: bool = False
    no_controlled_relaxation: bool = False


def emergency_stop_prefix():
    return {"type": "emergency_stop", "valid": True}


def _qdict(q=None, q_R=None, q_H=None, q_delta=None, q_C=None):
    if q is None:
        q={}
    if not isinstance(q, dict):
        q={"q_R": float(q)}
    return {
        "q_R": float(q.get("q_R", 0.0 if q_R is None else q_R)),
        "q_H": float(q.get("q_H", 0.0 if q_H is None else q_H)),
        "q_delta": float(q.get("q_delta", 0.0 if q_delta is None else q_delta)),
        "q_C": float(q.get("q_C", 0.0 if q_C is None else q_C)),
    }


def select_action(actions, profiles: Dict[str, np.ndarray], U_drv: np.ndarray, q: dict | float | None = None, masks: Dict[str, np.ndarray] | None = None, params: SelectorParams | dict | None = None, q_R=None, q_H=None, q_delta=None, q_C=None) -> dict:
    # Backward compatibility: select_action(actions, profiles, U, q_R, q_H, masks, params)
    if not (masks is None or isinstance(masks, dict)):
        legacy_q_H = masks
        legacy_masks = params if isinstance(params, dict) and "action_mask" in params else None
        legacy_params = q_R if isinstance(q_R, (SelectorParams, dict)) else None
        q_H = legacy_q_H
        masks = legacy_masks
        params = legacy_params
    if params is None: params=SelectorParams()
    if isinstance(params, dict): params=SelectorParams(**{k:v for k,v in params.items() if k in SelectorParams.__annotations__})
    if params.method == "ours" and params.lambda_U_selector > 0:
        raise ValueError("main method must not double-penalize U in selector; use an ablation flag")
    qv=_qdict(q, q_R, q_H, q_delta, q_C)
    if masks is None: masks={"action_mask": np.ones_like(np.asarray(U_drv), dtype=bool)}
    valid=np.asarray(masks["action_mask"], dtype=bool)
    if not valid.any():
        return {"action_index": -1, "action_prefix": emergency_stop_prefix(), "profile": {}, "witness": np.array([], dtype=np.int64), "status": "no_valid_action", "activated_recovery_constraint": True, "activated_harm_constraint": True, "activated_rule_constraint": True, "used_emergency_fallback": True}
    R=np.asarray(profiles["R"], dtype=np.float64)
    B=np.asarray(profiles.get("B", np.zeros_like(R)), dtype=np.float64)
    H=np.asarray(profiles.get("H", np.zeros_like(R)), dtype=np.float64)
    dH=np.asarray(profiles.get("dH", np.zeros_like(R)), dtype=np.float64)
    K_post=np.asarray(profiles.get("K_post", np.zeros_like(R)), dtype=np.float64)
    U_prof=np.asarray(profiles.get("U", np.zeros_like(R)), dtype=np.float64)
    C=np.asarray(profiles.get("C", np.zeros_like(R)), dtype=np.float64)
    U_drv=np.asarray(U_drv, dtype=np.float64)
    a_nom=masked_argmax(U_drv, valid)
    S_R=valid & (R - qv["q_R"] >= params.eta_R)
    if params.no_harm_constraint:
        S_H_abs=valid.copy(); S_H_gap=valid.copy()
    else:
        S_H_abs=valid & (H + qv["q_H"] <= params.eta_H)
        S_H_gap=valid & (dH + qv["q_delta"] <= params.epsilon_H)
    S_C=valid.copy() if params.no_rule_constraint else (valid & (C + qv["q_C"] <= 0.0))
    S=S_R & S_H_abs & S_H_gap & S_C
    def pack(idx:int,status:str,used_fallback:bool=False):
        prof={k:(v[idx] if hasattr(v,"__len__") and np.asarray(v).ndim>0 and idx>=0 else v) for k,v in profiles.items() if k != "witness"}
        wit=profiles.get("witness", np.array([]))[idx] if idx>=0 and "witness" in profiles else np.array([], dtype=np.int64)
        return {"action_index": int(idx), "action_prefix": actions[idx] if idx>=0 and actions is not None else emergency_stop_prefix(), "profile": prof, "witness": wit, "status": status, "activated_recovery_constraint": bool(not S_R[a_nom]), "activated_harm_constraint": bool((not S_H_abs[a_nom]) or (not S_H_gap[a_nom])), "activated_rule_constraint": bool(not S_C[a_nom]), "used_emergency_fallback": bool(used_fallback)}
    if S[a_nom]: return pack(a_nom, "nominal_accepted")
    if S.any():
        score=U_drv - params.lambda_B*B - (params.lambda_U or params.lambda_U_selector)*U_prof - params.lambda_K*K_post
        return pack(masked_argmax(score, S), "constrained")
    if params.no_controlled_relaxation:
        return pack(-1, "no_valid_action", used_fallback=True)
    # minimum-risk controlled relaxation: choose smallest calibrated violation.
    violation=(params.lambda_R*np.maximum(params.eta_R - R, 0.0) + params.lambda_H*np.maximum(dH - params.epsilon_H, 0.0) + params.lambda_C*np.maximum(C, 0.0) + params.lambda_K*K_post)
    return pack(int(np.argmin(np.where(valid, violation, np.inf))), "controlled_relaxation")
