from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal
import numpy as np
from recap.utils.masks import masked_argmax


@dataclass
class SelectorParams:
    eta_R: float = 0.70
    epsilon_H: float = 0.05
    lambda_B: float = 0.2
    lambda_K: float = 0.2
    lambda_H: float = 1.0
    lambda_U_selector: float = 0.0
    method: str = "ours"
    no_harm_constraint: bool = False
    no_controlled_relaxation: bool = False


def emergency_stop_prefix():
    return {"type": "emergency_stop", "valid": True}


def select_action(actions, profiles: Dict[str, np.ndarray], U_drv: np.ndarray, q_R: float, q_H: float, masks: Dict[str, np.ndarray], params: SelectorParams | dict | None = None) -> dict:
    if params is None:
        params = SelectorParams()
    if isinstance(params, dict):
        params = SelectorParams(**{k: v for k, v in params.items() if k in SelectorParams.__annotations__})
    if params.method == "ours" and params.lambda_U_selector > 0:
        raise ValueError("main method must not double-penalize U in selector; use an ablation flag")
    valid = np.asarray(masks["action_mask"], dtype=bool)
    if not valid.any():
        return {
            "action_index": -1,
            "action_prefix": emergency_stop_prefix(),
            "profile": {},
            "witness": np.array([], dtype=np.int64),
            "status": "no_valid_action",
            "activated_recovery_constraint": True,
            "activated_harm_constraint": True,
            "used_emergency_fallback": True,
        }
    R = np.asarray(profiles["R"], dtype=np.float64)
    B = np.asarray(profiles.get("B", np.zeros_like(R)), dtype=np.float64)
    dH = np.asarray(profiles.get("dH", np.zeros_like(R)), dtype=np.float64)
    K_post = np.asarray(profiles.get("K_post", np.zeros_like(R)), dtype=np.float64)
    U_prof = np.asarray(profiles.get("U", np.zeros_like(R)), dtype=np.float64)
    U_drv = np.asarray(U_drv, dtype=np.float64)

    a_nom = masked_argmax(U_drv, valid)
    S_R = valid & (R - q_R >= params.eta_R)
    if params.no_harm_constraint:
        S_H = valid.copy()
    else:
        S_H = valid & (dH + q_H <= params.epsilon_H)
    S = S_R & S_H

    def pack(idx: int, status: str, used_fallback: bool = False):
        return {
            "action_index": int(idx),
            "action_prefix": actions[idx] if idx >= 0 and actions is not None else emergency_stop_prefix(),
            "profile": {k: (v[idx] if hasattr(v, "__len__") and len(np.asarray(v).shape) > 0 else v) for k, v in profiles.items() if k not in ("witness",)},
            "witness": profiles.get("witness", np.array([]))[idx] if idx >= 0 and "witness" in profiles else np.array([], dtype=np.int64),
            "status": status,
            "activated_recovery_constraint": bool(not S_R[a_nom]),
            "activated_harm_constraint": bool(not S_H[a_nom]),
            "used_emergency_fallback": bool(used_fallback),
        }

    if S[a_nom]:
        return pack(a_nom, "nominal_accepted")
    if S.any():
        score = U_drv - params.lambda_B * B - params.lambda_K * K_post
        if params.lambda_U_selector > 0:
            score = score - params.lambda_U_selector * U_prof
        return pack(masked_argmax(score, S), "constrained")
    if params.no_controlled_relaxation:
        return pack(-1, "no_valid_action", used_fallback=True)
    score = R - params.lambda_H * np.maximum(dH - params.epsilon_H, 0.0) - params.lambda_K * K_post - params.lambda_B * B
    return pack(masked_argmax(score, valid), "controlled_relaxation")
