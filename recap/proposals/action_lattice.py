from __future__ import annotations

from typing import List
import numpy as np

from recap.utils.datatypes import EgoState, RouteInfo, ActionPrefix, MapFeatures
from .action_projection import validate_prefix


def _simulate_prefix(v0: float, lateral_offset: float, v_target: float, T_p: float, H_p: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(H_p + 1, dtype=np.float32) * dt
    # Smooth lateral cubic y = dy * smoothstep(t/T)
    s = np.clip(t / max(T_p, 1e-6), 0.0, 1.0)
    smooth = 3 * s**2 - 2 * s**3
    y = lateral_offset * smooth
    v = v0 + (v_target - v0) * s
    x = np.cumsum(np.r_[0.0, 0.5 * (v[1:] + v[:-1]) * dt]).astype(np.float32)
    dy_dt = np.gradient(y, dt)
    psi = np.arctan2(dy_dt, np.maximum(v, 1e-3)).astype(np.float32)
    a = np.gradient(v, dt).astype(np.float32)
    kappa = np.gradient(psi, dt) / np.maximum(v, 1e-3)
    states = np.stack([x, y, psi, v, a, kappa], axis=-1).astype(np.float32)
    controls = np.zeros((H_p, 3), dtype=np.float32)
    controls[:, 0] = a[:-1]
    controls[:, 1] = kappa[:-1]
    controls[:, 2] = np.gradient(a, dt)[:-1]
    return states, controls


def generate_lattice_actions(
    ego: EgoState,
    route_info: RouteInfo,
    map_features: MapFeatures | None = None,
    K_raw: int = 64,
    K: int = 32,
    H_p: int = 10,
    dt: float = 0.2,
    lateral_offsets: list[float] | None = None,
    terminal_speed_factors: list[float] | None = None,
) -> List[ActionPrefix]:
    T_p = H_p * dt
    lateral_offsets = lateral_offsets or [-1.5, 0.0, 1.5]
    terminal_speed_factors = terminal_speed_factors or [0.0, 0.5, 1.0, 1.2]
    speed_limit = route_info.speed_limit_mps
    candidates: List[ActionPrefix] = []
    aid = 0
    for dy in lateral_offsets:
        for sf in terminal_speed_factors:
            for anchor in [10.0, 20.0, 30.0]:
                vtar = float(np.clip(speed_limit * sf, 0.0, speed_limit + 3.0))
                states, controls = _simulate_prefix(ego.v, dy, vtar, T_p, H_p, dt)
                params = np.array([anchor, dy, vtar, controls[:, 0].mean(), controls[:, 1].mean(), T_p], dtype=np.float32)
                progress = states[-1, 0]
                jerk = np.mean(np.abs(controls[:, 2]))
                curv = np.mean(np.abs(controls[:, 1]))
                offroute = abs(dy)
                score = progress - 0.2 * jerk - 2.0 * curv - 0.3 * offroute
                pref = ActionPrefix(aid, True, "lattice", states, controls, params, [], float(score))
                candidates.append(validate_prefix(pref, map_features, speed_limit))
                aid += 1
    valid = [c for c in candidates if c.valid]
    invalid = [c for c in candidates if not c.valid]
    ordered = sorted(valid, key=lambda a: a.score_prop, reverse=True) + invalid
    kept = ordered[:K]
    # Pad to K so downstream tensors always have masks.
    while len(kept) < K:
        states, controls = _simulate_prefix(ego.v, 0.0, 0.0, T_p, H_p, dt)
        kept.append(ActionPrefix(len(kept), False, "pad", states, controls, np.zeros(6, dtype=np.float32), [], -1e9, "padding"))
    for i, a in enumerate(kept):
        a.action_id = i
    return kept


def actions_to_tensors(actions: List[ActionPrefix]) -> dict:
    return {
        "actions_states": np.stack([a.states for a in actions]).astype(np.float32),
        "actions_controls": np.stack([a.controls for a in actions]).astype(np.float32),
        "actions_params": np.stack([a.params for a in actions]).astype(np.float32),
        "action_mask": np.array([a.valid for a in actions], dtype=bool),
    }
