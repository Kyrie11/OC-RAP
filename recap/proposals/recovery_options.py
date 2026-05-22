from __future__ import annotations

from typing import List
import numpy as np

from recap.utils.datatypes import ActionPrefix, RecoveryOption, RouteInfo, MapFeatures

OPTION_TYPES = ["maintain", "stop", "lane", "route", "yield", "escape", "stabilize"]


def _roll_from_terminal(term: np.ndarray, target_y: float, target_speed: float, H_r: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    st = np.zeros((H_r + 1, 6), dtype=np.float32)
    st[0] = term.astype(np.float32)
    T = H_r * dt
    y0 = float(term[1])
    v0 = float(term[3])
    for k in range(1, H_r + 1):
        s = k / H_r
        smooth = 3 * s**2 - 2 * s**3
        st[k, 1] = y0 + (target_y - y0) * smooth
        st[k, 3] = v0 + (target_speed - v0) * s
        st[k, 0] = st[k - 1, 0] + 0.5 * (st[k, 3] + st[k - 1, 3]) * dt
        dy = st[k, 1] - st[k - 1, 1]
        dx = max(st[k, 0] - st[k - 1, 0], 1e-4)
        st[k, 2] = np.arctan2(dy, dx)
    st[:, 4] = np.gradient(st[:, 3], dt)
    st[:, 5] = np.gradient(st[:, 2], dt) / np.maximum(st[:, 3], 1e-3)
    # The recovery reference must begin exactly at the prefix terminal state.
    st[0] = term.astype(np.float32)
    controls = np.zeros((H_r, 3), dtype=np.float32)
    controls[:, 0] = st[:-1, 4]
    controls[:, 1] = st[:-1, 5]
    controls[:, 2] = np.gradient(st[:, 4], dt)[:-1]
    return st, controls


def generate_options_for_action(action: ActionPrefix, route_info: RouteInfo, map_features: MapFeatures | None = None, L: int = 12, H_r: int = 25, dt: float = 0.2) -> List[RecoveryOption]:
    term = action.states[-1]
    speed_limit = route_info.speed_limit_mps
    candidates: List[RecoveryOption] = []

    def add(kind: str, target_y: float, target_speed: float, valid: bool = True, conditional: bool = False, reason: str = ""):
        oid = len(candidates)
        st, ct = _roll_from_terminal(term, target_y, target_speed, H_r, dt)
        params = np.array([OPTION_TYPES.index(kind), target_y, target_speed, H_r * dt, st[-1, 0], st[-1, 1]], dtype=np.float32)
        candidates.append(RecoveryOption(oid, action.action_id, kind, bool(valid), H_r, st[-1, :3].copy(), float(target_speed), params, st, ct, None, reason, conditional))

    add("maintain", float(term[1]), min(float(term[3]), speed_limit), valid=True)
    # stop requires bounded deceleration
    d_stop = float(term[3] ** 2 / (2 * 3.0) + 2.0)
    required_decel = float(term[3] ** 2 / max(2 * d_stop, 1e-6))
    add("stop", float(term[1]), 0.0, valid=required_decel <= 6.0, reason="required_decel" if required_decel > 6.0 else "")
    add("lane", 0.0, min(float(term[3]), speed_limit), valid=abs(term[1]) <= 4.5)
    add("route", 0.0, min(max(float(term[3]), 2.0), speed_limit), valid=True)
    add("yield", float(term[1]), max(0.0, 0.5 * float(term[3])), valid=True)
    add("escape", 3.0 if term[1] <= 0 else -3.0, min(float(term[3]), speed_limit), valid=True)
    add("stabilize", 0.0, 0.0, valid=True, conditional=True)

    valid_by_type = {}
    for opt in candidates:
        if opt.valid and opt.type not in valid_by_type:
            valid_by_type[opt.type] = opt
    kept: List[RecoveryOption] = list(valid_by_type.values())
    rest = [o for o in candidates if o not in kept]
    kept.extend(rest)
    kept = kept[:L]
    while len(kept) < L:
        st, ct = _roll_from_terminal(term, float(term[1]), 0.0, H_r, dt)
        kept.append(RecoveryOption(len(kept), action.action_id, "pad", False, H_r, st[-1, :3].copy(), 0.0, np.zeros(6, dtype=np.float32), st, ct, None, "padding", False))
    for i, opt in enumerate(kept):
        opt.option_id = i
    return kept


def generate_recovery_options(actions: List[ActionPrefix], route_info: RouteInfo, map_features: MapFeatures | None = None, L: int = 12, H_r: int = 25, dt: float = 0.2) -> List[List[RecoveryOption]]:
    return [generate_options_for_action(a, route_info, map_features, L, H_r, dt) for a in actions]


def options_to_tensors(options: List[List[RecoveryOption]]) -> dict:
    return {
        "options_states_ref": np.stack([[o.states_ref for o in opts] for opts in options]).astype(np.float32),
        "options_controls_ref": np.stack([[o.controls_ref for o in opts] for opts in options]).astype(np.float32),
        "options_params": np.stack([[o.params for o in opts] for opts in options]).astype(np.float32),
        "option_mask": np.array([[o.valid for o in opts] for opts in options], dtype=bool),
    }
