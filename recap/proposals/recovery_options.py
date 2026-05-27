from __future__ import annotations

from typing import List
import numpy as np

from recap.utils.datatypes import ActionPrefix, RecoveryOption, RecoveryAffordanceToken, RouteInfo, MapFeatures

OPTION_TYPES = ["maintain", "stop", "lane", "route", "yield", "escape", "stabilize"]


def _roll_from_terminal(term: np.ndarray, target_y: float, target_speed: float, H_r: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    st = np.zeros((H_r + 1, 6), dtype=np.float32)
    st[0] = term.astype(np.float32)
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


def _reference_valid(st: np.ndarray, ct: np.ndarray, *, speed_limit: float) -> tuple[bool, str]:
    if not np.all(np.isfinite(st)) or not np.all(np.isfinite(ct)):
        return False, "nonfinite_reference"
    if float(np.nanmax(st[:, 3])) > speed_limit + 3.0:
        return False, "speed_limit_reference"
    # Match the paper/default dynamic bounds approximately.  This is a static
    # reference feasibility check; closed-loop teacher rollout remains the final
    # authority for option labels.
    if float(np.nanmax(ct[:, 0])) > 4.0 + 1e-4 or float(np.nanmin(ct[:, 0])) < -6.0 - 1e-4:
        return False, "accel_reference"
    # Jerk from a short polynomial reference is diagnostic, not an online mask:
    # teacher rollout determines final success.  Keep acceleration/curvature as
    # relaxed feasibility checks so stop/yield tokens are still evaluable.
    if float(np.nanmax(np.abs(st[:, 5]))) > 0.35 + 1e-4:
        return False, "curvature_reference"
    return True, ""


def generate_options_for_action(action: ActionPrefix, route_info: RouteInfo, map_features: MapFeatures | None = None, L: int = 12, H_r: int = 25, dt: float = 0.2) -> List[RecoveryOption]:
    term = action.states[-1]
    speed_limit = route_info.speed_limit_mps
    candidates: List[RecoveryOption] = []
    seen: set[tuple[str, float, float]] = set()

    def add(kind: str, target_y: float, target_speed: float, valid: bool = True, conditional: bool = False, reason: str = ""):
        target_y = float(target_y)
        target_speed = float(np.clip(target_speed, 0.0, speed_limit + 3.0))
        key = (kind, round(target_y, 2), round(target_speed, 2))
        if key in seen:
            return
        seen.add(key)
        oid = len(candidates)
        st, ct = _roll_from_terminal(term, target_y, target_speed, H_r, dt)
        ref_valid, ref_reason = _reference_valid(st, ct, speed_limit=speed_limit)
        opt_valid = bool(valid and ref_valid and action.valid)
        why = reason or (ref_reason if not ref_valid else ("invalid_action" if not action.valid else ""))
        params = np.array([OPTION_TYPES.index(kind), target_y, target_speed, H_r * dt, st[-1, 0], st[-1, 1]], dtype=np.float32)
        candidates.append(RecoveryOption(oid, action.action_id, kind, opt_valid, H_r, st[-1, :3].copy(), float(target_speed), params, st, ct, None, why, conditional))

    v_term = float(term[3])
    y_term = float(term[1])
    lane_width = 3.6
    current_speed = min(v_term, speed_limit)
    cruise_speed = min(max(v_term, 2.0), speed_limit)
    slow_speed = max(0.0, min(0.5 * v_term, speed_limit))

    # Preserve the seven semantic types, but instantiate multiple scene-grounded
    # anchors so L=12 is actually used.  The previous implementation produced
    # exactly seven valid options per action and five pads, which starved the
    # existential recovery operator and weakened same-root ranking labels.
    add("maintain", y_term, current_speed)
    add("maintain", y_term, min(speed_limit, max(current_speed, 0.8 * speed_limit)))
    d_stop = float(v_term ** 2 / (2 * 3.0) + 2.0)
    required_decel = float(v_term ** 2 / max(2 * d_stop, 1e-6))
    add("stop", y_term, 0.0, valid=required_decel <= 6.0, reason="required_decel" if required_decel > 6.0 else "")
    add("stop", 0.0, 0.0, valid=required_decel <= 6.0 and abs(y_term) <= 4.5, reason="required_decel_or_lane_return" if required_decel > 6.0 or abs(y_term) > 4.5 else "")
    add("lane", 0.0, current_speed, valid=abs(y_term) <= 5.5)
    add("lane", np.clip(y_term, -0.5 * lane_width, 0.5 * lane_width), slow_speed, valid=abs(y_term) <= 5.5)
    add("route", 0.0, cruise_speed)
    add("route", 0.0, slow_speed)
    add("yield", y_term, slow_speed)
    add("yield", 0.0, slow_speed, valid=abs(y_term) <= 5.5)
    escape_sign = 1.0 if y_term <= 0 else -1.0
    add("escape", escape_sign * 3.0, current_speed)
    add("escape", escape_sign * 4.5, slow_speed)
    add("stabilize", 0.0, 0.0, conditional=True)

    # First preserve one token per OC-RAP semantic tag when L allows.  Masking
    # indicates relaxed executability, not teacher success.
    required = ["maintain", "stop", "lane", "route", "yield", "escape", "stabilize"]
    first_by_type = {}
    for opt in candidates:
        if opt.type not in first_by_type:
            first_by_type[opt.type] = opt
    kept: List[RecoveryOption] = [first_by_type[t] for t in required if t in first_by_type][:L]
    rest = [o for o in candidates if o not in kept]
    rest = sorted(rest, key=lambda o: (not o.valid, abs(float(o.params[1])), -float(o.target_speed)))
    kept.extend(rest)
    kept = kept[:L]
    while len(kept) < L:
        st, ct = _roll_from_terminal(term, y_term, 0.0, H_r, dt)
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


def generate_recovery_affordances(actions: List[ActionPrefix], route_info: RouteInfo, map_features: MapFeatures | None = None, traffic_control=None, L: int = 12, H_r: int = 25, dt: float = 0.2) -> List[List[RecoveryOption]]:
    """OC-RAP name for executable recovery affordance tokens.

    Kept as RecoveryOption-compatible objects for older scripts/tests.
    """
    return generate_recovery_options(actions, route_info, map_features, L=L, H_r=H_r, dt=dt)
