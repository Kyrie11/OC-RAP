from __future__ import annotations

import math
import numpy as np
from ocrap.envs.scenario_motion import local_states_to_world
from ocrap.raster.geometry import oriented_box
from ocrap.utils.datatypes import ActionPrefix, EgoState, MapFeatures

DEFAULT_BOUNDS = {
    "a_max": 4.0,
    "brake_max": 6.0,
    "j_max": 6.0,
    "kappa_max": 0.25,
    "dkappa_max": 0.20,
    "v_min": 0.0,
    "speed_limit_margin": 3.0,
    "dt": 0.2,
    "wheelbase": 2.8,
    "ego_length": 4.7,
    "ego_width": 1.9,
    "route_y_max": 8.0,
    "heading_max_rad": math.radians(55.0),
}


def _wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def _limit_rate(values: np.ndarray, max_delta: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    if out.size <= 1:
        return out
    for i in range(1, out.shape[0]):
        out[i] = np.clip(out[i], out[i - 1] - max_delta, out[i - 1] + max_delta)
    return out


def _rollout_bicycle(initial_state: np.ndarray, controls: np.ndarray, speed_limit: float, bounds: dict) -> np.ndarray:
    """Forward-project controls with a kinematic-bicycle curvature model.

    This is a portable sequential projection used when a QP solver is not
    available.  It enforces the same dynamic limits that OC-RAP labels assume and
    returns dynamically consistent states instead of merely clipping an already
    sampled lattice trajectory.
    """
    dt = float(bounds.get("dt", 0.2))
    H = int(controls.shape[0])
    st = np.zeros((H + 1, 6), dtype=np.float32)
    st[0] = np.asarray(initial_state, dtype=np.float32)[:6]
    st[0, 3] = np.clip(st[0, 3], bounds["v_min"], speed_limit + bounds["speed_limit_margin"])
    st[0, 5] = np.clip(st[0, 5], -bounds["kappa_max"], bounds["kappa_max"])
    for k in range(H):
        a = float(controls[k, 0])
        kappa = float(controls[k, 1])
        v0 = float(st[k, 3])
        v1 = float(np.clip(v0 + a * dt, bounds["v_min"], speed_limit + bounds["speed_limit_margin"]))
        v_mid = 0.5 * (v0 + v1)
        psi0 = float(st[k, 2])
        psi1 = float(_wrap_angle(psi0 + v_mid * kappa * dt))
        psi_mid = 0.5 * (psi0 + psi1)
        st[k + 1, 0] = st[k, 0] + v_mid * math.cos(psi_mid) * dt
        st[k + 1, 1] = st[k, 1] + v_mid * math.sin(psi_mid) * dt
        st[k + 1, 2] = psi1
        st[k + 1, 3] = v1
        st[k + 1, 4] = a
        st[k + 1, 5] = kappa
    if H > 0:
        st[0, 4] = controls[0, 0]
        st[0, 5] = controls[0, 1]
    return st


def project_controls(states: np.ndarray, controls: np.ndarray, speed_limit: float = 13.9, bounds: dict | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Project a raw prefix to dynamically executable controls/states.

    The paper's projection Γ is a constrained optimization over dynamics,
    input/jerk/curvature-rate limits and swept geometry.  This implementation is
    a deterministic sequential approximation: clip input bounds, enforce rate
    limits, and then re-integrate a kinematic bicycle trajectory from the prefix
    initial state.  The returned ``violation`` is a soft penalty used for pruning.
    """
    bounds = {**DEFAULT_BOUNDS, **(bounds or {})}
    raw_st = np.asarray(states, dtype=np.float32)
    ct = np.asarray(controls, dtype=np.float32).copy()
    if raw_st.ndim != 2 or raw_st.shape[1] < 6:
        raise ValueError("states must have shape [H+1,6]")
    if ct.ndim != 2 or ct.shape[0] != raw_st.shape[0] - 1:
        raise ValueError("controls must have shape [H,>=2] matching states")
    raw_ct = ct.copy()
    ct[:, 0] = np.clip(ct[:, 0], -bounds["brake_max"], bounds["a_max"])
    ct[:, 1] = np.clip(ct[:, 1], -bounds["kappa_max"], bounds["kappa_max"])
    dt = float(bounds.get("dt", 0.2))
    ct[:, 0] = _limit_rate(ct[:, 0], float(bounds["j_max"]) * dt)
    ct[:, 1] = _limit_rate(ct[:, 1], float(bounds["dkappa_max"]) * dt)
    if ct.shape[1] > 2:
        ct[:, 2] = np.gradient(ct[:, 0], dt).astype(np.float32)
        ct[:, 2] = np.clip(ct[:, 2], -bounds["j_max"], bounds["j_max"])
    st = _rollout_bicycle(raw_st[0], ct, float(speed_limit), bounds)
    ctrl_delta = float(np.nanmean(np.abs(raw_ct[:, :2] - ct[:, :2]))) if raw_ct.size else 0.0
    state_delta = float(np.nanmean(np.abs(raw_st[:, :4] - st[:, :4]))) if raw_st.shape == st.shape else 0.0
    violation = ctrl_delta + 0.05 * state_delta
    return st.astype(np.float32), ct.astype(np.float32), violation


def _polygon_axes(poly: np.ndarray) -> list[np.ndarray]:
    axes: list[np.ndarray] = []
    p = np.asarray(poly, dtype=np.float32)
    for i in range(len(p)):
        edge = p[(i + 1) % len(p)] - p[i]
        n = np.array([-edge[1], edge[0]], dtype=np.float32)
        norm = float(np.linalg.norm(n))
        if norm > 1e-8:
            axes.append(n / norm)
    return axes


def _polygons_intersect(a: np.ndarray, b: np.ndarray) -> bool:
    """Convex polygon intersection via separating-axis theorem."""
    if len(a) < 3 or len(b) < 3:
        return False
    for axis in _polygon_axes(a) + _polygon_axes(b):
        pa = a @ axis
        pb = b @ axis
        if float(pa.max()) < float(pb.min()) or float(pb.max()) < float(pa.min()):
            return False
    return True


def _swept_static_collision(states_xyh: np.ndarray, obstacles: list[np.ndarray], *, length: float, width: float) -> bool:
    for row in states_xyh:
        box = oriented_box(np.asarray(row[:2], dtype=np.float32), float(row[2]), float(length), float(width))
        for obs in obstacles:
            poly = np.asarray(obs, dtype=np.float32)[:, :2]
            if len(poly) >= 3 and _polygons_intersect(box, poly):
                return True
    return False


def validate_prefix(prefix: ActionPrefix, map_features: MapFeatures | None = None, speed_limit: float = 13.9, ego: EgoState | None = None, bounds: dict | None = None) -> ActionPrefix:
    bounds_eff = {**DEFAULT_BOUNDS, **(bounds or {})}
    st, ct, violation = project_controls(prefix.states, prefix.controls, speed_limit, bounds_eff)
    valid = bool(prefix.valid)
    reason = prefix.mask_reason
    if np.nanmax(st[:, 3]) > speed_limit + bounds_eff["speed_limit_margin"] + 2.0:
        valid = False
        reason = "speed_limit_violation"
    if np.nanmax(np.abs(st[:, 5])) > bounds_eff["kappa_max"] + 1e-5:
        valid = False
        reason = "curvature_violation"
    if np.nanmax(np.abs(np.gradient(st[:, 5], bounds_eff["dt"]))) > bounds_eff["dkappa_max"] + 1e-3:
        valid = False
        reason = "curvature_rate_violation"
    if np.nanmax(np.abs(st[:, 2])) > bounds_eff["heading_max_rad"] or np.nanmax(np.abs(st[:, 1])) > bounds_eff["route_y_max"]:
        valid = False
        reason = "route_heading_or_lateral_violation"
    if map_features is not None and map_features.static_obstacles:
        # Check both the dynamically projected prefix and the raw proposal corridor.
        # A raw lattice path that crosses an obstacle should not be silently made
        # valid by a control projection that brakes/stops before reaching it.  The
        # latter may be useful as a new proposal, but the original prefix's swept
        # geometry still violated the static hard shell.
        check_states = local_states_to_world(ego, st) if ego is not None else st
        raw_states = local_states_to_world(ego, prefix.states) if ego is not None else prefix.states
        if _swept_static_collision(
            check_states[:, :3],
            map_features.static_obstacles,
            length=float(getattr(ego, "length", bounds_eff["ego_length"])),
            width=float(getattr(ego, "width", bounds_eff["ego_width"])),
        ) or _swept_static_collision(
            np.asarray(raw_states, dtype=np.float32)[:, :3],
            map_features.static_obstacles,
            length=float(getattr(ego, "length", bounds_eff["ego_length"])),
            width=float(getattr(ego, "width", bounds_eff["ego_width"])),
        ):
            valid = False
            reason = "static_collision"
    return ActionPrefix(prefix.action_id, valid, prefix.type, st, ct, prefix.params, prefix.swept_polygons, prefix.score_prop - violation, reason)
