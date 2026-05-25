from __future__ import annotations

import numpy as np
from recap.envs.scenario_motion import local_states_to_world
from recap.utils.datatypes import ActionPrefix, EgoState, MapFeatures

DEFAULT_BOUNDS = {
    "a_max": 4.0,
    "brake_max": 6.0,
    "j_max": 6.0,
    "kappa_max": 0.25,
    "dkappa_max": 0.20,
    "v_min": 0.0,
    "speed_limit_margin": 3.0,
}


def project_controls(states: np.ndarray, controls: np.ndarray, speed_limit: float = 13.9, bounds: dict | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    bounds = {**DEFAULT_BOUNDS, **(bounds or {})}
    st = np.asarray(states, dtype=np.float32).copy()
    ct = np.asarray(controls, dtype=np.float32).copy()
    violation = 0.0
    a_before = ct[:, 0].copy()
    ct[:, 0] = np.clip(ct[:, 0], -bounds["brake_max"], bounds["a_max"])
    ct[:, 1] = np.clip(ct[:, 1], -bounds["kappa_max"], bounds["kappa_max"])
    if ct.shape[1] > 2:
        ct[:, 2] = np.clip(ct[:, 2], -bounds["j_max"], bounds["j_max"])
    violation += float(np.mean(np.abs(a_before - ct[:, 0])))
    st[:, 3] = np.clip(st[:, 3], bounds["v_min"], speed_limit + bounds["speed_limit_margin"])
    st[:, 5] = np.clip(st[:, 5], -bounds["kappa_max"], bounds["kappa_max"])
    return st, ct, violation


def validate_prefix(prefix: ActionPrefix, map_features: MapFeatures | None = None, speed_limit: float = 13.9, ego: EgoState | None = None) -> ActionPrefix:
    st, ct, violation = project_controls(prefix.states, prefix.controls, speed_limit)
    valid = bool(prefix.valid)
    reason = prefix.mask_reason
    if np.nanmax(st[:, 3]) > speed_limit + 5.0:
        valid = False
        reason = "speed_limit_violation"
    if np.nanmax(np.abs(st[:, 5])) > DEFAULT_BOUNDS["kappa_max"] + 1e-5:
        valid = False
        reason = "curvature_violation"
    # Static collision rejection is only implemented for axis-aligned polygon bbox
    # overlap in this portable MVP. Dynamic conflicts are deliberately not rejected.
    if map_features is not None and map_features.static_obstacles:
        # Prefix states are ego-local.  Map features from WOMD/ScenarioNet are in
        # centralized world coordinates, so static-obstacle pruning must compare
        # in world coordinates when the root ego pose is available.
        check_states = local_states_to_world(ego, st) if ego is not None else st
        for obs in map_features.static_obstacles:
            omin, omax = np.min(obs, axis=0), np.max(obs, axis=0)
            for x, y in check_states[:, :2]:
                if omin[0] <= x <= omax[0] and omin[1] <= y <= omax[1]:
                    valid = False
                    reason = "static_collision"
                    break
    return ActionPrefix(prefix.action_id, valid, prefix.type, st, ct, prefix.params, prefix.swept_polygons, prefix.score_prop - violation, reason)
