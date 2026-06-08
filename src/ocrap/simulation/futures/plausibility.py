from __future__ import annotations

import numpy as np


def check_no_teleportation(states: np.ndarray, valid: np.ndarray, max_step_m: float = 6.0) -> bool:
    ok = valid.astype(bool)
    for a in range(states.shape[1]):
        idx = np.where(ok[:, a])[0]
        if len(idx) <= 1:
            continue
        d = np.linalg.norm(np.diff(states[idx, a, :2], axis=0), axis=-1)
        if np.any(d > max_step_m):
            return False
    return True


def check_speed_accel_bounds(states: np.ndarray, valid: np.ndarray, dt: float = 0.1, max_speed: float = 35.0, max_accel: float = 12.0) -> bool:
    speed = np.linalg.norm(states[..., 3:5], axis=-1)
    if np.any(speed[valid.astype(bool)] > max_speed):
        return False
    for a in range(states.shape[1]):
        idx = np.where(valid[:, a].astype(bool))[0]
        if len(idx) <= 2:
            continue
        v = speed[idx, a]
        acc = np.abs(np.diff(v) / max(dt, 1e-6))
        if np.any(acc > max_accel):
            return False
    return True


def check_spawn_from_unknown_if_hidden(metadata: dict, occ_mask: np.ndarray) -> bool:
    if not metadata.get("hidden_emergence", False):
        return True
    return bool(metadata.get("from_unknown_mask", False)) and not bool(metadata.get("hidden_invalid_spawn", False))


def check_no_visible_free_spawn(metadata: dict, occ_mask: np.ndarray) -> bool:
    if not metadata.get("hidden_emergence", False):
        return True
    return not bool(metadata.get("spawn_in_visible_free", False))


def check_lane_or_crosswalk_consistency(metadata: dict) -> bool:
    return not bool(metadata.get("lane_crosswalk_inconsistent", False))


def check_contact_surrogate_metadata(metadata: dict) -> bool:
    if metadata.get("targeted_type") == "contact_impulse_surrogate":
        return bool(metadata.get("contact_surrogate", False))
    return True


def run_plausibility_checks(states: np.ndarray, valid: np.ndarray, metadata: dict, occ_mask: np.ndarray, dt: float = 0.1) -> tuple[bool, list[str]]:
    failures: list[str] = []
    # The ego trajectory has already been replaced by the candidate prefix.
    # Plausibility checks here are meant to validate generated counterfactual
    # non-ego actors, especially hidden/targeted actors, not to reject a hard
    # but dynamically feasible candidate prefix because it diverges from the
    # logged SDC future at the stitch point.
    other_states = states[:, 1:] if states.ndim >= 3 and states.shape[1] > 1 else states[:, :0]
    other_valid = valid[:, 1:] if valid.ndim >= 2 and valid.shape[1] > 1 else valid[:, :0]
    if not check_no_teleportation(other_states, other_valid):
        failures.append("teleportation")
    if not check_speed_accel_bounds(other_states, other_valid, dt=dt):
        failures.append("speed_accel_bounds")
    if not check_spawn_from_unknown_if_hidden(metadata, occ_mask):
        failures.append("hidden_spawn_not_from_unknown")
    if not check_no_visible_free_spawn(metadata, occ_mask):
        failures.append("hidden_spawn_in_visible_free")
    if not check_lane_or_crosswalk_consistency(metadata):
        failures.append("lane_crosswalk_consistency")
    if not check_contact_surrogate_metadata(metadata):
        failures.append("missing_contact_surrogate_metadata")
    return len(failures) == 0, failures
