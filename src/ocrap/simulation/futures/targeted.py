from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from ocrap.data.schema import CandidatePrefix, CounterfactualFuture, SceneHistory
from ocrap.utils.seed import stable_seed

from .plausibility import run_plausibility_checks
from .replay import copy_future, inject_ego_prefix

TARGETED_KINDS = [
    "hidden_vehicle_yields",
    "hidden_vehicle_accelerates",
    "occluded_pedestrian_emerges",
    "adjacent_vehicle_cut_in",
    "rejoin_corridor_blocked",
    "low_friction_braking",
    "control_delay_noise",
    "contact_impulse_surrogate",
    "secondary_collision_approach",
]


def find_free_agent_slot(states: np.ndarray, valid: np.ndarray) -> int:
    unused = np.where(~valid.any(axis=0))[0]
    if len(unused):
        return int(unused[0])
    return int(states.shape[1] - 1)


def _cell_to_xy(iy: int, ix: int, occ_mask: np.ndarray, cfg: dict) -> tuple[float, float]:
    radius = float(cfg.get("local_radius_m", 80.0))
    res = float(cfg.get("bev_resolution_m", 1.0))
    x = (ix + 0.5) * res - radius
    y = (iy + 0.5) * res - radius
    return float(x), float(y)


def sample_unknown_spawn(history: SceneHistory, cfg: dict, rng: np.random.Generator) -> tuple[float, float, dict] | None:
    mask = history.occ_mask
    if mask.size == 0 or mask.shape[0] < 6:
        return None
    unknown = mask[2] > 0.5
    drivable = mask[5] > 0.5
    visible_free = mask[0] > 0.5
    occupied = mask[1] > 0.5
    route = mask[4] > 0.5
    legal = unknown & drivable & ~visible_free & ~occupied
    route_legal = legal & route
    cells = np.argwhere(route_legal if route_legal.any() else legal)
    if len(cells) == 0:
        return None
    # Prefer cells ahead and near the route, but sample stably.
    xy = np.array([_cell_to_xy(int(iy), int(ix), mask, cfg) for iy, ix in cells], dtype=np.float32)
    score = np.abs(xy[:, 1]) + 0.02 * np.maximum(-xy[:, 0], 0.0) - 0.01 * xy[:, 0]
    order = np.argsort(score, kind="mergesort")[: max(1, min(50, len(score)))]
    k = int(rng.choice(order))
    iy, ix = cells[k]
    x, y = _cell_to_xy(int(iy), int(ix), mask, cfg)
    meta = {"hidden_spawn_xy": [x, y], "hidden_spawn_cell": [int(iy), int(ix)], "from_unknown_mask": True, "hidden_invalid_spawn": False, "spawn_in_visible_free": False}
    return x, y, meta


def insert_agent(states: np.ndarray, valid: np.ndarray, slot: int, start_t: int, x: float, y: float, speed: float, heading: float, obj_type: float, length: float, width: float, accel: float = 0.0) -> None:
    start_t = max(0, min(int(start_t), states.shape[0] - 1))
    for t in range(start_t, states.shape[0]):
        dt = 0.1 * (t - start_t)
        v = max(0.0, speed + accel * dt)
        dist = speed * dt + 0.5 * accel * dt * dt
        states[t, slot, 0] = x + dist * math.cos(heading)
        states[t, slot, 1] = y + dist * math.sin(heading)
        states[t, slot, 2] = 0.0
        states[t, slot, 3] = v * math.cos(heading)
        states[t, slot, 4] = v * math.sin(heading)
        states[t, slot, 5] = accel * math.cos(heading)
        states[t, slot, 6] = accel * math.sin(heading)
        states[t, slot, 7] = heading
        states[t, slot, 8] = math.sin(heading)
        states[t, slot, 9] = math.cos(heading)
        states[t, slot, 10] = length
        states[t, slot, 11] = width
        states[t, slot, 12] = 1.5 if obj_type == 1 else 1.0
        states[t, slot, 13] = obj_type
        states[t, slot, 14] = 1.0
        states[t, slot, 15] = 0.9
        valid[t, slot] = True


def targeted_perturbation(history: SceneHistory, prefix: CandidatePrefix, total_steps: int, kind: str, idx: int, prior: float, cfg: dict) -> CounterfactualFuture | None:
    states, valid = copy_future(history, total_steps)
    inject_ego_prefix(states, valid, prefix)
    rng = np.random.default_rng(stable_seed("targeted", history.scene_id, history.time_index, prefix.macro_id, kind, idx))
    T_p = prefix.prefix_states.shape[0]
    hidden_start = min(total_steps - 1, T_p + int(cfg.get("hidden_emergence_delay_steps", 2)))
    ego_end = prefix.prefix_states[-1]
    slot = find_free_agent_slot(states, valid)
    meta: dict[str, object] = {
        "targeted_type": kind,
        "recovery_relevant": True,
        "prefix_steps": int(T_p),
        "hidden_start_step": int(hidden_start),
        "hidden_emergence_delay_steps": int(cfg.get("hidden_emergence_delay_steps", 2)),
        "hidden_start_ge_prefix_plus_delay": bool(hidden_start >= T_p + int(cfg.get("hidden_emergence_delay_steps", 2))),
    }

    hidden_kind = kind in {"hidden_vehicle_yields", "hidden_vehicle_accelerates", "occluded_pedestrian_emerges"}
    spawn = sample_unknown_spawn(history, cfg, rng) if hidden_kind else None
    if hidden_kind and spawn is None:
        meta.update({"hidden_emergence": False, "from_unknown_mask": False, "hidden_invalid_spawn": False, "degraded_from_hidden": True, "skip_reason": "no_legal_unknown_spawn"})
        # Degrade to a visible targeted cut-in rather than fabricating hidden provenance.
        kind = "adjacent_vehicle_cut_in"
    if kind == "hidden_vehicle_yields":
        x, y, smeta = spawn  # type: ignore[misc]
        heading = float(ego_end[4])
        insert_agent(states, valid, slot, hidden_start, x, y, max(0.0, float(ego_end[6]) - 2.0), heading, 1.0, 4.8, 2.0, accel=-0.3)
        meta.update(smeta)  # type: ignore[arg-type]
        meta.update({"hidden_emergence": True, "hidden_intent": "yield", "artifact_branch": "yield"})
    elif kind == "hidden_vehicle_accelerates":
        x, y, smeta = spawn  # type: ignore[misc]
        heading = float(ego_end[4])
        insert_agent(states, valid, slot, hidden_start, x, y, max(2.0, float(ego_end[6]) + 1.0), heading, 1.0, 4.8, 2.0, accel=1.2)
        meta.update(smeta)  # type: ignore[arg-type]
        meta.update({"hidden_emergence": True, "hidden_intent": "accelerate", "artifact_branch": "accelerate"})
    elif kind == "occluded_pedestrian_emerges":
        x, y, smeta = spawn if spawn is not None else (float(ego_end[0] + 9.0), float(ego_end[1] + 4.5), {"from_unknown_mask": False, "hidden_invalid_spawn": True})
        insert_agent(states, valid, slot, hidden_start, x, y, 1.5, float(ego_end[4] + math.pi / 2), 2.0, 0.7, 0.7)
        meta.update(smeta)  # type: ignore[arg-type]
        meta.update({"hidden_emergence": True, "hidden_intent": "pedestrian_emerge"})
    elif kind == "adjacent_vehicle_cut_in":
        insert_agent(states, valid, slot, max(0, T_p - 2), float(ego_end[0] + 7.0), float(ego_end[1] + 3.5), max(float(ego_end[6]), 3.0), float(ego_end[4]), 1.0, 4.8, 2.0)
        for t in range(max(0, T_p - 1), total_steps):
            frac = min(1.0, (t - T_p + 1) / 15.0)
            states[t, slot, 1] -= 3.0 * frac
        meta.update({"lateral_escape_blocked": True, "hidden_emergence": False})
    elif kind == "rejoin_corridor_blocked":
        insert_agent(states, valid, slot, max(0, T_p - 1), float(ego_end[0] + 12.0), float(ego_end[1]), max(1.0, float(ego_end[6]) * 0.4), float(ego_end[4]), 1.0, 4.8, 2.0)
        meta.update({"route_blocked": True, "rejoin_corridor_available": False})
    elif kind == "low_friction_braking":
        meta.update({"friction_factor": float(rng.uniform(0.35, 0.75)), "control_envelope_uncertain": True})
    elif kind == "control_delay_noise":
        meta.update({"control_delay_s": float(rng.choice([0.1, 0.2, 0.3])), "actuation_noise_std": 0.15, "control_envelope_uncertain": True})
    elif kind == "contact_impulse_surrogate":
        meta.update({"contact_surrogate": True, "yaw_rate_impulse": float(rng.choice([-0.55, 0.55])), "lateral_velocity_impulse": float(rng.choice([-1.5, 1.5]))})
    elif kind == "secondary_collision_approach":
        insert_agent(states, valid, slot, max(0, T_p - 1), float(ego_end[0] + 24.0), float(ego_end[1]), max(8.0, float(ego_end[6]) + 5.0), float(ego_end[4] + math.pi), 1.0, 4.8, 2.0)
        meta.update({"secondary_agent": True, "secondary_threat": True})
    else:
        raise ValueError(f"Unknown targeted perturbation kind {kind}")
    ok, failures = run_plausibility_checks(states, valid, meta, history.occ_mask, dt=1.0 / float(cfg.get("sample_rate_hz", 10.0)))
    meta["plausibility_passed"] = ok
    meta["plausibility_failures"] = failures
    if not ok and hidden_kind:
        # Do not silently keep invalid hidden futures.
        meta["hidden_invalid_spawn"] = True if "hidden_spawn_not_from_unknown" in failures or "hidden_spawn_in_visible_free" in failures else bool(meta.get("hidden_invalid_spawn", False))
        return None
    return CounterfactualFuture(idx, "targeted", prior, states, valid, meta)


def mine_artifact_futures(history: SceneHistory, prefix: CandidatePrefix, total_steps: int, start_id: int, prior_each: float, cfg: dict) -> list[CounterfactualFuture]:
    out: list[CounterfactualFuture] = []
    for kind in ["hidden_vehicle_yields", "hidden_vehicle_accelerates"]:
        fut = targeted_perturbation(history, prefix, total_steps, kind, start_id + len(out), prior_each, cfg)
        if fut is not None and fut.metadata.get("hidden_emergence") and fut.metadata.get("from_unknown_mask"):
            fut.metadata["artifact_mined"] = True
            fut.metadata["artifact_pair_key"] = f"{history.scene_id}:{history.time_index}:{prefix.macro_id}"
            out.append(fut)
    return out if len(out) == 2 else []
