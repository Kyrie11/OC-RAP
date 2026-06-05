from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np

from .schema import CandidatePrefix, CounterfactualFuture, SceneHistory


def _copy_future(history: SceneHistory, total_steps: int) -> tuple[np.ndarray, np.ndarray]:
    A = history.agent_history.shape[1]
    F = history.agent_history.shape[2]
    states = np.zeros((total_steps, A, F), dtype=np.float32)
    valid = np.zeros((total_steps, A), dtype=bool)
    avail = min(total_steps, history.future_agent_states.shape[0])
    if avail > 0:
        states[:avail] = history.future_agent_states[:avail]
        valid[:avail] = history.future_agent_valid[:avail].astype(bool)
    if avail < total_steps:
        if avail > 0:
            last = states[avail - 1].copy()
            last_valid = valid[avail - 1].copy()
        else:
            last = history.agent_history[-1].copy()
            last_valid = history.agent_valid[-1].astype(bool).copy()
        for t in range(avail, total_steps):
            states[t] = last
            states[t, :, 0] += last[:, 3] * 0.1 * (t - avail + 1)
            states[t, :, 1] += last[:, 4] * 0.1 * (t - avail + 1)
            valid[t] = last_valid
    return states, valid


def _inject_ego_prefix(states: np.ndarray, valid: np.ndarray, prefix: CandidatePrefix) -> None:
    T = min(prefix.prefix_states.shape[0], states.shape[0])
    for t in range(T):
        e = prefix.prefix_states[t]
        states[t, 0, 0] = e[0]
        states[t, 0, 1] = e[1]
        states[t, 0, 2] = 0.0
        states[t, 0, 3] = e[2]
        states[t, 0, 4] = e[3]
        states[t, 0, 5] = e[4]
        states[t, 0, 6] = e[7]
        states[t, 0, 7] = e[8]
        states[t, 0, 8] = 1.5
        states[t, 0, 9] = 1.0
        valid[t, 0] = True


def replay_future(history: SceneHistory, prefix: CandidatePrefix, total_steps: int, prior: float = 0.25) -> CounterfactualFuture:
    states, valid = _copy_future(history, total_steps)
    _inject_ego_prefix(states, valid, prefix)
    return CounterfactualFuture(0, "replay", prior, states, valid, {"anchor_logged": True})


def reactive_future(history: SceneHistory, prefix: CandidatePrefix, total_steps: int, idx: int, prior: float) -> CounterfactualFuture:
    states, valid = _copy_future(history, total_steps)
    _inject_ego_prefix(states, valid, prefix)
    rng = np.random.default_rng(abs(hash((history.scene_id, history.time_index, idx))) % (2**32))
    T_p = prefix.prefix_states.shape[0]
    for a in range(1, states.shape[1]):
        if not valid[0, a]:
            continue
        # IDM-like response to ego prefix for vehicle-like agents.
        typ = states[0, a, 9]
        reaction_gain = 0.2 + 0.15 * idx
        for t in range(1, total_steps):
            prev = states[t - 1, a].copy()
            ego = states[min(t, T_p - 1), 0]
            rel = prev[:2] - ego[:2]
            dist = np.linalg.norm(rel)
            v = math.hypot(prev[3], prev[4])
            heading = prev[5]
            if typ in (1, 2, 3) and dist < 25.0:
                closing = max(0.0, (ego[3] - prev[3]) * math.cos(heading) + (ego[4] - prev[4]) * math.sin(heading))
                decel = reaction_gain * closing / max(dist / 10.0, 1.0)
                v = max(0.0, v - decel * 0.1)
            else:
                v = max(0.0, v + rng.normal(0.0, 0.05))
            states[t, a, 3] = v * math.cos(heading)
            states[t, a, 4] = v * math.sin(heading)
            states[t, a, 0] = prev[0] + states[t, a, 3] * 0.1
            states[t, a, 1] = prev[1] + states[t, a, 4] * 0.1
            states[t, a, 5] = heading
            valid[t, a] = valid[t, a] or valid[t - 1, a]
    return CounterfactualFuture(idx, "reactive", prior, states, valid, {"reactive_policy": "idm_like", "reactive_variant": idx})


def _find_free_agent_slot(states: np.ndarray, valid: np.ndarray) -> int:
    unused = np.where(~valid.any(axis=0))[0]
    if len(unused) > 0:
        return int(unused[0])
    return int(states.shape[1] - 1)


def _insert_agent(states: np.ndarray, valid: np.ndarray, slot: int, start_t: int, x: float, y: float, speed: float, heading: float, obj_type: float, length: float, width: float) -> None:
    for t in range(start_t, states.shape[0]):
        dt = (t - start_t) * 0.1
        states[t, slot, 0] = x + speed * math.cos(heading) * dt
        states[t, slot, 1] = y + speed * math.sin(heading) * dt
        states[t, slot, 2] = 0.0
        states[t, slot, 3] = speed * math.cos(heading)
        states[t, slot, 4] = speed * math.sin(heading)
        states[t, slot, 5] = heading
        states[t, slot, 6] = length
        states[t, slot, 7] = width
        states[t, slot, 8] = 1.5
        states[t, slot, 9] = obj_type
        valid[t, slot] = True


def targeted_perturbation(history: SceneHistory, prefix: CandidatePrefix, total_steps: int, kind: str, idx: int, prior: float) -> CounterfactualFuture:
    states, valid = _copy_future(history, total_steps)
    _inject_ego_prefix(states, valid, prefix)
    rng = np.random.default_rng(abs(hash((history.scene_id, history.time_index, kind, idx))) % (2**32))
    meta: dict[str, object] = {"targeted_type": kind, "recovery_relevant": True}
    slot = _find_free_agent_slot(states, valid)
    T_p = prefix.prefix_states.shape[0]
    ego_end = prefix.prefix_states[-1]

    if kind == "hidden_vehicle_yields":
        _insert_agent(states, valid, slot, max(1, T_p - 2), ego_end[0] + 12.0, ego_end[1] - 3.0, max(0.0, ego_end[6] - 3.0), ego_end[4], 1.0, 4.8, 2.0)
        meta.update({"hidden_emergence": True, "from_unknown_mask": True, "hidden_intent": "yield"})
    elif kind == "hidden_vehicle_accelerates":
        _insert_agent(states, valid, slot, max(1, T_p - 2), ego_end[0] + 12.0, ego_end[1] - 3.0, ego_end[6] + 3.0, ego_end[4], 1.0, 4.8, 2.0)
        for t in range(max(1, T_p - 2), total_steps):
            states[t, slot, 3] += 2.0 * math.cos(ego_end[4])
            states[t, slot, 4] += 2.0 * math.sin(ego_end[4])
        meta.update({"hidden_emergence": True, "from_unknown_mask": True, "hidden_intent": "accelerate"})
    elif kind == "occluded_pedestrian_emerges":
        heading = ego_end[4] + math.pi / 2.0
        _insert_agent(states, valid, slot, max(1, T_p - 1), ego_end[0] + 8.0, ego_end[1] - 5.0, 1.6, heading, 2.0, 0.7, 0.7)
        meta.update({"hidden_emergence": True, "from_unknown_mask": True})
    elif kind == "adjacent_vehicle_cut_in":
        _insert_agent(states, valid, slot, 0, ego_end[0] + 6.0, ego_end[1] + 3.5, max(ego_end[6], 3.0), ego_end[4], 1.0, 4.8, 2.0)
        for t in range(T_p, total_steps):
            frac = min(1.0, (t - T_p) / 15.0)
            states[t, slot, 1] -= 3.0 * frac
        meta.update({"lateral_escape_blocked": True})
    elif kind == "rejoin_corridor_blocked":
        _insert_agent(states, valid, slot, max(0, T_p - 1), ego_end[0] + 15.0, ego_end[1], max(0.5, ego_end[6] * 0.4), ego_end[4], 1.0, 4.8, 2.0)
        meta.update({"route_blocked": True, "rejoin_corridor_available": False})
    elif kind == "low_friction_braking":
        meta.update({"friction_factor": float(rng.uniform(0.3, 0.8)), "control_envelope_uncertain": True})
    elif kind == "control_delay_noise":
        meta.update({"control_delay_s": float(rng.choice([0.1, 0.2, 0.3])), "actuation_noise_std": 0.15, "control_envelope_uncertain": True})
    elif kind == "contact_impulse_surrogate":
        meta.update({"contact_surrogate": True, "yaw_rate_impulse": float(rng.choice([-0.5, 0.5])), "lateral_velocity_impulse": float(rng.choice([-1.5, 1.5]))})
    elif kind == "secondary_collision_approach":
        _insert_agent(states, valid, slot, max(0, T_p - 1), ego_end[0] + 25.0, ego_end[1], max(8.0, ego_end[6] + 5.0), ego_end[4] + math.pi, 1.0, 4.8, 2.0)
        meta.update({"secondary_agent": True, "secondary_threat": True})
    else:
        raise ValueError(f"Unknown targeted perturbation kind {kind}")
    return CounterfactualFuture(idx, "targeted", prior, states, valid, meta)


def generate_counterfactual_futures(history: SceneHistory, prefix: CandidatePrefix, cfg: dict) -> list[CounterfactualFuture]:
    total_steps = int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * float(cfg.get("sample_rate_hz", 10))))
    futures: list[CounterfactualFuture] = []
    replay_prior = float(cfg.get("future_priors", {}).get("replay", 0.25))
    reactive_total = float(cfg.get("future_priors", {}).get("reactive", 0.35))
    targeted_total = float(cfg.get("future_priors", {}).get("targeted", 0.40))
    futures.append(replay_future(history, prefix, total_steps, replay_prior))
    n_reactive = int(cfg.get("num_reactive_futures", 4))
    for i in range(n_reactive):
        futures.append(reactive_future(history, prefix, total_steps, i + 1, reactive_total / max(n_reactive, 1)))
    kinds = [
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
    n_targeted = int(cfg.get("num_targeted_futures", 8))
    for i in range(n_targeted):
        kind = kinds[i % len(kinds)]
        futures.append(targeted_perturbation(history, prefix, total_steps, kind, len(futures), targeted_total / max(n_targeted, 1)))
    priors = np.asarray([f.prior for f in futures], dtype=np.float64)
    priors = priors / max(float(priors.sum()), 1e-8)
    for f, p in zip(futures, priors):
        f.prior = float(p)
    return futures
