from __future__ import annotations

import math

import numpy as np

from ocrap.data.schema import CandidatePrefix, F_CTRL, F_EGO, SceneHistory
from ocrap.planning.route_lattice import offset_route_points, project_to_route
from ocrap.planning.utility import nominal_utility
from ocrap.utils.seed import stable_seed

MACROS = ["nominal", "keep", "brake", "yield", "lane_shift", "merge", "pull_over", "stabilize", "perturb_nominal"]


def _macro_params(macro: str, variant: int, ego_speed: float, cfg: dict) -> np.ndarray:
    if macro in ("nominal", "keep"):
        return np.array([ego_speed, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if macro == "brake":
        return np.array([max(0.0, ego_speed - 2.0 - variant), 0.0, -2.0 - 0.5 * variant, 15.0, 0.0], dtype=np.float32)
    if macro == "yield":
        return np.array([max(0.0, ego_speed - 3.0), 0.0, -2.5, 10.0 + 2.0 * variant, 0.0], dtype=np.float32)
    if macro == "lane_shift":
        return np.array([ego_speed, (-1.0 if variant % 2 else 1.0) * 3.5, 0.0, 0.0, 1.0], dtype=np.float32)
    if macro == "merge":
        return np.array([ego_speed + 1.0, (-1.0 if variant % 2 else 1.0) * 2.0, 0.5, 0.0, 1.0], dtype=np.float32)
    if macro == "pull_over":
        return np.array([max(0.0, ego_speed - 3.0), -4.0, -2.0, 20.0, 1.0], dtype=np.float32)
    if macro == "stabilize":
        return np.array([max(0.0, ego_speed - 1.0), 0.0, -0.5, 0.0, 0.0], dtype=np.float32)
    # perturb nominal
    return np.array([max(0.0, ego_speed + (-1) ** variant * 0.8), 0.4 * ((variant % 3) - 1), 0.0, 0.0, 0.0], dtype=np.float32)


def _rollout(history: SceneHistory, macro: str, params: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    sr = float(cfg.get("sample_rate_hz", 10.0))
    dt = 1.0 / sr
    T_p = max(2, int(round(float(cfg.get("prefix_horizon_s", 1.0)) * sr)))
    ego = history.ego_state.astype(np.float32)
    route = history.route
    proj = project_to_route(ego[:2], route)
    v0 = max(0.0, float(ego[6]))
    target_v, d_lat, a_bias, stop_s, topology_required = [float(x) for x in params]
    a_nom = np.clip((target_v - v0) / max((T_p - 1) * dt, 1e-3) + 0.3 * a_bias, float(cfg.get("control_limits", {}).get("a_min", -6.0)), float(cfg.get("control_limits", {}).get("a_max", 3.0)))
    speeds = np.maximum(0.0, v0 + a_nom * np.arange(T_p, dtype=np.float32) * dt)
    ds = np.cumsum(speeds * dt)
    if macro in ("brake", "yield", "pull_over") and stop_s > 1e-3:
        ds = np.minimum(ds, stop_s)
        speeds = np.where(ds >= stop_s - 1e-3, 0.0, speeds)
    s_query = proj.s + ds
    ramp = np.linspace(0.0, 1.0, T_p, dtype=np.float32)
    lat = proj.d + d_lat * (3 * ramp**2 - 2 * ramp**3)
    pts, headings = offset_route_points(route, s_query, lat)
    states = np.zeros((T_p, F_EGO), dtype=np.float32)
    states[:, 0:2] = pts
    states[:, 4] = headings
    states[:, 6] = speeds
    states[:, 2] = speeds * np.cos(headings)
    states[:, 3] = speeds * np.sin(headings)
    states[:, 5] = np.gradient(headings, dt) if T_p > 2 else 0.0
    states[:, 7] = ego[7] if ego.shape[0] > 7 else 4.8
    states[:, 8] = ego[8] if ego.shape[0] > 8 else 2.0
    controls = np.zeros((T_p - 1, F_CTRL), dtype=np.float32)
    if T_p > 1:
        controls[:, 0] = np.diff(speeds) / dt
        controls[:, 1] = np.clip(np.arctan(float(cfg.get("wheelbase_m", 2.8)) * np.gradient(headings, dt)[:-1] / np.maximum(speeds[:-1], 0.5)), -0.7, 0.7)
        controls[1:, 2] = np.diff(controls[:, 0]) / dt
        controls[1:, 3] = np.diff(controls[:, 1]) / dt
    limits = cfg.get("control_limits", {})
    control_ok = bool(
        np.all(controls[:, 0] <= float(limits.get("a_max", 3.0)) + 1e-4)
        and np.all(controls[:, 0] >= float(limits.get("a_min", -6.0)) - 1e-4)
        and np.all(np.abs(controls[:, 1]) <= float(limits.get("delta_max", 0.55)) + 0.15)
        and np.all(np.abs(controls[:, 2]) <= float(limits.get("j_max", 6.0)) + 8.0)
        and np.all(np.abs(controls[:, 3]) <= float(limits.get("steer_rate_max", 0.5)) + 1.0)
    )
    max_dev = float(np.max(np.abs(lat)))
    route_topology_valid = bool(max_dev <= 4.5 or macro == "pull_over")
    wrong_way_hard = bool(np.any(np.cos(headings - proj.heading) < -0.2))
    offroad_hard = bool(max_dev > 6.0)
    finite = bool(np.isfinite(states).all() and np.isfinite(controls).all())
    # Prefix-level collision/contact labels: keep them as labels, not blanket filters.
    prefix_collision = False
    min_other = 99.0
    if history.agent_history.shape[1] > 1:
        cur = history.agent_history[-1, 1:, :2]
        val = history.agent_valid[-1, 1:].astype(bool)
        if val.any():
            d = np.linalg.norm(cur[val][None, :, :] - states[:, None, :2], axis=-1)
            min_other = float(np.min(d))
            prefix_collision = bool(min_other < 1.8)
    hard_violation = float(prefix_collision) * 1.0 + float(offroad_hard) * 2.0 + float(wrong_way_hard) * 2.0
    diagnostics = {
        "control_limits_satisfied": control_ok,
        "route_topology_valid": route_topology_valid,
        "wrong_way_hard": wrong_way_hard,
        "offroad_hard": offroad_hard,
        "prefix_collision": prefix_collision,
        "prefix_contact": bool(min_other < 0.8),
        "max_route_deviation": max_dev,
        "log_divergence": float(np.linalg.norm(states[-1, :2] - history.future_agent_states[min(T_p - 1, len(history.future_agent_states)-1), 0, :2])) if len(history.future_agent_states) else 0.0,
        "topology_required": bool(topology_required),
    }
    feasible = finite and control_ok and route_topology_valid and not wrong_way_hard and not offroad_hard
    diagnostics["feasible"] = feasible
    diagnostics["hard_violation"] = hard_violation
    return states, controls, diagnostics


def generate_candidate_prefixes(history: SceneHistory, cfg: dict) -> list[CandidatePrefix]:
    n = int(cfg.get("num_candidate_prefixes", 24))
    ego_speed = float(history.ego_state[6])
    prefixes: list[CandidatePrefix] = []
    macro_seq: list[tuple[str, int]] = [("nominal", 0)]
    for rep in range(max(1, n)):
        for m in MACROS[1:]:
            macro_seq.append((m, rep))
            if len(macro_seq) >= n:
                break
        if len(macro_seq) >= n:
            break
    while len(macro_seq) < n:
        macro_seq.append(("perturb_nominal", len(macro_seq)))
    for i, (macro, variant) in enumerate(macro_seq[:n]):
        params = _macro_params(macro, variant, ego_speed, cfg)
        states, controls, diag = _rollout(history, macro, params, cfg)
        if i >= len(MACROS) * max(1, variant):
            diag["padded_duplicate"] = i >= len(set(m for m, _ in macro_seq[:i+1]))
        util = nominal_utility(states, controls, diag, cfg)
        harm_proxy = float(diag.get("prefix_contact", False)) + 0.2 * float(diag.get("prefix_collision", False))
        prefixes.append(CandidatePrefix(i, macro, params, states, controls, util, bool(diag["feasible"]), float(diag["hard_violation"]), harm_proxy, diag))
    return prefixes
