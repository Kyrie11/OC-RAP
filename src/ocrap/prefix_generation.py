from __future__ import annotations

import math

import numpy as np

from .geometry import agent_state_to_box, box_signed_clearance, fast_box_signed_clearance, ego_state_to_box, nearest_polyline_distance, smooth_step, wrap_angle
from .schema import CandidatePrefix, SceneHistory


MACRO_NAMES = [
    "nominal",
    "keep",
    "brake",
    "yield",
    "lane_shift",
    "merge",
    "pull_over",
    "stabilize",
    "perturb_nominal",
]
MACRO_ID = {m: i for i, m in enumerate(MACRO_NAMES)}


def _ego_from_agent_state(agent_state: np.ndarray, yaw_rate: float = 0.0) -> np.ndarray:
    return np.array(
        [
            agent_state[0],
            agent_state[1],
            agent_state[3],
            agent_state[4],
            agent_state[5],
            yaw_rate,
            math.hypot(agent_state[3], agent_state[4]),
            agent_state[6],
            agent_state[7],
        ],
        dtype=np.float32,
    )


def integrate_prefix(
    ego0: np.ndarray,
    horizon_steps: int,
    dt: float,
    accel: float = 0.0,
    lateral_offset: float = 0.0,
    target_speed: float | None = None,
    heading_delta: float = 0.0,
    wheelbase: float = 2.8,
    control_limits: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    control_limits = control_limits or {}
    a_max = float(control_limits.get("a_max", 3.0))
    a_min = float(control_limits.get("a_min", -6.0))
    delta_max = float(control_limits.get("delta_max", 0.55))
    jerk_max = float(control_limits.get("j_max", 6.0))
    steer_rate_max = float(control_limits.get("steer_rate_max", 0.5))
    v_max = float(control_limits.get("v_max", 20.0))

    states = np.zeros((horizon_steps, 9), dtype=np.float32)
    controls = np.zeros((max(horizon_steps - 1, 1), 4), dtype=np.float32)
    x, y, vx, vy, psi, yaw_rate, v, length, width = [float(vv) for vv in ego0[:9]]
    acc_prev = 0.0
    steer_prev = 0.0
    for t in range(horizon_steps):
        s = t / max(horizon_steps - 1, 1)
        lat = float(lateral_offset) * float(smooth_step(np.array(s)))
        psi_cmd = psi + float(heading_delta) * float(smooth_step(np.array(s)))
        if target_speed is not None:
            desired_acc = (float(target_speed) - v) / max((horizon_steps - t) * dt, dt)
            acc_t = float(np.clip(desired_acc, a_min, a_max))
        else:
            acc_t = float(np.clip(accel, a_min, a_max))
        # smooth lateral maneuver: approximate curvature/yaw with lateral offset over horizon
        yaw_rate_cmd = (heading_delta / max(horizon_steps * dt, dt)) + 2.0 * lateral_offset / max((horizon_steps * dt) ** 2, dt)
        steer_t = float(np.clip(math.atan2(wheelbase * yaw_rate_cmd, max(v, 0.5)), -delta_max, delta_max))
        if t > 0:
            jerk = float(np.clip((acc_t - acc_prev) / dt, -jerk_max, jerk_max))
            steer_rate = float(np.clip((steer_t - steer_prev) / dt, -steer_rate_max, steer_rate_max))
            controls[t - 1] = [acc_t, steer_t, jerk, steer_rate]
        # lateral offset applied in local normal direction as a planned target, not as teleportation; we blend state output.
        states[t] = [x - math.sin(psi) * lat, y + math.cos(psi) * lat, vx, vy, psi_cmd, yaw_rate_cmd, v, length, width]
        if t < horizon_steps - 1:
            v = float(np.clip(v + acc_t * dt, 0.0, v_max))
            psi = float(wrap_angle(psi + v / wheelbase * math.tan(steer_t) * dt))
            x += v * math.cos(psi) * dt
            y += v * math.sin(psi) * dt
            vx, vy = v * math.cos(psi), v * math.sin(psi)
            acc_prev = acc_t
            steer_prev = steer_t
    return states, controls[: max(horizon_steps - 1, 0)]


def prefix_collision_proxy(prefix_states: np.ndarray, history: SceneHistory) -> tuple[float, float]:
    # Compare prefix endpoint and current visible boxes. Critical prefixes are retained but labeled.
    if history.agent_history.shape[1] <= 1:
        return 0.0, 0.0
    current_agents = history.agent_history[-1, 1:]
    current_valid = history.agent_valid[-1, 1:].astype(bool)
    min_clear = float("inf")
    for st in prefix_states:
        ego_box = ego_state_to_box(st)
        for ag, ok in zip(current_agents, current_valid):
            if not ok:
                continue
            clear = fast_box_signed_clearance(ego_box, agent_state_to_box(ag))
            min_clear = min(min_clear, clear)
    hard_violation = float(max(0.0, -min_clear)) if np.isfinite(min_clear) else 0.0
    harm_proxy = float(max(0.0, 2.0 - min_clear)) if np.isfinite(min_clear) else 0.0
    return hard_violation, harm_proxy


def nominal_utility(prefix_states: np.ndarray, prefix_controls: np.ndarray, history: SceneHistory, cfg: dict) -> float:
    weights = cfg.get("utility_weights", {})
    progress_w = float(weights.get("progress", 1.0))
    comfort_w = float(weights.get("comfort", 0.05))
    route_w = float(weights.get("route", 0.5))
    logdiv_w = float(weights.get("logdiv", 0.05))
    offroad_w = float(weights.get("offroad", 5.0))
    wrongway_w = float(weights.get("wrongway", 5.0))
    start = prefix_states[0, :2]
    end = prefix_states[-1, :2]
    progress = float(np.linalg.norm(end - start))
    comfort = float(np.mean(prefix_controls[:, 0] ** 2) + 0.5 * np.mean(prefix_controls[:, 2] ** 2) + np.mean(prefix_controls[:, 3] ** 2)) if prefix_controls.size else 0.0
    route_dev = nearest_polyline_distance(end, history.route[None, :, :] if history.route.ndim == 2 else history.route, None)
    if not np.isfinite(route_dev):
        route_dev = 0.0
    logdiv = 0.0
    if history.future_agent_states.shape[0] >= prefix_states.shape[0]:
        ego_logged = np.stack([_ego_from_agent_state(s) for s in history.future_agent_states[: prefix_states.shape[0], 0]], axis=0)
        logdiv = float(np.mean(np.linalg.norm(prefix_states[:, :2] - ego_logged[:, :2], axis=-1)))
    offroad = float(max(0.0, route_dev - float(cfg.get("route_width", 3.5))))
    wrongway = 0.0
    if history.route.size > 0 and len(history.route) > 1:
        route_dir = history.route[min(len(history.route) - 1, 1), :2] - history.route[0, :2]
        move_dir = end - start
        if np.linalg.norm(route_dir) > 1e-3 and np.linalg.norm(move_dir) > 1e-3:
            wrongway = float((route_dir @ move_dir) < 0.0)
    cost = -progress_w * progress + comfort_w * comfort + route_w * route_dev + logdiv_w * logdiv + offroad_w * offroad + wrongway_w * wrongway
    return -float(cost)


def generate_candidate_prefixes(history: SceneHistory, cfg: dict) -> list[CandidatePrefix]:
    dt = 1.0 / float(cfg.get("sample_rate_hz", 10))
    T_p = int(round(float(cfg.get("prefix_horizon_s", 1.0)) * float(cfg.get("sample_rate_hz", 10))))
    T_p = max(T_p, 2)
    ego0 = history.ego_state.astype(np.float32)
    speed_limit = float(history.metadata.get("speed_limit", cfg.get("speed_limit_default", 13.4)))
    control_limits = cfg.get("control_limits", {}).copy()
    control_limits["v_max"] = min(speed_limit + 2.0, float(control_limits.get("v_max", 20.0)))
    prefixes: list[CandidatePrefix] = []

    def add(macro: str, params: list[float], **kwargs) -> None:
        states, controls = integrate_prefix(ego0, T_p, dt, control_limits=control_limits, **kwargs)
        hard, harm = prefix_collision_proxy(states, history)
        feasible = bool(np.all(np.isfinite(states)) and np.all(np.isfinite(controls)))
        util = nominal_utility(states, controls, history, cfg)
        prefixes.append(
            CandidatePrefix(
                macro_id=MACRO_ID[macro],
                macro_name=macro,
                params=np.asarray(params + [0.0] * max(0, int(cfg.get("prefix_param_dim", 4)) - len(params)), dtype=np.float32)[: int(cfg.get("prefix_param_dim", 4))],
                prefix_states=states,
                prefix_controls=controls,
                utility=util,
                feasible=feasible,
                hard_violation=hard,
                harm_proxy=harm,
                diagnostics={"prefix_collision": hard > 0.0, "prefix_contact": hard > 0.0, "source_macro": macro},
            )
        )

    # Nominal: prefer logged future if available; otherwise keep-speed bicycle rollout.
    if history.future_agent_states.shape[0] >= T_p and np.all(history.future_agent_valid[:T_p, 0]):
        logged = np.stack([_ego_from_agent_state(s) for s in history.future_agent_states[:T_p, 0]], axis=0)
        controls = np.zeros((T_p - 1, 4), dtype=np.float32)
        speeds = logged[:, 6]
        headings = logged[:, 4]
        controls[:, 0] = np.diff(speeds) / dt
        controls[:, 1] = np.clip(np.diff(headings, prepend=headings[0])[1:] / dt * 0.2, -0.55, 0.55)
        controls[:, 2] = np.diff(controls[:, 0], prepend=controls[0, 0]) / dt
        controls[:, 3] = np.diff(controls[:, 1], prepend=controls[0, 1]) / dt
        hard, harm = prefix_collision_proxy(logged, history)
        prefixes.append(CandidatePrefix(MACRO_ID["nominal"], "nominal", np.zeros(int(cfg.get("prefix_param_dim", 4)), dtype=np.float32), logged, controls, nominal_utility(logged, controls, history, cfg), True, hard, harm, {"source_macro": "nominal", "logged": True, "prefix_collision": hard > 0.0}))
    else:
        add("nominal", [float(ego0[6])], target_speed=float(ego0[6]))

    v0 = float(ego0[6])
    for dv in (-1.0, 0.0, 1.0):
        add("keep", [v0 + dv], target_speed=float(np.clip(v0 + dv, 0.0, speed_limit + 2.0)))
    for dec in (-1.5, -3.0, -5.0):
        add("brake", [dec], accel=dec)
    for dec in (-1.5, -3.0):
        for gap in (6.0, 10.0):
            add("yield", [dec, gap], accel=dec, target_speed=max(0.0, v0 + dec))
    for dy in (-1.0, -0.5, 0.5, 1.0):
        add("lane_shift", [dy, 1.0], lateral_offset=dy)
    for dy in (-1.75, 1.75, -3.5, 3.5):
        add("merge", [dy, v0], lateral_offset=dy, target_speed=v0)
    shoulder_available = bool(history.metadata.get("shoulder_available", True))
    if shoulder_available:
        add("pull_over", [-2.5, 0.0], lateral_offset=-2.5, target_speed=0.0)
        add("pull_over", [2.5, 0.0], lateral_offset=2.5, target_speed=0.0)
    if history.metadata.get("near_contact", False) or history.metadata.get("post_contact", False):
        add("stabilize", [-2.0, 1.0], accel=-2.0, heading_delta=-0.1)
        add("stabilize", [-3.0, 1.5], accel=-3.0, heading_delta=0.1)
    for da in (-1.0, 0.0, 1.0):
        for dy in (-0.3, 0.3):
            add("perturb_nominal", [da, dy], accel=da, lateral_offset=dy)

    # Diversity + utility truncation while preserving nominal as index 0.
    target_n = int(cfg.get("num_candidate_prefixes", 24))
    unique: list[CandidatePrefix] = []
    seen = set()
    for p in prefixes:
        key = (p.macro_name, tuple(np.round(p.params, 2)))
        if key not in seen:
            unique.append(p)
            seen.add(key)
    nominal = unique[0]
    rest = sorted(unique[1:], key=lambda p: (p.hard_violation > 0.0, -p.utility, p.macro_name))
    out = [nominal] + rest[: max(0, target_n - 1)]
    while len(out) < target_n:
        base = rest[(len(out) - 1) % max(1, len(rest))] if rest else nominal
        clone = CandidatePrefix(base.macro_id, base.macro_name, base.params.copy(), base.prefix_states.copy(), base.prefix_controls.copy(), base.utility - 1e-3 * len(out), base.feasible, base.hard_violation, base.harm_proxy, {**base.diagnostics, "padded_duplicate": True})
        out.append(clone)
    return out[:target_n]
