from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ocrap.ocrap.geometry import agent_state_to_box, box_signed_clearance, fast_box_signed_clearance, ego_state_to_box, nearest_polyline_distance, wrap_angle
from ocrap.ocrap.schema import CandidatePrefix, CounterfactualFuture, RecoveryOption, SceneHistory


@dataclass
class TeacherDiagnostics:
    component_margins: dict[str, float]
    active: dict[str, bool]
    min_clearance: float
    max_delta_v: float
    max_yaw_rate: float
    stable_stop: bool
    route_rejoin_success: bool
    secondary_collision: bool


@dataclass
class TeacherRollout:
    margin: float
    ego_states: np.ndarray
    controls: np.ndarray
    diagnostics: TeacherDiagnostics


def _apply_contact_impulse(state: np.ndarray, future: CounterfactualFuture) -> np.ndarray:
    s = state.copy().astype(np.float32)
    if future.metadata.get("contact_surrogate", False):
        s[5] += float(future.metadata.get("yaw_rate_impulse", 0.0))
        lateral = float(future.metadata.get("lateral_velocity_impulse", 0.0))
        s[2] += -math.sin(float(s[4])) * lateral
        s[3] += math.cos(float(s[4])) * lateral
        s[6] = float(np.linalg.norm(s[2:4]))
    return s


def _controller(option: RecoveryOption, state: np.ndarray, step: int, cfg: dict) -> tuple[float, float]:
    mode = option.mode
    p = option.params
    dt = 1.0 / float(cfg.get("sample_rate_hz", 10))
    limits = cfg.get("control_limits", {})
    a_min = float(limits.get("a_min", -6.0))
    a_max = float(limits.get("a_max", 3.0))
    delta_max = float(limits.get("delta_max", 0.55))
    T_rec = int(round(float(cfg.get("recovery_horizon_s", 4.0)) * float(cfg.get("sample_rate_hz", 10))))
    v = float(state[6])
    if mode == "stop":
        a = float(p[0])
        steer = 0.0
    elif mode == "brake_lane":
        a = float(p[0]) if step * dt <= float(p[1]) else 0.0
        steer = -0.15 * float(state[5])
    elif mode == "lateral_escape":
        dy, vtar, Tlat = [float(x) for x in p[:3]]
        a = float(np.clip((vtar - v) / max(T_rec * dt, dt), a_min, a_max))
        steer = float(np.clip(2.0 * dy / max(Tlat**2, 0.2) / max(v, 1.0), -delta_max, delta_max))
    elif mode == "yield_rejoin":
        a_yield, gap, T_rejoin = [float(x) for x in p[:3]]
        a = a_yield if step * dt < T_rejoin * 0.5 else min(1.0, max(0.0, gap / 20.0))
        steer = -0.10 * float(state[5])
    elif mode == "pull_over":
        dy, s_shoulder, v_stop = [float(x) for x in p[:3]]
        a = float(np.clip((v_stop - v) / max(T_rec * dt, dt), a_min, a_max))
        steer = float(np.clip(2.0 * dy / max((T_rec * dt) ** 2, 0.2) / max(v, 1.0), -delta_max, delta_max))
    elif mode == "mitigate_contact":
        a_dec, delta_psi, v_impact = [float(x) for x in p[:3]]
        a = a_dec if v > v_impact else -0.5
        steer = float(np.clip(delta_psi, -delta_max, delta_max))
    elif mode == "post_contact_stabilize":
        kpsi, kr, adecay = [float(x) for x in p[:3]]
        a = adecay
        steer = float(np.clip(-kpsi * float(wrap_angle(state[4])) - kr * float(state[5]), -delta_max, delta_max))
    elif mode == "avoid_secondary":
        dy, a_dec, s_clear = [float(x) for x in p[:3]]
        a = a_dec
        steer = float(np.clip(1.5 * dy / max(v, 1.0), -delta_max, delta_max))
    else:
        raise ValueError(f"Unknown recovery mode {mode}")
    return float(np.clip(a, a_min, a_max)), float(np.clip(steer, -delta_max, delta_max))


def _step_bicycle(state: np.ndarray, accel: float, steer: float, dt: float, cfg: dict, friction_factor: float = 1.0) -> np.ndarray:
    limits = cfg.get("control_limits", {})
    wheelbase = float(cfg.get("wheelbase_m", 2.8))
    v_max = float(limits.get("v_max", 20.0))
    a_min = float(limits.get("a_min", -6.0)) * float(friction_factor)
    a_max = float(limits.get("a_max", 3.0)) * max(0.5, float(friction_factor))
    a = float(np.clip(accel, a_min, a_max))
    x, y, vx, vy, psi, yaw_rate, v, length, width = [float(z) for z in state[:9]]
    v = float(np.clip(v + a * dt, 0.0, v_max))
    yaw_rate = float(v / wheelbase * math.tan(steer))
    psi = float(wrap_angle(psi + yaw_rate * dt))
    x = x + v * math.cos(psi) * dt
    y = y + v * math.sin(psi) * dt
    vx = v * math.cos(psi)
    vy = v * math.sin(psi)
    return np.array([x, y, vx, vy, psi, yaw_rate, v, length, width], dtype=np.float32)


def _available_stop_distance(ego: np.ndarray, agent_states: np.ndarray, agent_valid: np.ndarray, cfg: dict) -> float:
    heading = float(ego[4])
    fwd = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
    lat = np.array([-math.sin(heading), math.cos(heading)], dtype=np.float64)
    best = float(cfg.get("default_available_distance_m", 60.0))
    for s, ok in zip(agent_states, agent_valid.astype(bool)):
        if not ok:
            continue
        b = agent_state_to_box(s)
        rel = b[:2] - ego[:2]
        lon = float(rel @ fwd)
        lateral = abs(float(rel @ lat))
        if 0.0 < lon < best and lateral < max(2.5, ego[8] + b[6]):
            best = lon - 0.5 * b[5]
    return max(0.0, best)


def _relative_delta_v(ego: np.ndarray, other: np.ndarray) -> float:
    rel_p = other[:2] - ego[:2]
    n = rel_p / max(np.linalg.norm(rel_p), 1e-6)
    ego_v = ego[2:4]
    obj_v = other[2:4]
    return float(max(0.0, (ego_v - obj_v) @ n))


def teacher_margin(history: SceneHistory, prefix: CandidatePrefix, future: CounterfactualFuture, option: RecoveryOption, cfg: dict) -> TeacherRollout:
    dt = 1.0 / float(cfg.get("sample_rate_hz", 10))
    T_rec = int(round(float(cfg.get("recovery_horizon_s", 4.0)) * float(cfg.get("sample_rate_hz", 10))))
    T_p = prefix.prefix_states.shape[0]
    limits = cfg.get("control_limits", {})
    scales = cfg.get("margin_scales", {})
    inactive = float(scales.get("inactive", 10.0))
    ego = _apply_contact_impulse(prefix.prefix_states[-1], future)
    friction = float(future.metadata.get("friction_factor", 1.0))
    delay_steps = int(round(float(future.metadata.get("control_delay_s", 0.0)) / dt))
    noise_std = float(future.metadata.get("actuation_noise_std", 0.0))
    rng = np.random.default_rng(abs(hash((history.scene_id, history.time_index, future.future_id, option.option_id))) % (2**32))

    ego_states = np.zeros((T_rec, 9), dtype=np.float32)
    controls = np.zeros((T_rec, 4), dtype=np.float32)
    min_clear = float("inf")
    max_delta_v = 0.0
    max_abs_yaw = abs(float(ego[5]))
    secondary_collision = False
    prev_a = 0.0
    prev_steer = 0.0
    delayed_control = (0.0, 0.0)

    clearance_slacks = []
    stop_slacks = []
    control_slacks = []
    route_slacks = []
    harm_slacks = []
    stab_slacks = []
    sec_slacks = []

    for tau in range(T_rec):
        a, steer = _controller(option, ego, tau, cfg)
        if noise_std > 0:
            a += float(rng.normal(0.0, noise_std))
            steer += float(rng.normal(0.0, noise_std * 0.1))
        if tau < delay_steps:
            a, steer = delayed_control
        else:
            delayed_control = (a, steer)
        jerk = (a - prev_a) / dt
        steer_rate = (steer - prev_steer) / dt
        controls[tau] = [a, steer, jerk, steer_rate]
        ego = _step_bicycle(ego, a, steer, dt, cfg, friction_factor=friction)
        ego_states[tau] = ego
        idx = min(T_p + tau, future.agent_states.shape[0] - 1)
        agents = future.agent_states[idx, 1:]
        valids = future.agent_valid[idx, 1:]
        ego_box = ego_state_to_box(ego)
        step_min = float("inf")
        step_delta_v = 0.0
        for ag, ok in zip(agents, valids.astype(bool)):
            if not ok:
                continue
            box = agent_state_to_box(ag)
            clear = fast_box_signed_clearance(ego_box, box)
            step_min = min(step_min, clear)
            if clear < 0.5:
                step_delta_v = max(step_delta_v, _relative_delta_v(ego_box, box))
        min_clear = min(min_clear, step_min)
        max_delta_v = max(max_delta_v, step_delta_v)
        max_abs_yaw = max(max_abs_yaw, abs(float(ego[5])))
        v = float(ego[6])
        d_safe = float(cfg.get("d_safe0_m", 1.0)) + float(cfg.get("safe_time_headway_s", 0.5)) * v
        clearance_slacks.append((step_min - d_safe) / float(scales.get("distance", 2.0)))
        s_avail = _available_stop_distance(ego, future.agent_states[idx, 1:], future.agent_valid[idx, 1:], cfg)
        a_comfort = abs(float(cfg.get("comfort_brake_mps2", -3.0)))
        s_req = v**2 / max(2.0 * a_comfort, 1e-6) + v * float(cfg.get("control_delay_s_default", 0.2))
        stop_slacks.append((s_avail - s_req) / float(scales.get("stop", 5.0)))
        ctrl = min(
            (float(limits.get("a_max", 3.0)) - abs(max(a, 0.0))) / float(scales.get("accel", 1.0)),
            (abs(float(limits.get("a_min", -6.0))) - abs(min(a, 0.0))) / float(scales.get("decel", 1.0)),
            (float(limits.get("delta_max", 0.55)) - abs(steer)) / float(scales.get("steer", 0.1)),
            (float(limits.get("j_max", 6.0)) - abs(jerk)) / float(scales.get("jerk", 2.0)),
            (float(limits.get("steer_rate_max", 0.5)) - abs(steer_rate)) / float(scales.get("steer_rate", 0.1)),
        )
        control_slacks.append(ctrl)
        route_dev = nearest_polyline_distance(ego[:2], history.route[None, :, :] if history.route.ndim == 2 else history.route, None)
        if not np.isfinite(route_dev):
            route_dev = 0.0
        route_slacks.append((float(cfg.get("route_dev_max_m", 2.5)) - route_dev) / float(scales.get("route", 1.0)))
        harm_slacks.append((float(cfg.get("delta_v_max_mps", 5.0)) - step_delta_v) / float(scales.get("delta_v", 2.0)))
        stab_slacks.append((float(cfg.get("yaw_rate_max_rps", 0.6)) - abs(float(ego[5]))) / float(scales.get("yaw", 0.2)))
        sec_slacks.append((step_min - (d_safe + 1.0)) / float(scales.get("distance", 2.0)))
        if future.metadata.get("secondary_threat", False) and step_min < 0.0:
            secondary_collision = True
        prev_a, prev_steer = a, steer

    contact_active = bool(prefix.hard_violation > 0.0 or future.metadata.get("contact_surrogate", False) or min_clear < 0.0)
    stability_active = bool(contact_active or future.metadata.get("contact_surrogate", False) or option.mode == "post_contact_stabilize")
    secondary_active = bool(future.metadata.get("secondary_threat", False) or option.mode == "avoid_secondary")
    active = {
        "clearance": True,
        "stop": True,
        "control": True,
        "route": True,
        "harm": contact_active,
        "stability": stability_active,
        "secondary": secondary_active,
    }
    comp = {
        "clearance": float(np.min(clearance_slacks)) if clearance_slacks else inactive,
        "stop": float(np.min(stop_slacks)) if stop_slacks else inactive,
        "control": float(np.min(control_slacks)) if control_slacks else inactive,
        "route": float(np.min(route_slacks)) if route_slacks else inactive,
        "harm": float(np.min(harm_slacks)) if active["harm"] else inactive,
        "stability": float(np.min(stab_slacks)) if active["stability"] else inactive,
        "secondary": float(np.min(sec_slacks)) if active["secondary"] else inactive,
    }
    margin = min(comp[k] if active[k] else inactive for k in comp.keys())
    stable_stop = bool(ego_states[-1, 6] < 0.5 and max_abs_yaw <= float(cfg.get("yaw_rate_max_rps", 0.6)))
    route_rejoin_success = bool(comp["route"] >= 0.0)
    diag = TeacherDiagnostics(comp, active, float(min_clear), float(max_delta_v), float(max_abs_yaw), stable_stop, route_rejoin_success, secondary_collision)
    return TeacherRollout(float(margin), ego_states, controls, diag)


def compute_future_option_margins(history: SceneHistory, prefix: CandidatePrefix, futures: list[CounterfactualFuture], options: list[RecoveryOption], cfg: dict) -> tuple[np.ndarray, list[list[TeacherDiagnostics]]]:
    M = np.zeros((len(futures), len(options)), dtype=np.float32)
    diags: list[list[TeacherDiagnostics]] = []
    for j, f in enumerate(futures):
        row = []
        for l, g in enumerate(options):
            if not g.valid:
                M[j, l] = -1e6
                row.append(TeacherDiagnostics({}, {}, float("inf"), 0.0, 0.0, False, False, False))
            else:
                rollout = teacher_margin(history, prefix, f, g, cfg)
                M[j, l] = rollout.margin
                row.append(rollout.diagnostics)
        diags.append(row)
    return M, diags
