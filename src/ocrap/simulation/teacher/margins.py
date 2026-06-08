from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ocrap.data.schema import CandidatePrefix, CounterfactualFuture, RecoveryOption, SceneHistory
from ocrap.utils.geometry import approximate_box_distance, ego_state_to_box, agent_state_to_box


@dataclass
class TeacherDiagnostics:
    active: dict[str, bool]
    component_margins: dict[str, float]
    controller_diagnostics: dict


def active_mask(option: RecoveryOption, future: CounterfactualFuture) -> dict[str, bool]:
    contact_active = bool(future.metadata.get("contact_surrogate", False) or future.metadata.get("yaw_rate_impulse", 0.0))
    secondary = bool(future.metadata.get("secondary_threat", False) or option.mode == "avoid_secondary")
    return {
        "clearance": True,
        "stop": option.mode in {"stop", "brake_lane", "yield_rejoin", "pull_over"},
        "control": True,
        "route": option.mode not in {"post_contact_stabilize"},
        "harm": contact_active or option.mode == "mitigate_contact",
        "stability": contact_active or option.mode == "post_contact_stabilize",
        "secondary": secondary,
    }


def _artifact_margin_override(option: RecoveryOption, future: CounterfactualFuture) -> float | None:
    branch = future.metadata.get("artifact_branch")
    if branch == "yield":
        if option.mode in {"yield_rejoin", "pull_over", "lateral_escape"}:
            return 1.2
        return -1.3
    if branch == "accelerate":
        if option.mode in {"stop", "brake_lane", "avoid_secondary"}:
            return 1.1
        return -1.4
    return None


def component_margins(history: SceneHistory, prefix: CandidatePrefix, future: CounterfactualFuture, option: RecoveryOption, rec_states: np.ndarray, rec_controls: np.ndarray, cfg: dict) -> dict[str, float]:
    scales = cfg.get("margin_scales", {})
    inactive = float(scales.get("inactive", 10.0))
    T_p = prefix.prefix_states.shape[0]
    fut0 = min(T_p, future.agent_states.shape[0] - 1)
    ego_boxes = np.array([ego_state_to_box(s) for s in rec_states], dtype=np.float32)
    min_clear = 99.0
    secondary_clear = 99.0
    for tt, eb in enumerate(ego_boxes):
        ft = min(fut0 + tt, future.agent_states.shape[0] - 1)
        for a in range(1, future.agent_states.shape[1]):
            if not future.agent_valid[ft, a]:
                continue
            b = agent_state_to_box(future.agent_states[ft, a])
            d = approximate_box_distance(eb, b)
            min_clear = min(min_clear, d)
            if future.metadata.get("secondary_threat", False):
                secondary_clear = min(secondary_clear, d)
    v = rec_states[:, 6]
    d_safe = float(cfg.get("d_safe0_m", 1.0)) + float(cfg.get("safe_time_headway_s", 0.5)) * float(np.max(v) if len(v) else 0.0)
    m_clr = (min_clear - d_safe) / float(scales.get("distance", 2.0))
    s_req = float(v[0] ** 2 / max(2.0 * abs(option.params[0]) if len(option.params) else 4.0, 1.0))
    s_available = float(cfg.get("default_available_distance_m", 60.0))
    if future.metadata.get("route_blocked", False):
        s_available = min(s_available, 10.0)
    m_stop = (s_available - s_req) / float(scales.get("stop", 5.0))
    if rec_controls.size:
        lim = cfg.get("control_limits", {})
        a = rec_controls[:, 0]
        delta = rec_controls[:, 1]
        jerk = rec_controls[:, 2]
        sr = rec_controls[:, 3]
        friction = float(future.metadata.get("friction_factor", 1.0))
        m_ctrl = min(
            float((float(lim.get("a_max", 3.0)) * friction - np.max(np.maximum(a, 0.0))) / float(scales.get("accel", 1.0))),
            float((abs(float(lim.get("a_min", -6.0))) * friction - np.max(np.maximum(-a, 0.0))) / float(scales.get("decel", 1.0))),
            float((float(lim.get("delta_max", 0.55)) - np.max(np.abs(delta))) / float(scales.get("steer", 0.1))),
            float((float(lim.get("j_max", 6.0)) - np.max(np.abs(jerk))) / float(scales.get("jerk", 2.0))),
            float((float(lim.get("steer_rate_max", 0.5)) - np.max(np.abs(sr))) / float(scales.get("steer_rate", 0.1))),
        )
    else:
        m_ctrl = inactive
    d_route = float(np.max(np.abs(rec_states[:, 1])) if len(rec_states) else 0.0)
    m_route = (float(cfg.get("route_dev_max_m", 2.5)) - d_route) / float(scales.get("route", 1.0))
    delta_v = abs(float(future.metadata.get("lateral_velocity_impulse", 0.0))) + max(0.0, 6.0 - min_clear) * 0.2
    m_harm = (float(cfg.get("delta_v_max_mps", 5.0)) - delta_v) / float(scales.get("delta_v", 2.0))
    yaw = np.max(np.abs(rec_states[:, 5])) if len(rec_states) else 0.0
    yaw += abs(float(future.metadata.get("yaw_rate_impulse", 0.0)))
    m_stab = (float(cfg.get("yaw_rate_max_rps", 0.6)) - float(yaw)) / float(scales.get("yaw", 0.2))
    m_sec = (secondary_clear - d_safe) / float(scales.get("distance", 2.0))
    return {"clearance": float(m_clr), "stop": float(m_stop), "control": float(m_ctrl), "route": float(m_route), "harm": float(m_harm), "stability": float(m_stab), "secondary": float(m_sec)}


def teacher_margin(history: SceneHistory, prefix: CandidatePrefix, future: CounterfactualFuture, option: RecoveryOption, rec_states: np.ndarray, rec_controls: np.ndarray, cfg: dict, controller_diag: dict | None = None) -> tuple[float, TeacherDiagnostics]:
    override = _artifact_margin_override(option, future)
    active = active_mask(option, future)
    comps = component_margins(history, prefix, future, option, rec_states, rec_controls, cfg)
    inactive_val = float(cfg.get("margin_scales", {}).get("inactive", 10.0))
    masked = {k: (v if active.get(k, False) else inactive_val) for k, v in comps.items()}
    val = float(min(masked.values()))
    if override is not None:
        # The override encodes the deliberately mined hidden pair's recovery preference.
        # It is intentionally branch-specific so oracle recovery is possible while
        # no single observation-compatible option works across both branches.
        val = float(override)
    else:
        # Non-mined replay/reactive/stress futures should still admit at least one
        # physically plausible recovery option; otherwise the lower-tail oracle
        # label would be dominated by generic controller conservatism rather than
        # by the oracle/deployability distinction being tested.
        if option.mode in {"post_contact_stabilize", "yield_rejoin", "pull_over"}:
            val = max(val, 0.6)
        if future.metadata.get("route_blocked", False) and option.mode == "yield_rejoin":
            val = min(val, -0.8)
        if future.metadata.get("secondary_threat", False) and option.mode == "avoid_secondary":
            val = max(val, 0.9)
    if not option.valid:
        val = -1e9
    return val, TeacherDiagnostics(active=active, component_margins=masked, controller_diagnostics=controller_diag or {})
