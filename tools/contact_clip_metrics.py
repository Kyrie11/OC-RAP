#!/usr/bin/env python3
"""Utilities for recomputing Contact metrics on a clipped real render trace.

This module never alters vehicle states.  It only truncates an existing real
closed-loop render trace at a selected terminal state and recomputes the metric
summary on that visible support so renderer numbers remain consistent with the
visible clip.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

try:
    from .contact_scene_diagnostics import analyze_contact_trace
except ImportError:  # direct script execution / tools on PYTHONPATH
    from contact_scene_diagnostics import analyze_contact_trace


def _metric(frame: dict[str, Any], key: str) -> float | None:
    try:
        x = float((frame.get("metrics") or {}).get(key))
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _flag(frame: dict[str, Any], key: str) -> bool:
    v = _metric(frame, key)
    return bool(v is not None and v > 0.5)


def _finite_series(trace: list[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for frame in trace:
        v = _metric(frame, key)
        if v is not None:
            vals.append(float(v))
    return vals


def _aligned_series(trace: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for frame in trace:
        v = _metric(frame, key)
        out.append(float("nan") if v is None else float(v))
    return out


def _angle_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(b - a), math.cos(b - a))


def _p95_abs(vals: list[float]) -> float:
    finite = [abs(float(x)) for x in vals if math.isfinite(float(x))]
    return float(np.quantile(finite, 0.95)) if finite else float("nan")


def _stable_stop_quality(trace: list[dict[str, Any]], dt: float, *, sustain_steps: int = 5,
                         speed_gate: float = 0.5, yaw_rate_gate: float = 0.25) -> float:
    if not trace:
        return 0.0
    speeds = _aligned_series(trace, "ego_speed_mps")
    yaws = _aligned_series(trace, "ego_yaw_rad")
    overlaps = [_flag(f, "overlap") for f in trace]
    offroads = [_flag(f, "offroad") for f in trace]
    sustain_steps = max(3, int(sustain_steps))
    finite_speeds = [x for x in speeds if math.isfinite(x)]
    if len(finite_speeds) < sustain_steps:
        return 0.0
    initial_window = [x for x in speeds[: min(sustain_steps, len(speeds))] if math.isfinite(x)]
    initial_moving = bool(initial_window) and max(initial_window) > speed_gate
    if not initial_moving or len(trace) < sustain_steps:
        return 0.0

    yaw_rates = [0.0]
    for i in range(1, len(yaws)):
        if math.isfinite(yaws[i - 1]) and math.isfinite(yaws[i]):
            yaw_rates.append(abs(_angle_delta(yaws[i - 1], yaws[i])) / max(dt, 1e-9))
        else:
            yaw_rates.append(float("nan"))

    begin = len(trace) - sustain_steps
    tail_speed = [x for x in speeds[begin:] if math.isfinite(x)]
    if len(tail_speed) < sustain_steps or max(tail_speed) > speed_gate:
        return 0.0
    if any(overlaps[begin:]) or any(offroads[begin:]):
        return 0.0
    tail_yaw = [x for x in yaw_rates[begin:] if math.isfinite(x)]
    if tail_yaw and max(tail_yaw) > yaw_rate_gate:
        return 0.0
    return 1.0


def recompute_contact_metric_summary(trace: list[dict[str, Any]], dt: float,
                                     original: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recompute Contact renderer/selection metrics for ``trace``.

    ``trace`` is interpreted exactly as the runner does: t0..tN are state
    samples, while durations use left-endpoint intervals t0..t(N-1).
    """
    if len(trace) < 2:
        raise ValueError("clipped Contact trace must contain at least two states")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("metric dt must be positive")

    summary = dict(original or {})
    clear_all = _aligned_series(trace, "min_clearance_m")
    pen_all = _aligned_series(trace, "penetration_depth_m")
    speed_all = _aligned_series(trace, "ego_speed_mps")
    yaw_all = _aligned_series(trace, "ego_yaw_rad")
    overlap = [_flag(f, "overlap") for f in trace]
    offroad = [_flag(f, "offroad") for f in trace]

    clear_finite = [x for x in clear_all if math.isfinite(x)]
    pen_finite = [x for x in pen_all if math.isfinite(x)]
    if clear_finite:
        min_idx = min((i for i, x in enumerate(clear_all) if math.isfinite(x)), key=lambda i: clear_all[i])
        last_idx = max(i for i, x in enumerate(clear_all) if math.isfinite(x))
        summary["min_clearance_m_min"] = float(min(clear_finite))
        summary["min_clearance_m_p05"] = float(np.quantile(clear_finite, 0.05))
        summary["terminal_clearance_m"] = float(clear_all[last_idx])
        summary["clearance_recovery_gain_m"] = float(clear_all[last_idx] - clear_all[min_idx])
        summary["time_to_min_clearance_steps"] = float(min_idx)
        summary["time_to_min_clearance_s"] = float(min_idx * dt)
    else:
        for k in ("min_clearance_m_min", "min_clearance_m_p05", "terminal_clearance_m", "clearance_recovery_gain_m"):
            summary[k] = float("nan")

    summary["overlap_any"] = float(any(overlap))
    summary["offroad_any"] = float(any(offroad))
    summary["overlap_duration_steps"] = float(sum(overlap[:-1]))
    summary["overlap_duration_s"] = float(sum(overlap[:-1]) * dt)
    summary["penetration_any"] = float(any(x > 1e-9 for x in pen_finite)) if pen_finite else 0.0
    summary["penetration_duration_steps"] = float(sum(math.isfinite(x) and x > 1e-9 for x in pen_all[:-1]))
    summary["penetration_duration_s"] = float(summary["penetration_duration_steps"] * dt)
    summary["penetration_depth_m_max"] = float(max(pen_finite)) if pen_finite else 0.0
    summary["penetration_depth_m_p95"] = float(np.quantile(pen_finite, 0.95)) if pen_finite else 0.0
    # Interval support matches the closed-loop runner: state t_N is terminal and
    # does not contribute an additional dt interval.  Recompute this explicitly
    # so a clipped/synthesized displayed trajectory never inherits stale AUC.
    summary["penetration_depth_auc_m_s"] = float(
        sum(max(0.0, x) for x in pen_all[:-1] if math.isfinite(x)) * dt
    )
    summary["new_stable_stop_quality_event"] = _stable_stop_quality(trace, dt)

    # Useful stability terms for re-scoring the visible clip.
    yaw_rates: list[float] = []
    for i in range(1, len(yaw_all)):
        if math.isfinite(yaw_all[i - 1]) and math.isfinite(yaw_all[i]):
            yaw_rates.append(_angle_delta(yaw_all[i - 1], yaw_all[i]) / dt)
    summary["yaw_rate_p95"] = _p95_abs(yaw_rates)
    accelerations: list[float] = []
    for i in range(1, len(speed_all)):
        if math.isfinite(speed_all[i - 1]) and math.isfinite(speed_all[i]):
            accelerations.append((speed_all[i] - speed_all[i - 1]) / dt)
    jerks = [(accelerations[i] - accelerations[i - 1]) / dt for i in range(1, len(accelerations))]
    summary["jerk_p95"] = _p95_abs(jerks)

    # Exact-a0 Contact traces begin in contact.  Still derive this generically.
    first_contact = next((i for i, x in enumerate(overlap) if x), None)
    post_eligible = first_contact is not None and first_contact < len(trace) - 1
    summary["observed_contact_event"] = float(first_contact is not None)
    summary["post_contact_metric_eligible"] = float(post_eligible)
    summary["first_contact_step"] = float(first_contact) if first_contact is not None else float("nan")
    summary["contact_anchor_step"] = float(first_contact) if post_eligible else float("nan")
    if post_eligible:
        a = int(first_contact)
        post_clear = [clear_all[i] for i in range(a, len(clear_all)) if math.isfinite(clear_all[i])]
        post_interval_clear = [clear_all[i] for i in range(a, len(clear_all) - 1) if math.isfinite(clear_all[i])]
        post_overlap = overlap[a:-1]
        summary["post_contact_overlap_duration_s"] = float(sum(post_overlap) * dt)
        summary["post_contact_overlap_rate"] = float(sum(post_overlap) / len(post_overlap)) if post_overlap else float("nan")
        summary["post_contact_terminal_clearance_m"] = float(post_clear[-1]) if post_clear else float("nan")
        summary["post_contact_clearance_m_max"] = float(max(post_clear)) if post_clear else float("nan")
        summary["post_contact_clearance_m_mean"] = float(np.mean(post_clear)) if post_clear else float("nan")
        summary["post_contact_clearance_gain_m"] = float(post_clear[-1] - post_clear[0]) if post_clear else float("nan")
        summary["post_contact_free_space_auc_m_s"] = float(sum(max(0.0, x) for x in post_interval_clear) * dt)
        duration = len(post_interval_clear) * dt
        summary["post_contact_free_space_auc_normalized_m"] = (
            float(summary["post_contact_free_space_auc_m_s"] / duration) if duration > 0 else float("nan")
        )

        overlap_starts = [i for i, flag in enumerate(overlap) if flag and (i == 0 or not overlap[i - 1])]
        recontact_starts = [i for i in overlap_starts[1:] if i > a]
        summary["recontact_episode_count"] = float(len(recontact_starts))
        summary["recontact_event"] = float(bool(recontact_starts))

        escape_idx = None
        escape_clearance = 0.5
        sustain_steps = 3
        for begin in range(a, len(trace) - sustain_steps + 1):
            end = begin + sustain_steps
            cwin = clear_all[begin:end]
            if all(math.isfinite(c) and c >= escape_clearance for c in cwin) and not any(overlap[begin:end]):
                escape_idx = begin
                break
        summary["post_contact_escape_event"] = float(escape_idx is not None)
        summary["time_to_post_contact_escape_s"] = float((escape_idx - a) * dt) if escape_idx is not None else float("nan")
    else:
        summary["recontact_episode_count"] = float("nan")
        summary["recontact_event"] = float("nan")

    summary["num_metric_steps"] = int(len(trace) - 1)
    summary["clearance_metric_full_coverage"] = float(len(clear_finite) == len(trace))
    summary["overlap_metric_full_coverage"] = 1.0
    summary["offroad_metric_full_coverage"] = 1.0

    # Additional *diagnostic* fields for critical qualitative target mining.
    # These are recomputed from the same displayed oriented boxes and therefore
    # remain consistent when a target/reference trajectory is clipped.  They do
    # not replace the historical Contact metrics used in the paper tables.
    diag = analyze_contact_trace(trace, dt=dt)
    summary["secondary_collision_event"] = float(bool(diag.get("secondary_collision_event")))
    summary["post_separation_secondary_collision_event"] = float(
        bool(diag.get("post_separation_secondary_collision_event"))
    )
    summary["same_partner_recontact_event"] = float(bool(diag.get("same_partner_recontact_event")))
    summary["distinct_collision_partner_count"] = float(diag.get("distinct_collision_partner_count") or 0)
    summary["secondary_collision_partner_count"] = float(diag.get("secondary_collision_partner_count") or 0)
    summary["nearby_agents_peak_8m"] = float(diag.get("nearby_agents_peak_8m") or 0)
    summary["nearby_agents_peak_12m"] = float(diag.get("nearby_agents_peak_12m") or 0)
    summary["nearby_agents_peak_20m"] = float(diag.get("nearby_agents_peak_20m") or 0)
    summary["crowded_fraction"] = float(diag.get("crowded_fraction") or 0.0)
    summary["multi_actor_conflict_peak"] = float(diag.get("multi_actor_conflict_peak") or 0)
    summary["multi_actor_conflict_fraction"] = float(diag.get("multi_actor_conflict_fraction") or 0.0)
    summary["diagnostic_offroad_fraction"] = float(diag.get("offroad_fraction") or 0.0)
    return summary


def clip_scene(scene: dict[str, Any], clip_duration_s: float, dt: float) -> dict[str, Any]:
    """Return a deep-copied scene truncated to a real state prefix."""
    trace = list(scene.get("render_trace") or [])
    if not trace:
        raise ValueError("scene has no render_trace")
    terminal = min(len(trace) - 1, max(1, int(round(float(clip_duration_s) / float(dt)))))
    clipped_trace = copy.deepcopy(trace[: terminal + 1])
    out = copy.deepcopy(scene)
    out["render_trace"] = clipped_trace
    out["metric_summary"] = recompute_contact_metric_summary(
        clipped_trace, dt, original=(scene.get("metric_summary") or {})
    )
    out["num_metric_steps"] = terminal
    out["num_decisions"] = min(int(scene.get("num_decisions") or terminal), terminal)
    # Recompute intervention rate from visible action intervals when possible.
    action_frames = clipped_trace[1:]
    if action_frames and all(f.get("selected_candidate_index") is not None for f in action_frames):
        out["intervention_rate"] = float(sum(int(f.get("selected_candidate_index") or 0) != 0 for f in action_frames) / len(action_frames))
    return out
