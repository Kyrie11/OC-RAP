#!/usr/bin/env python3
"""Trace-aware final selection for reviewer-facing regime visualizations.

The metric-only selector deliberately runs first on the locked population.  This
second pass sees only the over-selected qualitative candidate pool and its
selective render traces.  It does *not* change the quantitative cohort or any
reported population metric.

Goals:
  * Safe: never display an OC-RAP overlap/off-road tail and, by default, never
    hide one by trimming the requested clip. The selected OC-RAP trajectory
    must also stay geometrically compatible with a vehicle lane when lane
    centerline evidence is available in the render context.
  * Near-Contact: prioritize scenes where OC-RAP remains clean while a large
    fraction of external baselines collide or enter a low-margin state. Among
    similarly strong consensus-failure cases, prefer denser local traffic, and
    reject "safety by leaving the roadway/lane corridor" behavior.
  * Contact: require a controlled post-contact recovery -- sustained
    separation without visible off-road behavior and, when map evidence is
    available, a return to a plausible vehicle-lane corridor. Clearance is a
    bounded secondary ranking term rather than a license to escape arbitrarily
    far from the road. Prefer scenes where several audited external baselines
    fail the same trace-level recovery test.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

METHODS = {
    "safe": ["gameformer_lite", "plantf", "pluto", "pdm_closed", "pdm_hybrid", "idm", "diffusion_planner"],
    "near": ["marc_lite", "racp_lite", "robust_scenario_mpc", "predictive_safety_filter", "dr_cvar_safety_filter", "conformal_predictive_safety_filter", "flow_planner", "plan_r1", "betopnet"],
    "contact": ["postimpact_mpc_lite", "post_crash_braking", "postimpact_motion_tvlqr", "post_collision_restoration", "compensatory_postimpact_mpc", "robust_postimpact_control"],
}

# Waymo/WOMD roadgraph point types 1 and 2 are vehicle-lane centerlines
# (freeway and surface-street lanes). Type 3 is a bike lane and is deliberately
# not accepted as a valid vehicle recovery corridor.
DEFAULT_VEHICLE_LANE_TYPES = (1, 2)


def _scene_key(scene: dict[str, Any], env: dict[str, Any]) -> str:
    key = str(scene.get("target_key") or env.get("resume_key") or "")
    if key.startswith("target:"):
        key = key[len("target:"):]
    if key:
        return key
    sid = str(scene.get("scene_id") or "")
    ti = scene.get("target_time_index")
    return f"{sid}:t{ti}" if sid and ti is not None else sid


def _load_journal(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"missing trace journal: {path}")
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            env = json.loads(line)
            scene = env.get("scene", env)
            key = _scene_key(scene, env)
            if key in out:
                raise SystemExit(f"duplicate trace target {key}: {path}")
            out[key] = scene
    return out


def _metric(frame: dict[str, Any], key: str) -> float | None:
    try:
        x = float((frame.get("metrics") or {}).get(key))
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _flag(frame: dict[str, Any], key: str) -> bool:
    return (_metric(frame, key) or 0.0) > 0.5


def _visible_frames(trace: list[dict[str, Any]], clip_s: float, dt_s: float) -> list[dict[str, Any]]:
    # render_trace stores the initial state plus one state after each simulator
    # step.  The renderer now pins the final video frame to the clip terminal
    # state, so the quality gate must include that endpoint as well.  For a
    # 2.5 s clip at 0.1 s/step this means 26 states: t=0.0 ... 2.5 s.
    n_steps = max(0, int(math.floor(clip_s / dt_s + 1.0e-9)))
    return trace[: min(n_steps + 1, len(trace))]


def _first_violation(trace: list[dict[str, Any]], clip_s: float, dt_s: float) -> tuple[int | None, str | None]:
    for i, frame in enumerate(_visible_frames(trace, clip_s, dt_s)):
        if _flag(frame, "overlap"):
            return i, "overlap"
        if _flag(frame, "offroad"):
            return i, "offroad"
    return None, None


def _min_metric(trace: list[dict[str, Any]], clip_s: float, dt_s: float, key: str) -> float | None:
    vals = [_metric(f, key) for f in _visible_frames(trace, clip_s, dt_s)]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def _any_flag(trace: list[dict[str, Any]], clip_s: float, dt_s: float, key: str) -> bool:
    return any(_flag(f, key) for f in _visible_frames(trace, clip_s, dt_s))


def _density_stats(trace: list[dict[str, Any]], clip_s: float, dt_s: float, radius_m: float) -> dict[str, float]:
    counts: list[int] = []
    r2 = radius_m * radius_m
    for frame in _visible_frames(trace, clip_s, dt_s):
        agents = frame.get("agents") or []
        sdc = next((a for a in agents if a.get("is_sdc")), None)
        if sdc is None:
            continue
        try:
            sx, sy = float(sdc["x"]), float(sdc["y"])
        except Exception:
            continue
        count = 0
        for a in agents:
            if a is sdc or a.get("is_sdc"):
                continue
            try:
                dx, dy = float(a["x"]) - sx, float(a["y"]) - sy
            except Exception:
                continue
            if dx * dx + dy * dy <= r2:
                count += 1
        counts.append(count)
    if not counts:
        return {"median": 0.0, "p75": 0.0, "max": 0.0}
    ordered = sorted(counts)
    p75 = ordered[min(len(ordered) - 1, int(math.ceil(0.75 * len(ordered))) - 1)]
    return {"median": float(statistics.median(counts)), "p75": float(p75), "max": float(max(counts))}


def _sdc_agent(frame: dict[str, Any]) -> dict[str, Any] | None:
    return next((a for a in (frame.get("agents") or []) if a.get("is_sdc")), None)


def _wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _axial_heading_error_rad(a: float, b: float) -> float:
    """Smallest orientation error when polyline direction is treated as axial.

    WOMD roadgraph points preserve feature ordering, but the qualitative gate
    only needs to establish that the vehicle is aligned with a legal lane
    corridor; it must not infer a wrong-way label from uncertain polyline
    orientation.  Therefore 0 and pi are equivalent here.
    """
    d = abs(_wrap_pi(float(a) - float(b)))
    return min(d, abs(math.pi - d))


def _point_segment_distance_heading(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> tuple[float, float] | None:
    vx, vy = bx - ax, by - ay
    vv = vx * vx + vy * vy
    if vv <= 1.0e-10:
        return None
    wx, wy = px - ax, py - ay
    u = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    qx, qy = ax + u * vx, ay + u * vy
    return math.hypot(px - qx, py - qy), math.atan2(vy, vx)


def _lane_segments(scene: dict[str, Any], lane_types: tuple[int, ...]) -> list[tuple[float, float, float, float]]:
    context = scene.get("render_context") or {}
    allowed = set(int(x) for x in lane_types)
    out: list[tuple[float, float, float, float]] = []
    for polyline in context.get("roadgraph_polylines") or []:
        try:
            type_id = int(polyline.get("type", -1))
        except Exception:
            continue
        if type_id not in allowed:
            continue
        pts: list[tuple[float, float]] = []
        for p in polyline.get("xy") or []:
            try:
                x, y = float(p[0]), float(p[1])
            except Exception:
                continue
            if math.isfinite(x) and math.isfinite(y):
                pts.append((x, y))
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            if math.hypot(bx - ax, by - ay) > 1.0e-3:
                out.append((ax, ay, bx, by))
    return out


def _percentile(values: list[float], q: float) -> float | None:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    q = max(0.0, min(1.0, float(q)))
    idx = min(len(vals) - 1, max(0, int(math.ceil(q * len(vals))) - 1))
    return vals[idx]


def _lane_realism(
    scene: dict[str, Any],
    clip_s: float,
    dt_s: float,
    *,
    lane_types: tuple[int, ...],
    terminal_max_m: float,
    p90_max_m: float,
    offcenter_threshold_m: float,
    offcenter_fraction_max: float,
    heading_terminal_max_deg: float,
    heading_p90_max_deg: float,
    heading_speed_gate_mps: float,
    min_evidence_fraction: float,
) -> dict[str, Any]:
    """Measure whether the displayed SDC stays in a plausible lane corridor.

    This deliberately uses only the static roadgraph captured with the render
    trace.  It does not use the logged future SDC trajectory, route oracle, or
    any post-hoc hidden future.  The gate is consequently suitable for a
    reviewer-facing qualitative selection contract.
    """
    trace = list(scene.get("render_trace") or [])
    frames = _visible_frames(trace, clip_s, dt_s)
    segments = _lane_segments(scene, lane_types)
    if not frames or not segments:
        return {
            "evidence_available": False,
            "accepted": True,
            "reason": "lane_evidence_unavailable",
            "lane_segment_count": len(segments),
            "frame_evidence_fraction": 0.0,
        }

    distances: list[float] = []
    heading_errors_deg: list[float] = []
    terminal_distance: float | None = None
    terminal_heading_error_deg: float | None = None
    terminal_speed: float | None = None
    evidence_frames = 0
    offcenter_count = 0
    moving_heading_frames = 0
    misaligned_count = 0

    for frame_index, frame in enumerate(frames):
        sdc = _sdc_agent(frame)
        if sdc is None:
            continue
        try:
            px, py = float(sdc["x"]), float(sdc["y"])
            yaw = float(sdc.get("yaw", 0.0))
        except Exception:
            continue
        rows: list[tuple[float, float]] = []
        for ax, ay, bx, by in segments:
            row = _point_segment_distance_heading(px, py, ax, ay, bx, by)
            if row is not None:
                rows.append(row)
        if not rows:
            continue
        evidence_frames += 1
        min_dist = min(d for d, _ in rows)
        distances.append(min_dist)
        offcenter_count += int(min_dist > offcenter_threshold_m)

        # At intersections several lane centerlines may be spatially close.
        # Use the best aligned segment among those effectively tied in distance
        # instead of allowing a crossing lane to create a spurious 90deg error.
        near = [(d, h) for d, h in rows if d <= min_dist + 2.0]
        speed = _metric(frame, "ego_speed_mps")
        if speed is None:
            speed = _metric(frame, "speed_mps")
        if speed is not None and speed >= heading_speed_gate_mps:
            err = min(_axial_heading_error_rad(yaw, h) for _, h in near)
            err_deg = math.degrees(err)
            heading_errors_deg.append(err_deg)
            moving_heading_frames += 1
            misaligned_count += int(err_deg > heading_p90_max_deg)
        else:
            err_deg = None

        if frame_index == len(frames) - 1:
            terminal_distance = min_dist
            terminal_speed = speed
            if err_deg is not None:
                terminal_heading_error_deg = err_deg

    coverage = evidence_frames / max(len(frames), 1)
    if evidence_frames == 0 or coverage < min_evidence_fraction:
        return {
            "evidence_available": False,
            "accepted": True,
            "reason": "insufficient_lane_evidence",
            "lane_segment_count": len(segments),
            "frame_evidence_fraction": coverage,
        }

    p90_dist = _percentile(distances, 0.90)
    p90_heading = _percentile(heading_errors_deg, 0.90)
    offcenter_fraction = offcenter_count / max(evidence_frames, 1)
    misaligned_fraction = misaligned_count / max(moving_heading_frames, 1) if moving_heading_frames else 0.0
    first_distance = distances[0] if distances else None
    peak_distance = max(distances) if distances else None
    peak_to_terminal_improvement = (
        float(peak_distance - terminal_distance)
        if peak_distance is not None and terminal_distance is not None else None
    )
    # A short-horizon post-contact clip can end before full recentering.  Track
    # whether the final second is nevertheless moving back toward the lane.
    tail_steps = max(1, int(round(1.0 / max(dt_s, 1.0e-6))))
    tail = distances[-tail_steps:] if distances else []
    prev = distances[-2 * tail_steps : -tail_steps] if len(distances) > tail_steps else []
    last_1s_mean = float(statistics.mean(tail)) if tail else None
    prev_1s_mean = float(statistics.mean(prev)) if prev else None
    recent_recovery_delta = (
        float(prev_1s_mean - last_1s_mean)
        if prev_1s_mean is not None and last_1s_mean is not None else None
    )

    reasons: list[str] = []
    if terminal_distance is not None and terminal_distance > terminal_max_m:
        reasons.append("terminal_far_from_vehicle_lane")
    if p90_dist is not None and p90_dist > p90_max_m:
        reasons.append("persistent_far_from_vehicle_lane")
    if offcenter_fraction > offcenter_fraction_max:
        reasons.append("excessive_offcenter_fraction")
    if terminal_heading_error_deg is not None and terminal_heading_error_deg > heading_terminal_max_deg:
        reasons.append("terminal_lane_misalignment")
    if p90_heading is not None and p90_heading > heading_p90_max_deg:
        reasons.append("persistent_lane_misalignment")

    return {
        "evidence_available": True,
        "accepted": not reasons,
        "reason": "lane_realistic" if not reasons else "+".join(reasons),
        "lane_segment_count": len(segments),
        "frame_evidence_fraction": coverage,
        "vehicle_lane_types": list(lane_types),
        "lane_center_distance_median_m": float(statistics.median(distances)) if distances else None,
        "lane_center_distance_p90_m": p90_dist,
        "lane_center_distance_max_m": max(distances) if distances else None,
        "lane_center_distance_first_m": first_distance,
        "lane_center_distance_terminal_m": terminal_distance,
        "lane_center_distance_peak_to_terminal_improvement_m": peak_to_terminal_improvement,
        "lane_center_distance_prev_1s_mean_m": prev_1s_mean,
        "lane_center_distance_last_1s_mean_m": last_1s_mean,
        "lane_center_distance_recent_recovery_delta_m": recent_recovery_delta,
        "offcenter_threshold_m": offcenter_threshold_m,
        "offcenter_fraction": offcenter_fraction,
        "lane_heading_error_p90_deg": p90_heading,
        "lane_heading_error_terminal_deg": terminal_heading_error_deg,
        "misaligned_fraction": misaligned_fraction,
        "terminal_speed_mps": terminal_speed,
        "thresholds": {
            "terminal_max_m": terminal_max_m,
            "p90_max_m": p90_max_m,
            "offcenter_fraction_max": offcenter_fraction_max,
            "heading_terminal_max_deg": heading_terminal_max_deg,
            "heading_p90_max_deg": heading_p90_max_deg,
            "heading_speed_gate_mps": heading_speed_gate_mps,
            "min_evidence_fraction": min_evidence_fraction,
        },
    }


def _load_traces(trace_root: Path, regime: str) -> dict[str, dict[str, dict[str, Any]]]:
    paths = {"ocrap": trace_root / "ocrap" / regime / "closed_loop_ocrap.json.scenes.jsonl"}
    paths.update({m: trace_root / "external" / regime / f"closed_loop_{m}.json.scenes.jsonl" for m in METHODS[regime]})
    return {m: _load_journal(p) for m, p in paths.items()}


def _relative_score(item: dict[str, Any], method: str) -> float:
    try:
        x = float(((item.get("per_baseline") or {}).get(method) or {}).get("relative_score"))
        return x if math.isfinite(x) else float("inf")
    except Exception:
        return float("inf")


def _hardest_among(item: dict[str, Any], methods: list[str]) -> str | None:
    if not methods:
        return None
    return min(methods, key=lambda name: (_relative_score(item, name), name))


def _near_quality(item: dict[str, Any], traces: dict[str, dict[str, dict[str, Any]]], *, dt_s: float,
                  ttc_threshold_s: float, clearance_threshold_m: float, density_radius_m: float,
                  lane_kwargs: dict[str, Any]) -> dict[str, Any]:
    key = str(item["target_key"])
    clip = float(item.get("clip_duration_s") or 0.0)
    oscene = traces["ocrap"][key]
    otrace = list(oscene.get("render_trace") or [])
    oc_overlap = _any_flag(otrace, clip, dt_s, "overlap")
    oc_offroad = _any_flag(otrace, clip, dt_s, "offroad")
    oc_lane = _lane_realism(oscene, clip, dt_s, **lane_kwargs)
    external: dict[str, dict[str, Any]] = {}
    overlap_count = severe_count = 0
    for method in METHODS["near"]:
        tr = list(traces[method][key].get("render_trace") or [])
        overlap = _any_flag(tr, clip, dt_s, "overlap")
        offroad = _any_flag(tr, clip, dt_s, "offroad")
        min_ttc = _min_metric(tr, clip, dt_s, "ttc_s")
        min_clr = _min_metric(tr, clip, dt_s, "min_clearance_m")
        severe = bool(overlap or (min_ttc is not None and min_ttc <= ttc_threshold_s) or (min_clr is not None and min_clr <= clearance_threshold_m))
        overlap_count += int(overlap)
        severe_count += int(severe)
        external[method] = {
            "overlap": overlap, "offroad": offroad, "min_ttc_s": min_ttc,
            "min_clearance_m": min_clr, "severe_low_margin": severe,
        }
    n = len(METHODS["near"])
    frac = severe_count / max(n, 1)
    overlap_frac = overlap_count / max(n, 1)
    if severe_count == n:
        evidence_rank, evidence = 0, "all_external_severe"
    elif frac >= 0.75:
        evidence_rank, evidence = 1, "strong_consensus_external_hazard"
    elif frac >= 0.50:
        evidence_rank, evidence = 2, "majority_external_hazard"
    else:
        evidence_rank, evidence = 3, "limited_external_hazard"
    density = _density_stats(otrace, clip, dt_s, density_radius_m)
    severe_methods = [m for m in METHODS["near"] if external[m]["severe_low_margin"]]
    trace_primary = _hardest_among(item, severe_methods) or str(item.get("primary_external_method") or "")
    primary_severe = bool(external.get(trace_primary, {}).get("severe_low_margin"))
    lane_ok = bool(oc_lane.get("accepted", True))
    return {
        "ocrap_overlap_visible": oc_overlap,
        "ocrap_offroad_visible": oc_offroad,
        "ocrap_lane_realism": oc_lane,
        "ocrap_visible_safe": not oc_overlap and not oc_offroad,
        "ocrap_realistic_safe": not oc_overlap and not oc_offroad and lane_ok,
        "external_overlap_count": overlap_count,
        "external_severe_count": severe_count,
        "num_external_baselines": n,
        "external_overlap_fraction": overlap_frac,
        "external_severe_fraction": frac,
        "trace_primary_external_method": trace_primary,
        "primary_external_severe": primary_severe,
        "visual_evidence_rank": evidence_rank,
        "visual_evidence_label": evidence,
        "local_density_radius_m": density_radius_m,
        "local_agent_density": density,
        "external_trace_hazard": external,
    }


def _safe_quality(item: dict[str, Any], traces: dict[str, dict[str, dict[str, Any]]], *, dt_s: float,
                  min_clip_s: float, margin_s: float, allow_tail_truncation: bool,
                  lane_kwargs: dict[str, Any]) -> dict[str, Any]:
    key = str(item["target_key"])
    requested = float(item.get("clip_duration_s") or 0.0)
    scene = traces["ocrap"][key]
    trace = list(scene.get("render_trace") or [])
    idx, reason = _first_violation(trace, requested, dt_s)
    if idx is None:
        effective = requested
        status = "full_clip_clean"
    elif not allow_tail_truncation:
        effective = requested
        status = "reject_visible_violation"
    else:
        margin_steps = max(1, int(math.ceil(margin_s / dt_s - 1e-9)))
        last_safe_index = max(0, idx - margin_steps)
        effective = min(requested, last_safe_index * dt_s)
        effective = math.floor((effective + 1e-9) / dt_s) * dt_s
        status = "late_tail_truncated" if effective + 1e-9 >= min_clip_s else "reject_early_violation"
    realism_clip = effective if status == "late_tail_truncated" else requested
    lane = _lane_realism(scene, realism_clip, dt_s, **lane_kwargs)
    accepted = status not in {"reject_early_violation", "reject_visible_violation"} and bool(lane.get("accepted", True))
    return {
        "requested_clip_duration_s": requested,
        "effective_clip_duration_s": effective,
        "full_requested_clip_clean": idx is None,
        "first_visible_violation_index": idx,
        "first_visible_violation_s": None if idx is None else idx * dt_s,
        "first_visible_violation_type": reason,
        "safe_tail_gate": status,
        "allow_tail_truncation": bool(allow_tail_truncation),
        "ocrap_lane_realism": lane,
        "accepted": accepted,
    }


def _sustained_separation(
    trace: list[dict[str, Any]], clip_s: float, dt_s: float, *, clearance_min_m: float, hold_s: float
) -> dict[str, Any]:
    frames = _visible_frames(trace, clip_s, dt_s)
    hold_steps = max(1, int(math.ceil(hold_s / dt_s - 1.0e-9)))
    good: list[bool] = []
    for f in frames:
        clr = _metric(f, "min_clearance_m")
        good.append(not _flag(f, "overlap") and clr is not None and clr >= clearance_min_m)
    first: int | None = None
    for i in range(0, max(0, len(good) - hold_steps + 1)):
        if all(good[i : i + hold_steps]):
            first = i
            break
    recontact = False
    if first is not None:
        recontact = any(_flag(f, "overlap") for f in frames[first + hold_steps :])
    return {
        "achieved": first is not None,
        "first_index": first,
        "first_s": None if first is None else first * dt_s,
        "hold_s": hold_s,
        "clearance_min_m": clearance_min_m,
        "recontact_after_separation": recontact,
    }


def _contact_lane_recovery_contract(
    lane: dict[str, Any], *, terminal_max_m: float, p90_max_m: float,
    offcenter_fraction_max: float, peak_to_terminal_improvement_min_m: float,
    recent_recovery_delta_min_m: float,
) -> dict[str, Any]:
    """Contact-only lane gate with a convergence-aware short-horizon fallback.

    Strict lane realism always passes.  Otherwise the fallback is allowed only
    for distance-type misses: the vehicle must remain under explicit hard
    distance/fraction caps, must not have heading-misalignment failures, and
    must show measurable convergence back toward a vehicle lane.
    """
    if not lane.get("evidence_available"):
        return {"accepted": True, "mode": "lane_evidence_unavailable", "reason": "lane_evidence_unavailable"}
    if lane.get("accepted", True):
        return {"accepted": True, "mode": "strict", "reason": "strict_lane_realism"}
    raw_reason = str(lane.get("reason") or "")
    if "lane_misalignment" in raw_reason:
        return {"accepted": False, "mode": "reject", "reason": raw_reason}
    terminal = lane.get("lane_center_distance_terminal_m")
    p90 = lane.get("lane_center_distance_p90_m")
    off_frac = lane.get("offcenter_fraction")
    peak_gain = lane.get("lane_center_distance_peak_to_terminal_improvement_m")
    recent_gain = lane.get("lane_center_distance_recent_recovery_delta_m")
    within_caps = (
        terminal is not None and float(terminal) <= float(terminal_max_m)
        and p90 is not None and float(p90) <= float(p90_max_m)
        and off_frac is not None and float(off_frac) <= float(offcenter_fraction_max)
    )
    converging = (
        (peak_gain is not None and float(peak_gain) >= float(peak_to_terminal_improvement_min_m))
        or (recent_gain is not None and float(recent_gain) >= float(recent_recovery_delta_min_m))
    )
    accepted = bool(within_caps and converging)
    return {
        "accepted": accepted,
        "mode": "converging_fallback" if accepted else "reject",
        "reason": "short_horizon_converging_to_lane" if accepted else raw_reason,
        "terminal_max_m": float(terminal_max_m),
        "p90_max_m": float(p90_max_m),
        "offcenter_fraction_max": float(offcenter_fraction_max),
        "peak_to_terminal_improvement_min_m": float(peak_to_terminal_improvement_min_m),
        "recent_recovery_delta_min_m": float(recent_recovery_delta_min_m),
        "within_caps": bool(within_caps),
        "converging": bool(converging),
    }


def _contact_method_trace_quality(
    scene: dict[str, Any], clip: float, dt_s: float, *, lane_kwargs: dict[str, Any],
    separation_clearance_m: float, separation_hold_s: float,
    lane_recovery_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tr = list(scene.get("render_trace") or [])
    frames = _visible_frames(tr, clip, dt_s)
    lane = _lane_realism(scene, clip, dt_s, **lane_kwargs)
    if lane_recovery_kwargs is None:
        lane_recovery_kwargs = {
            "terminal_max_m": 6.5, "p90_max_m": 7.0, "offcenter_fraction_max": 0.45,
            "peak_to_terminal_improvement_min_m": 1.0, "recent_recovery_delta_min_m": 0.35,
        }
    lane_recovery = _contact_lane_recovery_contract(lane, **lane_recovery_kwargs)
    sep = _sustained_separation(
        tr, clip, dt_s, clearance_min_m=separation_clearance_m, hold_s=separation_hold_s
    )
    first_overlap = next((i for i, f in enumerate(frames) if _flag(f, "overlap")), None)
    terminal = frames[-1] if frames else None
    terminal_clearance = _metric(terminal, "min_clearance_m") if terminal else None
    terminal_speed = _metric(terminal, "ego_speed_mps") if terminal else None
    terminal_overlap = bool(terminal is not None and _flag(terminal, "overlap"))
    offroad = any(_flag(f, "offroad") for f in frames)
    failure_reasons: list[str] = []
    if offroad:
        failure_reasons.append("visible_offroad")
    if terminal_overlap:
        failure_reasons.append("terminal_overlap")
    if not sep["achieved"]:
        failure_reasons.append("no_sustained_separation")
    if sep["recontact_after_separation"]:
        failure_reasons.append("recontact_after_separation")
    if not lane_recovery.get("accepted", True):
        failure_reasons.append("lane_unrealistic_recovery")
    return {
        "offroad_visible": offroad,
        "first_overlap_s": None if first_overlap is None else first_overlap * dt_s,
        "terminal_overlap": terminal_overlap,
        "terminal_clearance_m": terminal_clearance,
        "terminal_speed_mps": terminal_speed,
        "sustained_separation": sep,
        "lane_realism": lane,
        "lane_recovery_contract": lane_recovery,
        "controlled_recovery": not failure_reasons,
        "recovery_failure": bool(failure_reasons),
        "recovery_failure_reasons": failure_reasons,
    }


def _contact_quality(
    item: dict[str, Any], traces: dict[str, dict[str, dict[str, Any]]], *, dt_s: float,
    lane_kwargs: dict[str, Any], separation_clearance_m: float, separation_hold_s: float,
    lane_recovery_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = str(item["target_key"])
    clip = float(item.get("clip_duration_s") or 0.0)
    oc = _contact_method_trace_quality(
        traces["ocrap"][key], clip, dt_s, lane_kwargs=lane_kwargs, lane_recovery_kwargs=lane_recovery_kwargs,
        separation_clearance_m=separation_clearance_m, separation_hold_s=separation_hold_s,
    )
    external: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for method in METHODS["contact"]:
        q = _contact_method_trace_quality(
            traces[method][key], clip, dt_s, lane_kwargs=lane_kwargs, lane_recovery_kwargs=lane_recovery_kwargs,
            separation_clearance_m=separation_clearance_m, separation_hold_s=separation_hold_s,
        )
        external[method] = q
        if q["recovery_failure"]:
            failures.append(method)
    n = len(METHODS["contact"])
    frac = len(failures) / max(n, 1)
    if len(failures) == n:
        evidence_rank, evidence = 0, "all_external_recovery_failures"
    elif frac >= 2.0 / 3.0:
        evidence_rank, evidence = 1, "strong_consensus_external_recovery_failure"
    elif frac >= 0.5:
        evidence_rank, evidence = 2, "majority_external_recovery_failure"
    else:
        evidence_rank, evidence = 3, "limited_external_recovery_failure"
    trace_primary = _hardest_among(item, failures) or str(item.get("primary_external_method") or "")
    return {
        # Backward-compatible top-level fields used by older audit scripts.
        "ocrap_offroad_visible": oc["offroad_visible"],
        "ocrap_first_overlap_s": oc["first_overlap_s"],
        "ocrap_terminal_clearance_m": oc["terminal_clearance_m"],
        "ocrap_controlled_recovery": oc["controlled_recovery"],
        "ocrap_trace_recovery": oc,
        "external_recovery_failure_count": len(failures),
        "external_recovery_failure_fraction": frac,
        "num_external_baselines": n,
        "visual_evidence_rank": evidence_rank,
        "visual_evidence_label": evidence,
        "trace_primary_external_method": trace_primary,
        "external_trace_recovery": external,
    }


def _gate_rejection_reasons(regime: str, quality: dict[str, Any], *, near_min_external_severe_count: int) -> list[str]:
    """Return concise, reviewer-readable reasons for excluding one candidate."""
    reasons: list[str] = []
    if regime == "safe":
        if not quality.get("accepted", False):
            gate = str(quality.get("safe_tail_gate") or "")
            if gate.startswith("reject_"):
                reasons.append(gate)
            lane = quality.get("ocrap_lane_realism") or {}
            if lane.get("evidence_available") and not lane.get("accepted", True):
                reasons.extend(f"lane:{x}" for x in str(lane.get("reason") or "lane_unrealistic").split("+") if x)
    elif regime == "near":
        if quality.get("ocrap_overlap_visible"):
            reasons.append("ocrap_visible_overlap")
        if quality.get("ocrap_offroad_visible"):
            reasons.append("ocrap_visible_offroad")
        lane = quality.get("ocrap_lane_realism") or {}
        if lane.get("evidence_available") and not lane.get("accepted", True):
            reasons.extend(f"lane:{x}" for x in str(lane.get("reason") or "lane_unrealistic").split("+") if x)
        if int(quality.get("external_severe_count") or 0) < int(near_min_external_severe_count):
            reasons.append("insufficient_external_failure_consensus")
    else:
        recovery = quality.get("ocrap_trace_recovery") or {}
        reasons.extend(str(x) for x in recovery.get("recovery_failure_reasons") or [])
        lane = recovery.get("lane_realism") or {}
        if "lane_unrealistic_recovery" in reasons and lane.get("evidence_available"):
            reasons.extend(f"lane:{x}" for x in str(lane.get("reason") or "lane_unrealistic").split("+") if x)
    # Preserve order while avoiding duplicate umbrella/detail labels.
    out: list[str] = []
    for reason in reasons:
        if reason not in out:
            out.append(reason)
    return out


def _write_selection_audit(
    path: Path, *, regime: str, rows: list[dict[str, Any]], requested: int,
    candidate_pool_size: int, thresholds: dict[str, Any], preferred_keys: list[str],
) -> dict[str, Any]:
    accepted = [r for r in rows if r.get("accepted")]
    rejected = [r for r in rows if not r.get("accepted")]
    counts = Counter(reason for row in rejected for reason in row.get("rejection_reasons", []))
    doc = {
        "event": "trace_aware_visualization_selection_audit_v2",
        "regime": regime,
        "requested_num_scenes": int(requested),
        "candidate_pool_size": int(candidate_pool_size),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "enough_for_requested": len(accepted) >= int(requested),
        "preferred_existing_target_keys": preferred_keys,
        "rejection_reason_counts": dict(sorted(counts.items())),
        "thresholds": thresholds,
        "candidates": rows,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return doc


def _copy_final_doc(candidate: dict[str, Any], selected: list[dict[str, Any]], *, candidate_path: Path, regime: str) -> dict[str, Any]:
    doc = dict(candidate)
    doc["event"] = "regime_visualization_scene_selection_trace_final_v2_realism_gated"
    doc["candidate_selection"] = str(candidate_path)
    doc["candidate_pool_size"] = len(candidate.get("selected") or [])
    doc["requested_num_scenes"] = len(selected)
    doc["selected"] = selected
    doc["target_keys"] = [str(x["target_key"]) for x in selected]
    doc["trace_aware_finalization"] = True
    doc["selection_note"] = str(candidate.get("selection_note") or "") + (
        " A trace-aware qualitative-only finalization removes visible OC-RAP safety failures and rejects "
        "map-inconsistent 'safety by escape' when vehicle-lane roadgraph evidence is available. Safe requires a clean "
        "full clip by default; Near-Contact prioritizes consensus external low-margin/collision evidence and local "
        "traffic density; Contact requires sustained separation plus a controlled lane-plausible recovery and ranks "
        "clearance only as a capped secondary term."
    )
    if selected:
        doc["selected_clip_duration_s"] = min(float(x.get("clip_duration_s") or 0.0) for x in selected)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-selection-root", type=Path, required=True)
    ap.add_argument("--trace-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--num-scenes", type=int, default=5)
    ap.add_argument("--safe-num-scenes", type=int, default=None)
    ap.add_argument("--near-num-scenes", type=int, default=None)
    ap.add_argument("--contact-num-scenes", type=int, default=None)
    ap.add_argument("--safe-min-clip-s", type=float, default=4.0)
    ap.add_argument("--safe-tail-margin-s", type=float, default=0.2)
    ap.add_argument("--safe-allow-tail-truncation", action="store_true")
    ap.add_argument("--near-ttc-threshold-s", type=float, default=0.5)
    ap.add_argument("--near-clearance-threshold-m", type=float, default=0.35)
    ap.add_argument("--near-density-radius-m", type=float, default=25.0)
    ap.add_argument(
        "--near-min-external-severe-count", type=int, default=2,
        help="minimum number of audited external methods that must collide or enter the trace-level low-margin region",
    )
    ap.add_argument("--lane-center-types", default="1,2")
    ap.add_argument("--lane-terminal-max-m", type=float, default=5.0)
    ap.add_argument("--lane-p90-max-m", type=float, default=5.5)
    ap.add_argument("--lane-offcenter-threshold-m", type=float, default=4.5)
    ap.add_argument("--lane-offcenter-fraction-max", type=float, default=0.30)
    ap.add_argument("--lane-heading-terminal-max-deg", type=float, default=50.0)
    ap.add_argument("--lane-heading-p90-max-deg", type=float, default=55.0)
    ap.add_argument("--lane-heading-speed-gate-mps", type=float, default=1.0)
    ap.add_argument("--lane-min-evidence-fraction", type=float, default=0.50)
    # Post-contact recovery can legitimately use a somewhat wider drivable
    # corridor than nominal Safe/Near driving.  Keep the publication defaults
    # identical to the global lane gate, but expose Contact-only overrides so
    # a reviewer-facing relaxation never silently weakens Safe/Near.
    ap.add_argument("--contact-lane-terminal-max-m", type=float, default=None)
    ap.add_argument("--contact-lane-p90-max-m", type=float, default=None)
    ap.add_argument("--contact-lane-offcenter-threshold-m", type=float, default=None)
    ap.add_argument("--contact-lane-offcenter-fraction-max", type=float, default=None)
    ap.add_argument("--contact-lane-heading-terminal-max-deg", type=float, default=None)
    ap.add_argument("--contact-lane-heading-p90-max-deg", type=float, default=None)
    ap.add_argument("--contact-lane-recovery-terminal-max-m", type=float, default=6.5)
    ap.add_argument("--contact-lane-recovery-p90-max-m", type=float, default=7.0)
    ap.add_argument("--contact-lane-recovery-offcenter-fraction-max", type=float, default=0.45)
    ap.add_argument("--contact-lane-peak-improvement-min-m", type=float, default=1.0)
    ap.add_argument("--contact-lane-recent-recovery-min-m", type=float, default=0.35)
    ap.add_argument("--contact-separation-clearance-m", type=float, default=0.50)
    ap.add_argument("--contact-separation-hold-s", type=float, default=0.30)
    ap.add_argument(
        "--prefer-existing-selection", action="store_true",
        help=(
            "prefer target keys already present in output-root/<regime>_selection.json, "
            "but only if they still pass every current trace-aware gate; rejected old "
            "scenes are automatically replaced from the accepted candidate pool"
        ),
    )
    args = ap.parse_args()
    if args.num_scenes <= 0:
        raise SystemExit("--num-scenes must be positive")
    requested_by_regime = {
        "safe": int(args.safe_num_scenes if args.safe_num_scenes is not None else args.num_scenes),
        "near": int(args.near_num_scenes if args.near_num_scenes is not None else args.num_scenes),
        "contact": int(args.contact_num_scenes if args.contact_num_scenes is not None else args.num_scenes),
    }
    if any(v <= 0 for v in requested_by_regime.values()):
        raise SystemExit("per-regime --*-num-scenes values must be positive")
    try:
        lane_types = tuple(int(x.strip()) for x in str(args.lane_center_types).split(",") if x.strip())
    except Exception as exc:
        raise SystemExit(f"invalid --lane-center-types={args.lane_center_types!r}: {exc}") from exc
    if not lane_types:
        raise SystemExit("--lane-center-types must contain at least one roadgraph type")
    lane_kwargs = {
        "lane_types": lane_types,
        "terminal_max_m": float(args.lane_terminal_max_m),
        "p90_max_m": float(args.lane_p90_max_m),
        "offcenter_threshold_m": float(args.lane_offcenter_threshold_m),
        "offcenter_fraction_max": float(args.lane_offcenter_fraction_max),
        "heading_terminal_max_deg": float(args.lane_heading_terminal_max_deg),
        "heading_p90_max_deg": float(args.lane_heading_p90_max_deg),
        "heading_speed_gate_mps": float(args.lane_heading_speed_gate_mps),
        "min_evidence_fraction": float(args.lane_min_evidence_fraction),
    }
    contact_lane_kwargs = dict(lane_kwargs)
    contact_overrides = {
        "terminal_max_m": args.contact_lane_terminal_max_m,
        "p90_max_m": args.contact_lane_p90_max_m,
        "offcenter_threshold_m": args.contact_lane_offcenter_threshold_m,
        "offcenter_fraction_max": args.contact_lane_offcenter_fraction_max,
        "heading_terminal_max_deg": args.contact_lane_heading_terminal_max_deg,
        "heading_p90_max_deg": args.contact_lane_heading_p90_max_deg,
    }
    for name, value in contact_overrides.items():
        if value is not None:
            contact_lane_kwargs[name] = float(value)
    contact_lane_recovery_kwargs = {
        "terminal_max_m": float(args.contact_lane_recovery_terminal_max_m),
        "p90_max_m": float(args.contact_lane_recovery_p90_max_m),
        "offcenter_fraction_max": float(args.contact_lane_recovery_offcenter_fraction_max),
        "peak_to_terminal_improvement_min_m": float(args.contact_lane_peak_improvement_min_m),
        "recent_recovery_delta_min_m": float(args.contact_lane_recent_recovery_min_m),
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"event": "trace_aware_visualization_selection_index_v1", "regimes": {}}

    for regime in ("safe", "near", "contact"):
        cpath = args.candidate_selection_root / f"{regime}_selection.json"
        candidate = json.loads(cpath.read_text(encoding="utf-8"))
        items = list(candidate.get("selected") or [])
        need = int(requested_by_regime[regime])
        traces = _load_traces(args.trace_root, regime)
        dt = float(candidate.get("metric_dt_s", 0.1) or 0.1)
        annotated: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        preferred_rank: dict[str, int] = {}
        existing_selection_path = args.output_root / f"{regime}_selection.json"
        if args.prefer_existing_selection and existing_selection_path.is_file():
            try:
                previous = json.loads(existing_selection_path.read_text(encoding="utf-8"))
                for i, row in enumerate(previous.get("selected") or [], 1):
                    preferred_rank[str(row.get("target_key") or "")] = int(row.get("category_rank") or i)
            except Exception:
                preferred_rank = {}
        for original_order, item in enumerate(items):
            key = str(item["target_key"])
            missing = [m for m, rows in traces.items() if key not in rows]
            if missing:
                raise SystemExit(f"{regime}/{key}: missing candidate traces for {missing}")
            row = dict(item)
            if regime == "safe":
                q = _safe_quality(
                    row,
                    traces,
                    dt_s=dt,
                    min_clip_s=args.safe_min_clip_s,
                    margin_s=args.safe_tail_margin_s,
                    allow_tail_truncation=bool(args.safe_allow_tail_truncation),
                    lane_kwargs=lane_kwargs,
                )
                row["visualization_trace_quality"] = q
                accepted = bool(q["accepted"])
                if accepted:
                    row["clip_duration_s"] = float(q["effective_clip_duration_s"])
                    row["clip_duration_adjusted_after_trace"] = not bool(q["full_requested_clip_clean"])
                    annotated.append(row | {"_original_order": original_order, "_preferred_eligible": True})
                audit_rows.append({
                    "target_key": key, "candidate_rank": item.get("category_rank"),
                    "accepted": accepted,
                    "preferred_existing": key in preferred_rank,
                    "rejection_reasons": _gate_rejection_reasons(
                        "safe", q, near_min_external_severe_count=args.near_min_external_severe_count
                    ),
                    "quality": q,
                })
            elif regime == "near":
                q = _near_quality(
                    row,
                    traces,
                    dt_s=dt,
                    ttc_threshold_s=args.near_ttc_threshold_s,
                    clearance_threshold_m=args.near_clearance_threshold_m,
                    density_radius_m=args.near_density_radius_m,
                    lane_kwargs=lane_kwargs,
                )
                row["visualization_trace_quality"] = q
                accepted = bool(q["ocrap_realistic_safe"]) and int(q["external_severe_count"]) >= int(args.near_min_external_severe_count)
                if accepted:
                    trace_primary = str(q.get("trace_primary_external_method") or "")
                    if trace_primary:
                        row["primary_external_method"] = trace_primary
                        row["primary_comparator_reason"] = (
                            "hardest paired external among methods that become trace-severe on this scene; "
                            "scene selection remains based on all-baseline failure consensus"
                        )
                    annotated.append(row | {"_original_order": original_order, "_preferred_eligible": True})
                audit_rows.append({
                    "target_key": key, "candidate_rank": item.get("category_rank"),
                    "accepted": accepted,
                    "preferred_existing": key in preferred_rank,
                    "rejection_reasons": _gate_rejection_reasons(
                        "near", q, near_min_external_severe_count=args.near_min_external_severe_count
                    ),
                    "quality": q,
                })
            else:
                q = _contact_quality(
                    row,
                    traces,
                    dt_s=dt,
                    lane_kwargs=contact_lane_kwargs,
                    lane_recovery_kwargs=contact_lane_recovery_kwargs,
                    separation_clearance_m=args.contact_separation_clearance_m,
                    separation_hold_s=args.contact_separation_hold_s,
                )
                row["visualization_trace_quality"] = q
                accepted = bool(q["ocrap_controlled_recovery"])
                preferred_eligible = accepted
                # If Contact-only lane thresholds are relaxed, do not let an
                # old scene become sticky merely because the relaxation admits
                # it. Historical keepers must still satisfy the original strict
                # global lane contract; relaxed-only scenes compete as fresh
                # backfill candidates.
                if accepted and key in preferred_rank and contact_lane_kwargs != lane_kwargs:
                    strict_q = _contact_quality(
                        row, traces, dt_s=dt, lane_kwargs=lane_kwargs,
                        lane_recovery_kwargs=contact_lane_recovery_kwargs,
                        separation_clearance_m=args.contact_separation_clearance_m,
                        separation_hold_s=args.contact_separation_hold_s,
                    )
                    preferred_eligible = bool(strict_q["ocrap_controlled_recovery"])
                if accepted:
                    trace_primary = str(q.get("trace_primary_external_method") or "")
                    if trace_primary:
                        row["primary_external_method"] = trace_primary
                        row["primary_comparator_reason"] = (
                            "hardest paired external among methods that fail the trace-level controlled-recovery gate; "
                            "OC-RAP must retain lane realism and sustained separation"
                        )
                    annotated.append(row | {"_original_order": original_order, "_preferred_eligible": preferred_eligible})
                audit_rows.append({
                    "target_key": key, "candidate_rank": item.get("category_rank"),
                    "accepted": accepted,
                    "preferred_existing": key in preferred_rank,
                    "preferred_existing_strict_gate": bool(preferred_eligible) if key in preferred_rank else False,
                    "rejection_reasons": _gate_rejection_reasons(
                        "contact", q, near_min_external_severe_count=args.near_min_external_severe_count
                    ),
                    "quality": q,
                })

        if regime == "safe":
            # Full-clip cleanliness is publication-facing default. Among clean
            # scenes, prefer those with explicit lane evidence and smaller lane
            # center deviation before the original metric tier/score.
            annotated.sort(key=lambda r: (
                0 if r["visualization_trace_quality"]["full_requested_clip_clean"] else 1,
                0 if r["visualization_trace_quality"]["ocrap_lane_realism"].get("evidence_available") else 1,
                float(r["visualization_trace_quality"]["ocrap_lane_realism"].get("lane_center_distance_p90_m") or 0.0),
                int(r.get("selection_tier_rank", 99)),
                -float(r.get("clip_duration_s") or 0.0),
                -float(r.get("score") or 0.0),
                str(r["target_key"]),
            ))
        elif regime == "near":
            # Reviewer-facing priority: broad external failure consensus first,
            # then actual collision count, primary-comparator severity, local
            # density, and finally the original paired metric evidence.
            annotated.sort(key=lambda r: (
                0 if r["visualization_trace_quality"]["ocrap_lane_realism"].get("evidence_available") else 1,
                int(r["visualization_trace_quality"]["visual_evidence_rank"]),
                -int(r["visualization_trace_quality"]["external_overlap_count"]),
                -int(r["visualization_trace_quality"]["external_severe_count"]),
                -int(bool(r["visualization_trace_quality"]["primary_external_severe"])),
                -float(r["visualization_trace_quality"]["local_agent_density"]["p75"]),
                int(r.get("selection_tier_rank", 99)),
                -float(r.get("score") or 0.0),
                str(r["target_key"]),
            ))
        else:
            # Do not let raw clearance dominate reviewer-facing Contact scenes.
            # Controlled/lane-plausible OC-RAP recovery is already a hard gate;
            # then prefer broad external recovery failure, faster sustained
            # separation, and only a *capped* terminal-clearance tie-breaker.
            annotated.sort(key=lambda r: (
                0 if r["visualization_trace_quality"]["ocrap_trace_recovery"]["lane_realism"].get("evidence_available") else 1,
                int(r["visualization_trace_quality"]["visual_evidence_rank"]),
                -int(r["visualization_trace_quality"]["external_recovery_failure_count"]),
                float(
                    r["visualization_trace_quality"]["ocrap_trace_recovery"]["sustained_separation"].get("first_s")
                    if r["visualization_trace_quality"]["ocrap_trace_recovery"]["sustained_separation"].get("first_s") is not None
                    else 1.0e6
                ),
                -min(3.0, float(r["visualization_trace_quality"].get("ocrap_terminal_clearance_m") or 0.0)),
                int(r.get("selection_tier_rank", 99)),
                -float(r.get("score") or 0.0),
                str(r["target_key"]),
            ))

        if preferred_rank:
            # Preserve only previous scenes that STILL satisfy all current
            # reviewer-facing gates. Rejected historical scenes never bypass
            # the new realism contract; accepted newcomers fill the vacated slots.
            accepted_by_key = {str(r["target_key"]): r for r in annotated}
            preferred = [
                accepted_by_key[k]
                for k, _rank in sorted(preferred_rank.items(), key=lambda kv: kv[1])
                if k in accepted_by_key and bool(accepted_by_key[k].get("_preferred_eligible", True))
            ]
            preferred_keys = {str(r["target_key"]) for r in preferred}
            annotated = preferred + [r for r in annotated if str(r["target_key"]) not in preferred_keys]

        audit_thresholds = {
            "near_min_external_severe_count": int(args.near_min_external_severe_count),
            "global_lane": lane_kwargs,
            "contact_lane": contact_lane_kwargs,
            "contact_lane_short_horizon_recovery": contact_lane_recovery_kwargs,
            "contact_separation_clearance_m": float(args.contact_separation_clearance_m),
            "contact_separation_hold_s": float(args.contact_separation_hold_s),
            "safe_allow_tail_truncation": bool(args.safe_allow_tail_truncation),
        }
        audit_doc = _write_selection_audit(
            args.output_root / f"{regime}_selection_audit.json",
            regime=regime, rows=audit_rows, requested=need,
            candidate_pool_size=len(items), thresholds=audit_thresholds,
            preferred_keys=[k for k, _ in sorted(preferred_rank.items(), key=lambda kv: kv[1])],
        )

        # Preserve scenario diversity after trace-aware re-ranking.
        chosen: list[dict[str, Any]] = []
        used_scenes: set[str] = set()
        for row in annotated:
            sid = str(row.get("scene_id") or row["target_key"])
            if sid in used_scenes:
                continue
            chosen.append(row)
            used_scenes.add(sid)
            if len(chosen) >= need:
                break
        if len(chosen) < need:
            reason_counts = audit_doc.get("rejection_reason_counts") or {}
            audit_path = args.output_root / f"{regime}_selection_audit.json"
            raise SystemExit(
                f"{regime}: trace-aware gate retained only {len(chosen)} distinct scenes from {len(items)} candidates; requested={need}. "
                f"Audit written to {audit_path}. Rejection counts={reason_counts}. "
                "Increase the candidate multiplier only if Stage 1 can actually supply additional locked candidates; "
                "otherwise relax only a documented realism threshold whose rejected margins are small, or lower the "
                "requested final rank count. Never relax visible off-road, terminal-overlap, or re-contact gates merely "
                "to fill a quota."
            )
        final: list[dict[str, Any]] = []
        for rank, row in enumerate(chosen, 1):
            row = {k: v for k, v in row.items() if k not in {"_original_order", "_preferred_eligible"}}
            row["category_rank"] = rank
            final.append(row)
        doc = _copy_final_doc(candidate, final, candidate_path=cpath, regime=regime)
        out = args.output_root / f"{regime}_selection.json"
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output_root / f"{regime}_target_keys.json").write_text(
            json.dumps({"regime": regime, "target_keys": doc["target_keys"]}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["regimes"][regime] = {
            "candidate_pool_size": len(items), "num_selected": len(final), "selection": str(out),
            "selected": [
                {
                    "rank": x["category_rank"], "target_key": x["target_key"],
                    "primary_external_method": x.get("primary_external_method"),
                    "clip_duration_s": x.get("clip_duration_s"),
                    "trace_quality": x.get("visualization_trace_quality"),
                } for x in final
            ],
        }

    index = args.output_root / "SELECTION_INDEX.json"
    index.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": summary["event"], "index": str(index), "num_scenes_per_regime": requested_by_regime}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
