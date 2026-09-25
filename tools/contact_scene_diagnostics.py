#!/usr/bin/env python3
"""Contact-scene diagnostics used by qualitative target mining and auditing.

The functions in this module are deliberately observation/render-trace based.
They do not modify trajectories and do not access latent future labels.  They
measure three properties that are useful when curating *aspirational* Contact
visualizations:

* crowding around the SDC;
* re-contact after an initial separation;
* contact with a new actor after the common observed-contact boundary.

The last item is reported separately from same-partner re-contact because a
secondary collision with a different actor is visually and physically more
severe than simply remaining in the original contact episode.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def _finite(x: Any, default: float | None = None) -> float | None:
    try:
        v = float(x)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def _flag(frame: dict[str, Any], key: str) -> bool:
    v = _finite((frame.get("metrics") or {}).get(key), 0.0)
    return bool(v is not None and v > 0.5)


def _sdc_agent(frame: dict[str, Any]) -> dict[str, Any] | None:
    for agent in frame.get("agents") or []:
        if agent.get("is_sdc"):
            return agent
    return None


def agent_key(agent: dict[str, Any], fallback_index: int | None = None) -> str:
    """Return a stable-enough actor key for one trace.

    Waymax/WOMD render records usually expose ``object_index``.  Several helper
    exporters use ``track_id``/``object_id`` instead, so accept all common names.
    """
    for name in ("object_id", "track_id", "object_index", "id", "agent_id"):
        if name in agent and agent.get(name) is not None:
            return f"{name}:{agent.get(name)}"
    if fallback_index is not None:
        return f"frame_index:{int(fallback_index)}"
    return "unknown"


def _box_corners(agent: dict[str, Any]) -> list[tuple[float, float]]:
    x = float(agent["x"])
    y = float(agent["y"])
    yaw = float(agent.get("yaw", 0.0))
    length = max(float(agent.get("length", 4.8)), 0.1)
    width = max(float(agent.get("width", 2.0)), 0.1)
    c, s = math.cos(yaw), math.sin(yaw)
    out: list[tuple[float, float]] = []
    for dx, dy in (
        (length / 2, width / 2),
        (length / 2, -width / 2),
        (-length / 2, -width / 2),
        (-length / 2, width / 2),
    ):
        out.append((x + c * dx - s * dy, y + s * dx + c * dy))
    return out


def _point_segment_distance(p, a, b) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy)


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d) -> bool:
    eps = 1e-9
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True

    def on(p, q, r):
        return (
            abs(_orient(p, q, r)) <= eps
            and min(p[0], q[0]) - eps <= r[0] <= max(p[0], q[0]) + eps
            and min(p[1], q[1]) - eps <= r[1] <= max(p[1], q[1]) + eps
        )

    return on(a, b, c) or on(a, b, d) or on(c, d, a) or on(c, d, b)


def _polygon_distance(pa, pb) -> float:
    ea = list(zip(pa, pa[1:] + pa[:1]))
    eb = list(zip(pb, pb[1:] + pb[:1]))
    if any(_segments_intersect(a, b, c, d) for a, b in ea for c, d in eb):
        return 0.0
    return min(
        min(_point_segment_distance(p, c, d) for p in pa for c, d in eb),
        min(_point_segment_distance(p, a, b) for p in pb for a, b in ea),
    )


def _sat_penetration(pa, pb) -> float:
    axes: list[tuple[float, float]] = []
    for poly in (pa, pb):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            ex, ey = x2 - x1, y2 - y1
            n = math.hypot(ex, ey)
            if n <= 1e-9:
                continue
            axis = (-ey / n, ex / n)
            if not any(abs(axis[0] * bx + axis[1] * by) > 0.999 for bx, by in axes):
                axes.append(axis)
    min_overlap = float("inf")
    for ax, ay in axes:
        aa = [x * ax + y * ay for x, y in pa]
        bb = [x * ax + y * ay for x, y in pb]
        overlap = min(max(aa), max(bb)) - max(min(aa), min(bb))
        if overlap < -1e-9:
            return 0.0
        min_overlap = min(min_overlap, max(0.0, overlap))
    return 0.0 if not math.isfinite(min_overlap) else float(min_overlap)


def signed_box_clearance(a: dict[str, Any], b: dict[str, Any]) -> float:
    pa, pb = _box_corners(a), _box_corners(b)
    pen = _sat_penetration(pa, pb)
    if pen > 1e-9:
        return -float(pen)
    return float(_polygon_distance(pa, pb))


def frame_actor_geometry(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-actor center distance and exact oriented-box clearance."""
    sdc = _sdc_agent(frame)
    if sdc is None:
        return []
    sx, sy = float(sdc["x"]), float(sdc["y"])
    out: list[dict[str, Any]] = []
    for i, agent in enumerate(frame.get("agents") or []):
        if agent.get("is_sdc"):
            continue
        try:
            cx = float(agent["x"])
            cy = float(agent["y"])
            clr = signed_box_clearance(sdc, agent)
        except Exception:
            continue
        out.append(
            {
                "actor_key": agent_key(agent, i),
                "center_distance_m": float(math.hypot(cx - sx, cy - sy)),
                "clearance_m": float(clr),
            }
        )
    out.sort(key=lambda x: (float(x["clearance_m"]), float(x["center_distance_m"]), str(x["actor_key"])))
    return out


def _quantile(values: Iterable[float], q: float) -> float | None:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.quantile(vals, q)) if vals else None


def analyze_contact_trace(
    trace: list[dict[str, Any]],
    dt: float = 0.1,
    *,
    crowd_radius_m: float = 12.0,
    crowd_min_agents: int = 3,
    conflict_clearance_m: float = 2.0,
) -> dict[str, Any]:
    """Compute crowding and secondary-contact diagnostics from visible states."""
    if not trace:
        return {"usable": False, "critical_tags": []}
    dt = float(dt)
    geometry = [frame_actor_geometry(frame) for frame in trace]
    metric_overlap = [_flag(frame, "overlap") for frame in trace]
    partner_sets: list[set[str]] = []
    for flag, rows in zip(metric_overlap, geometry):
        partners = {str(r["actor_key"]) for r in rows if float(r["clearance_m"]) < -1e-7}
        # Some recorded metrics mark an overlap when SAT penetration is almost
        # exactly zero.  Associate that event with the nearest touching actor so
        # partner-level diagnostics stay useful without overriding geometry.
        if flag and not partners and rows and float(rows[0]["clearance_m"]) <= 0.05:
            partners.add(str(rows[0]["actor_key"]))
        partner_sets.append(partners)
    overlap = [bool(metric_overlap[i] or partner_sets[i]) for i in range(len(trace))]
    first_contact = next((i for i, value in enumerate(overlap) if value), None)
    first_sep = None
    if first_contact is not None:
        first_sep = next((i for i in range(first_contact + 1, len(overlap)) if not overlap[i]), None)

    anchor_partners: set[str] = set()
    if first_contact is not None:
        anchor_partners = set(partner_sets[first_contact])
        if not anchor_partners and geometry[first_contact]:
            anchor_partners.add(str(geometry[first_contact][0]["actor_key"]))

    all_partners: set[str] = set().union(*partner_sets) if partner_sets else set()
    later_partners: set[str] = set()
    if first_contact is not None:
        for ps in partner_sets[first_contact + 1 :]:
            later_partners.update(ps)
    secondary_partners = later_partners - anchor_partners

    post_sep_partners: set[str] = set()
    if first_sep is not None:
        for ps in partner_sets[first_sep + 1 :]:
            post_sep_partners.update(ps)
    post_sep_new_partners = post_sep_partners - anchor_partners
    post_sep_same_partners = post_sep_partners & anchor_partners

    counts_8 = [sum(float(r["center_distance_m"]) <= 8.0 for r in rows) for rows in geometry]
    counts_12 = [sum(float(r["center_distance_m"]) <= 12.0 for r in rows) for rows in geometry]
    counts_20 = [sum(float(r["center_distance_m"]) <= 20.0 for r in rows) for rows in geometry]
    crowd_counts = [sum(float(r["center_distance_m"]) <= float(crowd_radius_m) for r in rows) for rows in geometry]
    conflict_counts = [sum(float(r["clearance_m"]) <= float(conflict_clearance_m) for r in rows) for rows in geometry]
    nearest_clear = [float(rows[0]["clearance_m"]) for rows in geometry if rows]
    noninitial_clear: list[float] = []
    if first_contact is not None:
        for rows in geometry[first_contact:]:
            vals = [float(r["clearance_m"]) for r in rows if str(r["actor_key"]) not in anchor_partners]
            if vals:
                noninitial_clear.append(min(vals))

    offroad = [_flag(frame, "offroad") or _flag(frame, "sdc_off_route") for frame in trace]
    clear = [_finite((frame.get("metrics") or {}).get("min_clearance_m"), None) for frame in trace]
    terminal_clearance = next((float(v) for v in reversed(clear) if v is not None), None)
    post_sep_overlap = bool(first_sep is not None and any(overlap[first_sep + 1 :]))
    initial_contact_end = first_sep if first_sep is not None else len(overlap)
    initial_contact_duration_s = None
    if first_contact is not None:
        initial_contact_duration_s = float(max(0, initial_contact_end - first_contact) * dt)

    tags: list[str] = []
    crowded_fraction = float(sum(c >= int(crowd_min_agents) for c in crowd_counts) / len(crowd_counts)) if crowd_counts else 0.0
    if max(crowd_counts or [0]) >= int(crowd_min_agents) or crowded_fraction >= 0.20:
        tags.append("crowded")
    if max(conflict_counts or [0]) >= 2:
        tags.append("multi_actor_conflict")
    if secondary_partners:
        tags.append("secondary_collision")
    if post_sep_new_partners:
        tags.append("post_separation_secondary_collision")
    if post_sep_same_partners:
        tags.append("recontact")
    elif post_sep_overlap:
        tags.append("recontact")
    if first_contact is not None and first_sep is None:
        tags.append("persistent_contact")
    offroad_fraction = float(sum(offroad) / len(offroad)) if offroad else 0.0
    if offroad_fraction > 0.0:
        tags.append("source_offroad")
    if terminal_clearance is not None and terminal_clearance < 1.0:
        tags.append("low_terminal_clearance")

    return {
        "usable": bool(first_contact is not None),
        "num_states": len(trace),
        "first_contact_index": first_contact,
        "first_separation_index": first_sep,
        "first_separation_s": None if first_sep is None else float(first_sep * dt),
        "persistent_initial_contact": bool(first_contact is not None and first_sep is None),
        "initial_contact_duration_s": initial_contact_duration_s,
        "post_separation_collision_event": bool(post_sep_overlap),
        "same_partner_recontact_event": bool(post_sep_same_partners),
        "secondary_collision_event": bool(secondary_partners),
        "post_separation_secondary_collision_event": bool(post_sep_new_partners),
        "anchor_collision_partners": sorted(anchor_partners),
        "secondary_collision_partners": sorted(secondary_partners),
        "post_separation_collision_partners": sorted(post_sep_partners),
        "distinct_collision_partner_count": int(len(all_partners)),
        "secondary_collision_partner_count": int(len(secondary_partners)),
        "overlap_duration_s": float(sum(overlap[:-1]) * dt) if len(overlap) > 1 else 0.0,
        "terminal_clearance_m": terminal_clearance,
        "nearest_clearance_p05_m": _quantile(nearest_clear, 0.05),
        "min_noninitial_partner_clearance_m": min(noninitial_clear) if noninitial_clear else None,
        "nearby_agents_peak_8m": int(max(counts_8 or [0])),
        "nearby_agents_peak_12m": int(max(counts_12 or [0])),
        "nearby_agents_peak_20m": int(max(counts_20 or [0])),
        "crowd_radius_m": float(crowd_radius_m),
        "crowd_min_agents": int(crowd_min_agents),
        "crowded_fraction": crowded_fraction,
        "conflict_clearance_m": float(conflict_clearance_m),
        "multi_actor_conflict_peak": int(max(conflict_counts or [0])),
        "multi_actor_conflict_fraction": float(sum(c >= 2 for c in conflict_counts) / len(conflict_counts)) if conflict_counts else 0.0,
        "offroad_fraction": offroad_fraction,
        "critical_tags": list(dict.fromkeys(tags)),
    }


def analyze_contact_scene(scene: dict[str, Any], dt: float = 0.1, **kwargs: Any) -> dict[str, Any]:
    return analyze_contact_trace(list(scene.get("render_trace") or []), dt=dt, **kwargs)


def criticality_score(
    ocrap_diag: dict[str, Any],
    baseline_diags: dict[str, dict[str, Any]] | None = None,
) -> float:
    """Deterministic source-only criticality score for qualitative mining.

    It intentionally rewards *difficult source scenes*.  It is never an
    evaluation metric and must not be used to claim empirical OC-RAP quality.
    """
    if not ocrap_diag.get("usable"):
        return -1e9
    score = 0.0
    score += 18.0 * float(bool(ocrap_diag.get("secondary_collision_event")))
    score += 12.0 * float(bool(ocrap_diag.get("post_separation_secondary_collision_event")))
    score += 9.0 * float(bool(ocrap_diag.get("same_partner_recontact_event")))
    score += 6.0 * float(bool(ocrap_diag.get("persistent_initial_contact")))
    score += 5.0 * max(0, int(ocrap_diag.get("distinct_collision_partner_count") or 0) - 1)
    score += 2.5 * max(0, int(ocrap_diag.get("nearby_agents_peak_12m") or 0) - 2)
    score += 2.0 * max(0, int(ocrap_diag.get("multi_actor_conflict_peak") or 0) - 1)
    score += 10.0 * min(1.0, float(ocrap_diag.get("crowded_fraction") or 0.0))
    score += 6.0 * min(0.5, float(ocrap_diag.get("offroad_fraction") or 0.0)) / 0.5
    term = _finite(ocrap_diag.get("terminal_clearance_m"), None)
    if term is not None:
        score += 3.0 * max(0.0, min(2.0, 1.5 - float(term)))

    for diag in (baseline_diags or {}).values():
        if not diag.get("usable"):
            continue
        score += 2.5 * float(bool(diag.get("persistent_initial_contact")))
        score += 2.0 * float(bool(diag.get("post_separation_collision_event")))
        score += 1.5 * float(bool(diag.get("secondary_collision_event")))
        score += 0.75 * max(0, int(diag.get("nearby_agents_peak_12m") or 0) - 2)
        score += min(2.0, float(diag.get("overlap_duration_s") or 0.0))
    return float(score)
