#!/usr/bin/env python3
"""Render synchronized single-model and OC-RAP-vs-best/worst videos.

For every selected scene, this tool creates:
  * one synchronized single-model video for OC-RAP;
  * one synchronized single-model video per external baseline;
  * OC-RAP (left) vs the per-scene best external baseline (right);
  * OC-RAP (left) vs the per-scene worst external baseline (right).

All videos for a scene use the same clip duration, simulation-time samples and
scene-wide fixed world-frame camera by default.  If a rollout terminates before
the selected clip duration, the final state is held and explicitly labeled.
The selector's >=5 s / >=3 s duration contract is therefore preserved in the
encoded video duration without silently comparing different time windows.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import animation, patches, transforms

try:
    from ocrap.external_baselines.provenance import find_provenance
except Exception:  # pragma: no cover - renderer still works outside installed package
    find_provenance = None


def _parse_trace_specs(specs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"invalid --trace {spec!r}; expected METHOD=SCENES.jsonl")
        name, raw = spec.split("=", 1)
        name = name.strip()
        if not name or name in out:
            raise SystemExit(f"duplicate/empty method in --trace {spec!r}")
        out[name] = Path(raw)
    if "ocrap" not in out:
        raise SystemExit("--trace must include ocrap=...")
    if len(out) < 2:
        raise SystemExit("at least one external baseline trace is required")
    return out


def _load_scenes(path: Path):
    if path.suffix == ".json" and not path.name.endswith(".scenes.jsonl"):
        alt = Path(str(path) + ".scenes.jsonl")
        if alt.is_file():
            path = alt
    direct: dict[str, dict[str, Any]] = {}
    by_scene_time: dict[tuple[str, str], dict[str, Any]] = {}
    by_scene: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            envelope = json.loads(line)
            scene = envelope.get("scene", envelope)
            key = str(scene.get("target_key") or envelope.get("resume_key") or "")
            if key.startswith("target:"):
                key = key[len("target:"):]
            scene_id = str(scene.get("scene_id") or "")
            raw_time = scene.get("target_time_index")
            time_index = str(raw_time if raw_time is not None else "")
            if not key:
                key = f"{scene_id}:t{time_index}" if scene_id and time_index else scene_id
            if not key:
                raise ValueError(f"scene without target key in {path}")
            if key in direct:
                raise ValueError(f"duplicate target key {key} in {path}")
            direct[key] = scene
            if scene_id:
                by_scene.setdefault(scene_id, []).append(scene)
                if time_index:
                    by_scene_time[(scene_id, time_index)] = scene
    if not direct:
        raise ValueError(f"empty scene journal: {path}")
    return direct, by_scene_time, by_scene


def _resolve_scene(item, loaded):
    direct, by_scene_time, by_scene = loaded
    key = str(item["target_key"])
    if key in direct:
        return direct[key], "target_key"
    scene_id = str(item.get("scene_id") or "")
    time_index = str(item.get("target_time_index") if item.get("target_time_index") is not None else "")
    if scene_id and time_index and (scene_id, time_index) in by_scene_time:
        return by_scene_time[(scene_id, time_index)], "scene_id_and_time"
    if scene_id and len(by_scene.get(scene_id, [])) == 1:
        return by_scene[scene_id][0], "unique_scene_id"
    return None, "unresolved"


def _safe_name(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return value.strip("._-") or "model"


def _display_name(method: str) -> str:
    if method == "ocrap":
        return "OC-RAP"
    if find_provenance is not None:
        item = find_provenance(method)
        if item is not None:
            return str(item.reporting_name or item.canonical_name)
    return method


def _metric_float(frame: dict[str, Any], key: str) -> float | None:
    try:
        value = float((frame.get("metrics", {}) or {}).get(key))
        return value if math.isfinite(value) else None
    except Exception:
        return None


def _scene_metric(scene: dict[str, Any], *keys: str) -> float | None:
    summary = scene.get("metric_summary", {}) or {}
    for key in keys:
        for source in (scene, summary):
            try:
                if key in source:
                    value = float(source[key])
                    if math.isfinite(value):
                        return value
            except Exception:
                pass
    return None


def _frame(trace: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return trace[min(max(index, 0), len(trace) - 1)]


def _sdc_agent(frame: dict[str, Any]) -> dict[str, Any] | None:
    for agent in frame.get("agents", []):
        if agent.get("is_sdc"):
            return agent
    return None


def _sdc(frame: dict[str, Any]) -> tuple[float, float] | None:
    agent = _sdc_agent(frame)
    if agent is None:
        return None
    try:
        return float(agent["x"]), float(agent["y"])
    except Exception:
        return None


def _box_corners(agent: dict[str, Any]) -> list[tuple[float, float]]:
    x = float(agent["x"]); y = float(agent["y"])
    half_l = max(float(agent["length"]), 0.1) / 2.0
    half_w = max(float(agent["width"]), 0.1) / 2.0
    yaw = float(agent["yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dx, dy in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)):
        out.append((x + c * dx - s * dy, y + s * dx + c * dy))
    return out


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_segment_distance(p, a, b) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    wx, wy = p[0] - a[0], p[1] - a[1]
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(wx, wy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    qx, qy = a[0] + t * vx, a[1] + t * vy
    return math.hypot(p[0] - qx, p[1] - qy)


def _segments_intersect(a, b, c, d) -> bool:
    eps = 1e-9
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    def on(p, q, r):
        return abs(_orient(p, q, r)) <= eps and min(p[0], q[0]) - eps <= r[0] <= max(p[0], q[0]) + eps and min(p[1], q[1]) - eps <= r[1] <= max(p[1], q[1]) + eps
    return on(a, b, c) or on(a, b, d) or on(c, d, a) or on(c, d, b)


def _polygon_distance(poly_a, poly_b) -> float:
    edges_a = list(zip(poly_a, poly_a[1:] + poly_a[:1]))
    edges_b = list(zip(poly_b, poly_b[1:] + poly_b[:1]))
    if any(_segments_intersect(a, b, c, d) for a, b in edges_a for c, d in edges_b):
        return 0.0
    return min(
        min(_point_segment_distance(p, c, d) for p in poly_a for c, d in edges_b),
        min(_point_segment_distance(p, a, b) for p in poly_b for a, b in edges_a),
    )


def _minimum_box_pair(frame: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float] | None:
    sdc = _sdc_agent(frame)
    if sdc is None:
        return None
    try:
        sdc_poly = _box_corners(sdc)
    except Exception:
        return None
    rows = []
    for other in frame.get("agents", []):
        if other.get("is_sdc"):
            continue
        try:
            dist = _polygon_distance(sdc_poly, _box_corners(other))
        except Exception:
            continue
        rows.append((dist, other))
    if not rows:
        return None
    dist, other = min(rows, key=lambda row: row[0])
    return sdc, other, float(dist)


def _draw_roadgraph(ax, context, center, radius):
    cx, cy = center
    for polyline in (context or {}).get("roadgraph_polylines", []):
        xy = polyline.get("xy") or []
        points = []
        for point in xy:
            try:
                x, y = float(point[0]), float(point[1])
            except Exception:
                continue
            if abs(x - cx) <= radius + 5.0 and abs(y - cy) <= radius + 5.0:
                points.append((x, y))
        if len(points) >= 2:
            ax.plot([p[0] for p in points], [p[1] for p in points], linewidth=0.65, alpha=0.35, zorder=0)


def _all_model_fixed_view(traces: dict[str, list[dict[str, Any]]], minimum_radius: float):
    points = [point for trace in traces.values() for frame in trace for point in [_sdc(frame)] if point is not None]
    if not points:
        return (0.0, 0.0), minimum_radius
    min_x, max_x = min(p[0] for p in points), max(p[0] for p in points)
    min_y, max_y = min(p[1] for p in points), max(p[1] for p in points)
    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    radius = max(minimum_radius, 0.55 * max(max_x - min_x, max_y - min_y) + 6.0)
    return center, radius


def _dynamic_center(traces: list[list[dict[str, Any]]], sim_index: int):
    points = [p for trace in traces for p in [_sdc(_frame(trace, sim_index))] if p is not None]
    if not points:
        return 0.0, 0.0
    return sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)


def _contact_marker(trace, regime):
    for row in trace:
        if (_metric_float(row, "overlap") or 0.0) > 0.5:
            return _sdc(row), "observed overlap"
    if regime == "contact" and trace:
        return _sdc(trace[0]), "post-contact rollout start"
    return None, None


def _yaw_rate(trace: list[dict[str, Any]], index: int, metric_dt_s: float) -> float | None:
    if index <= 0:
        return 0.0
    curr = _metric_float(_frame(trace, index), "ego_yaw_rad")
    prev = _metric_float(_frame(trace, index - 1), "ego_yaw_rad")
    if curr is None or prev is None:
        return None
    return math.atan2(math.sin(curr - prev), math.cos(curr - prev)) / metric_dt_s


def _draw_frame(ax, trace, sim_index, title, center, radius, contact_xy, contact_label, metric_dt_s, context=None):
    held = sim_index >= len(trace)
    row = _frame(trace, sim_index)
    cx, cy = center
    ax.clear(); ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(cx - radius, cx + radius); ax.set_ylim(cy - radius, cy + radius)
    _draw_roadgraph(ax, context, center, radius)
    ax.set_title(title + (" · final state held" if held else ""), fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(alpha=0.15)

    trail = [_sdc(r) for r in trace[:min(sim_index, len(trace) - 1) + 1]]
    trail = [p for p in trail if p is not None]
    if len(trail) >= 2:
        ax.plot([p[0] for p in trail], [p[1] for p in trail], linewidth=1.8, alpha=0.9, zorder=2)
    if contact_xy is not None:
        ax.scatter([contact_xy[0]], [contact_xy[1]], marker="x", s=80, linewidths=2, zorder=5)
        ax.annotate(contact_label or "contact", contact_xy, xytext=(6, 6), textcoords="offset points", fontsize=7,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75}, zorder=7)

    overlap = (_metric_float(row, "overlap") or 0.0) > 0.5
    offroad = (_metric_float(row, "offroad") or 0.0) > 0.5
    for agent in row.get("agents", []):
        try:
            x, y = float(agent["x"]), float(agent["y"])
            length, width, yaw = max(float(agent["length"]), 0.1), max(float(agent["width"]), 0.1), float(agent["yaw"])
        except Exception:
            continue
        if abs(x - cx) > radius + length or abs(y - cy) > radius + length:
            continue
        is_sdc = bool(agent.get("is_sdc"))
        edge = "red" if is_sdc and overlap else ("orange" if is_sdc and offroad else ("tab:blue" if is_sdc else "0.35"))
        rect = patches.Rectangle((x - length / 2, y - width / 2), length, width, fill=False,
                                 linewidth=2.4 if is_sdc else 0.8, edgecolor=edge, zorder=4 if is_sdc else 3)
        rect.set_transform(transforms.Affine2D().rotate_around(x, y, yaw) + ax.transData)
        ax.add_patch(rect)
        if is_sdc:
            ax.text(x, y, "SDC", fontsize=8, ha="center", va="center", zorder=6)

    reported = _metric_float(row, "min_clearance_m")
    pair = _minimum_box_pair(row)
    if pair is not None and reported is not None:
        sdc, other, rendered_dist = pair
        sx, sy, ox, oy = float(sdc["x"]), float(sdc["y"]), float(other["x"]), float(other["y"])
        ax.plot([sx, ox], [sy, oy], linestyle="--", linewidth=1.0, alpha=0.65, zorder=1)
        mismatch = abs(rendered_dist - max(0.0, reported)) > 0.15
        label = f"min box clearance={reported:.2f} m"
        if mismatch:
            label += f" (render geom {rendered_dist:.2f})"
        ax.annotate(label, ((sx + ox) / 2.0, (sy + oy) / 2.0), xytext=(4, 4), textcoords="offset points", fontsize=7,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.70}, zorder=7)

    lines = [
        f"t={row.get('time_index')}  macro={row.get('selected_macro', '')}",
        f"candidate={row.get('selected_candidate_index')}  reason={row.get('selection_reason', '')}",
    ]
    for key, label, unit in (("ttc_s", "TTC", "s"), ("min_clearance_m", "clearance", "m"),
                             ("ego_speed_mps", "speed", "m/s"), ("overlap", "overlap", ""), ("offroad", "offroad", "")):
        value = _metric_float(row, key)
        if value is not None:
            lines.append(f"{label}={value:.3f}{unit}")
    yr = _yaw_rate(trace, min(sim_index, len(trace) - 1), metric_dt_s)
    if yr is not None:
        lines.append(f"yaw_rate={yr:.3f} rad/s")
    ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=7.5,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82}, zorder=10)


def _sample_indices(frame_count: int, fps: int, metric_dt_s: float) -> list[int]:
    return [int(round((i / fps) / metric_dt_s)) for i in range(frame_count)]


def _validate_trace_time_alignment(traces: dict[str, list[dict[str, Any]]], item: dict[str, Any]) -> dict[str, Any]:
    starts: dict[str, int] = {}
    lengths: dict[str, int] = {}
    for method, trace in traces.items():
        times = []
        for row in trace:
            try:
                times.append(int(row["time_index"]))
            except Exception as exc:
                raise SystemExit(f"{item.get('target_key')}: render_trace for {method} lacks integer time_index") from exc
        if not times:
            raise SystemExit(f"{item.get('target_key')}: empty time-index sequence for {method}")
        bad = [(a, b) for a, b in zip(times, times[1:]) if b != a + 1]
        if bad:
            raise SystemExit(f"{item.get('target_key')}: non-consecutive render time_index for {method}; examples={bad[:5]}")
        starts[method] = times[0]
        lengths[method] = len(times)
    if len(set(starts.values())) != 1:
        raise SystemExit(f"{item.get('target_key')}: model traces do not start at the same simulator time: {starts}")
    expected = item.get("target_time_index")
    if expected is not None and int(expected) != next(iter(starts.values())):
        raise SystemExit(
            f"{item.get('target_key')}: trace start time {next(iter(starts.values()))} != selected target_time_index {expected}"
        )
    return {"start_time_index": next(iter(starts.values())), "trace_lengths": lengths, "consecutive": True}


def _series(trace: list[dict[str, Any]], key: str, sim_indices: list[int]) -> list[float]:
    values = []
    for idx in sim_indices:
        val = _metric_float(_frame(trace, idx), key)
        values.append(float("nan") if val is None else val)
    return values


def _draw_timeline(ax, twin, traces: dict[str, list[dict[str, Any]]], display: dict[str, str], frame_index: int,
                   sim_indices: list[int], fps: int, regime: str):
    ax.clear(); twin.clear()
    times = [i / fps for i in range(len(sim_indices))]
    visible = min(frame_index + 1, len(times))
    for method, trace in traces.items():
        ax.plot(times[:visible], _series(trace, "min_clearance_m", sim_indices)[:visible], label=f"{display[method]} clearance")
    if regime == "near":
        ax.axhline(2.0, linestyle="--", linewidth=0.9, alpha=0.6, label="2 m near-contact boundary")
    ax.set_xlim(0.0, max(times[-1] if times else 0.0, 0.1)); ax.set_xlabel("rollout video time [s]")
    ax.set_ylabel("box clearance [m]"); ax.grid(alpha=0.15)

    secondary_key = "ttc_s" if regime == "near" else "ego_speed_mps"
    secondary_label = "TTC [s]" if regime == "near" else "ego speed [m/s]"
    for method, trace in traces.items():
        twin.plot(times[:visible], _series(trace, secondary_key, sim_indices)[:visible], linestyle=":" if regime == "near" else "--",
                  alpha=0.65, label=f"{display[method]} {'TTC' if regime == 'near' else 'speed'}")
    if regime == "near":
        twin.axhline(3.0, linestyle=":", linewidth=0.9, alpha=0.45, label="3 s TTC boundary")
    twin.set_ylabel(secondary_label)

    # Overlap is encoded as sparse markers rather than full-height fill, which
    # remains readable with two traces and does not obscure clearance curves.
    overlap_y = ax.get_ylim()[0]
    for method, trace in traces.items():
        xs = [times[j] for j in range(visible) if (_metric_float(_frame(trace, sim_indices[j]), "overlap") or 0.0) > 0.5]
        if xs:
            ax.scatter(xs, [overlap_y] * len(xs), marker="x", s=18, label=f"{display[method]} overlap")
    handles, labels = ax.get_legend_handles_labels(); h2, l2 = twin.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, loc="upper right", fontsize=6.7, ncol=2)


def _fmt(value: float | None, unit: str = "", digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{unit}"


def _summary(scene: dict[str, Any], regime: str) -> str:
    if regime == "safe":
        parts = [
            f"NUP={_fmt(_scene_metric(scene, 'closed_loop_bounded_NUP'))}",
            f"intervention={_fmt(_scene_metric(scene, 'intervention_rate'))}",
            f"clearance p05={_fmt(_scene_metric(scene, 'min_clearance_m_p05', 'min_clearance_m_min'), ' m')}",
            f"TTC p05={_fmt(_scene_metric(scene, 'ttc_s_p05', 'ttc_s_min'), ' s')}",
            f"offroad={_fmt(_scene_metric(scene, 'offroad_any'), digits=0)}",
        ]
    elif regime == "near":
        parts = [
            f"TTC p05={_fmt(_scene_metric(scene, 'ttc_s_p05'), ' s')}",
            f"clearance p05={_fmt(_scene_metric(scene, 'min_clearance_m_p05'), ' m')}",
            f"critical-TTC exposure={_fmt(_scene_metric(scene, 'critical_ttc_exposure_duration_s'), ' s')}",
            f"near-zero exposure={_fmt(_scene_metric(scene, 'near_zero_clearance_exposure_rate'))}",
            f"NUP={_fmt(_scene_metric(scene, 'closed_loop_bounded_NUP'))}",
        ]
    else:
        parts = [
            f"terminal clearance={_fmt(_scene_metric(scene, 'post_contact_terminal_clearance_m'), ' m')}",
            f"free-space AUC={_fmt(_scene_metric(scene, 'post_contact_free_space_auc_normalized_m'), ' m')}",
            f"overlap duration={_fmt(_scene_metric(scene, 'post_contact_overlap_duration_s'), ' s')}",
            f"recontact={_fmt(_scene_metric(scene, 'recontact_event'), digits=0)}",
            f"stable stop={_fmt(_scene_metric(scene, 'new_stable_stop_quality_event'), digits=0)}",
            f"escape={_fmt(_scene_metric(scene, 'post_contact_escape_event'), digits=0)}",
        ]
    return " | ".join(parts)


def _save_animation(fig, update, frame_count, fps, output: Path, use_mp4: bool):
    anim = animation.FuncAnimation(fig, update, frames=frame_count, interval=1000 / fps, blit=False)
    if use_mp4:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2200, metadata={"artist": "OC-RAP regime visualization v51"})
        anim.save(output, writer=writer)
    else:
        writer = animation.PillowWriter(fps=fps)
        anim.save(output, writer=writer)
    plt.close(fig)


def _render_single(*, method, scene, trace, item, regime, display_name, context, center, radius,
                   camera_mode, sim_indices, fps, metric_dt_s, output, use_mp4):
    contact_xy, contact_label = _contact_marker(trace, regime)
    figure = plt.figure(figsize=(9.6, 7.8), dpi=100)
    grid = figure.add_gridspec(2, 1, height_ratios=[4.3, 1.35])
    map_ax = figure.add_subplot(grid[0, 0]); timeline_ax = figure.add_subplot(grid[1, 0]); timeline_twin = timeline_ax.twinx()
    figure.suptitle(f"{regime.upper()} · {display_name} · rank {item.get('category_rank')}\n{_summary(scene, regime)}", fontsize=10)
    figure.subplots_adjust(top=0.89, bottom=0.08, hspace=0.30)

    def update(frame_i):
        sim_idx = sim_indices[frame_i]
        view_center = _dynamic_center([trace], sim_idx) if camera_mode == "dynamic" else center
        _draw_frame(map_ax, trace, sim_idx, display_name, view_center, radius, contact_xy, contact_label, metric_dt_s, context)
        _draw_timeline(timeline_ax, timeline_twin, {method: trace}, {method: display_name}, frame_i, sim_indices, fps, regime)

    _save_animation(figure, update, len(sim_indices), fps, output, use_mp4)


def _render_pair(*, methods, scenes, traces, item, regime, displays, context, center, radius,
                 camera_mode, sim_indices, fps, metric_dt_s, output, use_mp4, comparator_role):
    ocrap, comparator = methods
    contact = {m: _contact_marker(traces[m], regime) for m in methods}
    figure = plt.figure(figsize=(14.0, 8.2), dpi=100)
    grid = figure.add_gridspec(2, 2, height_ratios=[4.2, 1.35])
    axes = {ocrap: figure.add_subplot(grid[0, 0]), comparator: figure.add_subplot(grid[0, 1])}
    timeline_ax = figure.add_subplot(grid[1, :]); timeline_twin = timeline_ax.twinx()
    figure.suptitle(
        f"{regime.upper()} · OC-RAP vs per-scene {comparator_role} external · rank {item.get('category_rank')}\n"
        f"OC-RAP: {_summary(scenes[ocrap], regime)}\n{displays[comparator]}: {_summary(scenes[comparator], regime)}",
        fontsize=9.5,
    )
    figure.subplots_adjust(top=0.84, bottom=0.08, hspace=0.32, wspace=0.18)

    def update(frame_i):
        sim_idx = sim_indices[frame_i]
        view_center = _dynamic_center([traces[m] for m in methods], sim_idx) if camera_mode == "dynamic" else center
        for method in methods:
            xy, label = contact[method]
            _draw_frame(axes[method], traces[method], sim_idx, displays[method], view_center, radius, xy, label, metric_dt_s, context)
        _draw_timeline(timeline_ax, timeline_twin, {m: traces[m] for m in methods}, displays, frame_i, sim_indices, fps, regime)

    _save_animation(figure, update, len(sim_indices), fps, output, use_mp4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", action="append", default=[], metavar="METHOD=SCENES.jsonl")
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--format", choices=("auto", "mp4", "gif"), default="auto")
    ap.add_argument("--view-radius-m", type=float, default=35.0)
    ap.add_argument("--camera-mode", choices=("fixed", "dynamic"), default="fixed")
    args = ap.parse_args()
    if args.fps <= 0 or args.view_radius_m <= 5.0:
        raise SystemExit("fps must be positive and view radius must exceed 5 m")

    paths = _parse_trace_specs(args.trace)
    loaded = {method: _load_scenes(path) for method, path in paths.items()}
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    regime = str(selection.get("regime") or "")
    if regime not in {"safe", "near", "contact"}:
        raise SystemExit(f"invalid/missing regime in selection: {regime!r}")
    selected = selection.get("selected") or []
    if not selected:
        raise SystemExit("selection contains no selected scenes")
    external_methods = [m for m in paths if m != "ocrap"]
    expected_external = list(selection.get("external_baselines") or [])
    if set(external_methods) != set(expected_external):
        raise SystemExit(f"trace baseline set does not match selection: traces={external_methods}, selection={expected_external}")

    use_mp4 = args.format == "mp4" or (args.format == "auto" and shutil.which("ffmpeg") is not None)
    if args.format == "mp4" and shutil.which("ffmpeg") is None:
        raise SystemExit("--format mp4 requested but ffmpeg is unavailable")
    suffix = ".mp4" if use_mp4 else ".gif"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_dt_s = float(selection.get("metric_dt_s", 0.1) or 0.1)
    records = []

    for item in selected:
        key = str(item["target_key"])
        resolved: dict[str, dict[str, Any]] = {}
        methods_resolution: dict[str, str] = {}
        for method in paths:
            scene, how = _resolve_scene(item, loaded[method])
            if scene is None:
                raise SystemExit(f"selected target {key} unresolved for {method}")
            trace = scene.get("render_trace") or []
            if not trace:
                raise SystemExit(f"selected target {key} has no render_trace for {method}; rerun selective traces with full scene detail")
            resolved[method] = scene
            methods_resolution[method] = how
        traces = {m: list(resolved[m]["render_trace"]) for m in paths}
        time_alignment = _validate_trace_time_alignment(traces, item)

        clip_duration_s = float(item.get("clip_duration_s", selection.get("selected_clip_duration_s", 5.0)))
        frame_count = max(1, int(round(clip_duration_s * args.fps)))
        sim_indices = _sample_indices(frame_count, args.fps, metric_dt_s)
        center, radius = _all_model_fixed_view(traces, args.view_radius_m)
        context = resolved["ocrap"].get("render_context") or next((resolved[m].get("render_context") for m in paths if resolved[m].get("render_context")), {})
        rank = int(item.get("category_rank", len(records) + 1))
        scene_dir = args.output_dir / regime / f"rank_{rank:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        displays = {m: _display_name(m) for m in paths}
        outputs = []

        for method in paths:
            filename = scene_dir / f"{regime}__rank_{rank:02d}__{_safe_name(method)}{suffix}"
            _render_single(method=method, scene=resolved[method], trace=traces[method], item=item, regime=regime,
                           display_name=displays[method], context=context, center=center, radius=radius,
                           camera_mode=args.camera_mode, sim_indices=sim_indices, fps=args.fps, metric_dt_s=metric_dt_s,
                           output=filename, use_mp4=use_mp4)
            outputs.append({"type": "single", "method": method, "path": str(filename)})

        best = str(item.get("best_external_method") or "")
        worst = str(item.get("worst_external_method") or "")
        if best not in external_methods or worst not in external_methods:
            raise SystemExit(f"selection comparator missing from external trace set for {key}: best={best}, worst={worst}")
        for role, comparator in (("best", best), ("worst", worst)):
            filename = scene_dir / f"{regime}__rank_{rank:02d}__ocrap_vs_{role}__{_safe_name(comparator)}{suffix}"
            pair_methods = ["ocrap", comparator]
            _render_pair(methods=pair_methods, scenes={m: resolved[m] for m in pair_methods}, traces={m: traces[m] for m in pair_methods},
                         item=item, regime=regime, displays={m: displays[m] for m in pair_methods}, context=context,
                         center=center, radius=radius, camera_mode=args.camera_mode, sim_indices=sim_indices, fps=args.fps,
                         metric_dt_s=metric_dt_s, output=filename, use_mp4=use_mp4, comparator_role=role)
            outputs.append({"type": f"pair_{role}", "method": comparator, "path": str(filename)})

        expected_count = len(external_methods) + 3
        if len(outputs) != expected_count:
            raise RuntimeError(f"internal video count mismatch for {key}: {len(outputs)} vs {expected_count}")
        records.append({
            "target_key": key,
            "rank": rank,
            "scene_id": item.get("scene_id"),
            "target_time_index": item.get("target_time_index"),
            "clip_duration_s": clip_duration_s,
            "encoded_frames": frame_count,
            "fps": args.fps,
            "metric_dt_s": metric_dt_s,
            "camera_mode": args.camera_mode,
            "camera_center_xy": list(center),
            "camera_radius_m": radius,
            "best_external_method": best,
            "worst_external_method": worst,
            "resolution": methods_resolution,
            "time_alignment": time_alignment,
            "raw_trace_frames": {m: len(traces[m]) for m in paths},
            "held_final_state": {m: max(sim_indices) >= len(traces[m]) for m in paths},
            "num_videos": len(outputs),
            "videos": outputs,
        })

    expected_total = len(selected) * (len(external_methods) + 3)
    actual_total = sum(r["num_videos"] for r in records)
    if actual_total != expected_total:
        raise RuntimeError(f"total video count mismatch: {actual_total} vs {expected_total}")
    index = {
        "event": "regime_visualization_video_index_v51",
        "regime": regime,
        "selection": str(args.selection),
        "external_baselines": external_methods,
        "num_external_baselines": len(external_methods),
        "num_selected_scenes": len(selected),
        "videos_per_scene": len(external_methods) + 3,
        "num_videos": actual_total,
        "format": suffix.lstrip("."),
        "synchronization_contract": "same selected target, same video-time sample, same simulation dt, same scene-wide fixed camera unless camera_mode=dynamic",
        "pair_layout": "OC-RAP left; per-scene external comparator right",
        "records": records,
    }
    index_path = args.output_dir / f"{regime.upper()}_VIDEO_INDEX.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": index["event"], "regime": regime, "num_videos": actual_total, "index": str(index_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
