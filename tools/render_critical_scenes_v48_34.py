#!/usr/bin/env python3
"""Render paired selected closed-loop traces as side-by-side MP4/GIF videos.

The renderer uses one shared world-frame camera for both policies by default,
marks observed overlap (or the post-contact rollout start), draws the reported
box-clearance relation without treating it as a radial free-space circle, and
adds a synchronized metric timeline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import animation, patches, transforms


def _load_scenes(path: Path):
    if path.suffix == ".json" and not path.name.endswith(".scenes.jsonl"):
        alternative = Path(str(path) + ".scenes.jsonl")
        if alternative.is_file():
            path = alternative
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
    return direct, by_scene_time, by_scene


def _resolve_scene(item, direct, by_scene_time, by_scene):
    key = str(item["target_key"])
    scene = direct.get(key)
    if scene is not None:
        return scene
    scene_id = str(item.get("scene_id") or "")
    time_index = str(item.get("target_time_index") if item.get("target_time_index") is not None else "")
    if scene_id and time_index:
        scene = by_scene_time.get((scene_id, time_index))
    if scene is None and scene_id and len(by_scene.get(scene_id, [])) == 1:
        scene = by_scene[scene_id][0]
    return scene


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


def _frame(trace: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return trace[min(index, len(trace) - 1)]


def _dynamic_center(control_trace, method_trace, index):
    points = [point for point in (_sdc(_frame(control_trace, index)), _sdc(_frame(method_trace, index))) if point is not None]
    if not points:
        return 0.0, 0.0
    return sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)


def _fixed_shared_view(control_trace, method_trace, minimum_radius: float) -> tuple[tuple[float, float], float]:
    points = [
        point
        for trace in (control_trace, method_trace)
        for frame in trace
        for point in [_sdc(frame)]
        if point is not None
    ]
    if not points:
        return (0.0, 0.0), minimum_radius
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    radius = max(minimum_radius, 0.55 * max(max_x - min_x, max_y - min_y) + 6.0)
    return center, radius


def _contact_marker(trace, regime):
    for frame in trace:
        metrics = frame.get("metrics", {}) or {}
        try:
            overlap = float(metrics.get("overlap", metrics.get("overlap_any", 0.0)))
        except Exception:
            overlap = 0.0
        if overlap > 0.5:
            return _sdc(frame), "observed overlap"
    # Contact targets can begin after the initiating collision.  The first
    # simulated state is an anchor for the post-contact rollout, not proof of
    # the collision's exact world-space location.
    if regime == "contact" and trace:
        return _sdc(trace[0]), "post-contact rollout start"
    return None, None


def _metric_float(frame, key):
    try:
        value = float((frame.get("metrics", {}) or {}).get(key))
        return value if math.isfinite(value) else None
    except Exception:
        return None


def _yaw_rate(trace: list[dict[str, Any]], index: int, fps: int) -> float | None:
    if index <= 0:
        return 0.0
    current = _metric_float(_frame(trace, index), "ego_yaw_rad")
    previous = _metric_float(_frame(trace, index - 1), "ego_yaw_rad")
    if current is None or previous is None:
        return None
    delta = math.atan2(math.sin(current - previous), math.cos(current - previous))
    return delta * fps


def _nearest_other(frame: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    sdc = _sdc_agent(frame)
    if sdc is None:
        return None
    try:
        sx, sy = float(sdc["x"]), float(sdc["y"])
    except Exception:
        return None
    others = []
    for agent in frame.get("agents", []):
        if agent.get("is_sdc"):
            continue
        try:
            distance2 = (float(agent["x"]) - sx) ** 2 + (float(agent["y"]) - sy) ** 2
        except Exception:
            continue
        others.append((distance2, agent))
    if not others:
        return None
    return sdc, min(others, key=lambda item: item[0])[1]


def _draw_roadgraph(ax, context, center, radius):
    center_x, center_y = center
    for polyline in (context or {}).get("roadgraph_polylines", []):
        xy = polyline.get("xy") or []
        if len(xy) < 2:
            continue
        points = []
        for point in xy:
            try:
                x, y = float(point[0]), float(point[1])
            except Exception:
                continue
            if abs(x - center_x) <= radius + 5.0 and abs(y - center_y) <= radius + 5.0:
                points.append((x, y))
        if len(points) >= 2:
            ax.plot([point[0] for point in points], [point[1] for point in points], linewidth=0.65, alpha=0.35, zorder=0)


def _draw_frame(ax, trace, index, title, center, radius, contact_xy, contact_label, fps, context=None):
    held = index >= len(trace)
    frame = _frame(trace, index)
    center_x, center_y = center
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(center_x - radius, center_x + radius)
    ax.set_ylim(center_y - radius, center_y + radius)
    _draw_roadgraph(ax, context, center, radius)
    ax.set_title(title + (" · final state held" if held else ""))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.15)

    trail = [_sdc(row) for row in trace[:min(index, len(trace) - 1) + 1]]
    trail = [point for point in trail if point]
    if len(trail) >= 2:
        ax.plot([point[0] for point in trail], [point[1] for point in trail], linewidth=1.8, alpha=0.9, zorder=2)
    if contact_xy is not None:
        ax.scatter([contact_xy[0]], [contact_xy[1]], marker="x", s=80, linewidths=2, zorder=5)
        ax.annotate(
            contact_label or "contact",
            contact_xy,
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
            zorder=7,
        )

    overlap = (_metric_float(frame, "overlap") or 0.0) > 0.5
    offroad = (_metric_float(frame, "offroad") or 0.0) > 0.5
    for agent in frame.get("agents", []):
        try:
            x, y = float(agent["x"]), float(agent["y"])
            length = max(float(agent["length"]), 0.1)
            width = max(float(agent["width"]), 0.1)
            yaw = float(agent["yaw"])
        except Exception:
            continue
        if abs(x - center_x) > radius + length or abs(y - center_y) > radius + length:
            continue
        is_sdc = bool(agent.get("is_sdc"))
        edge = "red" if is_sdc and overlap else ("orange" if is_sdc and offroad else ("tab:blue" if is_sdc else "0.35"))
        rectangle = patches.Rectangle(
            (x - length / 2, y - width / 2), length, width,
            fill=False, linewidth=2.4 if is_sdc else 0.8, edgecolor=edge,
            zorder=4 if is_sdc else 3,
        )
        rectangle.set_transform(transforms.Affine2D().rotate_around(x, y, yaw) + ax.transData)
        ax.add_patch(rectangle)
        if is_sdc:
            ax.text(x, y, "SDC", fontsize=8, ha="center", va="center", zorder=6)

    clearance = _metric_float(frame, "min_clearance_m")
    nearest = _nearest_other(frame)
    if nearest is not None and clearance is not None:
        sdc_agent, other = nearest
        sx, sy = float(sdc_agent["x"]), float(sdc_agent["y"])
        ox, oy = float(other["x"]), float(other["y"])
        ax.plot([sx, ox], [sy, oy], linestyle="--", linewidth=1.0, alpha=0.65, zorder=1)
        ax.annotate(
            f"reported box clearance {clearance:.2f} m",
            ((sx + ox) / 2.0, (sy + oy) / 2.0),
            xytext=(4, 4), textcoords="offset points", fontsize=7,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.70}, zorder=7,
        )

    lines = [
        f"t={frame.get('time_index')} macro={frame.get('selected_macro', '')}",
        f"candidate={frame.get('selected_candidate_index')} reason={frame.get('selection_reason', '')}",
    ]
    for key, label, unit in (
        ("ttc_s", "TTC", "s"),
        ("min_clearance_m", "clearance", "m"),
        ("ego_speed_mps", "speed", "m/s"),
        ("overlap", "overlap", ""),
        ("offroad", "offroad", ""),
    ):
        value = _metric_float(frame, key)
        if value is not None:
            lines.append(f"{label}={value:.3f}{unit}")
    yaw_rate = _yaw_rate(trace, min(index, len(trace) - 1), fps)
    if yaw_rate is not None:
        lines.append(f"yaw_rate={yaw_rate:.3f}rad/s")
    ax.text(
        0.01, 0.99, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82}, zorder=10,
    )


def _series(trace: list[dict[str, Any]], key: str, frame_count: int) -> list[float]:
    values = []
    for index in range(frame_count):
        value = _metric_float(_frame(trace, index), key)
        values.append(float("nan") if value is None else value)
    return values


def _draw_timeline(ax, control_trace, method_trace, index, frames, fps, regime, control_name, method_name):
    ax.clear()
    times = [step / fps for step in range(frames)]
    visible = min(index + 1, frames)
    control_clearance = _series(control_trace, "min_clearance_m", frames)
    method_clearance = _series(method_trace, "min_clearance_m", frames)
    ax.plot(times[:visible], control_clearance[:visible], label=f"{control_name} clearance")
    ax.plot(times[:visible], method_clearance[:visible], label=f"{method_name} clearance")
    if regime == "near":
        ax.axhline(2.0, linestyle="--", linewidth=0.9, alpha=0.6, label="near-contact threshold 2 m")
    overlap_control = [(_metric_float(_frame(control_trace, step), "overlap") or 0.0) > 0.5 for step in range(frames)]
    overlap_method = [(_metric_float(_frame(method_trace, step), "overlap") or 0.0) > 0.5 for step in range(frames)]
    y_min, y_max = ax.get_ylim()
    ax.fill_between(times[:visible], y_min, y_max, where=overlap_control[:visible], alpha=0.08, step="post", label=f"{control_name} overlap")
    ax.fill_between(times[:visible], y_min, y_max, where=overlap_method[:visible], alpha=0.08, step="post", label=f"{method_name} overlap")
    ax.set_xlim(0.0, max(times[-1] if times else 0.0, 0.1))
    ax.set_xlabel("rollout time [s]")
    ax.set_ylabel("box clearance [m]")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def _delta_text(item, regime):
    terms = item.get("terms", {}) or {}
    keys = (
        (("ttc_p05_s", "ΔTTCp05", "s"), ("clearance_p05_m", "Δclearancep05", "m"), ("terminal_clearance_m", "Δterminal clearance", "m"), ("critical_ttc_exposure_s", "Δcritical exposure", "s"))
        if regime == "near"
        else (("post_contact_terminal_clearance_m", "Δterminal clearance", "m"), ("post_contact_free_space_auc_normalized_m", "Δfree-space AUC", "m"), ("post_contact_clearance_gain_m", "Δclearance gain", "m"), ("post_contact_overlap_duration_s", "Δoverlap duration", "s"), ("new_stable_stop_quality_event", "Δnew stable stop", ""), ("recontact_event", "Δrecontact", ""), ("yaw_rate_p95", "Δyaw-rate p95", "rad/s"), ("jerk_p95", "Δjerk p95", "m/s³"))
    )
    parts = []
    for key, label, unit in keys:
        try:
            value = float(terms[key])
            if math.isfinite(value):
                parts.append(f"{label}={value:+.2f}{unit}")
        except Exception:
            pass
    return " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-scenes", type=Path, required=True)
    ap.add_argument("--control-scenes", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--format", choices=("auto", "mp4", "gif"), default="auto")
    ap.add_argument("--method-name", default="OC-RAP")
    ap.add_argument("--control-name", default="Comparator")
    ap.add_argument("--view-radius-m", type=float, default=35.0)
    ap.add_argument("--camera-mode", choices=("fixed", "dynamic"), default="fixed")
    args = ap.parse_args()
    if args.fps <= 0 or args.view_radius_m <= 5:
        raise SystemExit("fps must be positive and view radius >5m")

    method, method_scene_time, method_scene = _load_scenes(args.method_scenes)
    control, control_scene_time, control_scene = _load_scenes(args.control_scenes)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_records = []
    csv_rows = []
    used_stems: set[str] = set()
    use_mp4 = args.format == "mp4" or (args.format == "auto" and shutil.which("ffmpeg") is not None)
    if args.format == "mp4" and shutil.which("ffmpeg") is None:
        raise SystemExit("--format mp4 requested but ffmpeg is unavailable")

    for item in selection.get("selected", []):
        key = str(item["target_key"])
        method_scene_row = _resolve_scene(item, method, method_scene_time, method_scene)
        control_scene_row = _resolve_scene(item, control, control_scene_time, control_scene)
        if method_scene_row is None or control_scene_row is None:
            raise SystemExit(f"selected scene {key} missing from paired traces")
        method_trace = method_scene_row.get("render_trace") or []
        control_trace = control_scene_row.get("render_trace") or []
        if not method_trace or not control_trace:
            raise SystemExit(f"scene {key} has no render_trace; run the selective trace stage with closed_loop.render_trace=true")

        frames = max(len(method_trace), len(control_trace))
        figure = plt.figure(figsize=(12.8, 7.8), dpi=100)
        grid = figure.add_gridspec(2, 2, height_ratios=[4.0, 1.25])
        axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
        timeline_axis = figure.add_subplot(grid[1, :])
        regime = str(selection.get("regime"))
        control_contact, control_contact_label = _contact_marker(control_trace, regime)
        method_contact, method_contact_label = _contact_marker(method_trace, regime)
        fixed_center, fixed_radius = _fixed_shared_view(control_trace, method_trace, args.view_radius_m)

        def update(frame_index):
            if args.camera_mode == "dynamic":
                center, radius = _dynamic_center(control_trace, method_trace, frame_index), args.view_radius_m
            else:
                center, radius = fixed_center, fixed_radius
            _draw_frame(axes[0], control_trace, frame_index, args.control_name, center, radius, control_contact, control_contact_label, args.fps, control_scene_row.get("render_context"))
            _draw_frame(axes[1], method_trace, frame_index, args.method_name, center, radius, method_contact, method_contact_label, args.fps, method_scene_row.get("render_context"))
            _draw_timeline(timeline_axis, control_trace, method_trace, frame_index, frames, args.fps, regime, args.control_name, args.method_name)
            delta = _delta_text(item, regime)
            tier = str(item.get("selection_tier") or "strict_material_improvement")
            profile = str(item.get("evidence_profile") or "unspecified_profile")
            improvements = ", ".join(item.get("material_improvements") or []) or "positive non-regressive score"
            figure.suptitle(
                f"{regime} | rank {item.get('category_rank')} | {profile} | {tier}\n"
                f"{key} | {improvements}\n{delta}",
                fontsize=10.0,
            )
            figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
            return []

        video = animation.FuncAnimation(figure, update, frames=frames, interval=1000 / args.fps, blit=False)
        scene_id = str(method_scene_row.get("scene_id") or "scene")
        time_index = method_scene_row.get("target_time_index")
        key_hash = hashlib.sha1(key.encode()).hexdigest()[:8]
        safe_scene_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in scene_id)
        stem = f"{regime}_{item.get('category_rank', 0)}_{safe_scene_id}_t{time_index}_{key_hash}"
        if stem in used_stems:
            raise SystemExit(f"duplicate output stem for {key}")
        used_stems.add(stem)
        if use_mp4:
            output = args.output_dir / f"{stem}.mp4"
            writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400, extra_args=["-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        else:
            output = args.output_dir / f"{stem}.gif"
            writer = animation.PillowWriter(fps=args.fps)
        video.save(output, writer=writer)
        plt.close(figure)

        record = {
            "target_key": key,
            "source_target_key": item.get("source_target_key"),
            "scene_id": method_scene_row.get("scene_id"),
            "target_time_index": method_scene_row.get("target_time_index"),
            "category": item.get("category"),
            "category_rank": item.get("category_rank"),
            "selection_tier": item.get("selection_tier"),
            "evidence_profile": item.get("evidence_profile"),
            "selection_score": item.get("score"),
            "material_improvements": item.get("material_improvements"),
            "selection_terms": item.get("terms"),
            "video": str(output),
            "num_control_frames": len(control_trace),
            "num_method_frames": len(method_trace),
            "fps": args.fps,
            "camera_mode": args.camera_mode,
            "view_center_xy": [fixed_center[0], fixed_center[1]],
            "view_radius_m": fixed_radius if args.camera_mode == "fixed" else args.view_radius_m,
        }
        index_records.append(record)
        csv_rows.append({key_name: record.get(key_name) for key_name in (
            "target_key", "source_target_key", "scene_id", "target_time_index", "category", "category_rank",
            "selection_tier", "evidence_profile", "selection_score", "video", "num_control_frames",
            "num_method_frames", "fps", "camera_mode", "view_radius_m",
        )})

    document = {
        "event": "critical_scene_recovery_videos_v50",
        "exploratory_qualitative_only": True,
        "paper_population_claim_allowed": False,
        "rendering_scope": "paired fixed-world camera, nearby WOMD roadgraph polylines, oriented agent boxes, ego trails, nearest-agent connector with reported box clearance, observed-overlap/post-contact-start marker, and synchronized clearance timeline; no raster image dependency",
        "method_scenes": str(args.method_scenes),
        "control_scenes": str(args.control_scenes),
        "selection": str(args.selection),
        "method_name": args.method_name,
        "control_name": args.control_name,
        "camera_mode": args.camera_mode,
        "videos": index_records,
    }
    (args.output_dir / "VIDEO_INDEX.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "target_key", "source_target_key", "scene_id", "target_time_index", "category", "category_rank",
        "selection_tier", "evidence_profile", "selection_score", "video", "num_control_frames",
        "num_method_frames", "fps", "camera_mode", "view_radius_m",
    ]
    with (args.output_dir / "VIDEO_INDEX.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"event": document["event"], "num_videos": len(index_records), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
