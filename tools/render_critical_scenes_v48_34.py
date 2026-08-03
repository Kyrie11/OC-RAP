#!/usr/bin/env python3
"""Render paired closed-loop agent-box traces as side-by-side MP4/GIF videos."""
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


def _load_scenes(path: Path) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if path.suffix == ".json" and not path.name.endswith(".scenes.jsonl"):
        alt = Path(str(path) + ".scenes.jsonl")
        if alt.is_file():
            path = alt
    out: dict[str, dict[str, Any]] = {}
    by_scene_time: dict[tuple[str, str], dict[str, Any]] = {}
    by_scene: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            s = e.get("scene", e)
            key = str(s.get("target_key") or e.get("resume_key") or "")
            scene_id = str(s.get("scene_id") or "")
            time_value = s.get("target_time_index")
            time_index = str(time_value if time_value is not None else "")
            if not key:
                key = f"{scene_id}:t{time_index}" if scene_id and time_index else scene_id
            if not key:
                raise ValueError(f"scene without target key in {path}")
            if key in out:
                raise ValueError(f"duplicate target key {key} in {path}")
            out[key] = s
            if scene_id:
                by_scene.setdefault(scene_id, []).append(s)
                if time_index:
                    by_scene_time[(scene_id, time_index)] = s
    return out, by_scene_time, by_scene


def _bounds(*traces: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for trace in traces:
        for frame in trace:
            for a in frame.get("agents", []):
                try:
                    xs.append(float(a["x"])); ys.append(float(a["y"]))
                except Exception:
                    continue
    if not xs:
        return -20.0, 20.0, -20.0, 20.0
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, 20.0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    pad = 0.15 * span
    return cx - span / 2 - pad, cx + span / 2 + pad, cy - span / 2 - pad, cy + span / 2 + pad


def _sdc_xy(frame: dict[str, Any]) -> tuple[float, float] | None:
    for agent in frame.get("agents", []):
        if agent.get("is_sdc"):
            try:
                return float(agent["x"]), float(agent["y"])
            except Exception:
                return None
    return None


def _draw_frame(ax, trace: list[dict[str, Any]], i: int, title: str, bounds: tuple[float, float, float, float]) -> None:
    frame = trace[min(i, len(trace) - 1)]
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    ax.set_title(title); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    trail = [_sdc_xy(x) for x in trace[: min(i, len(trace) - 1) + 1]]
    trail = [x for x in trail if x is not None]
    if len(trail) >= 2:
        ax.plot([x[0] for x in trail], [x[1] for x in trail], linewidth=1.5, alpha=0.8)
    for agent in frame.get("agents", []):
        try:
            x, y = float(agent["x"]), float(agent["y"])
            length = max(float(agent["length"]), 0.1)
            width = max(float(agent["width"]), 0.1)
            yaw = float(agent["yaw"])
        except Exception:
            continue
        is_sdc = bool(agent.get("is_sdc"))
        rect = patches.Rectangle((x - length / 2, y - width / 2), length, width, fill=False, linewidth=2.0 if is_sdc else 0.8)
        rect.set_transform(transforms.Affine2D().rotate_around(x, y, yaw) + ax.transData)
        ax.add_patch(rect)
        if is_sdc:
            ax.text(x, y, "SDC", fontsize=8, ha="center", va="center")
    m = frame.get("metrics", {}) or {}
    lines = [
        f"t={frame.get('time_index')} macro={frame.get('selected_macro', '')}",
        f"candidate={frame.get('selected_candidate_index')} reason={frame.get('selection_reason', '')}",
    ]
    for key, label in (("ttc_s", "TTC"), ("min_clearance_m", "clearance"), ("overlap", "overlap"), ("offroad", "offroad")):
        try:
            value = float(m[key])
        except Exception:
            continue
        if math.isfinite(value):
            lines.append(f"{label}={value:.3f}")
    ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75})


def _resolve_scene(item: dict[str, Any], direct, by_scene_time, by_scene) -> dict[str, Any] | None:
    key = str(item["target_key"])
    scene = direct.get(key)
    if scene is not None:
        return scene
    scene_id = str(item.get("scene_id") or "")
    time_index = str(item.get("target_time_index") if item.get("target_time_index") is not None else "")
    if scene_id and time_index:
        scene = by_scene_time.get((scene_id, time_index))
    if scene is None and scene_id:
        candidates = by_scene.get(scene_id, [])
        if len(candidates) == 1:
            scene = candidates[0]
    return scene


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-scenes", type=Path, required=True)
    ap.add_argument("--control-scenes", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--format", choices=("auto", "mp4", "gif"), default="auto")
    args = ap.parse_args()
    if args.fps <= 0:
        raise SystemExit("fps must be positive")

    method, method_st, method_s = _load_scenes(args.method_scenes)
    control, control_st, control_s = _load_scenes(args.control_scenes)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    used_stems: set[str] = set()

    use_mp4 = args.format == "mp4" or (args.format == "auto" and shutil.which("ffmpeg") is not None)
    if args.format == "mp4" and shutil.which("ffmpeg") is None:
        raise SystemExit("--format mp4 requested but ffmpeg is unavailable")

    for item in selection.get("selected", []):
        key = str(item["target_key"])
        method_scene = _resolve_scene(item, method, method_st, method_s)
        control_scene = _resolve_scene(item, control, control_st, control_s)
        if method_scene is None or control_scene is None:
            raise SystemExit(f"selected scene {key} missing from paired traces")
        mt = method_scene.get("render_trace") or []
        ct = control_scene.get("render_trace") or []
        if not mt or not ct:
            raise SystemExit(f"scene {key} has no render_trace; rerun closed-loop with closed_loop.render_trace=true")
        frames = max(len(mt), len(ct)); bounds = _bounds(mt, ct)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

        def update(i: int):
            _draw_frame(axes[0], ct, i, "Scalar control", bounds)
            _draw_frame(axes[1], mt, i, "OC-RAP", bounds)
            fig.suptitle(f"{selection.get('regime')} | {item.get('category')} | {key}")
            return []

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
        scene_id = str(method_scene.get("scene_id") or "scene")
        target_time = method_scene.get("target_time_index")
        key_hash = hashlib.sha1(key.encode()).hexdigest()[:8]
        stem = f"{selection.get('regime')}_{item.get('category_rank', 0)}_{item.get('category')}_{scene_id}_t{target_time}_{key_hash}"
        if stem in used_stems:
            raise SystemExit(f"duplicate output stem for {key}")
        used_stems.add(stem)
        if use_mp4:
            out = args.output_dir / f"{stem}.mp4"
            writer = animation.FFMpegWriter(fps=args.fps, bitrate=2200, extra_args=["-preset", "veryfast", "-pix_fmt", "yuv420p"])
        else:
            out = args.output_dir / f"{stem}.gif"
            writer = animation.PillowWriter(fps=args.fps)
        ani.save(out, writer=writer)
        plt.close(fig)
        record = {
            "target_key": key,
            "scene_id": method_scene.get("scene_id"),
            "target_time_index": method_scene.get("target_time_index"),
            "category": item.get("category"),
            "category_rank": item.get("category_rank"),
            "selection_score": item.get("score"),
            "selection_terms": item.get("terms"),
            "video": str(out),
            "num_control_frames": len(ct),
            "num_method_frames": len(mt),
            "fps": args.fps,
        }
        index.append(record)
        csv_rows.append({k: record.get(k) for k in ("target_key", "scene_id", "target_time_index", "category", "category_rank", "selection_score", "video", "num_control_frames", "num_method_frames", "fps")})

    index_doc = {
        "event": "v48_34_1_critical_scene_videos",
        "version": "v48.34.1-RC30-MODEL-CONTRACT-HOTFIX",
        "exploratory_only": True,
        "paper_claim_allowed": False,
        "method_scenes": str(args.method_scenes),
        "control_scenes": str(args.control_scenes),
        "selection": str(args.selection),
        "videos": index,
    }
    (args.output_dir / "VIDEO_INDEX.json").write_text(json.dumps(index_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "VIDEO_INDEX.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["target_key", "scene_id", "target_time_index", "category", "category_rank", "selection_score", "video", "num_control_frames", "num_method_frames", "fps"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(csv_rows)
    print(json.dumps({"event": index_doc["event"], "num_videos": len(index), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
