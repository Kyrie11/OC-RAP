#!/usr/bin/env python3
"""Render paired closed-loop agent-box traces as side-by-side MP4/GIF videos."""
from __future__ import annotations

import argparse
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
            e = json.loads(line); s = e.get("scene", e)
            key = str(s.get("target_key") or s.get("scene_id") or e.get("resume_key"))
            out[key] = s
            scene_id = str(s.get("scene_id") or "")
            time_index = str(s.get("target_time_index") if s.get("target_time_index") is not None else "")
            if scene_id:
                by_scene.setdefault(scene_id, []).append(s)
                if time_index:
                    by_scene_time[(scene_id, time_index)] = s
    return out, by_scene_time, by_scene


def _bounds(*traces: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for trace in traces:
        for frame in trace:
            for a in frame.get("agents", []):
                xs.append(float(a["x"])); ys.append(float(a["y"]))
    if not xs:
        return -20.0, 20.0, -20.0, 20.0
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    span = max(xmax-xmin, ymax-ymin, 20.0)
    cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
    pad = 0.15 * span
    return cx-span/2-pad, cx+span/2+pad, cy-span/2-pad, cy+span/2+pad


def _draw_frame(ax, frame: dict[str, Any], title: str, bounds) -> None:
    ax.clear(); ax.set_aspect("equal", adjustable="box"); ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    ax.set_title(title); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    for agent in frame.get("agents", []):
        x, y = float(agent["x"]), float(agent["y"])
        length, width, yaw = max(float(agent["length"]), .1), max(float(agent["width"]), .1), float(agent["yaw"])
        rect = patches.Rectangle((x-length/2, y-width/2), length, width, fill=False, linewidth=2.0 if agent.get("is_sdc") else 0.8)
        rect.set_transform(transforms.Affine2D().rotate_around(x, y, yaw) + ax.transData)
        ax.add_patch(rect)
        if agent.get("is_sdc"):
            ax.text(x, y, "SDC", fontsize=8, ha="center", va="center")
    m = frame.get("metrics", {}) or {}
    lines = [
        f"t={frame.get('time_index')} macro={frame.get('selected_macro','')}",
        f"candidate={frame.get('selected_candidate_index')} reason={frame.get('selection_reason','')}",
    ]
    for key, label in (("ttc_s","TTC"),("min_clearance_m","clearance"),("overlap","overlap"),("offroad","offroad")):
        if key in m and math.isfinite(float(m[key])):
            lines.append(f"{label}={float(m[key]):.3f}")
    ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8,
            bbox={"boxstyle":"round", "facecolor":"white", "alpha":0.75})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-scenes", type=Path, required=True)
    ap.add_argument("--control-scenes", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    method, method_st, method_s = _load_scenes(args.method_scenes)
    control, control_st, control_s = _load_scenes(args.control_scenes)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for item in selection.get("selected", []):
        key = str(item["target_key"])
        method_scene = method.get(key)
        control_scene = control.get(key)
        if method_scene is None or control_scene is None:
            scene_id = str(item.get("scene_id") or "")
            time_index = str(item.get("target_time_index") if item.get("target_time_index") is not None else "")
            if scene_id and time_index:
                method_scene = method_st.get((scene_id, time_index))
                control_scene = control_st.get((scene_id, time_index))
            if (method_scene is None or control_scene is None) and scene_id:
                mm = method_s.get(scene_id, [])
                cc = control_s.get(scene_id, [])
                if len(mm) == 1 and len(cc) == 1:
                    method_scene, control_scene = mm[0], cc[0]
        if method_scene is None or control_scene is None:
            raise SystemExit(f"selected scene {key} missing from paired traces")
        mt = method_scene.get("render_trace") or []
        ct = control_scene.get("render_trace") or []
        if not mt or not ct:
            raise SystemExit(f"scene {key} has no render_trace; rerun closed-loop with closed_loop.render_trace=true")
        frames = max(len(mt), len(ct)); bounds = _bounds(mt, ct)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        def update(i: int):
            _draw_frame(axes[0], ct[min(i, len(ct)-1)], "Control", bounds)
            _draw_frame(axes[1], mt[min(i, len(mt)-1)], "OC-RAP", bounds)
            fig.suptitle(f"{selection.get('regime')} | {item.get('category')} | {key}")
            return []
        ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000/max(args.fps,1), blit=False)
        stem = f"{selection.get('regime')}_{item.get('category_rank',0)}_{item.get('category')}_{method_scene.get('scene_id','scene')}"
        if shutil.which("ffmpeg"):
            out = args.output_dir / f"{stem}.mp4"
            ani.save(out, writer=animation.FFMpegWriter(fps=max(args.fps,1), bitrate=2400))
        else:
            out = args.output_dir / f"{stem}.gif"
            ani.save(out, writer=animation.PillowWriter(fps=max(args.fps,1)))
        plt.close(fig)
        index.append({"target_key":key,"scene_id":method_scene.get("scene_id"),"category":item.get("category"),"video":str(out)})
    index_doc={"event":"v48_34_critical_scene_videos","exploratory_only":True,"paper_claim_allowed":False,"videos":index}
    (args.output_dir/"VIDEO_INDEX.json").write_text(json.dumps(index_doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"event":index_doc["event"],"num_videos":len(index),"output_dir":str(args.output_dir)}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
