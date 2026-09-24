#!/usr/bin/env python3
"""Render event-aligned static small-multiple figures from selective trace reruns.

Outputs per selected target:
  * main pair figure: OC-RAP vs the pre-recorded primary external comparator,
    2 rows x 4 synchronized keyframes;
  * appendix overview: OC-RAP + all audited external baselines,
    N rows x 3 synchronized keyframes.

The script never re-selects scenes.  It consumes the exact target-locked selection
artifact and selective traces used by the video renderer.  Keyframe times are
chosen symmetrically from all methods displayed in the figure (not from OC-RAP
alone): start, pre-critical, critical, end for pair figures; start, critical, end
for all-method figures.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from render_regime_visualization_videos import (
    _all_model_fixed_view,
    _contact_marker,
    _display_name,
    _draw_frame,
    _prepare_roadgraph_segments,
    _load_scenes,
    _metric_float,
    _parse_trace_specs,
    _resolve_scene,
)


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _frame(trace: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    return trace[min(max(idx, 0), len(trace) - 1)]


def _critical_index(traces: dict[str, list[dict[str, Any]]], regime: str, max_idx: int) -> int:
    # Contact: actual simulator overlap has priority; this never treats Contact
    # bucket membership itself as proof that impact occurred.
    if regime == "contact":
        for idx in range(max_idx + 1):
            if any((_metric_float(_frame(t, idx), "overlap") or 0.0) > 0.5 for t in traces.values()):
                return idx

    best_idx, best_value = 0, float("inf")
    metric = "ttc_s" if regime == "near" else "min_clearance_m"
    for idx in range(max_idx + 1):
        vals = []
        for trace in traces.values():
            v = _finite(_metric_float(_frame(trace, idx), metric))
            if v is not None:
                vals.append(v)
        if vals and min(vals) < best_value:
            best_idx, best_value = idx, min(vals)
    if best_value < float("inf"):
        return best_idx

    # Fallback if TTC is absent/sentinel-heavy.
    if metric != "min_clearance_m":
        for idx in range(max_idx + 1):
            vals = [_finite(_metric_float(_frame(t, idx), "min_clearance_m")) for t in traces.values()]
            vals = [v for v in vals if v is not None]
            if vals and min(vals) < best_value:
                best_idx, best_value = idx, min(vals)
    return best_idx


def _keyframes(traces: dict[str, list[dict[str, Any]]], regime: str, clip_duration_s: float, dt_s: float, count: int) -> list[int]:
    max_by_clip = max(0, int(math.floor(clip_duration_s / dt_s + 1e-9)) - 1)
    max_by_trace = max(max(len(t) for t in traces.values()) - 1, 0)
    max_idx = min(max_by_clip, max_by_trace)
    crit = _critical_index(traces, regime, max_idx)
    if count == 3:
        raw = [0, crit, max_idx]
    else:
        pre = max(0, crit - max(1, int(round(1.0 / dt_s))))
        raw = [0, pre, crit, max_idx]

    # Ensure monotonically ordered distinct columns.  Degenerate very-short clips
    # are filled by evenly spaced indices without changing the shared time basis.
    result = sorted(set(raw))
    if len(result) < count and max_idx > 0:
        for j in range(count):
            result.append(int(round(j * max_idx / max(count - 1, 1))))
            result = sorted(set(result))
            if len(result) >= count:
                break
    while len(result) < count:
        result.append(result[-1] if result else 0)
    return result[:count]


def _figure_outputs_complete(output_stem: Path) -> bool:
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    if not png.is_file() or not pdf.is_file() or png.stat().st_size < 4096 or pdf.stat().st_size < 4096:
        return False
    try:
        from PIL import Image
        with Image.open(png) as im:
            im.verify()
        raw = pdf.read_bytes()
        return raw.startswith(b"%PDF") and b"%%EOF" in raw[-2048:]
    except Exception:
        return False


def _render_grid(*, methods: list[str], traces: dict[str, list[dict[str, Any]]], displays: dict[str, str],
                 regime: str, context: dict[str, Any], keyframes: list[int], dt_s: float,
                 minimum_radius: float, title: str, output_stem: Path, force: bool = False) -> list[str]:
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    if not force and _figure_outputs_complete(output_stem):
        print(f"[FIG][REUSE] {png} + {pdf}", flush=True)
        return [str(png), str(pdf)]
    started = time.monotonic()
    print(f"[FIG][START] {regime} {output_stem.name} rows={len(methods)} cols={len(keyframes)}", flush=True)
    center, radius = _all_model_fixed_view({m: traces[m] for m in methods}, minimum_radius)
    road_segments = _prepare_roadgraph_segments(context, center, radius)
    contacts = {m: _contact_marker(traces[m], regime) for m in methods}
    rows, cols = len(methods), len(keyframes)
    width = 3.05 * cols
    # Reserve an explicit header band above the top-row time labels.  The old
    # 2x4 layout used suptitle y=0.995 with top=0.93 and bbox_inches="tight",
    # which visibly merged titles such as "CONTACT: OC-RAP vs APF + TVLQR"
    # with the t=... labels.
    is_main_pair = rows == 2 and cols == 4
    height = max(3.0, 1.90 * rows + (1.25 if is_main_pair else 0.95))
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)

    for r, method in enumerate(methods):
        for c, idx in enumerate(keyframes):
            xy, label, event_idx = contacts[method]
            _draw_frame(
                axes[r][c], traces[method], idx, "", center, radius, xy, label, dt_s, context, event_idx,
                show_hud=False, show_axes=False, show_clearance_annotation=False,
                roadgraph_segments=road_segments,
            )
            if r == 0:
                axes[r][c].set_title(f"t = {idx * dt_s:.1f} s", fontsize=9.5, fontweight="bold", pad=8.0)
            if c == 0:
                axes[r][c].text(
                    -0.045, 0.5, displays[method], transform=axes[r][c].transAxes,
                    rotation=90, ha="right", va="center", fontsize=9.2, fontweight="bold",
                )

    fig.suptitle(title, fontsize=11.0, fontweight="bold", y=0.985)
    if regime in {"near", "contact"}:
        # The blue X is a history marker after the first observed overlap, not
        # a claim that the vehicles are still overlapping in the displayed
        # frame. Make that semantics explicit in static reviewer-facing figures.
        fig.text(
            0.5, 0.010,
            "× = first observed-overlap location (history marker); red SDC = overlap at the displayed frame",
            ha="center", va="bottom", fontsize=7.2,
        )
    fig.subplots_adjust(
        left=0.065, right=0.995, top=(0.855 if is_main_pair else 0.925),
        bottom=(0.045 if regime in {"near", "contact"} else 0.025), wspace=0.035, hspace=0.08,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    print(f"[FIG][SAVE] {png}", flush=True)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"[FIG][SAVE] {pdf}", flush=True)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    elapsed = time.monotonic() - started
    print(f"[FIG][DONE] {output_stem.name} elapsed={elapsed:.1f}s png={png.stat().st_size/(1024*1024):.1f}MiB pdf={pdf.stat().st_size/(1024*1024):.1f}MiB", flush=True)
    return [str(png), str(pdf)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", action="append", default=[], metavar="METHOD=SCENES.jsonl")
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--view-radius-m", type=float, default=35.0)
    ap.add_argument("--force", action="store_true", help="Re-render figures even when existing PNG/PDF outputs validate.")
    args = ap.parse_args()

    paths = _parse_trace_specs(args.trace)
    print(f"[FIG][LOAD] loading {len(paths)} method journals", flush=True)
    load_started = time.monotonic()
    loaded = {m: _load_scenes(p) for m, p in paths.items()}
    print(f"[FIG][LOAD] done elapsed={time.monotonic()-load_started:.1f}s", flush=True)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    regime = str(selection.get("regime") or "")
    if regime not in {"safe", "near", "contact"}:
        raise SystemExit(f"invalid regime: {regime!r}")
    expected = set(selection.get("external_baselines") or [])
    if set(paths) - {"ocrap"} != expected:
        raise SystemExit("trace baseline set does not match selection artifact")
    dt_s = float(selection.get("metric_dt_s", 0.1) or 0.1)
    records = []

    selected_items = selection.get("selected") or []
    for scene_no, item in enumerate(selected_items, 1):
        scene_started = time.monotonic()
        print(f"[FIG][SCENE] regime={regime} scene={scene_no}/{len(selected_items)} target={item.get('target_key')}", flush=True)
        resolved = {}
        for method in paths:
            scene, how = _resolve_scene(item, loaded[method])
            if scene is None:
                raise SystemExit(f"cannot resolve {item.get('target_key')} for {method}")
            trace = list(scene.get("render_trace") or [])
            if not trace:
                raise SystemExit(f"missing render_trace for {item.get('target_key')} / {method}")
            resolved[method] = (scene, trace, how)
        traces = {m: resolved[m][1] for m in paths}
        context = resolved["ocrap"][0].get("render_context") or next(
            (resolved[m][0].get("render_context") for m in paths if resolved[m][0].get("render_context")), {}
        )
        displays = {m: _display_name(m) for m in paths}
        primary = str(item.get("primary_external_method") or item.get("hardest_external_method") or item.get("best_external_method") or "")
        if primary not in paths:
            raise SystemExit(f"primary comparator missing for {item.get('target_key')}: {primary}")
        rank = int(item.get("category_rank", len(records) + 1))
        clip = float(item.get("clip_duration_s", selection.get("selected_clip_duration_s", 5.0)))
        pair_methods = ["ocrap", primary]
        pair_kf = _keyframes({m: traces[m] for m in pair_methods}, regime, clip, dt_s, 4)
        all_methods = ["ocrap"] + list(selection.get("external_baselines") or [])
        all_kf = _keyframes({m: traces[m] for m in all_methods}, regime, clip, dt_s, 3)
        scene_dir = args.output_dir / regime / f"rank_{rank:02d}"
        regime_title = "NEAR-CONTACT" if regime == "near" else ("CONTACT" if regime == "contact" else "SAFE")
        pair_files = _render_grid(
            methods=pair_methods, traces=traces, displays=displays, regime=regime, context=context,
            keyframes=pair_kf, dt_s=dt_s, minimum_radius=args.view_radius_m,
            title=f"{regime_title}: OC-RAP vs {displays[primary]}",
            output_stem=scene_dir / f"{regime}__rank_{rank:02d}__paper_pair_2x4",
            force=args.force,
        )
        all_files = _render_grid(
            methods=all_methods, traces=traces, displays=displays, regime=regime, context=context,
            keyframes=all_kf, dt_s=dt_s, minimum_radius=args.view_radius_m,
            title=f"{regime_title}: target-locked all-method comparison",
            output_stem=scene_dir / f"{regime}__rank_{rank:02d}__appendix_all_{len(all_methods)}x3",
            force=args.force,
        )
        print(f"[FIG][SCENE-DONE] regime={regime} scene={scene_no}/{len(selected_items)} elapsed={time.monotonic()-scene_started:.1f}s", flush=True)
        records.append({
            "target_key": item.get("target_key"), "rank": rank, "primary_external_method": primary,
            "pair_keyframe_indices": pair_kf, "all_method_keyframe_indices": all_kf,
            "metric_dt_s": dt_s, "pair_files": pair_files, "all_method_files": all_files,
        })

    index = {
        "event": "regime_paper_figure_index_v124",
        "regime": regime,
        "selection": str(args.selection),
        "selection_is_posthoc_qualitative": bool(selection.get("exploratory_qualitative_only", True)),
        "main_figure_policy": "2x4 pair: rows=methods, columns=start/pre-critical/critical/end; critical time chosen symmetrically from displayed methods",
        "appendix_policy": "Nx3 all-method small multiples: rows=OC-RAP+all audited baselines, columns=start/critical/end",
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{regime.upper()}_PAPER_FIGURE_INDEX.json"
    out.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": index["event"], "regime": regime, "num_scenes": len(records), "index": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
