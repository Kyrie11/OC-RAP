#!/usr/bin/env python3
"""Render paired OC-RAP critical-scene diagnostics from closed-loop JSON files."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return doc


def _key(scene: dict[str, Any]) -> str:
    return str(scene.get("target_key") or f"{scene.get('bucket_name','')}|{scene.get('scene_id','')}|{scene.get('target_time_index','')}")


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _metric(scene: dict[str, Any], name: str) -> float | None:
    value = scene.get(name) if name in scene else (scene.get("metric_summary") or {}).get(name)
    return _finite(value)


def _trace(scene: dict[str, Any]) -> list[dict[str, float]]:
    raw = scene.get("metric_trace")
    if isinstance(raw, list) and raw:
        return [dict(x) for x in raw if isinstance(x, dict)]
    return [dict(d.get("metrics_after_step") or {}) for d in scene.get("decisions", [])]


def _series(scene: dict[str, Any], name: str) -> np.ndarray:
    vals = []
    for row in _trace(scene):
        value = _finite(row.get(name))
        vals.append(np.nan if value is None else value)
    return np.asarray(vals, dtype=float)


def _dt(scene: dict[str, Any]) -> float:
    return _finite(scene.get("trace_dt_s")) or 0.1


def _xy(scene: dict[str, Any]) -> np.ndarray | None:
    raw = scene.get("state_xy_trace")
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    arr = arr[:, :2]
    return arr[np.all(np.isfinite(arr), axis=1)]


def _slug(text: str, limit: int = 100) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return (cleaned or "scene")[:limit]


def _intervention_steps(scene: dict[str, Any]) -> list[int]:
    return [i for i, d in enumerate(scene.get("decisions", [])) if int(d.get("selected_candidate_index", 0) or 0) != 0]


def _macro_trace(scene: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    names = [str(d.get("selected_macro", "unknown")) for d in scene.get("decisions", [])]
    unique = ["nominal"] + sorted({x for x in names if x != "nominal"})
    mapping = {name: i for i, name in enumerate(unique)}
    return np.asarray([mapping.get(x, 0) for x in names], dtype=float), unique


def _add_intervention_markers(axes: list[Any], method_scene: dict[str, Any], dt: float) -> None:
    for step in _intervention_steps(method_scene):
        x = step * dt
        for ax in axes:
            ax.axvline(x, alpha=0.18, linewidth=1.0)


def _derived_series(scene: dict[str, Any], name: str, dt: float) -> np.ndarray:
    """Return a stored trace or a physically interpretable finite-difference trace."""
    if name not in {"ego_accel_mps2", "ego_jerk_mps3", "ego_yaw_rate_radps"}:
        return _series(scene, name)
    if name in {"ego_accel_mps2", "ego_jerk_mps3"}:
        speed = _series(scene, "ego_speed_mps")
        if speed.size == 0:
            return speed
        accel = np.gradient(speed, max(dt, 1.0e-6))
        return accel if name == "ego_accel_mps2" else np.gradient(accel, max(dt, 1.0e-6))
    yaw = _series(scene, "ego_yaw_rad")
    if yaw.size == 0:
        return yaw
    return np.gradient(np.unwrap(yaw), max(dt, 1.0e-6))


def _add_contact_anchor(axes: list[Any], scene: dict[str, Any], dt: float) -> None:
    anchor = _metric(scene, "contact_anchor_step")
    if anchor is None:
        return
    x = max(0.0, anchor * dt)
    for ax in axes:
        ax.axvline(x, linestyle=":", linewidth=1.2, alpha=0.65)


def _plot_scene_timeseries(control: dict[str, Any], method: dict[str, Any], regime: str, output: Path, title: str, dpi: int) -> None:
    import matplotlib.pyplot as plt

    dt = min(_dt(control), _dt(method))
    if regime == "near_contact":
        metric_rows = [
            ("min_clearance_m", "Minimum clearance (m)", 2.0, False),
            ("ttc_s", "TTC (s)", 3.0, False),
            ("ego_speed_mps", "Ego speed (m/s)", None, False),
            ("ego_accel_mps2", "Longitudinal acceleration proxy (m/s²)", 0.0, False),
        ]
    else:
        metric_rows = [
            ("min_clearance_m", "Post-contact clearance (m)", 0.5, False),
            ("overlap", "Overlap/re-contact flag", None, False),
            ("ego_speed_mps", "Ego speed (m/s)", 0.5, False),
            ("ego_yaw_rate_radps", "Absolute yaw rate (rad/s)", 0.25, True),
            ("ego_accel_mps2", "Absolute acceleration proxy (m/s²)", None, True),
        ]

    n_axes = len(metric_rows) + 1
    fig, axes = plt.subplots(n_axes, 1, figsize=(11, 2.6 * n_axes + 1.0), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (metric, label, threshold, absolute) in zip(axes[:-1], metric_rows):
        c = _derived_series(control, metric, dt)
        m = _derived_series(method, metric, dt)
        if absolute:
            c = np.abs(c)
            m = np.abs(m)
        tc = np.arange(c.size) * dt
        tm = np.arange(m.size) * dt
        ax.plot(tc, c, label="control")
        ax.plot(tm, m, label="OC-RAP")
        if threshold is not None:
            ax.axhline(threshold, linestyle="--", linewidth=1.0, alpha=0.7, label=f"reference={threshold:g}")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    macro, macro_names = _macro_trace(method)
    t_macro = np.arange(macro.size) * dt
    axes[-1].step(t_macro, macro, where="post", label="OC-RAP selected macro")
    axes[-1].set_yticks(range(len(macro_names)), labels=macro_names)
    axes[-1].set_ylabel("Selected macro")
    axes[-1].set_xlabel("Closed-loop time after target start (s)")
    axes[-1].grid(alpha=0.25)
    axes[-1].legend(loc="best")
    _add_intervention_markers(list(axes), method, dt)
    if regime == "contact":
        _add_contact_anchor(list(axes), control, dt)
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory(control: dict[str, Any], method: dict[str, Any], output: Path, title: str, dpi: int) -> bool:
    import matplotlib.pyplot as plt

    c = _xy(control)
    m = _xy(method)
    if c is None or m is None or c.size == 0 or m.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.plot(c[:, 0], c[:, 1], marker=".", label="control")
    ax.plot(m[:, 0], m[:, 1], marker=".", label="OC-RAP")
    ax.scatter([c[0, 0], m[0, 0]], [c[0, 1], m[0, 1]], marker="o", label="start")
    ax.scatter([c[-1, 0], m[-1, 0]], [c[-1, 1], m[-1, 1]], marker="x", label="end")
    ax.set_xlabel("Global x (m)")
    ax.set_ylabel("Global y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(title + " — ego trajectory")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_aggregate(control_scenes: dict[str, dict[str, Any]], method_scenes: dict[str, dict[str, Any]], selected: list[dict[str, Any]], regime: str, output_dir: Path, dpi: int) -> list[str]:
    import matplotlib.pyplot as plt

    if regime == "near_contact":
        metrics = [
            ("min_clearance_m_min", "Minimum clearance (m)", "higher"),
            ("ttc_s_min", "Minimum TTC (s)", "higher"),
            ("clearance_deficit_auc_m_s", "Clearance deficit AUC", "lower"),
            ("ttc_deficit_auc_s2", "TTC deficit AUC", "lower"),
        ]
    else:
        metrics = [
            ("overlap_duration_s", "Overlap duration (s)", "lower"),
            ("recontact_event", "Recontact event", "lower"),
            ("post_contact_free_space_auc_m_s", "Post-contact free-space AUC", "higher"),
            ("post_contact_clearance_m_mean", "Post-contact mean clearance (m)", "higher"),
        ]
    paths: list[str] = []
    for metric, label, direction in metrics:
        pairs = []
        for key in sorted(set(control_scenes) & set(method_scenes)):
            c = _metric(control_scenes[key], metric)
            m = _metric(method_scenes[key], metric)
            if c is not None and m is not None:
                pairs.append((c, m))
        if not pairs:
            continue
        arr = np.asarray(pairs, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.scatter(arr[:, 0], arr[:, 1], alpha=0.65)
        lo = float(np.nanmin(arr)); hi = float(np.nanmax(arr))
        if math.isclose(lo, hi):
            lo -= 0.5; hi += 0.5
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, label="no change")
        ax.set_xlabel("Control")
        ax.set_ylabel("OC-RAP")
        ax.set_title(f"Paired scene scatter: {label}\n({direction} is better)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        path = output_dir / f"aggregate_scatter__{_slug(metric)}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path.name)

    if selected:
        labels = [_slug(str(row.get("scene_id") or row.get("key")), 26) for row in selected]
        values = [float(row.get("improvement_score", 0.0)) for row in selected]
        fig, ax = plt.subplots(figsize=(max(9, len(values) * 0.55), 5), constrained_layout=True)
        ax.bar(np.arange(len(values)), values)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xticks(np.arange(len(values)), labels=labels, rotation=55, ha="right")
        ax.set_ylabel("Signed physical improvement score")
        ax.set_title("Selected critical scenes: positive=improvement, negative=regression")
        ax.grid(axis="y", alpha=0.25)
        path = output_dir / "selected_scene_signed_scores.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path.name)
    return paths


def _write_scene_summary(path: Path, entry: dict[str, Any], trajectory_available: bool) -> None:
    lines = [
        f"# Critical scene: {entry.get('scene_id')}",
        "",
        f"- Category: `{entry.get('first_selected_category', '')}`",
        f"- Target: `{entry.get('target_key') or entry.get('key')}`",
        f"- Method intervention rate: `{entry.get('method_intervention_rate', 0):.6f}`",
        f"- Method macros: `{', '.join(entry.get('method_macros') or [])}`",
        f"- Improvement score: `{entry.get('improvement_score', 0):+.4f}`",
        f"- Criticality score: `{entry.get('criticality_score', 0):.4f}`",
        f"- Reason: {entry.get('reason', '')}",
        f"- Ego trajectory rendered: `{trajectory_available}`",
        "",
        "## Key deltas (OC-RAP - control)",
        "",
        "| Metric | Control | OC-RAP | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric, delta in sorted((entry.get("deltas_method_minus_control") or {}).items()):
        if delta is None:
            continue
        c = (entry.get("control_metrics") or {}).get(metric)
        m = (entry.get("method_metrics") or {}).get(metric)
        lines.append(f"| {metric} | {c:.6g} | {m:.6g} | {delta:+.6g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize paired critical closed-loop scenes selected by select_critical_closed_loop_scenes.py")
    ap.add_argument("control", type=Path)
    ap.add_argument("method", type=Path)
    ap.add_argument("critical", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-scenes", type=int, default=24)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    control_doc = _load(args.control)
    method_doc = _load(args.method)
    critical_doc = _load(args.critical)
    regime = str(critical_doc.get("regime") or "near_contact")
    control_scenes = {_key(s): s for s in control_doc.get("scenes", [])}
    method_scenes = {_key(s): s for s in method_doc.get("scenes", [])}
    selected = list(critical_doc.get("selected") or [])[: max(0, args.max_scenes)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[dict[str, Any]] = []
    for rank, entry in enumerate(selected, start=1):
        key = str(entry.get("key"))
        control = control_scenes.get(key)
        method = method_scenes.get(key)
        if control is None or method is None:
            continue
        scene_dir = args.output_dir / f"{rank:02d}__{_slug(str(entry.get('scene_id') or key))}__t{entry.get('target_time_index')}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        title = f"{regime}: {entry.get('scene_id')} @ t={entry.get('target_time_index')} | score={entry.get('improvement_score', 0):+.3f}"
        _plot_scene_timeseries(control, method, regime, scene_dir / "paired_timeseries.png", title, args.dpi)
        trajectory_available = _plot_trajectory(control, method, scene_dir / "ego_trajectory.png", title, args.dpi)
        (scene_dir / "critical_entry.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        _write_scene_summary(scene_dir / "README.md", entry, trajectory_available)
        rendered.append({
            "rank": rank,
            "key": key,
            "scene_id": entry.get("scene_id"),
            "category": entry.get("first_selected_category"),
            "score": entry.get("improvement_score"),
            "reason": entry.get("reason"),
            "directory": scene_dir.name,
            "trajectory_available": trajectory_available,
        })

    aggregate_paths = _plot_aggregate(control_scenes, method_scenes, selected, regime, args.output_dir, args.dpi)
    with (args.output_dir / "rendered_scenes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "key", "scene_id", "category", "score", "reason", "directory", "trajectory_available"])
        writer.writeheader(); writer.writerows(rendered)

    rows = []
    for item in rendered:
        d = html.escape(str(item["directory"]))
        rows.append(
            "<tr>"
            f"<td>{item['rank']}</td><td>{html.escape(str(item['category']))}</td>"
            f"<td><code>{html.escape(str(item['scene_id']))}</code></td><td>{float(item['score']):+.4f}</td>"
            f"<td>{html.escape(str(item['reason']))}</td>"
            f"<td><a href='{d}/paired_timeseries.png'>timeseries</a> | <a href='{d}/README.md'>metrics</a>"
            + (f" | <a href='{d}/ego_trajectory.png'>trajectory</a>" if item["trajectory_available"] else "")
            + "</td></tr>"
        )
    aggregate_html = " ".join(f"<a href='{html.escape(p)}'>{html.escape(p)}</a>" for p in aggregate_paths)
    page = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>OC-RAP critical closed-loop scenes</title>
<style>body{{font-family:Arial,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.45rem;vertical-align:top}}th{{background:#eee}}code{{font-size:.9em}}</style></head>
<body><h1>OC-RAP critical closed-loop scenes — {html.escape(regime)}</h1>
<p><strong>Exploratory only:</strong> the v48.33 deployment gate did not pass. The report intentionally includes both improvements and regressions.</p>
<p>Aggregate plots: {aggregate_html}</p>
<table><thead><tr><th>#</th><th>Category</th><th>Scene</th><th>Score</th><th>Reason</th><th>Files</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    (args.output_dir / "VISUALIZATION_COMPLETE.json").write_text(json.dumps({
        "regime": regime,
        "control": str(args.control),
        "method": str(args.method),
        "critical": str(args.critical),
        "rendered_scenes": len(rendered),
        "aggregate_plots": aggregate_paths,
        "exploratory_only": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "rendered": len(rendered), "index": str(args.output_dir / 'index.html')}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
