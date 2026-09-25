#!/usr/bin/env python3
"""Materialize clipped real Contact traces and clip-consistent selection metrics.

Input traces are never altered in place.  Only selected real prefix states are
copied into a separate target-display trace tree.  Contact panel metrics and
paired relative scores are recomputed on exactly the visible state support.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from contact_clip_metrics import clip_scene  # noqa: E402
import select_regime_visualization_scenes as selector  # noqa: E402

CONTACT_METHODS = (
    "postimpact_mpc_lite",
    "post_crash_braking",
    "postimpact_motion_tvlqr",
    "post_collision_restoration",
    "compensatory_postimpact_mpc",
    "robust_postimpact_control",
)


def _scene_key(scene: dict[str, Any], env: dict[str, Any]) -> str:
    key = str(scene.get("target_key") or env.get("resume_key") or "")
    if key.startswith("target:"):
        key = key[len("target:"):]
    if key:
        return key
    sid = str(scene.get("scene_id") or "")
    ti = scene.get("target_time_index")
    return f"{sid}:t{ti}" if sid and ti is not None else sid


def _load_envelopes(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            env = json.loads(line)
            scene = env.get("scene", env)
            if not isinstance(scene, dict):
                continue
            key = _scene_key(scene, env)
            if not key:
                raise SystemExit(f"scene without target key: {path}")
            if key in out:
                raise SystemExit(f"duplicate target {key}: {path}")
            out[key] = env
    return out


def _write_clipped_journal(src: Path, dst: Path, selected: dict[str, float], dt: float) -> dict[str, dict[str, Any]]:
    envs = _load_envelopes(src)
    missing = sorted(set(selected) - set(envs))
    if missing:
        raise SystemExit(f"missing selected targets in {src}: {missing}")
    clipped: dict[str, dict[str, Any]] = {}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for key, clip_s in selected.items():
            env = envs[key]
            scene = env.get("scene", env)
            new_scene = clip_scene(scene, clip_s, dt)
            new_env = dict(env)
            if "scene" in env:
                new_env["scene"] = new_scene
            else:
                new_env = new_scene
            f.write(json.dumps(new_env, ensure_ascii=False, separators=(",", ":")) + "\n")
            clipped[key] = new_scene
    return clipped


def _metric_snapshot(scene: dict[str, Any]) -> dict[str, float | None]:
    return selector._metric_snapshot("contact", scene)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--source-trace-root", type=Path, required=True)
    ap.add_argument("--output-trace-root", type=Path, required=True)
    ap.add_argument("--output-selection", type=Path, required=True)
    ap.add_argument("--metrics-output-json", type=Path, default=None)
    ap.add_argument("--metrics-output-csv", type=Path, default=None)
    args = ap.parse_args()

    sel = json.loads(args.selection.read_text(encoding="utf-8"))
    if str(sel.get("regime") or "") != "contact":
        raise SystemExit("selection must be Contact")
    items = list(sel.get("selected") or [])
    if not items:
        raise SystemExit("selection is empty")
    dt = float(sel.get("metric_dt_s", 0.1) or 0.1)
    selected = {str(x["target_key"]): float(x["clip_duration_s"]) for x in items}

    source_paths = {"ocrap": args.source_trace_root / "ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"}
    source_paths.update({m: args.source_trace_root / f"external/contact/closed_loop_{m}.json.scenes.jsonl" for m in CONTACT_METHODS})
    output_paths = {"ocrap": args.output_trace_root / "ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"}
    output_paths.update({m: args.output_trace_root / f"external/contact/closed_loop_{m}.json.scenes.jsonl" for m in CONTACT_METHODS})

    clipped: dict[str, dict[str, dict[str, Any]]] = {}
    for method in ("ocrap", *CONTACT_METHODS):
        if not source_paths[method].is_file():
            raise SystemExit(f"missing source trace: {source_paths[method]}")
        clipped[method] = _write_clipped_journal(source_paths[method], output_paths[method], selected, dt)

    thresholds = SimpleNamespace(**selector.DEFAULT_THRESHOLDS)
    new_items: list[dict[str, Any]] = []
    for item in items:
        key = str(item["target_key"])
        x = dict(item)
        o_scene = clipped["ocrap"][key]
        ext_scenes = {m: clipped[m][key] for m in CONTACT_METHODS}
        x["ocrap_metrics"] = _metric_snapshot(o_scene)
        x["external_metrics"] = {m: _metric_snapshot(ext_scenes[m]) for m in CONTACT_METHODS}
        # Do not let full-rollout-only optional terms leak into visible-clip
        # scores.  The required Contact geometry/recovery terms and the
        # recomputed yaw/jerk terms remain available in metric_summary.
        def score_scene(scene):
            z = dict(scene)
            z["route_progression"] = None
            z["closed_loop_bounded_NUP"] = None
            return z
        o_score_scene = score_scene(o_scene)
        ext_score_scenes = {m: score_scene(ext_scenes[m]) for m in CONTACT_METHODS}
        x["ocrap_absolute_score"] = float(selector._contact_absolute(o_score_scene))
        ext_abs = {m: float(selector._contact_absolute(ext_score_scenes[m])) for m in CONTACT_METHODS}
        x["external_absolute_scores"] = ext_abs
        x["best_external_method"] = max(ext_abs, key=lambda m: (ext_abs[m], m))
        x["worst_external_method"] = min(ext_abs, key=lambda m: (ext_abs[m], m))
        per: dict[str, Any] = {}
        for m in CONTACT_METHODS:
            ev = selector._evaluate_contact_surrogate(o_score_scene, ext_score_scenes[m], thresholds)
            # The selector evaluates method-control.  Here method=OC-RAP,
            # control=external so positive score means favorable to OC-RAP.
            per[m] = {
                "relative_score": float(ev["score"]),
                "material_improvements": ev.get("material") or [],
                "regression_reasons": ev.get("regressions") or [],
                "missing_required_metrics": ev.get("missing") or [],
                "evidence_profile": ev.get("evidence_profile"),
                "terms": ev.get("terms") or {},
                "external_absolute_score": ext_abs[m],
            }
        x["per_baseline"] = per
        primary = str(x.get("primary_external_method") or "")
        if primary not in per:
            primary = min(CONTACT_METHODS, key=lambda m: (float(per[m]["relative_score"]), m))
            x["primary_external_method"] = primary
        x["evidence_profile"] = per[primary].get("evidence_profile")
        x["num_nonregressive_external_comparisons"] = int(sum(
            not (per[m].get("regression_reasons") or []) and not (per[m].get("missing_required_metrics") or [])
            for m in CONTACT_METHODS
        ))
        x["display_metrics_recomputed_on_visible_clip"] = True
        new_items.append(x)

    out = dict(sel)
    out["event"] = "contact_target_display_materialized_v1"
    out["selected"] = new_items
    out["target_keys"] = [str(x["target_key"]) for x in new_items]
    out["selected_clip_duration_s"] = min(float(x["clip_duration_s"]) for x in new_items)
    out["num_external_baselines"] = len(CONTACT_METHODS)
    out["display_metrics_recomputed_on_visible_clip"] = True
    args.output_selection.parent.mkdir(parents=True, exist_ok=True)
    args.output_selection.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metric_rows: list[dict[str, Any]] = []
    for item in new_items:
        key = str(item["target_key"])
        for method in ("ocrap", *CONTACT_METHODS):
            scene = clipped[method][key]
            ms = dict(scene.get("metric_summary") or {})
            metric_rows.append({
                "target_key": key, "method": method, "clip_duration_s": float(item["clip_duration_s"]),
                "source_criticality_score": float(item.get("source_criticality_score") or 0.0),
                "source_critical_tags": ",".join(item.get("source_critical_tags") or []),
                "overlap_duration_s": ms.get("overlap_duration_s"),
                "penetration_depth_auc_m_s": ms.get("penetration_depth_auc_m_s"),
                "terminal_clearance_m": ms.get("terminal_clearance_m"),
                "recontact_event": ms.get("recontact_event"),
                "secondary_collision_event": ms.get("secondary_collision_event"),
                "post_separation_secondary_collision_event": ms.get("post_separation_secondary_collision_event"),
                "same_partner_recontact_event": ms.get("same_partner_recontact_event"),
                "distinct_collision_partner_count": ms.get("distinct_collision_partner_count"),
                "nearby_agents_peak_12m": ms.get("nearby_agents_peak_12m"),
                "crowded_fraction": ms.get("crowded_fraction"),
                "diagnostic_offroad_fraction": ms.get("diagnostic_offroad_fraction"),
                "reference_trajectory": bool(scene.get("reference_trajectory")),
            })
    if args.metrics_output_json is not None:
        args.metrics_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output_json.write_text(json.dumps({
            "event":"contact_target_display_metrics_v1",
            "reference_visualization_only": bool(sel.get("reference_visualization_only")),
            "note":"Metrics are recomputed on displayed materialized states. OC-RAP rows may describe an aspirational reference trajectory, not an empirical rollout.",
            "rows": metric_rows}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if args.metrics_output_csv is not None:
        args.metrics_output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_output_csv.open("w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()) if metric_rows else ["target_key","method"])
            w.writeheader(); w.writerows(metric_rows)

    reference_mode = bool(sel.get("reference_visualization_only")) or any(
        bool(clipped["ocrap"][str(x["target_key"])].get("reference_trajectory"))
        or str(clipped["ocrap"][str(x["target_key"])].get("method") or "") == "ocrap_reference"
        for x in items
    )
    provenance = {
        "event": "contact_target_display_provenance_v2",
        "trajectory_states_modified": bool(reference_mode),
        "reference_visualization_only": bool(reference_mode),
        "empirical_ocrap_relabelled": False,
        "source_trace_root": str(args.source_trace_root),
        "materialized_trace_root": str(args.output_trace_root),
        "selection_source": str(args.selection),
        "selection_materialized": str(args.output_selection),
        "metric_support": "visible materialized state support t0..tN; durations use t0..t(N-1); box geometry metrics recomputed on displayed states",
        "note": (
            "Separate aspirational/reference visualization artifact; generated SDC states are not an empirical OC-RAP rollout."
            if reference_mode else
            "Separate target/template visualization artifact using empirical states. Do not substitute for the reported empirical full-rollout results."
        ),
    }
    (args.output_selection.parent / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": out["event"], "num_scenes": len(new_items), "output": str(args.output_selection)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
