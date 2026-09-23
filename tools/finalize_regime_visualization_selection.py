#!/usr/bin/env python3
"""Trace-aware final selection for reviewer-facing regime visualizations.

The metric-only selector deliberately runs first on the locked population.  This
second pass sees only the over-selected qualitative candidate pool and its
selective render traces.  It does *not* change the quantitative cohort or any
reported population metric.

Goals:
  * Safe: never display an OC-RAP overlap/off-road tail. Prefer candidates that
    stay clean for the complete requested clip; if necessary, a late violation
    may be removed by shortening the displayed clip, but never below the
    configured minimum.
  * Near-Contact: prioritize scenes where OC-RAP remains clean while a large
    fraction of external baselines collide or enter a low-margin state. Among
    similarly strong consensus-failure cases, prefer denser local traffic.
  * Contact: preserve the metric selector's ordering while attaching trace
    quality diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

METHODS = {
    "safe": ["gameformer_lite", "plantf", "pluto", "pdm_closed", "pdm_hybrid", "idm", "diffusion_planner"],
    "near": ["marc_lite", "racp_lite", "robust_scenario_mpc", "predictive_safety_filter", "dr_cvar_safety_filter", "conformal_predictive_safety_filter", "flow_planner", "plan_r1", "betopnet"],
    "contact": ["postimpact_mpc_lite", "post_crash_braking", "postimpact_motion_tvlqr", "post_collision_restoration", "compensatory_postimpact_mpc", "robust_postimpact_control"],
}


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
    # Renderer samples [0, clip) at video cadence. Keep the quality gate aligned
    # with what is actually visible instead of rejecting an undisplayed endpoint.
    n = max(1, int(math.floor(clip_s / dt_s + 1.0e-9)))
    return trace[: min(n, len(trace))]


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


def _load_traces(trace_root: Path, regime: str) -> dict[str, dict[str, dict[str, Any]]]:
    paths = {"ocrap": trace_root / "ocrap" / regime / "closed_loop_ocrap.json.scenes.jsonl"}
    paths.update({m: trace_root / "external" / regime / f"closed_loop_{m}.json.scenes.jsonl" for m in METHODS[regime]})
    return {m: _load_journal(p) for m, p in paths.items()}


def _near_quality(item: dict[str, Any], traces: dict[str, dict[str, dict[str, Any]]], *, dt_s: float,
                  ttc_threshold_s: float, clearance_threshold_m: float, density_radius_m: float) -> dict[str, Any]:
    key = str(item["target_key"])
    clip = float(item.get("clip_duration_s") or 0.0)
    otrace = list(traces["ocrap"][key].get("render_trace") or [])
    oc_overlap = _any_flag(otrace, clip, dt_s, "overlap")
    oc_offroad = _any_flag(otrace, clip, dt_s, "offroad")
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
    primary = str(item.get("primary_external_method") or "")
    primary_severe = bool(external.get(primary, {}).get("severe_low_margin"))
    return {
        "ocrap_overlap_visible": oc_overlap,
        "ocrap_offroad_visible": oc_offroad,
        "ocrap_visible_safe": not oc_overlap and not oc_offroad,
        "external_overlap_count": overlap_count,
        "external_severe_count": severe_count,
        "num_external_baselines": n,
        "external_overlap_fraction": overlap_frac,
        "external_severe_fraction": frac,
        "primary_external_severe": primary_severe,
        "visual_evidence_rank": evidence_rank,
        "visual_evidence_label": evidence,
        "local_density_radius_m": density_radius_m,
        "local_agent_density": density,
        "external_trace_hazard": external,
    }


def _safe_quality(item: dict[str, Any], traces: dict[str, dict[str, dict[str, Any]]], *, dt_s: float,
                  min_clip_s: float, margin_s: float) -> dict[str, Any]:
    key = str(item["target_key"])
    requested = float(item.get("clip_duration_s") or 0.0)
    trace = list(traces["ocrap"][key].get("render_trace") or [])
    idx, reason = _first_violation(trace, requested, dt_s)
    if idx is None:
        effective = requested
        status = "full_clip_clean"
    else:
        margin_steps = max(1, int(math.ceil(margin_s / dt_s - 1e-9)))
        last_safe_index = max(0, idx - margin_steps)
        effective = min(requested, last_safe_index * dt_s)
        effective = math.floor((effective + 1e-9) / dt_s) * dt_s
        status = "late_tail_truncated" if effective + 1e-9 >= min_clip_s else "reject_early_violation"
    return {
        "requested_clip_duration_s": requested,
        "effective_clip_duration_s": effective,
        "full_requested_clip_clean": idx is None,
        "first_visible_violation_index": idx,
        "first_visible_violation_s": None if idx is None else idx * dt_s,
        "first_visible_violation_type": reason,
        "safe_tail_gate": status,
        "accepted": status != "reject_early_violation",
    }


def _contact_quality(item: dict[str, Any], traces: dict[str, dict[str, dict[str, Any]]], *, dt_s: float) -> dict[str, Any]:
    key = str(item["target_key"])
    clip = float(item.get("clip_duration_s") or 0.0)
    tr = list(traces["ocrap"][key].get("render_trace") or [])
    first_overlap = next((i for i, f in enumerate(_visible_frames(tr, clip, dt_s)) if _flag(f, "overlap")), None)
    return {
        "ocrap_offroad_visible": _any_flag(tr, clip, dt_s, "offroad"),
        "ocrap_first_overlap_s": None if first_overlap is None else first_overlap * dt_s,
        "ocrap_terminal_clearance_m": _metric(_visible_frames(tr, clip, dt_s)[-1], "min_clearance_m") if tr else None,
    }


def _copy_final_doc(candidate: dict[str, Any], selected: list[dict[str, Any]], *, candidate_path: Path, regime: str) -> dict[str, Any]:
    doc = dict(candidate)
    doc["event"] = "regime_visualization_scene_selection_trace_final_v1"
    doc["candidate_selection"] = str(candidate_path)
    doc["candidate_pool_size"] = len(candidate.get("selected") or [])
    doc["requested_num_scenes"] = len(selected)
    doc["selected"] = selected
    doc["target_keys"] = [str(x["target_key"]) for x in selected]
    doc["trace_aware_finalization"] = True
    doc["selection_note"] = str(candidate.get("selection_note") or "") + (
        " A trace-aware qualitative-only finalization then removes visible OC-RAP safety failures; "
        "Near-Contact additionally prioritizes consensus external low-margin/collision evidence and local traffic density."
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
    ap.add_argument("--safe-min-clip-s", type=float, default=4.0)
    ap.add_argument("--safe-tail-margin-s", type=float, default=0.2)
    ap.add_argument("--near-ttc-threshold-s", type=float, default=0.5)
    ap.add_argument("--near-clearance-threshold-m", type=float, default=0.35)
    ap.add_argument("--near-density-radius-m", type=float, default=25.0)
    args = ap.parse_args()
    if args.num_scenes <= 0:
        raise SystemExit("--num-scenes must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"event": "trace_aware_visualization_selection_index_v1", "regimes": {}}

    for regime in ("safe", "near", "contact"):
        cpath = args.candidate_selection_root / f"{regime}_selection.json"
        candidate = json.loads(cpath.read_text(encoding="utf-8"))
        items = list(candidate.get("selected") or [])
        traces = _load_traces(args.trace_root, regime)
        dt = float(candidate.get("metric_dt_s", 0.1) or 0.1)
        annotated: list[dict[str, Any]] = []
        for original_order, item in enumerate(items):
            key = str(item["target_key"])
            missing = [m for m, rows in traces.items() if key not in rows]
            if missing:
                raise SystemExit(f"{regime}/{key}: missing candidate traces for {missing}")
            row = dict(item)
            if regime == "safe":
                q = _safe_quality(row, traces, dt_s=dt, min_clip_s=args.safe_min_clip_s, margin_s=args.safe_tail_margin_s)
                row["visualization_trace_quality"] = q
                if q["accepted"]:
                    row["clip_duration_s"] = float(q["effective_clip_duration_s"])
                    row["clip_duration_adjusted_after_trace"] = not bool(q["full_requested_clip_clean"])
                    annotated.append(row | {"_original_order": original_order})
            elif regime == "near":
                q = _near_quality(row, traces, dt_s=dt, ttc_threshold_s=args.near_ttc_threshold_s,
                                  clearance_threshold_m=args.near_clearance_threshold_m, density_radius_m=args.near_density_radius_m)
                row["visualization_trace_quality"] = q
                if q["ocrap_visible_safe"]:
                    annotated.append(row | {"_original_order": original_order})
            else:
                q = _contact_quality(row, traces, dt_s=dt)
                row["visualization_trace_quality"] = q
                annotated.append(row | {"_original_order": original_order})

        if regime == "safe":
            # Prefer candidates requiring no truncation, then the original metric
            # tier/score.  Late-tail truncation is a fallback rather than the norm.
            annotated.sort(key=lambda r: (
                0 if r["visualization_trace_quality"]["full_requested_clip_clean"] else 1,
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
            annotated.sort(key=lambda r: int(r.get("_original_order", 0)))

        # Preserve scenario diversity after trace-aware re-ranking.
        chosen: list[dict[str, Any]] = []
        used_scenes: set[str] = set()
        for row in annotated:
            sid = str(row.get("scene_id") or row["target_key"])
            if sid in used_scenes:
                continue
            chosen.append(row)
            used_scenes.add(sid)
            if len(chosen) >= args.num_scenes:
                break
        if len(chosen) < args.num_scenes:
            raise SystemExit(
                f"{regime}: trace-aware gate retained only {len(chosen)} distinct scenes from {len(items)} candidates; "
                "increase VIS_CANDIDATE_MULTIPLIER or relax the explicit visualization thresholds."
            )
        final: list[dict[str, Any]] = []
        for rank, row in enumerate(chosen, 1):
            row = {k: v for k, v in row.items() if k != "_original_order"}
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
    print(json.dumps({"event": summary["event"], "index": str(index), "num_scenes_per_regime": args.num_scenes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
