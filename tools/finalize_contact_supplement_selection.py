#!/usr/bin/env python3
"""Keep every reviewer-safe Contact supplement scene from an expanded exact-a0 pool.

This is intentionally qualitative-only.  It never changes the frozen publication
Contact cohort or population tables.  Candidate scenes must first pass the
metric-tier filter in select_regime_visualization_scenes.py; this pass then
requires OC-RAP controlled recovery from full render traces and at least a
configurable amount of external-baseline recovery-failure evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import finalize_regime_visualization_selection as base  # noqa: E402


def _main_keys(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    rows = d.get("selected") if isinstance(d, dict) else []
    return {str(x.get("target_key") or "") for x in (rows or []) if isinstance(x, dict) and x.get("target_key")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-selection", type=Path, required=True)
    ap.add_argument("--trace-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--audit-output", type=Path, required=True)
    ap.add_argument("--exclude-selection", type=Path, default=None,
                    help="Optional main reviewer-safe Contact selection; accepted keys there are not duplicated in the supplement.")
    ap.add_argument("--min-external-recovery-failures", type=int, default=1)
    ap.add_argument("--min-clip-duration-s", type=float, default=2.5)
    ap.add_argument("--max-clip-duration-s", type=float, default=4.0)
    ap.add_argument("--lane-terminal-max-m", type=float, default=5.0)
    ap.add_argument("--lane-p90-max-m", type=float, default=5.5)
    ap.add_argument("--lane-offcenter-threshold-m", type=float, default=4.5)
    ap.add_argument("--lane-offcenter-fraction-max", type=float, default=0.30)
    ap.add_argument("--lane-heading-terminal-max-deg", type=float, default=50.0)
    ap.add_argument("--lane-heading-p90-max-deg", type=float, default=55.0)
    ap.add_argument("--lane-heading-speed-gate-mps", type=float, default=1.0)
    ap.add_argument("--lane-min-evidence-fraction", type=float, default=0.50)
    ap.add_argument("--lane-recovery-terminal-max-m", type=float, default=6.5)
    ap.add_argument("--lane-recovery-p90-max-m", type=float, default=7.0)
    ap.add_argument("--lane-recovery-offcenter-fraction-max", type=float, default=0.45)
    ap.add_argument("--lane-peak-improvement-min-m", type=float, default=1.0)
    ap.add_argument("--lane-recent-recovery-min-m", type=float, default=0.35)
    ap.add_argument("--separation-clearance-m", type=float, default=0.50)
    ap.add_argument("--separation-hold-s", type=float, default=0.30)
    args = ap.parse_args()

    if args.min_external_recovery_failures < 0:
        raise SystemExit("--min-external-recovery-failures must be >= 0")
    if not (0 < args.min_clip_duration_s <= args.max_clip_duration_s):
        raise SystemExit("require 0 < min clip duration <= max clip duration")

    candidate = json.loads(args.candidate_selection.read_text(encoding="utf-8"))
    if str(candidate.get("regime") or "") != "contact":
        raise SystemExit("candidate selection must be Contact")
    items = list(candidate.get("selected") or [])
    traces = base._load_traces(args.trace_root, "contact")
    dt = float(candidate.get("metric_dt_s", 0.1) or 0.1)
    lane_kwargs: dict[str, Any] = {
        "lane_types": base.DEFAULT_VEHICLE_LANE_TYPES,
        "terminal_max_m": float(args.lane_terminal_max_m),
        "p90_max_m": float(args.lane_p90_max_m),
        "offcenter_threshold_m": float(args.lane_offcenter_threshold_m),
        "offcenter_fraction_max": float(args.lane_offcenter_fraction_max),
        "heading_terminal_max_deg": float(args.lane_heading_terminal_max_deg),
        "heading_p90_max_deg": float(args.lane_heading_p90_max_deg),
        "heading_speed_gate_mps": float(args.lane_heading_speed_gate_mps),
        "min_evidence_fraction": float(args.lane_min_evidence_fraction),
    }
    lane_recovery_kwargs = {
        "terminal_max_m": float(args.lane_recovery_terminal_max_m),
        "p90_max_m": float(args.lane_recovery_p90_max_m),
        "offcenter_fraction_max": float(args.lane_recovery_offcenter_fraction_max),
        "peak_to_terminal_improvement_min_m": float(args.lane_peak_improvement_min_m),
        "recent_recovery_delta_min_m": float(args.lane_recent_recovery_min_m),
    }
    excluded = _main_keys(args.exclude_selection)
    accepted: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for item in items:
        row = dict(item)
        key = str(row["target_key"])
        available = row.get("available_future_s")
        try:
            available_s = float(available)
        except Exception:
            available_s = 0.0
        clip = min(float(args.max_clip_duration_s), available_s)
        if clip + 1e-9 < float(args.min_clip_duration_s):
            audit_rows.append({"target_key": key, "accepted": False, "rejection_reasons": ["insufficient_post_contact_horizon"], "available_future_s": available_s})
            continue
        row["clip_duration_s"] = clip
        missing = [m for m, scenes in traces.items() if key not in scenes]
        if missing:
            raise SystemExit(f"{key}: missing traces for {missing}")
        q = base._contact_quality(
            row, traces, dt_s=dt, lane_kwargs=lane_kwargs,
            lane_recovery_kwargs=lane_recovery_kwargs,
            separation_clearance_m=float(args.separation_clearance_m),
            separation_hold_s=float(args.separation_hold_s),
        )
        reasons = base._gate_rejection_reasons("contact", q, near_min_external_severe_count=0)
        if int(q.get("external_recovery_failure_count") or 0) < int(args.min_external_recovery_failures):
            reasons.append("insufficient_external_recovery_failure_evidence")
        if key in excluded:
            reasons.append("already_in_main_reviewer_safe_selection")
        ok = bool(q.get("ocrap_controlled_recovery")) and not reasons
        if ok:
            trace_primary = str(q.get("trace_primary_external_method") or "")
            if trace_primary:
                row["primary_external_method"] = trace_primary
                row["primary_comparator_reason"] = (
                    "hardest paired external among methods that fail the same trace-level controlled-recovery gate"
                )
            row["visualization_trace_quality"] = q
            accepted.append(row)
        audit_rows.append({
            "target_key": key,
            "candidate_rank": item.get("category_rank"),
            "selection_tier_rank": item.get("selection_tier_rank"),
            "selection_tier": item.get("selection_tier"),
            "clip_duration_s": clip,
            "accepted": ok,
            "rejection_reasons": reasons,
            "quality": q,
        })

    accepted.sort(key=lambda r: (
        int(r["visualization_trace_quality"]["visual_evidence_rank"]),
        -int(r["visualization_trace_quality"]["external_recovery_failure_count"]),
        float(r["visualization_trace_quality"]["ocrap_trace_recovery"]["sustained_separation"].get("first_s")
              if r["visualization_trace_quality"]["ocrap_trace_recovery"]["sustained_separation"].get("first_s") is not None else 1e6),
        -min(3.0, float(r["visualization_trace_quality"].get("ocrap_terminal_clearance_m") or 0.0)),
        int(r.get("selection_tier_rank", 99)),
        -float(r.get("score") or 0.0),
        str(r["target_key"]),
    ))
    final: list[dict[str, Any]] = []
    used_scenes: set[str] = set()
    for row in accepted:
        sid = str(row.get("scene_id") or row["target_key"])
        if sid in used_scenes:
            continue
        used_scenes.add(sid)
        clean = dict(row)
        clean["category_rank"] = len(final) + 1
        final.append(clean)

    out_doc = dict(candidate)
    out_doc.update({
        "event": "contact_qualitative_supplement_selection_v1",
        "exploratory_qualitative_only": True,
        "paper_population_claim_allowed": False,
        "selection_note": (
            "Expanded exact-a0 Contact pool for qualitative supplement only. Every retained scene passes the same "
            "controlled-recovery/off-road/re-contact/lane-realism gate used for reviewer-facing visualization, has "
            "metric-stage tier <= the configured supplement cap, and is not part of the main reviewer-safe Contact selection."
        ),
        "supplement_keep_all_accepted": True,
        "supplement_min_external_recovery_failures": int(args.min_external_recovery_failures),
        "selected": final,
        "target_keys": [r["target_key"] for r in final],
        "requested_num_scenes": None,
        "selected_clip_duration_s": None,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from collections import Counter
    counts = Counter(reason for x in audit_rows if not x.get("accepted") for reason in x.get("rejection_reasons", []))
    audit = {
        "event": "contact_qualitative_supplement_audit_v1",
        "candidate_count": len(items),
        "accepted_count": len(final),
        "excluded_main_count": sum("already_in_main_reviewer_safe_selection" in (x.get("rejection_reasons") or []) for x in audit_rows),
        "rejection_reason_counts": dict(sorted(counts.items())),
        "min_external_recovery_failures": int(args.min_external_recovery_failures),
        "clip_duration_range_s": [float(args.min_clip_duration_s), float(args.max_clip_duration_s)],
        "candidates": audit_rows,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": out_doc["event"], "accepted": len(final), "candidates": len(items),
        "output": str(args.output), "audit": str(args.audit_output),
    }))
    if not final:
        raise SystemExit("no new Contact supplement scene passed the strict qualitative gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
