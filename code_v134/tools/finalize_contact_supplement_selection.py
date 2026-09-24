#!/usr/bin/env python3
"""Keep reviewer-safe Contact supplement scenes from an expanded exact-a0 pool.

This tool is qualitative-only: it never changes the frozen publication Contact
cohort or population tables. OC-RAP must always pass the hard reality contract
(no visible off-road, no terminal overlap, no re-contact after sustained
separation, and lane-realistic recovery). Comparative evidence can be either:

1. failure evidence: at least one external method fails that same controlled-
   recovery contract; or
2. temporal-majority evidence: an external method may also recover, but OC-RAP
   is better on clearance for most of the visible post-contact trace and shows
   a material recovery advantage in terminal clearance, separation time, or
   overlap duration.

The second mode intentionally allows short intervals where OC-RAP is not the
best method, while retaining hard physical/road constraints.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import finalize_regime_visualization_selection as base  # noqa: E402


def _selection_keys(paths: list[Path] | None) -> set[str]:
    out: set[str] = set()
    for path in paths or []:
        if path is None or not path.is_file():
            continue
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = d.get("selected") if isinstance(d, dict) else []
        out.update(
            str(x.get("target_key") or "")
            for x in (rows or [])
            if isinstance(x, dict) and x.get("target_key")
        )
    return out


def _finite(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _temporal_pair_evidence(
    o_scene: dict[str, Any], e_scene: dict[str, Any], *, clip_s: float, dt_s: float,
    o_quality: dict[str, Any], e_quality: dict[str, Any],
    win_margin_m: float, noninferior_margin_m: float,
    min_win_fraction: float, min_noninferior_fraction: float,
    min_mean_clearance_gain_m: float, min_terminal_gain_m: float,
    min_separation_lead_s: float, min_overlap_reduction_s: float,
    min_valid_frames: int,
) -> dict[str, Any]:
    """Trace-level majority evidence for one OC-RAP/external pair.

    The exact-a0 initial state is common to both methods and carries no policy
    evidence, so comparison starts from the first post-action render state.
    """
    oframes = base._visible_frames(list(o_scene.get("render_trace") or []), clip_s, dt_s)
    eframes = base._visible_frames(list(e_scene.get("render_trace") or []), clip_s, dt_s)
    n = min(len(oframes), len(eframes))
    diffs: list[float] = []
    overlap_o = overlap_e = 0
    for i in range(1, n):
        of, ef = oframes[i], eframes[i]
        overlap_o += int(base._flag(of, "overlap"))
        overlap_e += int(base._flag(ef, "overlap"))
        oc = base._metric(of, "min_clearance_m")
        ec = base._metric(ef, "min_clearance_m")
        if oc is not None and ec is not None:
            diffs.append(float(oc - ec))

    valid = len(diffs)
    win_fraction = (
        sum(d >= float(win_margin_m) for d in diffs) / valid if valid else 0.0
    )
    noninferior_fraction = (
        sum(d >= -float(noninferior_margin_m) for d in diffs) / valid if valid else 0.0
    )
    mean_gain = float(statistics.mean(diffs)) if diffs else None
    median_gain = float(statistics.median(diffs)) if diffs else None
    terminal_gain = diffs[-1] if diffs else None
    overlap_reduction_s = float((overlap_e - overlap_o) * dt_s)

    o_sep = ((o_quality.get("sustained_separation") or {}).get("first_s"))
    e_sep = ((e_quality.get("sustained_separation") or {}).get("first_s"))
    o_sep_f = _finite(o_sep)
    e_sep_f = _finite(e_sep)
    separation_lead_s: float | None
    if o_sep_f is None:
        separation_lead_s = None
    elif e_sep_f is None:
        separation_lead_s = max(0.0, float(clip_s) - o_sep_f)
    else:
        separation_lead_s = float(e_sep_f - o_sep_f)

    material_terms: list[str] = []
    if terminal_gain is not None and terminal_gain >= float(min_terminal_gain_m):
        material_terms.append("terminal_clearance_gain")
    if separation_lead_s is not None and separation_lead_s >= float(min_separation_lead_s):
        material_terms.append("earlier_sustained_separation")
    if overlap_reduction_s >= float(min_overlap_reduction_s):
        material_terms.append("shorter_overlap_duration")

    temporal_majority = bool(
        valid >= int(min_valid_frames)
        and win_fraction >= float(min_win_fraction)
        and noninferior_fraction >= float(min_noninferior_fraction)
        and mean_gain is not None and mean_gain >= float(min_mean_clearance_gain_m)
        and material_terms
    )
    return {
        "valid_frames": valid,
        "compared_post_action_frames": max(0, n - 1),
        "clearance_win_margin_m": float(win_margin_m),
        "clearance_noninferior_margin_m": float(noninferior_margin_m),
        "clearance_win_fraction": float(win_fraction),
        "clearance_noninferior_fraction": float(noninferior_fraction),
        "clearance_mean_gain_m": mean_gain,
        "clearance_median_gain_m": median_gain,
        "terminal_clearance_gain_m": terminal_gain,
        "overlap_duration_reduction_s": overlap_reduction_s,
        "sustained_separation_lead_s": separation_lead_s,
        "material_advantages": material_terms,
        "temporal_majority_advantage": temporal_majority,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-selection", type=Path, required=True)
    ap.add_argument("--trace-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--audit-output", type=Path, required=True)
    ap.add_argument("--exclude-selection", type=Path, action="append", default=[],
                    help="Selection whose target keys must not be duplicated; repeatable.")
    ap.add_argument("--comparative-mode", choices=("failure_only", "failure_or_temporal"), default="failure_or_temporal")
    ap.add_argument("--min-external-recovery-failures", type=int, default=0,
                    help="Hard minimum external controlled-recovery failures. Keep 0 for temporal-majority discovery.")
    ap.add_argument("--min-comparative-evidence-methods", type=int, default=1,
                    help="Minimum external methods with either recovery failure or temporal-majority advantage.")
    ap.add_argument("--min-clip-duration-s", type=float, default=2.5)
    ap.add_argument("--max-clip-duration-s", type=float, default=4.0)

    # Temporal-majority comparison defaults intentionally permit some losing
    # frames while requiring a majority win plus broad non-inferiority.
    ap.add_argument("--temporal-clearance-win-margin-m", type=float, default=0.10)
    ap.add_argument("--temporal-clearance-noninferior-margin-m", type=float, default=0.10)
    ap.add_argument("--temporal-min-win-fraction", type=float, default=0.55)
    ap.add_argument("--temporal-min-noninferior-fraction", type=float, default=0.75)
    ap.add_argument("--temporal-min-mean-clearance-gain-m", type=float, default=0.10)
    ap.add_argument("--temporal-min-terminal-gain-m", type=float, default=0.30)
    ap.add_argument("--temporal-min-separation-lead-s", type=float, default=0.20)
    ap.add_argument("--temporal-min-overlap-reduction-s", type=float, default=0.20)
    ap.add_argument("--temporal-min-valid-frames", type=int, default=8)

    # Hard realism / recovery contract. These remain identical to the strict
    # reviewer-facing Contact gate unless explicitly documented otherwise.
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

    if args.min_external_recovery_failures < 0 or args.min_comparative_evidence_methods < 0:
        raise SystemExit("comparative evidence counts must be >= 0")
    if not (0 < args.min_clip_duration_s <= args.max_clip_duration_s):
        raise SystemExit("require 0 < min clip duration <= max clip duration")
    for name in ("temporal_min_win_fraction", "temporal_min_noninferior_fraction"):
        v = float(getattr(args, name))
        if not 0.0 <= v <= 1.0:
            raise SystemExit(f"--{name.replace('_','-')} must be in [0,1]")

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
    excluded = _selection_keys(args.exclude_selection)
    accepted: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for item in items:
        row = dict(item)
        key = str(row["target_key"])
        available_s = _finite(row.get("available_future_s")) or 0.0
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
        ocq = q.get("ocrap_trace_recovery") or {}
        temporal: dict[str, dict[str, Any]] = {}
        evidence_methods: list[str] = []
        temporal_methods: list[str] = []
        failure_methods: list[str] = []
        for method in base.METHODS["contact"]:
            eq = (q.get("external_trace_recovery") or {}).get(method) or {}
            if eq.get("recovery_failure"):
                failure_methods.append(method)
            tev = _temporal_pair_evidence(
                traces["ocrap"][key], traces[method][key], clip_s=clip, dt_s=dt,
                o_quality=ocq, e_quality=eq,
                win_margin_m=float(args.temporal_clearance_win_margin_m),
                noninferior_margin_m=float(args.temporal_clearance_noninferior_margin_m),
                min_win_fraction=float(args.temporal_min_win_fraction),
                min_noninferior_fraction=float(args.temporal_min_noninferior_fraction),
                min_mean_clearance_gain_m=float(args.temporal_min_mean_clearance_gain_m),
                min_terminal_gain_m=float(args.temporal_min_terminal_gain_m),
                min_separation_lead_s=float(args.temporal_min_separation_lead_s),
                min_overlap_reduction_s=float(args.temporal_min_overlap_reduction_s),
                min_valid_frames=int(args.temporal_min_valid_frames),
            )
            temporal[method] = tev
            if tev["temporal_majority_advantage"]:
                temporal_methods.append(method)
            if eq.get("recovery_failure") or (
                args.comparative_mode == "failure_or_temporal" and tev["temporal_majority_advantage"]
            ):
                evidence_methods.append(method)

        if len(failure_methods) < int(args.min_external_recovery_failures):
            reasons.append("insufficient_external_recovery_failure_evidence")
        if len(evidence_methods) < int(args.min_comparative_evidence_methods):
            reasons.append("insufficient_external_comparative_evidence")
        if key in excluded:
            reasons.append("already_in_excluded_selection")

        # Preserve order / uniqueness in reasons.
        reasons = list(dict.fromkeys(reasons))
        ok = bool(q.get("ocrap_controlled_recovery")) and not reasons
        q["external_temporal_advantage"] = temporal
        q["external_temporal_advantage_methods"] = temporal_methods
        q["external_comparative_evidence_methods"] = evidence_methods
        q["external_comparative_evidence_count"] = len(evidence_methods)
        q["comparative_mode"] = args.comparative_mode

        if ok:
            trace_primary = base._hardest_among(row, evidence_methods) or str(row.get("primary_external_method") or "")
            if trace_primary:
                row["primary_external_method"] = trace_primary
                if trace_primary in failure_methods:
                    reason = "hardest paired external among methods that fail the strict trace-level controlled-recovery gate"
                else:
                    reason = "hardest paired external among methods for which OC-RAP has majority trace-level post-contact advantage"
                row["primary_comparator_reason"] = reason
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
            "failure_evidence_methods": failure_methods,
            "temporal_majority_methods": temporal_methods,
            "comparative_evidence_methods": evidence_methods,
            "quality": q,
        })

    def best_temporal_strength(row: dict[str, Any]) -> float:
        vals = []
        for ev in (row["visualization_trace_quality"].get("external_temporal_advantage") or {}).values():
            if ev.get("temporal_majority_advantage"):
                vals.append(float(ev.get("clearance_win_fraction") or 0.0))
        return max(vals) if vals else 0.0

    accepted.sort(key=lambda r: (
        int(r["visualization_trace_quality"]["visual_evidence_rank"]),
        -int(r["visualization_trace_quality"].get("external_recovery_failure_count") or 0),
        -int(r["visualization_trace_quality"].get("external_comparative_evidence_count") or 0),
        -best_temporal_strength(r),
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
        "event": "contact_qualitative_supplement_selection_v2_temporal_majority",
        "exploratory_qualitative_only": True,
        "paper_population_claim_allowed": False,
        "selection_note": (
            "Expanded exact-a0 Contact pool for qualitative supplement only. Every retained scene passes hard OC-RAP "
            "controlled-recovery/off-road/re-contact/lane-realism constraints. Comparative evidence may be either an "
            "external controlled-recovery failure or a majority-of-time post-contact clearance advantage with a material "
            "terminal/separation/overlap benefit. Temporary non-optimal OC-RAP frames are explicitly permitted."
        ),
        "supplement_keep_all_accepted": True,
        "supplement_comparative_mode": args.comparative_mode,
        "supplement_min_external_recovery_failures": int(args.min_external_recovery_failures),
        "supplement_min_comparative_evidence_methods": int(args.min_comparative_evidence_methods),
        "temporal_advantage_thresholds": {
            "clearance_win_margin_m": float(args.temporal_clearance_win_margin_m),
            "clearance_noninferior_margin_m": float(args.temporal_clearance_noninferior_margin_m),
            "min_win_fraction": float(args.temporal_min_win_fraction),
            "min_noninferior_fraction": float(args.temporal_min_noninferior_fraction),
            "min_mean_clearance_gain_m": float(args.temporal_min_mean_clearance_gain_m),
            "min_terminal_gain_m": float(args.temporal_min_terminal_gain_m),
            "min_separation_lead_s": float(args.temporal_min_separation_lead_s),
            "min_overlap_reduction_s": float(args.temporal_min_overlap_reduction_s),
            "min_valid_frames": int(args.temporal_min_valid_frames),
        },
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
        "event": "contact_qualitative_supplement_audit_v2_temporal_majority",
        "candidate_count": len(items),
        "accepted_count": len(final),
        "excluded_count": sum("already_in_excluded_selection" in (x.get("rejection_reasons") or []) for x in audit_rows),
        "rejection_reason_counts": dict(sorted(counts.items())),
        "comparative_mode": args.comparative_mode,
        "min_external_recovery_failures": int(args.min_external_recovery_failures),
        "min_comparative_evidence_methods": int(args.min_comparative_evidence_methods),
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
        raise SystemExit("no new Contact supplement scene passed the temporal-majority qualitative gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
