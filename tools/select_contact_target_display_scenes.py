#!/usr/bin/env python3
"""Select five realistic Contact target-display scenes from existing real traces.

The selector never edits trajectories.  It may shorten the visible prefix of a
real closed-loop trace, choosing the longest prefix that satisfies the same
hard physical realism contract used by the reviewer-safe Contact gate.  This is
intended for a separate target/template visualization tree and must not replace
reported empirical results.
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
import finalize_contact_supplement_selection as supp  # noqa: E402


def _keys_from_selection(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    return [str(x.get("target_key")) for x in (d.get("selected") or []) if isinstance(x, dict) and x.get("target_key")]


def _finite(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _evaluate(
    item: dict[str, Any], traces: dict[str, dict[str, dict[str, Any]]], *, clip: float, dt: float,
    lane_kwargs: dict[str, Any], lane_recovery_kwargs: dict[str, Any],
    separation_clearance_m: float, separation_hold_s: float,
    min_terminal_clearance_m: float, min_post_separation_clearance_m: float,
    reject_any_recontact_after_first_separation: bool,
    min_comparative_methods: int,
    temporal_win_margin_m: float, temporal_noninferior_margin_m: float,
    temporal_min_win_fraction: float, temporal_min_noninferior_fraction: float,
    temporal_min_mean_gain_m: float, temporal_min_terminal_gain_m: float,
    temporal_min_separation_lead_s: float, temporal_min_overlap_reduction_s: float,
    temporal_min_valid_frames: int,
) -> tuple[bool, dict[str, Any], list[str], list[str], list[str], list[str]]:
    row = dict(item)
    row["clip_duration_s"] = float(clip)
    q = base._contact_quality(
        row, traces, dt_s=dt, lane_kwargs=lane_kwargs,
        lane_recovery_kwargs=lane_recovery_kwargs,
        separation_clearance_m=float(separation_clearance_m),
        separation_hold_s=float(separation_hold_s),
    )
    reasons = base._gate_rejection_reasons("contact", q, near_min_external_severe_count=0)
    ocq = q.get("ocrap_trace_recovery") or {}
    # Target-display quality is stricter than the ordinary qualitative gate on
    # secondary contact: once the rollout first separates from the initial
    # contact episode, any later overlap within the visible window is rejected.
    key = str(item["target_key"])
    oframes = base._visible_frames(list(traces["ocrap"][key].get("render_trace") or []), float(clip), dt)
    overlaps = [base._flag(f, "overlap") for f in oframes]
    first_contact = next((i for i, flag in enumerate(overlaps) if flag), None)
    first_sep = None if first_contact is None else next((i for i in range(first_contact + 1, len(overlaps)) if not overlaps[i]), None)
    any_recontact = bool(first_sep is not None and any(overlaps[first_sep + 1:]))
    if reject_any_recontact_after_first_separation and any_recontact:
        reasons.append("any_recontact_after_first_separation")
    terminal_clearance = ocq.get("terminal_clearance_m")
    if terminal_clearance is None or float(terminal_clearance) < float(min_terminal_clearance_m):
        reasons.append("insufficient_terminal_clearance")
    sep_idx = ((ocq.get("sustained_separation") or {}).get("first_index"))
    if sep_idx is not None:
        vals = [base._metric(f, "min_clearance_m") for f in oframes[int(sep_idx):]]
        vals = [float(v) for v in vals if v is not None]
        if vals and min(vals) < float(min_post_separation_clearance_m):
            reasons.append("post_separation_clearance_dip")
    failures: list[str] = []
    temporal_methods: list[str] = []
    evidence: list[str] = []
    temporal: dict[str, dict[str, Any]] = {}
    for method in base.METHODS["contact"]:
        eq = (q.get("external_trace_recovery") or {}).get(method) or {}
        if eq.get("recovery_failure"):
            failures.append(method)
        tev = supp._temporal_pair_evidence(
            traces["ocrap"][key], traces[method][key], clip_s=float(clip), dt_s=dt,
            o_quality=ocq, e_quality=eq,
            win_margin_m=float(temporal_win_margin_m),
            noninferior_margin_m=float(temporal_noninferior_margin_m),
            min_win_fraction=float(temporal_min_win_fraction),
            min_noninferior_fraction=float(temporal_min_noninferior_fraction),
            min_mean_clearance_gain_m=float(temporal_min_mean_gain_m),
            min_terminal_gain_m=float(temporal_min_terminal_gain_m),
            min_separation_lead_s=float(temporal_min_separation_lead_s),
            min_overlap_reduction_s=float(temporal_min_overlap_reduction_s),
            min_valid_frames=int(temporal_min_valid_frames),
        )
        temporal[method] = tev
        if tev.get("temporal_majority_advantage"):
            temporal_methods.append(method)
        if eq.get("recovery_failure") or tev.get("temporal_majority_advantage"):
            evidence.append(method)
    if len(evidence) < int(min_comparative_methods):
        reasons.append("insufficient_external_comparative_evidence")
    reasons = list(dict.fromkeys(reasons))
    q["external_temporal_advantage"] = temporal
    q["external_temporal_advantage_methods"] = temporal_methods
    q["external_comparative_evidence_methods"] = evidence
    q["external_comparative_evidence_count"] = len(evidence)
    ok = bool(q.get("ocrap_controlled_recovery")) and not reasons
    return ok, q, reasons, evidence, failures, temporal_methods


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-selection", type=Path, required=True)
    ap.add_argument("--trace-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--audit-output", type=Path, required=True)
    ap.add_argument("--preferred-selection", type=Path, default=None)
    ap.add_argument("--num-scenes", type=int, default=5)
    ap.add_argument("--min-clip-duration-s", type=float, default=2.5)
    ap.add_argument("--max-clip-duration-s", type=float, default=4.0)
    ap.add_argument("--clip-step-s", type=float, default=0.1)
    ap.add_argument("--min-comparative-evidence-methods", type=int, default=1)
    ap.add_argument("--min-terminal-clearance-m", type=float, default=0.50)
    ap.add_argument("--min-post-separation-clearance-m", type=float, default=0.25)
    ap.add_argument("--allow-recontact-after-first-separation", action="store_true")

    # Slightly permissive comparative thresholds are allowed for target-display
    # discovery, while hard OC-RAP reality/recovery gates remain unchanged.
    ap.add_argument("--temporal-clearance-win-margin-m", type=float, default=0.05)
    ap.add_argument("--temporal-clearance-noninferior-margin-m", type=float, default=0.15)
    ap.add_argument("--temporal-min-win-fraction", type=float, default=0.50)
    ap.add_argument("--temporal-min-noninferior-fraction", type=float, default=0.70)
    ap.add_argument("--temporal-min-mean-clearance-gain-m", type=float, default=0.05)
    ap.add_argument("--temporal-min-terminal-gain-m", type=float, default=0.15)
    ap.add_argument("--temporal-min-separation-lead-s", type=float, default=0.10)
    ap.add_argument("--temporal-min-overlap-reduction-s", type=float, default=0.10)
    ap.add_argument("--temporal-min-valid-frames", type=int, default=6)

    # Hard reality contract: intentionally the same conservative defaults used
    # by the reviewer-safe Contact finalizer.
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
    ap.add_argument("--require-exact-count", action="store_true")
    args = ap.parse_args()

    if args.num_scenes <= 0:
        raise SystemExit("--num-scenes must be positive")
    if not (0 < args.min_clip_duration_s <= args.max_clip_duration_s):
        raise SystemExit("invalid clip duration range")
    if args.clip_step_s <= 0:
        raise SystemExit("clip step must be positive")

    candidate = json.loads(args.candidate_selection.read_text(encoding="utf-8"))
    if str(candidate.get("regime") or "") != "contact":
        raise SystemExit("candidate selection must be Contact")
    items = list(candidate.get("selected") or [])
    traces = base._load_traces(args.trace_root, "contact")
    dt = float(candidate.get("metric_dt_s", 0.1) or 0.1)
    preferred = _keys_from_selection(args.preferred_selection)
    preferred_rank = {k: i for i, k in enumerate(preferred)}

    lane_kwargs = {
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

    accepted: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for item in items:
        key = str(item["target_key"])
        missing = [m for m, rows in traces.items() if key not in rows]
        if missing:
            raise SystemExit(f"{key}: missing traces for {missing}")
        available = _finite(item.get("available_future_s"))
        if available is None:
            available = max(0.0, (len(traces["ocrap"][key].get("render_trace") or []) - 1) * dt)
        full_clip = min(float(args.max_clip_duration_s), float(available))
        attempts: list[dict[str, Any]] = []
        chosen: dict[str, Any] | None = None
        clip = full_clip
        while clip + 1e-9 >= float(args.min_clip_duration_s):
            clip = round(clip / dt) * dt
            ok, q, reasons, evidence, failures, temporal = _evaluate(
                item, traces, clip=clip, dt=dt,
                lane_kwargs=lane_kwargs, lane_recovery_kwargs=lane_recovery_kwargs,
                separation_clearance_m=float(args.separation_clearance_m),
                separation_hold_s=float(args.separation_hold_s),
                min_terminal_clearance_m=float(args.min_terminal_clearance_m),
                min_post_separation_clearance_m=float(args.min_post_separation_clearance_m),
                reject_any_recontact_after_first_separation=not bool(args.allow_recontact_after_first_separation),
                min_comparative_methods=int(args.min_comparative_evidence_methods),
                temporal_win_margin_m=float(args.temporal_clearance_win_margin_m),
                temporal_noninferior_margin_m=float(args.temporal_clearance_noninferior_margin_m),
                temporal_min_win_fraction=float(args.temporal_min_win_fraction),
                temporal_min_noninferior_fraction=float(args.temporal_min_noninferior_fraction),
                temporal_min_mean_gain_m=float(args.temporal_min_mean_clearance_gain_m),
                temporal_min_terminal_gain_m=float(args.temporal_min_terminal_gain_m),
                temporal_min_separation_lead_s=float(args.temporal_min_separation_lead_s),
                temporal_min_overlap_reduction_s=float(args.temporal_min_overlap_reduction_s),
                temporal_min_valid_frames=int(args.temporal_min_valid_frames),
            )
            attempts.append({
                "clip_duration_s": float(clip), "accepted": bool(ok),
                "rejection_reasons": reasons, "failure_methods": failures,
                "temporal_methods": temporal, "comparative_methods": evidence,
                "quality": q,
            })
            if ok:
                row = dict(item)
                row["clip_duration_s"] = float(clip)
                row["display_full_available_clip_s"] = float(full_clip)
                row["display_window_trimmed"] = bool(clip + 1e-9 < full_clip)
                row["visualization_trace_quality"] = q
                trace_primary = base._hardest_among(row, evidence) or str(row.get("primary_external_method") or "")
                if trace_primary:
                    row["primary_external_method"] = trace_primary
                    row["primary_comparator_reason"] = (
                        "hardest paired external among methods with trace-level Contact comparison evidence"
                    )
                chosen = row
                break  # descending search => longest passing real prefix
            clip -= float(args.clip_step_s)
        if chosen is not None:
            accepted.append(chosen)
        audits.append({
            "target_key": key,
            "accepted": chosen is not None,
            "chosen_clip_duration_s": None if chosen is None else chosen["clip_duration_s"],
            "full_available_clip_s": full_clip,
            "preferred_existing": key in preferred_rank,
            "clip_search": attempts,
        })

    def temporal_strength(row: dict[str, Any]) -> float:
        vals = []
        for ev in (row["visualization_trace_quality"].get("external_temporal_advantage") or {}).values():
            if ev.get("temporal_majority_advantage"):
                vals.append(float(ev.get("clearance_win_fraction") or 0.0))
        return max(vals) if vals else 0.0

    accepted.sort(key=lambda r: (
        0 if (str(r["target_key"]) in preferred_rank and not r.get("display_window_trimmed")) else 1,
        preferred_rank.get(str(r["target_key"]), 10**6),
        0 if not r.get("display_window_trimmed") else 1,
        int(r["visualization_trace_quality"].get("visual_evidence_rank") or 99),
        -int(r["visualization_trace_quality"].get("external_recovery_failure_count") or 0),
        -int(r["visualization_trace_quality"].get("external_comparative_evidence_count") or 0),
        -temporal_strength(r),
        -float(r.get("clip_duration_s") or 0.0),
        float((r["visualization_trace_quality"].get("ocrap_trace_recovery") or {}).get("sustained_separation", {}).get("first_s") or 1e6),
        -min(3.0, float(r["visualization_trace_quality"].get("ocrap_terminal_clearance_m") or 0.0)),
        str(r["target_key"]),
    ))

    final: list[dict[str, Any]] = []
    used_scenes: set[str] = set()
    for row in accepted:
        sid = str(row.get("scene_id") or row["target_key"])
        if sid in used_scenes:
            continue
        used_scenes.add(sid)
        x = dict(row)
        x["category_rank"] = len(final) + 1
        final.append(x)
        if len(final) >= int(args.num_scenes):
            break

    out = dict(candidate)
    out.update({
        "event": "contact_target_display_selection_v1",
        "regime": "contact",
        "exploratory_qualitative_only": True,
        "paper_population_claim_allowed": False,
        "target_display_real_trace_only": True,
        "requested_num_scenes": int(args.num_scenes),
        "selected": final,
        "target_keys": [str(x["target_key"]) for x in final],
        "selected_clip_duration_s": None,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "event": "contact_target_display_selection_audit_v1",
        "candidate_count": len(items),
        "accepted_candidate_count": len(accepted),
        "selected_count": len(final),
        "requested_count": int(args.num_scenes),
        "hard_reality_contract_unchanged": True,
        "trajectory_states_modified": False,
        "candidates": audits,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": out["event"], "selected": len(final), "requested": args.num_scenes, "output": str(args.output)}))
    if args.require_exact_count and len(final) != int(args.num_scenes):
        raise SystemExit(f"only {len(final)} realistic target-display scenes passed; requested {args.num_scenes}")
    if not final:
        raise SystemExit("no realistic Contact target-display scenes passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
