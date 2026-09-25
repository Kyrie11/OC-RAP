#!/usr/bin/env python3
"""Select five high-quality Contact target-display scenes from empirical/reference traces.

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
from contact_scene_diagnostics import analyze_contact_trace  # noqa: E402


def _keys_from_selection(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(d, list):
        return [str(x) for x in d if isinstance(x, str) and x]
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
    max_sustained_separation_s: float | None,
    reject_any_recontact_after_first_separation: bool,
    min_comparative_methods: int, min_temporal_advantage_methods: int,
    min_dominance_methods: int, min_terminal_advantage_methods: int,
    min_overlap_advantage_methods: int, min_separation_advantage_methods: int,
    terminal_advantage_margin_m: float, overlap_advantage_margin_s: float, separation_advantage_margin_s: float,
    max_empirical_source_offroad_fraction: float | None,
    allow_source_offroad_repair: bool, max_repairable_source_offroad_fraction: float | None,
    temporal_win_margin_m: float, temporal_noninferior_margin_m: float,
    temporal_min_win_fraction: float, temporal_min_noninferior_fraction: float,
    temporal_min_mean_gain_m: float, temporal_min_terminal_gain_m: float,
    temporal_min_separation_lead_s: float, temporal_min_overlap_reduction_s: float,
    temporal_min_valid_frames: int,
    allow_reference_deviation_template: bool,
    reference_max_deviation_mean_m: float | None,
    reference_max_deviation_max_m: float | None,
    min_primary_pair_score: float | None,
    min_median_pair_score: float | None,
    min_material_comparisons: int,
) -> tuple[bool, dict[str, Any], list[str], list[str], list[str], list[str]]:
    row = dict(item)
    row["clip_duration_s"] = float(clip)
    reference_scene = traces["ocrap"][str(item["target_key"])]
    reference_quality = reference_scene.get("reference_quality") or {}
    q = base._contact_quality(
        row, traces, dt_s=dt, lane_kwargs=lane_kwargs,
        lane_recovery_kwargs=lane_recovery_kwargs,
        separation_clearance_m=float(separation_clearance_m),
        separation_hold_s=float(separation_hold_s),
    )
    reasons = base._gate_rejection_reasons("contact", q, near_min_external_severe_count=0)
    if min_primary_pair_score is not None:
        primary_pair = _finite(item.get("primary_pair_score"))
        if primary_pair is None or primary_pair < float(min_primary_pair_score) - 1e-9:
            reasons.append("weak_primary_pair_score")
    if min_median_pair_score is not None:
        median_pair = _finite(item.get("median_pair_score"))
        if median_pair is None or median_pair < float(min_median_pair_score) - 1e-9:
            reasons.append("weak_median_pair_score")
    material_count = int(item.get("num_material_external_comparisons") or 0)
    if material_count < int(min_material_comparisons):
        reasons.append("insufficient_material_external_comparisons")
    if bool(reference_scene.get("reference_trajectory")) and reference_quality:
        # The final displayed window is independently re-evaluated below.  Do
        # not reject a clean *visible prefix* merely because the longer
        # synthesized trajectory later violates a full-trace quality flag.
        # The only full-trace property that must survive clipping is the local
        # deviation envelope: without that, the target ceases to be a local
        # recovery correction around empirical OC-RAP.
        deviation_mean = _finite(reference_quality.get("deviation_mean_m"))
        deviation_max = _finite(reference_quality.get("deviation_max_m"))
        deviation_failed = bool(reference_quality.get("deviation_contract_failed"))
        if deviation_failed:
            relax = bool(allow_reference_deviation_template)
            if relax and reference_max_deviation_mean_m is not None:
                relax = relax and deviation_mean is not None and deviation_mean <= float(reference_max_deviation_mean_m) + 1e-9
            if relax and reference_max_deviation_max_m is not None:
                relax = relax and deviation_max is not None and deviation_max <= float(reference_max_deviation_max_m) + 1e-9
            if not relax:
                reasons.append("reference_deviation_contract_failed")
        q["reference_full_trace_clean"] = bool(reference_quality.get("clean"))
        q["reference_deviation_contract_failed"] = bool(deviation_failed)
    empirical_quality = (reference_quality.get("empirical_quality") or {}) if isinstance(reference_quality, dict) else {}
    if max_empirical_source_offroad_fraction is not None and empirical_quality:
        frac = empirical_quality.get("offroad_proxy_fraction")
        if frac is not None and float(frac) > float(max_empirical_source_offroad_fraction) + 1e-9:
            repairable = bool(
                allow_source_offroad_repair
                and bool(reference_scene.get("reference_trajectory"))
                and max_repairable_source_offroad_fraction is not None
                and float(frac) <= float(max_repairable_source_offroad_fraction) + 1e-9
                and float(reference_quality.get("offroad_proxy_fraction") or 0.0) <= 1e-9
                and not bool(reference_quality.get("secondary_collision_event"))
                and not bool(reference_quality.get("recontact"))
            )
            if not repairable:
                reasons.append("empirical_source_excessive_offroad")
            else:
                q["empirical_source_offroad_repaired_for_target"] = True
    ocq = q.get("ocrap_trace_recovery") or {}
    # Target-display quality is stricter than the ordinary qualitative gate on
    # secondary contact: once the rollout first separates from the initial
    # contact episode, any later overlap within the visible window is rejected.
    key = str(item["target_key"])
    oframes = base._visible_frames(list(traces["ocrap"][key].get("render_trace") or []), float(clip), dt)
    display_diag = analyze_contact_trace(oframes, dt=dt)
    q["ocrap_display_diagnostics"] = display_diag
    if bool(display_diag.get("secondary_collision_event")):
        reasons.append("secondary_collision_with_new_actor")
    overlaps = [base._flag(f, "overlap") for f in oframes]
    first_contact = next((i for i, flag in enumerate(overlaps) if flag), None)
    first_sep = None if first_contact is None else next((i for i in range(first_contact + 1, len(overlaps)) if not overlaps[i]), None)
    any_recontact = bool(first_sep is not None and any(overlaps[first_sep + 1:]))
    if reject_any_recontact_after_first_separation and any_recontact:
        reasons.append("any_recontact_after_first_separation")
    terminal_clearance = ocq.get("terminal_clearance_m")
    if terminal_clearance is None or float(terminal_clearance) < float(min_terminal_clearance_m):
        reasons.append("insufficient_terminal_clearance")
    sep_info = (ocq.get("sustained_separation") or {})
    sep_time = sep_info.get("first_s")
    if max_sustained_separation_s is not None:
        if sep_time is None or float(sep_time) > float(max_sustained_separation_s) + 1e-9:
            reasons.append("sustained_separation_too_late")
    sep_idx = sep_info.get("first_index")
    if sep_idx is not None:
        vals = [base._metric(f, "min_clearance_m") for f in oframes[int(sep_idx):]]
        vals = [float(v) for v in vals if v is not None]
        if vals and min(vals) < float(min_post_separation_clearance_m):
            reasons.append("post_separation_clearance_dip")
    failures: list[str] = []
    temporal_methods: list[str] = []
    evidence: list[str] = []
    dominance_methods: list[str] = []
    terminal_advantage_methods: list[str] = []
    overlap_advantage_methods: list[str] = []
    separation_advantage_methods: list[str] = []
    pair_dominance: dict[str, dict[str, Any]] = {}
    temporal: dict[str, dict[str, Any]] = {}
    o_overlap_duration = float(sum(base._flag(f, "overlap") for f in oframes[:-1]) * dt) if len(oframes) > 1 else 0.0
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

        eframes = base._visible_frames(list(traces[method][key].get("render_trace") or []), float(clip), dt)
        e_overlap_duration = float(sum(base._flag(f, "overlap") for f in eframes[:-1]) * dt) if len(eframes) > 1 else 0.0
        o_terminal = _finite(ocq.get("terminal_clearance_m"))
        e_terminal = _finite(eq.get("terminal_clearance_m"))
        terminal_gain = None if o_terminal is None or e_terminal is None else float(o_terminal - e_terminal)
        o_sep = _finite((ocq.get("sustained_separation") or {}).get("first_s"))
        e_sep = _finite((eq.get("sustained_separation") or {}).get("first_s"))
        if o_sep is None:
            separation_lead = None
        elif e_sep is None:
            separation_lead = max(0.0, float(clip) - o_sep)
        else:
            separation_lead = float(e_sep - o_sep)
        overlap_reduction = float(e_overlap_duration - o_overlap_duration)
        term_ok = terminal_gain is not None and terminal_gain >= float(terminal_advantage_margin_m)
        overlap_ok = overlap_reduction >= float(overlap_advantage_margin_s)
        sep_ok = separation_lead is not None and separation_lead >= float(separation_advantage_margin_s)
        temporal_ok = bool(tev.get("temporal_majority_advantage"))
        if term_ok: terminal_advantage_methods.append(method)
        if overlap_ok: overlap_advantage_methods.append(method)
        if sep_ok: separation_advantage_methods.append(method)
        dominance_votes = int(term_ok) + int(overlap_ok) + int(sep_ok) + int(temporal_ok)
        dominated = bool(eq.get("recovery_failure") and dominance_votes >= 1) or dominance_votes >= 2
        if dominated:
            dominance_methods.append(method)
        pair_dominance[method] = {
            "terminal_clearance_gain_m": terminal_gain,
            "overlap_duration_reduction_s": overlap_reduction,
            "sustained_separation_lead_s": separation_lead,
            "temporal_majority_advantage": temporal_ok,
            "dominance_votes": dominance_votes,
            "dominated": dominated,
        }
    if len(evidence) < int(min_comparative_methods):
        reasons.append("insufficient_external_comparative_evidence")
    if len(temporal_methods) < int(min_temporal_advantage_methods):
        reasons.append("insufficient_temporal_clearance_advantage")
    if len(dominance_methods) < int(min_dominance_methods):
        reasons.append("insufficient_multimetric_dominance")
    if len(terminal_advantage_methods) < int(min_terminal_advantage_methods):
        reasons.append("insufficient_terminal_clearance_advantage")
    if len(overlap_advantage_methods) < int(min_overlap_advantage_methods):
        reasons.append("insufficient_overlap_duration_advantage")
    if len(separation_advantage_methods) < int(min_separation_advantage_methods):
        reasons.append("insufficient_early_separation_advantage")
    reasons = list(dict.fromkeys(reasons))
    q["external_temporal_advantage"] = temporal
    q["external_temporal_advantage_methods"] = temporal_methods
    q["external_comparative_evidence_methods"] = evidence
    q["external_comparative_evidence_count"] = len(evidence)
    q["external_dominance_methods"] = dominance_methods
    q["external_dominance_count"] = len(dominance_methods)
    q["external_terminal_advantage_methods"] = terminal_advantage_methods
    q["external_overlap_advantage_methods"] = overlap_advantage_methods
    q["external_separation_advantage_methods"] = separation_advantage_methods
    q["external_pair_dominance"] = pair_dominance
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
    ap.add_argument("--max-sustained-separation-s", type=float, default=None)
    ap.add_argument("--enable-reference-quality-fallback", action="store_true",
                    help="reference visualization only: if a scene misses the strict target timing/clearance margins, retry with explicitly bounded fallback margins while keeping collision/recontact/offroad/lane gates unchanged")
    ap.add_argument("--fallback-max-sustained-separation-s", type=float, default=1.5)
    ap.add_argument("--fallback-min-post-separation-clearance-m", type=float, default=0.50)
    ap.add_argument("--min-temporal-advantage-methods", type=int, default=0)
    ap.add_argument("--min-dominance-methods", type=int, default=0, help="minimum external methods beaten on at least two post-contact dimensions")
    ap.add_argument("--min-terminal-advantage-methods", type=int, default=0)
    ap.add_argument("--min-overlap-advantage-methods", type=int, default=0)
    ap.add_argument("--min-separation-advantage-methods", type=int, default=0)
    ap.add_argument("--terminal-advantage-margin-m", type=float, default=0.25)
    ap.add_argument("--overlap-advantage-margin-s", type=float, default=0.10)
    ap.add_argument("--separation-advantage-margin-s", type=float, default=0.10)
    ap.add_argument("--max-empirical-source-offroad-fraction", type=float, default=None, help="reference mode only: reject source scenes whose empirical OC-RAP is already severely off-road unless explicitly repairable")
    ap.add_argument("--allow-source-offroad-repair", action="store_true", help="reference mode only: allow a moderately off-road empirical source if the synthesized displayed trace repairs it and passes target gates")
    ap.add_argument("--max-repairable-source-offroad-fraction", type=float, default=0.45)
    ap.add_argument("--criticality-audit", type=Path, default=None, help="optional contact_reference_synthesis_subset_v2 audit used only to prioritize diverse critical source scenes")
    ap.add_argument("--criticality-coverage-tags", default="secondary_collision,crowded,recontact,source_offroad")
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
    ap.add_argument("--allow-reference-deviation-template", action="store_true",
                    help="for reference/template OC-RAP trajectories only, allow scenes that fail only the deviation-contract subgate provided their physical/lane gates still pass")
    ap.add_argument("--reference-max-deviation-mean-m", type=float, default=3.2)
    ap.add_argument("--reference-max-deviation-max-m", type=float, default=6.5)
    ap.add_argument("--min-primary-pair-score", type=float, default=0.0,
                    help="reject candidates whose broad-pool primary paired score is below this threshold")
    ap.add_argument("--min-median-pair-score", type=float, default=0.0,
                    help="reject candidates whose broad-pool median paired score is below this threshold")
    ap.add_argument("--min-material-comparisons", type=int, default=0,
                    help="reject candidates that do not materially improve over at least this many baselines in the broad pool")
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
    criticality_by_key: dict[str, dict[str, Any]] = {}
    if args.criticality_audit is not None and args.criticality_audit.is_file():
        cd = json.loads(args.criticality_audit.read_text(encoding="utf-8"))
        for r in (cd.get("rows") or []):
            if isinstance(r, dict) and r.get("target_key"):
                criticality_by_key[str(r["target_key"])] = r
    coverage_tags = [x.strip() for x in str(args.criticality_coverage_tags).split(",") if x.strip()]

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
    for raw_item in items:
        item = dict(raw_item)
        key = str(item["target_key"])
        cr = criticality_by_key.get(key) or {}
        item["source_criticality_score"] = float(cr.get("criticality_score") or 0.0)
        item["source_critical_tags"] = list(cr.get("critical_tags") or [])
        item["repairable_source_offroad"] = bool(cr.get("repairable_source_offroad"))
        missing = [m for m, rows in traces.items() if key not in rows]
        if missing:
            audits.append({
                "target_key": key,
                "accepted": False,
                "chosen_clip_duration_s": None,
                "full_available_clip_s": None,
                "preferred_existing": key in preferred_rank,
                "missing_traces": missing,
                "clip_search": [],
            })
            continue
        available = _finite(item.get("available_future_s"))
        if available is None:
            available = max(0.0, (len(traces["ocrap"][key].get("render_trace") or []) - 1) * dt)
        full_clip = min(float(args.max_clip_duration_s), float(available))
        attempts: list[dict[str, Any]] = []
        chosen: dict[str, Any] | None = None
        strict_max_sep = None if args.max_sustained_separation_s is None else float(args.max_sustained_separation_s)
        strict_min_post = float(args.min_post_separation_clearance_m)
        tiers=[("strict",strict_max_sep,strict_min_post)]
        if bool(args.enable_reference_quality_fallback) and bool(candidate.get("reference_visualization_only")):
            fb_max=float(args.fallback_max_sustained_separation_s)
            if strict_max_sep is not None: fb_max=max(fb_max,strict_max_sep)
            fb_post=min(float(args.fallback_min_post_separation_clearance_m),strict_min_post)
            if fb_max!=strict_max_sep or abs(fb_post-strict_min_post)>1e-12:
                tiers.append(("reference_fallback",fb_max,fb_post))
        for tier_name,tier_max_sep,tier_min_post in tiers:
          clip = full_clip
          while clip + 1e-9 >= float(args.min_clip_duration_s):
            clip = round(clip / dt) * dt
            ok, q, reasons, evidence, failures, temporal = _evaluate(
                item, traces, clip=clip, dt=dt,
                lane_kwargs=lane_kwargs, lane_recovery_kwargs=lane_recovery_kwargs,
                separation_clearance_m=float(args.separation_clearance_m),
                separation_hold_s=float(args.separation_hold_s),
                min_terminal_clearance_m=float(args.min_terminal_clearance_m),
                min_post_separation_clearance_m=float(tier_min_post),
                max_sustained_separation_s=tier_max_sep,
                reject_any_recontact_after_first_separation=not bool(args.allow_recontact_after_first_separation),
                min_comparative_methods=int(args.min_comparative_evidence_methods),
                min_temporal_advantage_methods=int(args.min_temporal_advantage_methods),
                min_dominance_methods=int(args.min_dominance_methods),
                min_terminal_advantage_methods=int(args.min_terminal_advantage_methods),
                min_overlap_advantage_methods=int(args.min_overlap_advantage_methods),
                min_separation_advantage_methods=int(args.min_separation_advantage_methods),
                terminal_advantage_margin_m=float(args.terminal_advantage_margin_m),
                overlap_advantage_margin_s=float(args.overlap_advantage_margin_s),
                separation_advantage_margin_s=float(args.separation_advantage_margin_s),
                max_empirical_source_offroad_fraction=(None if args.max_empirical_source_offroad_fraction is None else float(args.max_empirical_source_offroad_fraction)),
                allow_source_offroad_repair=bool(args.allow_source_offroad_repair),
                max_repairable_source_offroad_fraction=(None if args.max_repairable_source_offroad_fraction is None else float(args.max_repairable_source_offroad_fraction)),
                temporal_win_margin_m=float(args.temporal_clearance_win_margin_m),
                temporal_noninferior_margin_m=float(args.temporal_clearance_noninferior_margin_m),
                temporal_min_win_fraction=float(args.temporal_min_win_fraction),
                temporal_min_noninferior_fraction=float(args.temporal_min_noninferior_fraction),
                temporal_min_mean_gain_m=float(args.temporal_min_mean_clearance_gain_m),
                temporal_min_terminal_gain_m=float(args.temporal_min_terminal_gain_m),
                temporal_min_separation_lead_s=float(args.temporal_min_separation_lead_s),
                temporal_min_overlap_reduction_s=float(args.temporal_min_overlap_reduction_s),
                temporal_min_valid_frames=int(args.temporal_min_valid_frames),
                allow_reference_deviation_template=bool(args.allow_reference_deviation_template),
                reference_max_deviation_mean_m=(None if args.reference_max_deviation_mean_m is None else float(args.reference_max_deviation_mean_m)),
                reference_max_deviation_max_m=(None if args.reference_max_deviation_max_m is None else float(args.reference_max_deviation_max_m)),
                min_primary_pair_score=(None if args.min_primary_pair_score is None else float(args.min_primary_pair_score)),
                min_median_pair_score=(None if args.min_median_pair_score is None else float(args.min_median_pair_score)),
                min_material_comparisons=int(args.min_material_comparisons),
            )
            attempts.append({
                "clip_duration_s": float(clip), "accepted": bool(ok),
                "quality_tier":tier_name,
                "max_sustained_separation_s":tier_max_sep,
                "min_post_separation_clearance_m":float(tier_min_post),
                "rejection_reasons": reasons, "failure_methods": failures,
                "temporal_methods": temporal, "comparative_methods": evidence,
                "quality": q,
            })
            if ok:
                row = dict(item)
                row["clip_duration_s"] = float(clip)
                row["display_full_available_clip_s"] = float(full_clip)
                row["display_window_trimmed"] = bool(clip + 1e-9 < full_clip)
                row["selection_quality_tier"] = tier_name
                row["selection_contract"] = {
                    "max_sustained_separation_s":tier_max_sep,
                    "min_post_separation_clearance_m":float(tier_min_post),
                    "min_terminal_clearance_m":float(args.min_terminal_clearance_m),
                }
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
              break
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
        0 if r.get("selection_quality_tier") == "strict" else 1,
        0 if not r.get("display_window_trimmed") else 1,
        -float(r.get("source_criticality_score") or 0.0),
        int(r["visualization_trace_quality"].get("visual_evidence_rank") or 99),
        -int(r["visualization_trace_quality"].get("external_dominance_count") or 0),
        -len(r["visualization_trace_quality"].get("external_terminal_advantage_methods") or []),
        -len(r["visualization_trace_quality"].get("external_overlap_advantage_methods") or []),
        -len(r["visualization_trace_quality"].get("external_separation_advantage_methods") or []),
        -int(r["visualization_trace_quality"].get("external_recovery_failure_count") or 0),
        -int(r["visualization_trace_quality"].get("external_comparative_evidence_count") or 0),
        -int(r.get("num_material_external_comparisons") or 0),
        -float(_finite(r.get("primary_pair_score")) or -1e9),
        -float(_finite(r.get("median_pair_score")) or -1e9),
        -temporal_strength(r),
        float(((r["visualization_trace_quality"].get("ocrap_trace_recovery") or {}).get("lane_realism") or {}).get("lane_center_distance_p90_m") or 0.0),
        -float(r.get("clip_duration_s") or 0.0),
        float((r["visualization_trace_quality"].get("ocrap_trace_recovery") or {}).get("sustained_separation", {}).get("first_s") or 1e6),
        -min(3.0, float(r["visualization_trace_quality"].get("ocrap_terminal_clearance_m") or 0.0)),
        str(r["target_key"]),
    ))

    final: list[dict[str, Any]] = []
    used_scenes: set[str] = set()
    def _add(row: dict[str, Any]) -> bool:
        sid = str(row.get("scene_id") or row["target_key"])
        if sid in used_scenes or any(str(x["target_key"]) == str(row["target_key"]) for x in final):
            return False
        used_scenes.add(sid); x=dict(row); x["category_rank"]=len(final)+1; final.append(x); return True
    # Preserve explicitly preferred good example(s), then cover distinct source
    # failure modes before filling by overall visual/criticality ordering.
    for row in accepted:
        if str(row["target_key"]) in preferred_rank and not row.get("display_window_trimmed"):
            _add(row)
            if len(final) >= int(args.num_scenes): break
    for tag in coverage_tags:
        if len(final) >= int(args.num_scenes): break
        row = next((r for r in accepted if tag in (r.get("source_critical_tags") or []) and str(r["target_key"]) not in {str(x["target_key"]) for x in final}), None)
        if row is not None: _add(row)
    for row in accepted:
        if len(final) >= int(args.num_scenes): break
        _add(row)

    out = dict(candidate)
    out.update({
        "event": "contact_target_display_selection_v1",
        "regime": "contact",
        "exploratory_qualitative_only": True,
        "paper_population_claim_allowed": False,
        "target_display_real_trace_only": not bool(candidate.get("reference_visualization_only")),
        "reference_visualization_only": bool(candidate.get("reference_visualization_only")),
        "display_name_overrides": candidate.get("display_name_overrides") or {},
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
        "hard_reality_contract_unchanged": not any(r.get("selection_quality_tier") == "reference_fallback" for r in final),
        "collision_recontact_offroad_lane_gates_unchanged": True,
        "reference_quality_fallback_enabled": bool(args.enable_reference_quality_fallback),
        "reference_quality_fallback_selected_count": sum(r.get("selection_quality_tier") == "reference_fallback" for r in final),
        "strict_max_sustained_separation_s": args.max_sustained_separation_s,
        "strict_min_post_separation_clearance_m": float(args.min_post_separation_clearance_m),
        "fallback_max_sustained_separation_s": float(args.fallback_max_sustained_separation_s),
        "fallback_min_post_separation_clearance_m": float(args.fallback_min_post_separation_clearance_m),
        "trajectory_states_modified": bool(candidate.get("reference_visualization_only")),
        "reference_visualization_only": bool(candidate.get("reference_visualization_only")),
        "criticality_coverage_tags": coverage_tags,
        "allow_source_offroad_repair": bool(args.allow_source_offroad_repair),
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
