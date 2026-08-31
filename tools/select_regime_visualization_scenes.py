#!/usr/bin/env python3
"""Select qualitative scenes for Safe / Near-Contact / Contact visualization.

The selector is intentionally metric-only.  It consumes the full closed-loop
scene journals for OC-RAP and *all* main-table external baselines, chooses five
scene-time targets per regime, and records a per-scene best/worst external
comparator for later video rendering.

Selection contract
------------------
Safe:
  rank high-quality OC-RAP closed-loop behavior (NUP, clearance, TTC, progress,
  low unnecessary intervention) with hard collision/off-road guards.  Relative
  external performance is disclosed but is not the primary ranking signal.
Near-Contact / Contact:
  rank robust relative gains against the complete external-baseline set.  The
  strongest tier requires a material gain over the per-scene best external
  comparator and no unsafe regression.  Lower tiers are deterministic fallbacks
  and are explicitly labeled in the output.
Duration:
  first require enough WOMD future horizon for >=5 s clips.  If fewer than the
  requested count are available, automatically fall back to >=3 s.  The chosen
  threshold is stored in the selection artifact and enforced by the renderer.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Reuse the already-audited paired near/contact effect contract so the new
# multi-baseline selector stays numerically consistent with the previous v50
# qualitative selector.
from select_critical_scenes_v48_34 import _evaluate as _pair_evaluate  # type: ignore


DEFAULT_THRESHOLDS = dict(
    minimum_positive_score=0.0,
    min_near_ttc_gain_s=0.25,
    min_near_clearance_gain_m=0.25,
    min_near_exposure_reduction_s=0.20,
    min_near_near_zero_reduction_rate=0.02,
    max_near_ttc_regression_s=0.10,
    max_near_terminal_ttc_regression_s=2.0,
    max_near_clearance_regression_m=0.10,
    max_near_terminal_clearance_regression_m=0.50,
    max_near_exposure_regression_s=0.10,
    max_near_near_zero_regression_rate=0.05,
    min_contact_terminal_clearance_gain_m=0.50,
    min_contact_auc_gain_m=0.50,
    min_contact_clearance_gain_m=0.25,
    min_contact_overlap_duration_reduction_s=0.20,
    min_contact_yaw_rate_reduction_radps=0.10,
    min_contact_jerk_reduction_mps3=1.0,
    max_contact_terminal_clearance_regression_m=0.10,
    max_contact_auc_regression_m=0.25,
    max_contact_overlap_duration_regression_s=0.10,
    max_contact_yaw_rate_regression_radps=0.50,
    max_contact_jerk_regression_mps3=4.0,
    max_contact_route_progress_regression_m=2.0,
)


def _finite_or_none(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _finite(value: Any, default: float = 0.0) -> float:
    x = _finite_or_none(value)
    return default if x is None else x


def _bounded(value: Any, scale: float, limit: float = 3.0) -> float:
    return max(-limit, min(limit, _finite(value) / max(scale, 1.0e-9)))


def _scene_key(scene: dict[str, Any], envelope: dict[str, Any]) -> str:
    key = str(scene.get("target_key") or envelope.get("resume_key") or "")
    if key.startswith("target:"):
        key = key[len("target:"):]
    if key:
        return key
    scene_id = str(scene.get("scene_id") or "")
    target_time = scene.get("target_time_index")
    return f"{scene_id}:t{target_time}" if scene_id and target_time is not None else scene_id


def _load_scenes(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".json" and not path.name.endswith(".scenes.jsonl"):
        candidate = Path(str(path) + ".scenes.jsonl")
        if candidate.is_file():
            path = candidate
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            envelope = json.loads(line)
            scene = envelope.get("scene", envelope)
            key = _scene_key(scene, envelope)
            if not key:
                raise ValueError(f"scene without target key in {path}")
            if key in rows:
                raise ValueError(f"duplicate scene key {key} in {path}")
            # Metric-only selection: deliberately discard heavyweight traces.
            rows[key] = {
                name: value
                for name, value in scene.items()
                if name not in {"decisions", "render_trace", "render_context", "render_trace_schema", "state_xy_trace"}
                and not str(name).endswith("_trace")
            }
    if not rows:
        raise ValueError(f"empty scene journal: {path}")
    return rows


def _metric(scene: dict[str, Any], *names: str) -> float | None:
    summary = scene.get("metric_summary", {}) or {}
    for name in names:
        if name in scene:
            value = _finite_or_none(scene.get(name))
            if value is not None:
                return value
        value = _finite_or_none(summary.get(name))
        if value is not None:
            return value
    return None


def _duration_available_s(scene: dict[str, Any], horizon_steps: int, dt_s: float) -> float | None:
    t = _finite_or_none(scene.get("target_time_index"))
    if t is None:
        return None
    # WOMD has horizon_steps states.  From state t there are horizon_steps-1-t
    # 0.1 s transitions remaining.
    return max(0.0, (float(horizon_steps - 1) - t) * dt_s)


def _safe_absolute(scene: dict[str, Any]) -> tuple[float, dict[str, float | None], list[str]]:
    terms = {
        "nup": _metric(scene, "closed_loop_bounded_NUP"),
        "intervention_rate": _metric(scene, "intervention_rate"),
        "clearance_p05_m": _metric(scene, "min_clearance_m_p05", "min_clearance_m_min"),
        "ttc_p05_s": _metric(scene, "ttc_s_p05", "ttc_s_min"),
        "route_progression_m": _metric(scene, "route_progression_m", "route_progression"),
        "overlap_any": _metric(scene, "overlap_any"),
        "offroad_any": _metric(scene, "offroad_any"),
        "jerk_p95": _metric(scene, "jerk_p95"),
        "yaw_rate_p95": _metric(scene, "yaw_rate_p95"),
    }
    missing = [k for k in ("nup", "intervention_rate", "clearance_p05_m", "ttc_p05_s", "overlap_any", "offroad_any") if terms[k] is None]
    # Bounded dimensionless quality: NUP dominates, while geometric margin and
    # nominal continuity distinguish otherwise-safe clips.  Sentinel TTC cannot
    # dominate because every continuous term is clipped.
    score = (
        2.2 * _bounded((_finite(terms["nup"]) - 0.90), 0.10, 1.5)
        + 0.9 * _bounded(terms["clearance_p05_m"], 2.0, 2.0)
        + 0.7 * _bounded(terms["ttc_p05_s"], 4.0, 2.0)
        + 0.25 * _bounded(terms["route_progression_m"], 10.0, 2.0)
        - 0.8 * _bounded(terms["intervention_rate"], 0.20, 2.0)
        - 4.0 * _finite(terms["overlap_any"])
        - 3.0 * _finite(terms["offroad_any"])
        - 0.15 * _bounded(terms["jerk_p95"], 4.0, 2.0)
        - 0.10 * _bounded(terms["yaw_rate_p95"], 0.5, 2.0)
    )
    return float(score), terms, missing


def _near_absolute(scene: dict[str, Any]) -> float:
    return float(
        1.35 * _bounded(_metric(scene, "ttc_s_p05"), 3.0, 2.5)
        + 1.20 * _bounded(_metric(scene, "min_clearance_m_p05"), 2.0, 2.5)
        + 0.55 * _bounded(_metric(scene, "terminal_ttc_s"), 4.0, 2.0)
        + 0.65 * _bounded(_metric(scene, "terminal_clearance_m"), 2.0, 2.0)
        - 0.85 * _bounded(_metric(scene, "critical_ttc_exposure_duration_s"), 1.0, 3.0)
        - 0.60 * _bounded(_metric(scene, "near_zero_clearance_exposure_rate"), 0.10, 3.0)
        + 0.35 * _bounded((_finite(_metric(scene, "closed_loop_bounded_NUP")) - 0.90), 0.10, 1.5)
        - 4.0 * _finite(_metric(scene, "overlap_any"))
        - 2.5 * _finite(_metric(scene, "offroad_any"))
    )


def _contact_absolute(scene: dict[str, Any]) -> float:
    return float(
        1.30 * _bounded(_metric(scene, "post_contact_terminal_clearance_m"), 1.0, 3.0)
        + 1.05 * _bounded(_metric(scene, "post_contact_free_space_auc_normalized_m"), 1.0, 3.0)
        + 0.75 * _bounded(_metric(scene, "post_contact_clearance_gain_m"), 0.75, 3.0)
        + 1.20 * _finite(_metric(scene, "post_contact_escape_event"))
        + 1.20 * _finite(_metric(scene, "new_stable_stop_quality_event"))
        - 2.75 * _finite(_metric(scene, "recontact_event"))
        - 1.20 * _bounded(_metric(scene, "post_contact_overlap_duration_s"), 0.5, 3.0)
        - 2.5 * _finite(_metric(scene, "offroad_any"))
        - 0.20 * _bounded(_metric(scene, "yaw_rate_p95"), 0.5, 2.0)
        - 0.10 * _bounded(_metric(scene, "jerk_p95"), 4.0, 2.0)
    )


def _absolute_score(regime: str, scene: dict[str, Any]) -> float:
    if regime == "safe":
        return _safe_absolute(scene)[0]
    if regime == "near":
        return _near_absolute(scene)
    if regime == "contact":
        return _contact_absolute(scene)
    raise ValueError(regime)


def _tier_safe(terms: dict[str, float | None], missing: list[str], score: float, gap_to_best: float) -> tuple[int, str]:
    unsafe = _finite(terms.get("overlap_any")) > 0.5 or _finite(terms.get("offroad_any")) > 0.5
    high = (
        not missing and not unsafe
        and _finite(terms.get("nup")) >= 0.95
        and _finite(terms.get("intervention_rate"), 1.0) <= 0.20
    )
    if high and gap_to_best >= -1.0:
        return 0, "safe_high_quality_noninferior"
    if not missing and not unsafe:
        return 1, "safe_high_quality"
    return 2, "safe_best_available"


def _paired_rows(
    regime: str,
    ocrap: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    baseline_names = list(baselines)
    threshold_ns = SimpleNamespace(**{k: getattr(args, k) for k in DEFAULT_THRESHOLDS})
    rows: list[dict[str, Any]] = []
    for key in sorted(ocrap):
        method_scene = ocrap[key]
        external_quality = {name: _absolute_score(regime, baselines[name][key]) for name in baseline_names}
        best_external = max(external_quality, key=lambda name: (external_quality[name], name))
        worst_external = min(external_quality, key=lambda name: (external_quality[name], name))
        ocrap_quality = _absolute_score(regime, method_scene)
        duration = _duration_available_s(method_scene, args.scenario_horizon_steps, args.metric_dt_s)

        common = {
            "target_key": key,
            "scene_id": method_scene.get("scene_id"),
            "source_scenario_index": method_scene.get("source_scenario_index"),
            "target_time_index": method_scene.get("target_time_index"),
            "regime": regime,
            "available_future_s": duration,
            "ocrap_absolute_score": ocrap_quality,
            "external_absolute_scores": external_quality,
            "best_external_method": best_external,
            "worst_external_method": worst_external,
        }

        if regime == "safe":
            safe_score, terms, missing = _safe_absolute(method_scene)
            best_gap = safe_score - external_quality[best_external]
            # Safe selection is primarily absolute OC-RAP quality.  A small
            # non-inferiority term breaks ties without turning Safe examples into
            # cherry-picked superiority demonstrations.
            selection_score = safe_score + 0.20 * max(-2.0, min(2.0, best_gap))
            tier_rank, tier = _tier_safe(terms, missing, safe_score, best_gap)
            rows.append(common | {
                "score": float(selection_score),
                "selection_tier_rank": tier_rank,
                "selection_tier": tier,
                "evidence_profile": "nominal_preservation",
                "material_improvements": [],
                "regression_reasons": [],
                "missing_required_metrics": missing,
                "terms": terms,
                "best_external_gap": float(best_gap),
                "per_baseline": {
                    name: {
                        "external_absolute_score": external_quality[name],
                        "safe_quality_gap": safe_score - external_quality[name],
                    }
                    for name in baseline_names
                },
            })
            continue

        per_baseline: dict[str, dict[str, Any]] = {}
        pair_scores: list[float] = []
        material_count = 0
        nonregressive_count = 0
        for name in baseline_names:
            evaluation = _pair_evaluate(regime, method_scene, baselines[name][key], threshold_ns)
            pair_scores.append(float(evaluation["score"]))
            material = list(evaluation["material"])
            regressions = list(evaluation["regressions"])
            missing = list(evaluation["missing"])
            if material:
                material_count += 1
            if not regressions and not missing:
                nonregressive_count += 1
            per_baseline[name] = {
                "relative_score": float(evaluation["score"]),
                "material_improvements": material,
                "regression_reasons": regressions,
                "missing_required_metrics": missing,
                "evidence_profile": evaluation["evidence_profile"],
                "terms": evaluation["terms"],
                "external_absolute_score": external_quality[name],
            }

        hardest = min(baseline_names, key=lambda name: (per_baseline[name]["relative_score"], name))
        pair_best = per_baseline[best_external]
        worst_pair_score = min(pair_scores)
        median_pair_score = float(statistics.median(pair_scores))
        best_pair_score = max(pair_scores)
        robust_score = 0.55 * worst_pair_score + 0.35 * median_pair_score + 0.10 * best_pair_score
        majority = math.ceil(len(baseline_names) / 2)
        no_unsafe_vs_best = not pair_best["regression_reasons"] and not pair_best["missing_required_metrics"]
        no_unsafe_any = all(not row["regression_reasons"] for row in per_baseline.values())
        strict = (
            no_unsafe_vs_best
            and bool(pair_best["material_improvements"])
            and pair_best["relative_score"] > 0.0
            and material_count >= majority
            and nonregressive_count == len(baseline_names)
        )
        if strict:
            tier_rank, tier = 0, "beats_scene_best_strict"
        elif no_unsafe_vs_best and material_count >= majority and median_pair_score > 0.0:
            tier_rank, tier = 1, "majority_material_gain"
        elif no_unsafe_any and median_pair_score >= 0.0:
            tier_rank, tier = 2, "all_nonregressive_best_available"
        else:
            tier_rank, tier = 3, "best_available"

        rows.append(common | {
            "score": float(robust_score),
            "selection_tier_rank": tier_rank,
            "selection_tier": tier,
            "evidence_profile": str(pair_best.get("evidence_profile") or "balanced"),
            "material_improvements": list(pair_best["material_improvements"]),
            "regression_reasons": list(pair_best["regression_reasons"]),
            "missing_required_metrics": list(pair_best["missing_required_metrics"]),
            "terms": pair_best["terms"],
            "best_external_gap": float(pair_best["relative_score"]),
            "hardest_external_method": hardest,
            "worst_pair_score": float(worst_pair_score),
            "median_pair_score": float(median_pair_score),
            "best_pair_score": float(best_pair_score),
            "num_material_external_comparisons": int(material_count),
            "num_nonregressive_external_comparisons": int(nonregressive_count),
            "per_baseline": per_baseline,
        })
    return rows


def _select_diverse(rows: list[dict[str, Any]], count: int, diversify: bool) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: (int(r["selection_tier_rank"]), -float(r["score"]), str(r["target_key"])))
    selected: list[dict[str, Any]] = []
    used_scenes: set[str] = set()
    used_profiles: set[str] = set()

    def add(row: dict[str, Any]) -> bool:
        scene_id = str(row.get("scene_id") or row["target_key"])
        if scene_id in used_scenes:
            return False
        selected.append(row)
        used_scenes.add(scene_id)
        used_profiles.add(str(row.get("evidence_profile") or ""))
        return True

    if diversify:
        for row in ordered:
            profile = str(row.get("evidence_profile") or "")
            if profile in used_profiles:
                continue
            add(row)
            if len(selected) >= count:
                return selected
    for row in ordered:
        add(row)
        if len(selected) >= count:
            break
    return selected


def _parse_baseline_specs(specs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"invalid --baseline {spec!r}; expected METHOD=SCENES.jsonl")
        name, raw = spec.split("=", 1)
        name = name.strip()
        if not name or name in out:
            raise SystemExit(f"duplicate/empty baseline name in {spec!r}")
        out[name] = Path(raw)
    if not out:
        raise SystemExit("at least one --baseline is required")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", choices=("safe", "near", "contact"), required=True)
    ap.add_argument("--ocrap-scenes", type=Path, required=True)
    ap.add_argument("--baseline", action="append", default=[], metavar="METHOD=SCENES.jsonl")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-keys-output", type=Path)
    ap.add_argument("--num-scenes", type=int, default=5)
    ap.add_argument("--min-duration-s", type=float, default=5.0)
    ap.add_argument("--fallback-min-duration-s", type=float, default=3.0)
    ap.add_argument("--scenario-horizon-steps", type=int, default=91)
    ap.add_argument("--metric-dt-s", type=float, default=0.1)
    ap.add_argument("--diversify-evidence-profiles", action=argparse.BooleanOptionalAction, default=True)
    for key, value in DEFAULT_THRESHOLDS.items():
        ap.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    args = ap.parse_args()

    if args.num_scenes <= 0:
        raise SystemExit("--num-scenes must be positive")
    if args.fallback_min_duration_s <= 0 or args.min_duration_s < args.fallback_min_duration_s:
        raise SystemExit("duration thresholds must satisfy min >= fallback > 0")
    if args.scenario_horizon_steps < 2 or args.metric_dt_s <= 0:
        raise SystemExit("invalid scenario horizon / metric dt")

    baseline_paths = _parse_baseline_specs(args.baseline)
    ocrap = _load_scenes(args.ocrap_scenes)
    baselines = {name: _load_scenes(path) for name, path in baseline_paths.items()}
    reference_keys = set(ocrap)
    mismatch = {
        name: {
            "ocrap_only": sorted(reference_keys - set(rows))[:10],
            "baseline_only": sorted(set(rows) - reference_keys)[:10],
        }
        for name, rows in baselines.items()
        if set(rows) != reference_keys
    }
    if mismatch:
        raise SystemExit(f"unpaired target sets: {json.dumps(mismatch, ensure_ascii=False)}")

    rows = _paired_rows(args.regime, ocrap, baselines, args)
    long_rows = [r for r in rows if r["available_future_s"] is not None and float(r["available_future_s"]) + 1e-9 >= args.min_duration_s]
    fallback_rows = [r for r in rows if r["available_future_s"] is not None and float(r["available_future_s"]) + 1e-9 >= args.fallback_min_duration_s]
    if len(long_rows) >= args.num_scenes:
        duration_threshold = float(args.min_duration_s)
        duration_pool = long_rows
        duration_mode = "preferred"
    else:
        duration_threshold = float(args.fallback_min_duration_s)
        duration_pool = fallback_rows
        duration_mode = "fallback"
    if len(duration_pool) < args.num_scenes:
        raise SystemExit(
            f"{args.regime}: need {args.num_scenes} scenes with >= {duration_threshold:.1f}s future horizon, "
            f"found {len(duration_pool)} (>= {args.min_duration_s:.1f}s: {len(long_rows)}, "
            f">= {args.fallback_min_duration_s:.1f}s: {len(fallback_rows)})"
        )

    selected = _select_diverse(
        duration_pool,
        args.num_scenes,
        args.diversify_evidence_profiles and args.regime != "safe",
    )
    if len(selected) != args.num_scenes:
        raise SystemExit(f"{args.regime}: diversity filtering yielded only {len(selected)} scenes")
    selected = [
        row | {
            "category": "visualization_example",
            "category_rank": rank,
            "clip_duration_s": duration_threshold,
            "duration_selection_mode": duration_mode,
        }
        for rank, row in enumerate(selected, 1)
    ]

    doc = {
        "event": "regime_visualization_scene_selection_v51",
        "regime": args.regime,
        "exploratory_qualitative_only": True,
        "paper_population_claim_allowed": False,
        "selection_note": (
            "Safe is ranked by high absolute OC-RAP closed-loop quality with safety guards; "
            "Near/Contact are ranked by robust multi-baseline relative effects.  Selection is post-hoc qualitative evidence."
        ),
        "num_external_baselines": len(baselines),
        "external_baselines": list(baselines),
        "num_paired_targets": len(rows),
        "requested_num_scenes": args.num_scenes,
        "preferred_min_duration_s": args.min_duration_s,
        "fallback_min_duration_s": args.fallback_min_duration_s,
        "num_preferred_duration_candidates": len(long_rows),
        "num_fallback_duration_candidates": len(fallback_rows),
        "selected_clip_duration_s": duration_threshold,
        "duration_selection_mode": duration_mode,
        "scenario_horizon_steps": args.scenario_horizon_steps,
        "metric_dt_s": args.metric_dt_s,
        "selected": selected,
        "target_keys": [r["target_key"] for r in selected],
        "all_scene_scores": sorted(rows, key=lambda r: (int(r["selection_tier_rank"]), -float(r["score"]), str(r["target_key"]))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.target_keys_output:
        args.target_keys_output.parent.mkdir(parents=True, exist_ok=True)
        args.target_keys_output.write_text(
            json.dumps({"regime": args.regime, "target_keys": doc["target_keys"]}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "event": doc["event"],
        "regime": args.regime,
        "selected": len(selected),
        "clip_duration_s": duration_threshold,
        "duration_mode": duration_mode,
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
