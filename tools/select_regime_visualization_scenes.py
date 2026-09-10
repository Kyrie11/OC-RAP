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
  rank robust relative gains against the complete external-baseline set. Contact uses generic physical recovery on the current counterfactual contact-surrogate cohort; observed-contact-only diagnostics are not required for selection.  The
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
# multi-baseline selector stays numerically consistent with the previous validated
# qualitative selector.
from critical_scene_metrics import _evaluate as _pair_evaluate  # type: ignore


# Compact full-run metrics carried into the rendering artifact.  These are
# deliberately the same endpoint families used in the paper protocol; the
# renderer labels them as full-run selection metrics so they are not confused
# with the longer selective trace rerun used only for animation.
METRIC_SPECS: dict[str, tuple[tuple[str, tuple[str, ...], str, str], ...]] = {
    "safe": (
        ("nup", ("closed_loop_bounded_NUP",), "NUP", "higher"),
        ("intervention_rate", ("intervention_rate",), "Intervention", "lower"),
        ("clearance_p05_m", ("min_clearance_m_p05", "min_clearance_m_min"), "Clearance p05", "higher"),
        ("ttc_p05_s", ("ttc_s_p05", "ttc_s_min"), "TTC p05", "higher"),
        ("overlap_any", ("overlap_any",), "Overlap", "lower"),
        ("offroad_any", ("offroad_any",), "Off-road", "lower"),
    ),
    "near": (
        ("ttc_p05_s", ("ttc_s_p05",), "TTC p05", "higher"),
        ("clearance_p05_m", ("min_clearance_m_p05",), "Clearance p05", "higher"),
        ("critical_ttc_exposure_s", ("critical_ttc_exposure_duration_s",), "Critical-TTC exposure", "lower"),
        ("near_zero_clearance_rate", ("near_zero_clearance_exposure_rate",), "Near-zero exposure", "lower"),
        ("overlap_any", ("overlap_any",), "Overlap", "lower"),
        ("offroad_any", ("offroad_any",), "Off-road", "lower"),
    ),
    "contact": (
        ("clearance_p05_m", ("min_clearance_m_p05",), "Clearance p05", "higher"),
        ("terminal_clearance_m", ("terminal_clearance_m",), "Terminal clearance", "higher"),
        ("clearance_gain_m", ("clearance_recovery_gain_m",), "Clearance recovery", "higher"),
        ("overlap_duration_s", ("overlap_duration_s",), "Overlap duration", "lower"),
        ("penetration_duration_s", ("penetration_duration_s",), "Penetration duration", "lower"),
        ("max_penetration_depth_m", ("penetration_depth_m_max",), "Max penetration", "lower"),
        ("stable_stop", ("new_stable_stop_quality_event",), "Stable stop", "higher"),
        ("overlap_any", ("overlap_any",), "Overlap", "lower"),
        ("offroad_any", ("offroad_any",), "Off-road", "lower"),
    ),
}


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
    min_contact_penetration_duration_reduction_s=0.10,
    min_contact_penetration_depth_reduction_m=0.05,
    max_contact_penetration_duration_regression_s=0.10,
    max_contact_penetration_depth_regression_m=0.05,
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


def _metric_snapshot(regime: str, scene: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, names, _label, _direction in METRIC_SPECS[regime]:
        out[key] = _metric(scene, *names)
    return out


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
    """Absolute quality on the current Contact *surrogate* cohort.

    The dataset builder does not materialize a physical post-impact impulse in
    raw WOMD replay, so population qualitative ranking must not fabricate a
    contact anchor.  Use generic physical recovery/stability metrics that are
    defined for every paired target; observed-contact diagnostics remain
    supplementary when an actual overlap occurs.
    """
    return float(
        1.10 * _bounded(_metric(scene, "min_clearance_m_p05"), 1.0, 3.0)
        + 1.10 * _bounded(_metric(scene, "terminal_clearance_m"), 2.0, 3.0)
        + 0.80 * _bounded(_metric(scene, "clearance_recovery_gain_m"), 1.0, 3.0)
        - 1.10 * _bounded(_metric(scene, "overlap_duration_s"), 0.5, 3.0)
        - 1.00 * _bounded(_metric(scene, "penetration_duration_s"), 0.25, 3.0)
        - 0.85 * _bounded(_metric(scene, "penetration_depth_m_max"), 0.25, 3.0)
        + 1.00 * _finite(_metric(scene, "new_stable_stop_quality_event"))
        - 3.25 * _finite(_metric(scene, "overlap_any"))
        - 2.50 * _finite(_metric(scene, "offroad_any"))
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


def _delta_metric(method: dict[str, Any], control: dict[str, Any], *names: str) -> float | None:
    a = _metric(method, *names)
    b = _metric(control, *names)
    return None if a is None or b is None else float(a - b)


def _evaluate_contact_surrogate(
    method: dict[str, Any],
    control: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Paired Contact-surrogate score with no fabricated post-impact anchor."""
    raw_terms = {
        "clearance_p05_m": _delta_metric(method, control, "min_clearance_m_p05"),
        "terminal_clearance_m": _delta_metric(method, control, "terminal_clearance_m"),
        "clearance_gain_m": _delta_metric(method, control, "clearance_recovery_gain_m"),
        "overlap_duration_s": _delta_metric(method, control, "overlap_duration_s"),
        "penetration_duration_s": _delta_metric(method, control, "penetration_duration_s"),
        "max_penetration_depth_m": _delta_metric(method, control, "penetration_depth_m_max"),
        "stable_stop": _delta_metric(method, control, "new_stable_stop_quality_event"),
        "overlap_any": _delta_metric(method, control, "overlap_any"),
        "offroad_any": _delta_metric(method, control, "offroad_any"),
        "yaw_rate_p95": _delta_metric(method, control, "yaw_rate_p95"),
        "jerk_p95": _delta_metric(method, control, "jerk_p95"),
        "route_progression_m": _delta_metric(method, control, "route_progression_m"),
        "bounded_nup": _delta_metric(method, control, "closed_loop_bounded_NUP"),
    }
    required = (
        "clearance_p05_m", "terminal_clearance_m", "clearance_gain_m",
        "overlap_duration_s", "penetration_duration_s", "max_penetration_depth_m",
        "stable_stop", "overlap_any", "offroad_any",
    )
    missing = [name for name in required if raw_terms[name] is None]
    terms = raw_terms
    components = {
        "clearance_p05": 1.20 * _bounded(terms["clearance_p05_m"], 0.5),
        "terminal_clearance": 1.10 * _bounded(terms["terminal_clearance_m"], 0.75),
        "clearance_recovery": 0.80 * _bounded(terms["clearance_gain_m"], 0.5),
        "overlap_duration": -1.15 * _bounded(terms["overlap_duration_s"], 0.20),
        "penetration_duration": -1.10 * _bounded(terms["penetration_duration_s"], 0.10),
        "penetration_depth": -0.95 * _bounded(terms["max_penetration_depth_m"], 0.10),
        "stable_stop": 1.20 * _finite(terms["stable_stop"]),
        "overlap_any": -2.50 * _finite(terms["overlap_any"]),
        "offroad_any": -2.50 * _finite(terms["offroad_any"]),
        "yaw_stability": -0.35 * _bounded(terms["yaw_rate_p95"], 0.25, 2.0),
        "jerk_stability": -0.20 * _bounded(terms["jerk_p95"], 2.0, 2.0),
        "route_progress": 0.25 * _bounded(terms["route_progression_m"], 0.5, 2.0),
        "bounded_nup": 0.20 * _bounded(terms["bounded_nup"], 0.10, 2.0),
    }
    material: list[str] = []
    if _finite(terms["terminal_clearance_m"]) >= args.min_contact_terminal_clearance_gain_m:
        material.append("terminal_clearance")
    if _finite(terms["clearance_gain_m"]) >= args.min_contact_clearance_gain_m:
        material.append("clearance_recovery")
    if _finite(terms["overlap_duration_s"]) <= -args.min_contact_overlap_duration_reduction_s:
        material.append("overlap_duration_reduced")
    if _finite(terms["penetration_duration_s"]) <= -args.min_contact_penetration_duration_reduction_s:
        material.append("penetration_duration_reduced")
    if _finite(terms["max_penetration_depth_m"]) <= -args.min_contact_penetration_depth_reduction_m:
        material.append("penetration_depth_reduced")
    if _finite(terms["stable_stop"]) > 0.0:
        material.append("new_stable_stop")

    regressions: list[str] = []
    guards = {
        "terminal_clearance_regression": _finite(terms["terminal_clearance_m"]) < -args.max_contact_terminal_clearance_regression_m,
        "overlap_duration_regression": _finite(terms["overlap_duration_s"]) > args.max_contact_overlap_duration_regression_s,
        "penetration_duration_regression": _finite(terms["penetration_duration_s"]) > args.max_contact_penetration_duration_regression_s,
        "penetration_depth_regression": _finite(terms["max_penetration_depth_m"]) > args.max_contact_penetration_depth_regression_m,
        "new_overlap_regression": _finite(terms["overlap_any"]) > 1.0e-9,
        "offroad_regression": _finite(terms["offroad_any"]) > 1.0e-9,
    }
    if terms["yaw_rate_p95"] is not None:
        guards["yaw_rate_regression"] = _finite(terms["yaw_rate_p95"]) > args.max_contact_yaw_rate_regression_radps
    if terms["jerk_p95"] is not None:
        guards["jerk_regression"] = _finite(terms["jerk_p95"]) > args.max_contact_jerk_regression_mps3
    if terms["route_progression_m"] is not None:
        guards["route_progress_regression"] = _finite(terms["route_progression_m"]) < -args.max_contact_route_progress_regression_m
    regressions.extend(name for name, failed in guards.items() if failed)

    profile_effects = {
        "clearance_recovery": max(_bounded(terms["terminal_clearance_m"], 0.75), _bounded(terms["clearance_gain_m"], 0.5)),
        "penetration_avoidance": max(_bounded(-_finite(terms["penetration_duration_s"]), 0.10), _bounded(-_finite(terms["max_penetration_depth_m"]), 0.10)),
        "overlap_avoidance": _bounded(-_finite(terms["overlap_duration_s"]), 0.20),
        "stabilization": _finite(terms["stable_stop"]),
    }
    profile, effect = max(profile_effects.items(), key=lambda item: (item[1], item[0]))
    if effect <= 0.0:
        profile = "balanced_nonregressive"
    return {
        "score": float(sum(components.values())),
        "score_components": components,
        "terms": terms,
        "material": material,
        "regressions": sorted(set(regressions)),
        "missing": missing,
        "evidence_profile": profile,
    }


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
        external_metrics = {name: _metric_snapshot(regime, baselines[name][key]) for name in baseline_names}
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
            "ocrap_metrics": _metric_snapshot(regime, method_scene),
            "external_metrics": external_metrics,
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
                "primary_external_method": best_external,
                "primary_comparator_reason": "highest per-scene external absolute Safe quality",
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
            evaluation = (
                _evaluate_contact_surrogate(method_scene, baselines[name][key], threshold_ns)
                if regime == "contact"
                else _pair_evaluate(regime, method_scene, baselines[name][key], threshold_ns)
            )
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

        # For Near/Contact the reviewer-facing comparator must be the *hardest*
        # baseline under the same paired critical-safety score used for selection,
        # not the weakest method and not an unrelated scalar-quality winner.
        # Positive relative_score means OC-RAP is better, so the smallest value
        # is the strongest/hardest external comparator on this target.
        hardest = min(baseline_names, key=lambda name: (per_baseline[name]["relative_score"], name))
        pair_best = per_baseline[best_external]
        pair_primary = per_baseline[hardest]
        worst_pair_score = min(pair_scores)
        median_pair_score = float(statistics.median(pair_scores))
        best_pair_score = max(pair_scores)
        robust_score = 0.55 * worst_pair_score + 0.35 * median_pair_score + 0.10 * best_pair_score
        majority = math.ceil(len(baseline_names) / 2)
        no_unsafe_vs_primary = not pair_primary["regression_reasons"] and not pair_primary["missing_required_metrics"]
        no_unsafe_any = all(not row["regression_reasons"] and not row["missing_required_metrics"] for row in per_baseline.values())
        strict = (
            no_unsafe_vs_primary
            and bool(pair_primary["material_improvements"])
            and pair_primary["relative_score"] > 0.0
            and material_count >= majority
            and nonregressive_count == len(baseline_names)
        )
        if strict:
            tier_rank, tier = 0, "beats_hardest_external_strict"
        elif no_unsafe_vs_primary and material_count >= majority and median_pair_score > 0.0:
            tier_rank, tier = 1, "majority_material_gain"
        elif no_unsafe_any and median_pair_score >= 0.0:
            tier_rank, tier = 2, "all_nonregressive_best_available"
        else:
            tier_rank, tier = 3, "best_available"

        rows.append(common | {
            "score": float(robust_score),
            "selection_tier_rank": tier_rank,
            "selection_tier": tier,
            "evidence_profile": str(pair_primary.get("evidence_profile") or "balanced"),
            "material_improvements": list(pair_primary["material_improvements"]),
            "regression_reasons": list(pair_primary["regression_reasons"]),
            "missing_required_metrics": list(pair_primary["missing_required_metrics"]),
            "terms": pair_primary["terms"],
            "best_external_gap": float(pair_best["relative_score"]),
            "primary_external_method": hardest,
            "primary_comparator_reason": "lowest paired critical-safety score across all external baselines (hardest to beat)",
            "primary_pair_score": float(pair_primary["relative_score"]),
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


def _global_external_ranking(regime: str, rows: list[dict[str, Any]], baseline_names: list[str]) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for name in baseline_names:
        if regime == "safe":
            values = [float(r["external_absolute_scores"][name]) for r in rows]
            score = float(statistics.median(values))
            ranking.append({"method": name, "median_external_absolute_score": score, "sort_value": -score})
        else:
            values = [float(r["per_baseline"][name]["relative_score"]) for r in rows]
            # Lower OC-RAP-minus-baseline score means a harder external comparator.
            score = float(statistics.median(values))
            ranking.append({"method": name, "median_ocrap_relative_score": score, "sort_value": score})
    ranking.sort(key=lambda r: (float(r["sort_value"]), str(r["method"])))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
        row.pop("sort_value", None)
    return ranking


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
    ap.add_argument(
        "--max-selected-tier-rank", type=int, default=None,
        help="Fail closed if fewer than --num-scenes candidates exist at or above this evidence tier (0=strict, 1=strong).",
    )
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
    tier_rows = rows
    if args.max_selected_tier_rank is not None:
        tier_rows = [r for r in rows if int(r["selection_tier_rank"]) <= int(args.max_selected_tier_rank)]
    long_rows = [r for r in tier_rows if r["available_future_s"] is not None and float(r["available_future_s"]) + 1e-9 >= args.min_duration_s]
    fallback_rows = [r for r in tier_rows if r["available_future_s"] is not None and float(r["available_future_s"]) + 1e-9 >= args.fallback_min_duration_s]
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

    global_ranking = _global_external_ranking(args.regime, rows, list(baselines))
    global_strongest = str(global_ranking[0]["method"]) if global_ranking else None
    selected = [row | {"global_strongest_external_method": global_strongest} for row in selected]

    doc = {
        "event": "regime_visualization_scene_selection_v52",
        "regime": args.regime,
        "exploratory_qualitative_only": True,
        "paper_population_claim_allowed": False,
        "selection_note": (
            "All main-table external baselines participate in selection. Safe is ranked by high absolute OC-RAP closed-loop quality with safety guards; "
            "Near/Contact are ranked by robust multi-baseline relative effects and the reviewer-facing comparator is the per-scene hardest baseline. "
            "For Contact, current test_contact is a counterfactual contact-surrogate cohort, so selection uses generic physical recovery/overlap/penetration/stability metrics rather than fabricated post-contact anchors. "
            "Selection is post-hoc qualitative evidence and does not replace population-level tables."
        ),
        "max_selected_tier_rank": args.max_selected_tier_rank,
        "global_external_ranking": global_ranking,
        "global_strongest_external_method": global_strongest,
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
