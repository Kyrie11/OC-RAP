#!/usr/bin/env python3
"""Deterministically select paired Near/Contact qualitative examples.

The selector consumes metric-only scene journals.  It uses bounded,
regime-specific effect scores, explicit safety/regression guards and optional
mechanism diversity.  Rendering traces are recorded only for the selected
keys, so qualitative videos remain cheap and reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REQUIRED = {
    "near": {
        "ttc_s_p05", "terminal_ttc_s", "min_clearance_m_p05", "terminal_clearance_m",
        "critical_ttc_exposure_duration_s", "near_zero_clearance_exposure_rate",
        "overlap_any", "offroad_any",
    },
    "contact": {
        "post_contact_terminal_clearance_m", "post_contact_free_space_auc_normalized_m",
        "post_contact_clearance_gain_m", "post_contact_overlap_duration_s",
        "post_contact_escape_event", "recontact_event",
        "new_stable_stop_quality_event", "offroad_any",
    },
}


def _finite_or_none(x: Any) -> float | None:
    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def _finite(x: Any, default: float = 0.0) -> float:
    value = _finite_or_none(x)
    return default if value is None else value


def _scaled(value: Any, scale: float, limit: float = 3.0) -> float:
    """Dimensionless bounded effect used to prevent sentinel TTC domination."""
    x = _finite(value) / max(float(scale), 1.0e-9)
    return max(-limit, min(limit, x))


def _scene_rows(path: Path) -> dict[str, dict[str, Any]]:
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
            key = str(scene.get("target_key") or envelope.get("resume_key") or "")
            if key.startswith("target:"):
                key = key[len("target:"):]
            if not key:
                scene_id = str(scene.get("scene_id") or "")
                target_time = scene.get("target_time_index")
                key = f"{scene_id}:t{target_time}" if scene_id and target_time is not None else scene_id
            if not key:
                raise ValueError(f"scene row without target/scene key in {path}")
            if key in rows:
                raise ValueError(f"duplicate scene key {key} in {path}")
            rows[key] = {
                name: value for name, value in scene.items()
                if name not in {"decisions", "render_trace", "render_context", "render_trace_schema", "state_xy_trace"}
                and not str(name).endswith("_trace")
            }
    if not rows:
        raise ValueError(f"empty paired scene journal: {path}")
    return rows


def _metric(scene: dict[str, Any], name: str) -> float | None:
    if name in scene:
        return _finite_or_none(scene.get(name))
    return _finite_or_none((scene.get("metric_summary", {}) or {}).get(name))


def _delta(method: dict[str, Any], control: dict[str, Any], name: str) -> float | None:
    method_value = _metric(method, name)
    control_value = _metric(control, name)
    return None if method_value is None or control_value is None else method_value - control_value


def _unsafe_regressions(regime: str, terms: dict[str, float | None]) -> list[str]:
    names = ("overlap_any", "offroad_any") if regime == "near" else ("offroad_any", "recontact_event")
    return [name for name in names if _finite(terms.get(name)) > 1.0e-9]


def _near_profile(terms: dict[str, float | None]) -> str:
    effects = {
        "ttc_margin": max(_scaled(terms.get("ttc_p05_s"), 1.0), _scaled(terms.get("terminal_ttc_s"), 2.0)),
        "geometric_clearance": max(_scaled(terms.get("clearance_p05_m"), 0.5), _scaled(terms.get("terminal_clearance_m"), 0.75)),
        "reduced_exposure": max(_scaled(-_finite(terms.get("critical_ttc_exposure_s")), 0.5), _scaled(-_finite(terms.get("near_zero_clearance_rate")), 0.05)),
    }
    profile, effect = max(effects.items(), key=lambda item: (item[1], item[0]))
    return profile if effect > 0 else "balanced_nonregressive"


def _contact_profile(terms: dict[str, float | None]) -> str:
    effects = {
        "separation_recovery": max(
            _scaled(terms.get("post_contact_terminal_clearance_m"), 0.75),
            _scaled(terms.get("post_contact_free_space_auc_normalized_m"), 0.75),
            _scaled(terms.get("post_contact_clearance_gain_m"), 0.5),
        ),
        "overlap_escape": max(
            _finite(terms.get("post_contact_escape_event")),
            _scaled(-_finite(terms.get("post_contact_overlap_duration_s")), 0.2),
        ),
        "recontact_avoidance": -_finite(terms.get("recontact_event")),
        "stable_stop": _finite(terms.get("new_stable_stop_quality_event")),
        "controlled_continuation": max(
            _scaled(-_finite(terms.get("yaw_rate_p95")), 0.25),
            _scaled(-_finite(terms.get("jerk_p95")), 2.0),
            _scaled(terms.get("route_progression_m"), 0.5),
        ),
    }
    profile, effect = max(effects.items(), key=lambda item: (item[1], item[0]))
    return profile if effect > 0 else "balanced_nonregressive"


def _evaluate(
    regime: str,
    method: dict[str, Any],
    control: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    missing = [name for name in sorted(REQUIRED[regime]) if _metric(method, name) is None or _metric(control, name) is None]
    common: dict[str, float | None] = {
        "bounded_nup": _delta(method, control, "closed_loop_bounded_NUP"),
        "intervention_rate": _finite_or_none(method.get("intervention_rate")),
        "overlap_any": _delta(method, control, "overlap_any"),
        "offroad_any": _delta(method, control, "offroad_any"),
    }
    material: list[str] = []
    regressions: list[str] = []
    components: dict[str, float] = {}

    if regime == "near":
        terms = common | {
            "ttc_p05_s": _delta(method, control, "ttc_s_p05"),
            "terminal_ttc_s": _delta(method, control, "terminal_ttc_s"),
            "clearance_p05_m": _delta(method, control, "min_clearance_m_p05"),
            "terminal_clearance_m": _delta(method, control, "terminal_clearance_m"),
            "critical_ttc_exposure_s": _delta(method, control, "critical_ttc_exposure_duration_s"),
            "near_zero_clearance_rate": _delta(method, control, "near_zero_clearance_exposure_rate"),
        }
        components = {
            "ttc_p05": 1.40 * _scaled(terms["ttc_p05_s"], 1.0),
            "terminal_ttc": 0.35 * _scaled(terms["terminal_ttc_s"], 2.0, 2.0),
            "clearance_p05": 1.20 * _scaled(terms["clearance_p05_m"], 0.5),
            "terminal_clearance": 0.70 * _scaled(terms["terminal_clearance_m"], 0.75),
            "critical_exposure": -1.20 * _scaled(terms["critical_ttc_exposure_s"], 0.5),
            "near_zero_exposure": -0.70 * _scaled(terms["near_zero_clearance_rate"], 0.05),
            "bounded_nup": 0.25 * _scaled(terms["bounded_nup"], 0.10, 2.0),
        }
        if _finite(terms["ttc_p05_s"]) >= args.min_near_ttc_gain_s:
            material.append("ttc_p05")
        if _finite(terms["clearance_p05_m"]) >= args.min_near_clearance_gain_m:
            material.append("clearance_p05")
        if _finite(terms["terminal_clearance_m"]) >= args.min_near_clearance_gain_m:
            material.append("terminal_clearance")
        if _finite(terms["critical_ttc_exposure_s"]) <= -args.min_near_exposure_reduction_s:
            material.append("critical_exposure")
        if _finite(terms["near_zero_clearance_rate"]) <= -args.min_near_near_zero_reduction_rate:
            material.append("near_zero_exposure")
        guards = {
            "ttc_p05_regression": _finite(terms["ttc_p05_s"]) < -args.max_near_ttc_regression_s,
            "terminal_ttc_regression": _finite(terms["terminal_ttc_s"]) < -args.max_near_terminal_ttc_regression_s,
            "clearance_p05_regression": _finite(terms["clearance_p05_m"]) < -args.max_near_clearance_regression_m,
            "terminal_clearance_regression": _finite(terms["terminal_clearance_m"]) < -args.max_near_terminal_clearance_regression_m,
            "critical_exposure_regression": _finite(terms["critical_ttc_exposure_s"]) > args.max_near_exposure_regression_s,
            "near_zero_exposure_regression": _finite(terms["near_zero_clearance_rate"]) > args.max_near_near_zero_regression_rate,
        }
        regressions.extend(name for name, failed in guards.items() if failed)
        profile = _near_profile(terms)
    elif regime == "contact":
        terms = common | {
            "post_contact_terminal_clearance_m": _delta(method, control, "post_contact_terminal_clearance_m"),
            "post_contact_free_space_auc_normalized_m": _delta(method, control, "post_contact_free_space_auc_normalized_m"),
            "post_contact_clearance_gain_m": _delta(method, control, "post_contact_clearance_gain_m"),
            "ttc_recovery_gain_s": _delta(method, control, "ttc_recovery_gain_s"),
            "post_contact_overlap_duration_s": _delta(method, control, "post_contact_overlap_duration_s"),
            "new_stable_stop_quality_event": _delta(method, control, "new_stable_stop_quality_event"),
            "post_contact_escape_event": _delta(method, control, "post_contact_escape_event"),
            "recontact_event": _delta(method, control, "recontact_event"),
            "yaw_rate_p95": _delta(method, control, "yaw_rate_p95"),
            "jerk_p95": _delta(method, control, "jerk_p95"),
            "route_progression_m": _delta(method, control, "route_progression_m"),
        }
        components = {
            "terminal_clearance": 1.30 * _scaled(terms["post_contact_terminal_clearance_m"], 0.75),
            "free_space_auc": 1.00 * _scaled(terms["post_contact_free_space_auc_normalized_m"], 0.75),
            "clearance_gain": 0.80 * _scaled(terms["post_contact_clearance_gain_m"], 0.5),
            "overlap_duration": -1.25 * _scaled(terms["post_contact_overlap_duration_s"], 0.2),
            "stable_stop": 1.50 * _finite(terms["new_stable_stop_quality_event"]),
            "escape": 1.20 * _finite(terms["post_contact_escape_event"]),
            "recontact": -3.00 * _finite(terms["recontact_event"]),
            # TTC is useful but horizon/sentinel values must never dominate the contact story.
            "ttc_recovery": 0.15 * _scaled(terms["ttc_recovery_gain_s"], 3.0, 2.0),
            "yaw_stability": -0.50 * _scaled(terms["yaw_rate_p95"], 0.25, 2.0),
            "jerk_stability": -0.25 * _scaled(terms["jerk_p95"], 2.0, 2.0),
            "route_progress": 0.30 * _scaled(terms["route_progression_m"], 0.5, 2.0),
            "bounded_nup": 0.20 * _scaled(terms["bounded_nup"], 0.10, 2.0),
        }
        if _finite(terms["post_contact_terminal_clearance_m"]) >= args.min_contact_terminal_clearance_gain_m:
            material.append("terminal_clearance")
        if _finite(terms["post_contact_free_space_auc_normalized_m"]) >= args.min_contact_auc_gain_m:
            material.append("free_space_auc")
        if _finite(terms["post_contact_clearance_gain_m"]) >= args.min_contact_clearance_gain_m:
            material.append("clearance_gain")
        if _finite(terms["post_contact_escape_event"]) > 0:
            material.append("escape_event")
        if _finite(terms["post_contact_overlap_duration_s"]) <= -args.min_contact_overlap_duration_reduction_s:
            material.append("overlap_duration_reduced")
        if _finite(terms["new_stable_stop_quality_event"]) > 0:
            material.append("new_stable_stop")
        if _finite(terms["recontact_event"]) < 0:
            material.append("recontact_avoided")
        if (_finite(terms["yaw_rate_p95"]) <= -args.min_contact_yaw_rate_reduction_radps
                or _finite(terms["jerk_p95"]) <= -args.min_contact_jerk_reduction_mps3):
            material.append("dynamic_stability")
        guards = {
            "terminal_clearance_regression": _finite(terms["post_contact_terminal_clearance_m"]) < -args.max_contact_terminal_clearance_regression_m,
            "free_space_auc_regression": _finite(terms["post_contact_free_space_auc_normalized_m"]) < -args.max_contact_auc_regression_m,
            "overlap_duration_regression": _finite(terms["post_contact_overlap_duration_s"]) > args.max_contact_overlap_duration_regression_s,
        }
        # Optional metrics only guard a scene when both journals actually expose them.
        if terms["yaw_rate_p95"] is not None:
            guards["yaw_rate_regression"] = _finite(terms["yaw_rate_p95"]) > args.max_contact_yaw_rate_regression_radps
        if terms["jerk_p95"] is not None:
            guards["jerk_regression"] = _finite(terms["jerk_p95"]) > args.max_contact_jerk_regression_mps3
        if terms["route_progression_m"] is not None:
            guards["route_progress_regression"] = _finite(terms["route_progression_m"]) < -args.max_contact_route_progress_regression_m
        regressions.extend(name for name, failed in guards.items() if failed)
        profile = _contact_profile(terms)
    else:
        raise ValueError(regime)

    regressions.extend(_unsafe_regressions(regime, terms))
    score = float(sum(components.values()))
    intervention = _finite_or_none(method.get("intervention_rate"))
    fallback_eligible = (
        not missing
        and intervention is not None
        and intervention > 0
        and score >= args.minimum_positive_score
        and not regressions
    )
    return {
        "score": score,
        "score_components": components,
        "terms": terms,
        "eligible": bool(fallback_eligible and material),
        "fallback_eligible": bool(fallback_eligible),
        "missing": missing,
        "material": material,
        "regressions": sorted(set(regressions)),
        "evidence_profile": profile,
    }


def _take_diverse(
    pool: list[dict[str, Any]],
    count: int,
    max_per_scene: int,
    diversify_profiles: bool,
    already: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected = list(already or [])
    selected_keys = {str(row["target_key"]) for row in selected}
    scene_counts: dict[str, int] = {}
    used_profiles: set[str] = set()
    for row in selected:
        scene_id = str(row.get("scene_id") or row["target_key"])
        scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
        used_profiles.add(str(row.get("evidence_profile") or ""))

    ordered = sorted(pool, key=lambda row: (-float(row["score"]), str(row["target_key"])))

    def add(row: dict[str, Any]) -> bool:
        key = str(row["target_key"])
        scene_id = str(row.get("scene_id") or key)
        if key in selected_keys or scene_counts.get(scene_id, 0) >= max(max_per_scene, 1):
            return False
        selected.append(row)
        selected_keys.add(key)
        scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
        used_profiles.add(str(row.get("evidence_profile") or ""))
        return True

    if diversify_profiles:
        for row in ordered:
            profile = str(row.get("evidence_profile") or "")
            if profile in used_profiles:
                continue
            add(row)
            if len(selected) >= max(count, 0):
                return selected
    for row in ordered:
        add(row)
        if len(selected) >= max(count, 0):
            break
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-scenes", type=Path, required=True)
    ap.add_argument("--control-scenes", type=Path, required=True)
    ap.add_argument("--regime", choices=("near", "contact"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-keys-output", type=Path)
    ap.add_argument("--num-positive", type=int, default=5)
    ap.add_argument("--num-failure", type=int, default=0)
    ap.add_argument("--max-per-scene", type=int, default=1)
    ap.add_argument("--minimum-positive-score", type=float, default=0.0)
    ap.add_argument("--require-exact-positive-count", action="store_true")
    ap.add_argument("--fallback-topk-nonregressive", action="store_true")
    ap.add_argument("--diversify-evidence-profiles", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--min-near-ttc-gain-s", type=float, default=0.25)
    ap.add_argument("--min-near-clearance-gain-m", type=float, default=0.25)
    ap.add_argument("--min-near-exposure-reduction-s", type=float, default=0.20)
    ap.add_argument("--min-near-near-zero-reduction-rate", type=float, default=0.02)
    ap.add_argument("--max-near-ttc-regression-s", type=float, default=0.10)
    ap.add_argument("--max-near-terminal-ttc-regression-s", type=float, default=2.0)
    ap.add_argument("--max-near-clearance-regression-m", type=float, default=0.10)
    ap.add_argument("--max-near-terminal-clearance-regression-m", type=float, default=0.50)
    ap.add_argument("--max-near-exposure-regression-s", type=float, default=0.10)
    ap.add_argument("--max-near-near-zero-regression-rate", type=float, default=0.05)

    ap.add_argument("--min-contact-terminal-clearance-gain-m", type=float, default=0.50)
    ap.add_argument("--min-contact-auc-gain-m", type=float, default=0.50)
    ap.add_argument("--min-contact-clearance-gain-m", type=float, default=0.25)
    ap.add_argument("--min-contact-overlap-duration-reduction-s", type=float, default=0.20)
    ap.add_argument("--min-contact-yaw-rate-reduction-radps", type=float, default=0.10)
    ap.add_argument("--min-contact-jerk-reduction-mps3", type=float, default=1.0)
    ap.add_argument("--max-contact-terminal-clearance-regression-m", type=float, default=0.10)
    ap.add_argument("--max-contact-auc-regression-m", type=float, default=0.25)
    ap.add_argument("--max-contact-overlap-duration-regression-s", type=float, default=0.10)
    ap.add_argument("--max-contact-yaw-rate-regression-radps", type=float, default=0.50)
    ap.add_argument("--max-contact-jerk-regression-mps3", type=float, default=4.0)
    ap.add_argument("--max-contact-route-progress-regression-m", type=float, default=2.0)
    args = ap.parse_args()

    method = _scene_rows(args.method_scenes)
    control = _scene_rows(args.control_scenes)
    if set(method) != set(control):
        raise SystemExit(
            f"unpaired scene sets: method_only={sorted(set(method)-set(control))[:10]} "
            f"control_only={sorted(set(control)-set(method))[:10]}"
        )

    rows: list[dict[str, Any]] = []
    for key in sorted(method):
        evaluation = _evaluate(args.regime, method[key], control[key], args)
        rows.append({
            "target_key": key,
            "scene_id": method[key].get("scene_id"),
            "source_scenario_index": method[key].get("source_scenario_index"),
            "target_time_index": method[key].get("target_time_index"),
            "regime": args.regime,
            "score": evaluation["score"],
            "score_components": evaluation["score_components"],
            "eligible_positive_example": evaluation["eligible"],
            "fallback_nonregressive_example": evaluation["fallback_eligible"],
            "missing_required_metrics": evaluation["missing"],
            "material_improvements": evaluation["material"],
            "regression_reasons": evaluation["regressions"],
            "evidence_profile": evaluation["evidence_profile"],
            "method_intervention_rate": _finite_or_none(method[key].get("intervention_rate")),
            "terms": evaluation["terms"],
        })

    positive_pool = [row for row in rows if row["eligible_positive_example"]]
    positive = _take_diverse(
        positive_pool,
        max(args.num_positive, 0),
        args.max_per_scene,
        args.diversify_evidence_profiles,
    )
    strict_count = len(positive)
    if args.fallback_topk_nonregressive and len(positive) < max(args.num_positive, 0):
        fallback_pool = [row for row in rows if row["fallback_nonregressive_example"]]
        positive = _take_diverse(
            fallback_pool,
            max(args.num_positive, 0),
            args.max_per_scene,
            args.diversify_evidence_profiles,
            already=positive,
        )
    strict_keys = {str(row["target_key"]) for row in positive_pool}
    positive = [
        {
            **row,
            "selection_tier": "strict_material_improvement" if str(row["target_key"]) in strict_keys else "best_available_nonregressive",
        }
        for row in positive
    ]
    if args.require_exact_positive_count and len(positive) != args.num_positive:
        raise SystemExit(
            f"requested {args.num_positive} positive/non-regressive {args.regime} scenes, "
            f"found {len(positive)} after diversity filtering (strict={strict_count})"
        )

    failure: list[dict[str, Any]] = []
    if args.num_failure > 0:
        positive_keys = {str(row["target_key"]) for row in positive}
        failure_pool = [row for row in rows if str(row["target_key"]) not in positive_keys]
        # Failures are deliberately not profile-diversified: show the most severe cases.
        failure = sorted(failure_pool, key=lambda row: (float(row["score"]), str(row["target_key"])))[:args.num_failure]

    selected: list[dict[str, Any]] = []
    for category, items in (("positive_toy_example", positive), ("failure_case", failure)):
        for rank, row in enumerate(items, 1):
            selected.append({**row, "category": category, "category_rank": rank})

    doc = {
        "event": "v50_critical_scene_selection",
        "regime": args.regime,
        "exploratory_qualitative_only": True,
        "paper_population_claim_allowed": False,
        "selection_process": "deterministic post-hoc paired selection with bounded effect scoring, explicit non-regression guards, evidence-profile diversity and target-key tie breaking",
        "not_population_level_evidence": True,
        "score_contract": "bounded_dimensionless_v2",
        "diversity_max_per_scene": args.max_per_scene,
        "diversify_evidence_profiles": args.diversify_evidence_profiles,
        "minimum_positive_score": args.minimum_positive_score,
        "num_paired_scenes": len(rows),
        "num_positive_eligible_scenes": len(positive_pool),
        "num_strict_selected_scenes": sum(row["selection_tier"] == "strict_material_improvement" for row in positive),
        "num_fallback_selected_scenes": sum(row["selection_tier"] == "best_available_nonregressive" for row in positive),
        "required_metrics": sorted(REQUIRED[args.regime]),
        "thresholds": {
            key: value for key, value in vars(args).items()
            if key.startswith(("min_", "max_")) and key != "max_per_scene"
        },
        "selected": selected,
        "target_keys": [row["target_key"] for row in selected],
        "all_scene_scores": sorted(rows, key=lambda row: (-float(row["score"]), str(row["target_key"]))),
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
        "output": str(args.output),
        "selected": len(selected),
        "positive_eligible": len(positive_pool),
        "evidence_profiles": sorted({str(row.get("evidence_profile")) for row in positive}),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
