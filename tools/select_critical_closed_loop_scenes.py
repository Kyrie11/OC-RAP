#!/usr/bin/env python3
"""Select paired critical closed-loop scenes for qualitative inspection.

The selector is deliberately two-sided: it reports the strongest improvements,
the strongest regressions, and the most critical baseline scenes.  This avoids
cherry-picking only positive OC-RAP examples when the deployment gate has not
passed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


NEAR_SPECS: dict[str, tuple[str, float, float]] = {
    # metric: (direction, normalization scale, weight)
    "min_clearance_m_min": ("higher", 0.50, 2.4),
    "min_clearance_m_p05": ("higher", 0.50, 1.4),
    "ttc_s_min": ("higher", 1.00, 1.8),
    "ttc_s_p05": ("higher", 1.00, 0.8),
    "near_contact_exposure_rate": ("lower", 0.25, 1.2),
    "critical_ttc_exposure_rate": ("lower", 0.25, 1.0),
    "near_contact_exposure_duration_s": ("lower", 1.00, 1.1),
    "critical_ttc_exposure_duration_s": ("lower", 1.00, 0.8),
    "clearance_deficit_auc_m_s": ("lower", 1.00, 1.8),
    "ttc_deficit_auc_s2": ("lower", 1.00, 1.2),
    "clearance_recovery_gain_m": ("higher", 0.50, 0.8),
    "ttc_recovery_gain_s": ("higher", 1.00, 0.5),
    "terminal_clearance_m": ("higher", 0.50, 0.7),
    "terminal_ttc_s": ("higher", 1.00, 0.4),
    "overlap_duration_s": ("lower", 0.20, 4.0),
    "secondary_overlap_event": ("lower", 1.00, 6.0),
    "offroad_any": ("lower", 1.00, 5.0),
}

CONTACT_SPECS: dict[str, tuple[str, float, float]] = {
    "overlap_duration_s": ("lower", 0.30, 2.0),
    "longest_overlap_run_s": ("lower", 0.30, 1.5),
    "secondary_overlap_event": ("lower", 1.00, 5.0),
    "recontact_event": ("lower", 1.00, 6.0),
    "recontact_episode_count": ("lower", 1.00, 2.5),
    "post_contact_clearance_m_mean": ("higher", 0.50, 1.8),
    "post_contact_clearance_m_max": ("higher", 0.75, 1.0),
    "post_contact_terminal_clearance_m": ("higher", 0.50, 1.2),
    "post_contact_clearance_gain_m": ("higher", 0.50, 1.2),
    "post_contact_free_space_auc_m_s": ("higher", 1.00, 2.0),
    "post_contact_free_space_auc_normalized_m": ("higher", 0.50, 1.5),
    "post_contact_clearance_deficit_auc_m_s": ("lower", 1.00, 1.5),
    "post_contact_escape_event": ("higher", 1.00, 3.0),
    "time_to_post_contact_escape_s": ("lower", 1.00, 0.8),
    "new_stable_stop_event": ("higher", 1.00, 1.5),
    "new_stable_stop_quality_event": ("higher", 1.00, 2.0),
    "time_to_stable_stop_quality_s": ("lower", 1.00, 0.5),
    "yaw_rate_p95": ("lower", 0.25, 1.0),
    "yaw_rate_max_abs": ("lower", 0.50, 0.7),
    "jerk_p95": ("lower", 2.00, 0.5),
    "offroad_any": ("lower", 1.00, 5.0),
}

ALL_METRICS = tuple(sorted(set(NEAR_SPECS) | set(CONTACT_SPECS) | {
    "overlap_any", "min_clearance_m_min", "ttc_s_min", "near_contact_exposure_rate",
    "critical_ttc_exposure_rate", "contact_anchor_step", "first_contact_step",
}))


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"Expected a JSON object: {path}")
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
    if name in scene:
        value = scene.get(name)
    else:
        value = (scene.get("metric_summary") or {}).get(name)
    return _finite(value)


def _macro_summary(scene: dict[str, Any]) -> list[str]:
    counts = scene.get("macro_counts") or {}
    return [f"{name}:{int(count)}" for name, count in sorted(counts.items()) if int(count) > 0]


def _intervened(scene: dict[str, Any]) -> bool:
    rate = _finite(scene.get("intervention_rate")) or 0.0
    if rate > 0.0:
        return True
    return any(int(d.get("selected_candidate_index", 0) or 0) != 0 for d in scene.get("decisions", []))


def _signed_benefit(control: float, method: float, direction: str, scale: float) -> float:
    raw = method - control
    if direction == "lower":
        raw = -raw
    return raw / max(scale, 1.0e-9)


def _criticality_near(scene: dict[str, Any]) -> float:
    clearance = _metric(scene, "min_clearance_m_min")
    ttc = _metric(scene, "ttc_s_min")
    exposure = _metric(scene, "near_contact_exposure_rate") or 0.0
    ttc_exposure = _metric(scene, "critical_ttc_exposure_rate") or 0.0
    overlap = _metric(scene, "overlap_any") or 0.0
    c_term = max(0.0, 2.0 - clearance) / 2.0 if clearance is not None else 0.0
    t_term = max(0.0, 3.0 - ttc) / 3.0 if ttc is not None else 0.0
    return 2.0 * c_term + 1.5 * t_term + exposure + 0.75 * ttc_exposure + 3.0 * overlap


def _criticality_contact(scene: dict[str, Any]) -> float:
    overlap = _metric(scene, "overlap_any") or 0.0
    overlap_dur = _metric(scene, "overlap_duration_s") or 0.0
    recontact = _metric(scene, "recontact_event") or 0.0
    secondary = _metric(scene, "secondary_overlap_event") or 0.0
    offroad = _metric(scene, "offroad_any") or 0.0
    yaw = _metric(scene, "yaw_rate_p95") or 0.0
    contact_target = 1.0 if scene.get("post_contact_target") else 0.0
    return contact_target + 2.0 * overlap + min(2.0, overlap_dur) + 3.0 * recontact + 2.0 * secondary + 2.0 * offroad + min(1.0, yaw)


def _score_pair(control: dict[str, Any], method: dict[str, Any], regime: str) -> dict[str, Any]:
    specs = NEAR_SPECS if regime == "near_contact" else CONTACT_SPECS
    contributions: list[tuple[str, float]] = []
    control_metrics: dict[str, float | None] = {}
    method_metrics: dict[str, float | None] = {}
    deltas: dict[str, float | None] = {}
    improvement = 0.0
    for name in ALL_METRICS:
        c = _metric(control, name)
        m = _metric(method, name)
        control_metrics[name] = c
        method_metrics[name] = m
        deltas[name] = None if c is None or m is None else m - c
    for name, (direction, scale, weight) in specs.items():
        c = control_metrics.get(name)
        m = method_metrics.get(name)
        if c is None or m is None:
            continue
        contribution = weight * _signed_benefit(c, m, direction, scale)
        # Prevent a single noisy continuous metric from completely dominating.
        contribution = max(-8.0, min(8.0, contribution))
        improvement += contribution
        contributions.append((name, contribution))

    new_overlap = (method_metrics.get("overlap_any") or 0.0) > (control_metrics.get("overlap_any") or 0.0) + 0.5
    new_secondary = (method_metrics.get("secondary_overlap_event") or 0.0) > (control_metrics.get("secondary_overlap_event") or 0.0) + 0.5
    new_recontact = (method_metrics.get("recontact_event") or 0.0) > (control_metrics.get("recontact_event") or 0.0) + 0.5
    new_offroad = (method_metrics.get("offroad_any") or 0.0) > (control_metrics.get("offroad_any") or 0.0) + 0.5
    hard_penalty = 12.0 * sum((new_overlap, new_secondary, new_recontact, new_offroad))
    improvement -= hard_penalty
    criticality = _criticality_near(control) if regime == "near_contact" else _criticality_contact(control)
    # Improvements in already-critical scenes are more informative, but the
    # physical metric change remains the dominant factor.
    critical_benefit_score = improvement + 0.35 * criticality
    regression_score = max(0.0, -improvement) + hard_penalty
    sorted_contrib = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)
    reason_parts = [f"{name} {'improves' if val > 0 else 'regresses'} ({val:+.2f})" for name, val in sorted_contrib[:3] if abs(val) > 1e-9]
    if not reason_parts:
        reason_parts = ["paired physical metrics are effectively unchanged"]
    flags = {
        "new_overlap": new_overlap,
        "new_secondary_overlap": new_secondary,
        "new_recontact": new_recontact,
        "new_offroad": new_offroad,
    }
    return {
        "key": _key(control),
        "scene_id": str(control.get("scene_id", "")),
        "target_key": control.get("target_key"),
        "target_time_index": control.get("target_time_index"),
        "bucket_name": control.get("bucket_name"),
        "canonical_regime": control.get("canonical_regime") or regime,
        "post_contact_target": bool(control.get("post_contact_target") or method.get("post_contact_target")),
        "method_intervened": _intervened(method),
        "control_intervention_rate": _finite(control.get("intervention_rate")) or 0.0,
        "method_intervention_rate": _finite(method.get("intervention_rate")) or 0.0,
        "control_macros": _macro_summary(control),
        "method_macros": _macro_summary(method),
        "criticality_score": criticality,
        "improvement_score": improvement,
        "critical_benefit_score": critical_benefit_score,
        "regression_score": regression_score,
        "hard_safety_penalty": hard_penalty,
        "safety_flags": flags,
        "reason": "; ".join(reason_parts),
        "control_metrics": control_metrics,
        "method_metrics": method_metrics,
        "deltas_method_minus_control": deltas,
    }


def _take(rows: Iterable[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    return list(rows)[: max(0, n)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "key", "scene_id", "target_time_index", "method_intervened",
        "control_intervention_rate", "method_intervention_rate", "criticality_score",
        "improvement_score", "critical_benefit_score", "regression_score", "reason",
    ]
    metric_fields = [f"control__{m}" for m in ALL_METRICS] + [f"method__{m}" for m in ALL_METRICS] + [f"delta__{m}" for m in ALL_METRICS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields + metric_fields)
        writer.writeheader()
        for row in rows:
            flat = {name: row.get(name) for name in fields}
            for metric in ALL_METRICS:
                flat[f"control__{metric}"] = row["control_metrics"].get(metric)
                flat[f"method__{metric}"] = row["method_metrics"].get(metric)
                flat[f"delta__{metric}"] = row["deltas_method_minus_control"].get(metric)
            writer.writerow(flat)


def main() -> int:
    ap = argparse.ArgumentParser(description="Select paired critical near-contact/contact scenes without positive-only cherry-picking.")
    ap.add_argument("control", type=Path, help="Scalar/nominal paired closed-loop JSON")
    ap.add_argument("method", type=Path, help="OC-RAP paired closed-loop JSON")
    ap.add_argument("--regime", choices=("near_contact", "contact"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--top-k-each", type=int, default=8)
    ap.add_argument("--max-selected", type=int, default=24)
    ap.add_argument("--allow-no-intervention-improvements", action="store_true")
    args = ap.parse_args()

    control_doc = _load(args.control)
    method_doc = _load(args.method)
    control_scenes = {_key(s): s for s in control_doc.get("scenes", [])}
    method_scenes = {_key(s): s for s in method_doc.get("scenes", [])}
    common = sorted(set(control_scenes) & set(method_scenes))
    if not common:
        raise SystemExit("No paired target keys. Run control and method on the same bucket targets, seed, and start time indices.")

    rows = [_score_pair(control_scenes[key], method_scenes[key], args.regime) for key in common]
    improvement_pool = rows if args.allow_no_intervention_improvements else [r for r in rows if r["method_intervened"]]
    improvements = _take(sorted([r for r in improvement_pool if r["improvement_score"] > 1.0e-9], key=lambda r: (r["critical_benefit_score"], r["improvement_score"]), reverse=True), args.top_k_each)
    regressions = _take(sorted([r for r in rows if r["regression_score"] > 1.0e-9], key=lambda r: (r["regression_score"], r["hard_safety_penalty"]), reverse=True), args.top_k_each)
    critical = _take(sorted(rows, key=lambda r: r["criticality_score"], reverse=True), args.top_k_each)
    interventions = _take(sorted([r for r in rows if r["method_intervened"]], key=lambda r: (r["method_intervention_rate"], abs(r["improvement_score"])), reverse=True), args.top_k_each)

    categories = {
        "largest_improvements": improvements,
        "largest_regressions": regressions,
        "most_critical_control": critical,
        "largest_interventions": interventions,
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category in ("largest_improvements", "largest_regressions", "most_critical_control", "largest_interventions"):
        for row in categories[category]:
            if row["key"] in seen:
                continue
            selected.append({**row, "first_selected_category": category})
            seen.add(row["key"])
            if len(selected) >= args.max_selected:
                break
        if len(selected) >= args.max_selected:
            break

    report = {
        "version": 1,
        "regime": args.regime,
        "control": str(args.control),
        "method": str(args.method),
        "num_control_scenes": len(control_scenes),
        "num_method_scenes": len(method_scenes),
        "num_paired_scenes": len(common),
        "num_method_intervention_scenes": sum(bool(r["method_intervened"]) for r in rows),
        "selection_policy": {
            "two_sided": True,
            "top_k_each": args.top_k_each,
            "max_selected": args.max_selected,
            "improvement_requires_method_intervention": not args.allow_no_intervention_improvements,
            "warning": "The deployment gate has not passed; these are exploratory diagnostic cases, not deployment evidence.",
        },
        "categories": categories,
        "selected_keys": [r["key"] for r in selected],
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    _write_csv(args.output.with_suffix(".all_pairs.csv"), rows)
    _write_csv(args.output.with_suffix(".selected.csv"), selected)
    print(json.dumps({
        "output": str(args.output),
        "paired": len(common),
        "intervention_scenes": report["num_method_intervention_scenes"],
        "selected": len(selected),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
