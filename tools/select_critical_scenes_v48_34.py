#!/usr/bin/env python3
"""Select auditable Near/Contact toy examples from paired closed-loop traces.

Positive examples require all regime-critical metrics to be present, an actual
OC-RAP intervention, positive composite progress, and no new safety regression.
Failure examples are selected from the remaining scenes and cannot duplicate a
positive example.
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
        "post_contact_clearance_gain_m", "post_contact_escape_event", "recontact_event",
        "stable_stop_quality_event", "overlap_any", "offroad_any",
    },
}


def _finite_or_none(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _finite(x: Any, default: float = 0.0) -> float:
    v = _finite_or_none(x)
    return default if v is None else v


def _scene_rows(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".json" and not path.name.endswith(".scenes.jsonl"):
        candidate = Path(str(path) + ".scenes.jsonl")
        if candidate.is_file():
            path = candidate
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            envelope = json.loads(line)
            scene = envelope.get("scene", envelope)
            key = str(scene.get("target_key") or envelope.get("resume_key") or "")
            if not key:
                scene_id = str(scene.get("scene_id") or "")
                target_time = scene.get("target_time_index")
                key = f"{scene_id}:t{target_time}" if scene_id and target_time is not None else scene_id
            if not key:
                raise ValueError(f"scene row without target/scene key in {path}")
            if key in rows:
                raise ValueError(f"duplicate scene key {key} in {path}")
            rows[key] = scene
    if not rows:
        raise ValueError(f"empty paired scene journal: {path}")
    return rows


def _metric(scene: dict[str, Any], name: str) -> float | None:
    if name == "closed_loop_bounded_NUP":
        return _finite_or_none(scene.get(name))
    return _finite_or_none((scene.get("metric_summary", {}) or {}).get(name))


def _delta(method: dict[str, Any], control: dict[str, Any], name: str) -> float | None:
    a = _metric(method, name)
    b = _metric(control, name)
    return None if a is None or b is None else a - b


def _unsafe_regression(method: dict[str, Any], control: dict[str, Any]) -> bool:
    bad = ("overlap_any", "offroad_any", "recontact_event")
    for name in bad:
        delta = _delta(method, control, name)
        if delta is not None and delta > 1.0e-9:
            return True
    return False


def _score(regime: str, method: dict[str, Any], control: dict[str, Any]) -> tuple[float, dict[str, float | None], bool, list[str]]:
    missing: list[str] = []
    for name in sorted(REQUIRED[regime]):
        if _metric(method, name) is None or _metric(control, name) is None:
            missing.append(name)
    common: dict[str, float | None] = {
        "bounded_nup": _delta(method, control, "closed_loop_bounded_NUP"),
        "intervention_rate": _finite_or_none(method.get("intervention_rate")),
        "overlap_any": _delta(method, control, "overlap_any"),
        "offroad_any": _delta(method, control, "offroad_any"),
    }
    if regime == "near":
        terms = common | {
            "ttc_p05_s": _delta(method, control, "ttc_s_p05"),
            "terminal_ttc_s": _delta(method, control, "terminal_ttc_s"),
            "clearance_p05_m": _delta(method, control, "min_clearance_m_p05"),
            "terminal_clearance_m": _delta(method, control, "terminal_clearance_m"),
            "critical_ttc_exposure_s": _delta(method, control, "critical_ttc_exposure_duration_s"),
            "near_zero_clearance_rate": _delta(method, control, "near_zero_clearance_exposure_rate"),
        }
        score = (
            1.5 * _finite(terms["ttc_p05_s"]) + 0.5 * _finite(terms["terminal_ttc_s"])
            + 1.0 * _finite(terms["clearance_p05_m"]) + 0.5 * _finite(terms["terminal_clearance_m"])
            - 1.5 * _finite(terms["critical_ttc_exposure_s"]) - 1.0 * _finite(terms["near_zero_clearance_rate"])
            + 0.25 * _finite(terms["bounded_nup"])
        )
    elif regime == "contact":
        terms = common | {
            "post_contact_terminal_clearance_m": _delta(method, control, "post_contact_terminal_clearance_m"),
            "post_contact_free_space_auc_normalized_m": _delta(method, control, "post_contact_free_space_auc_normalized_m"),
            "post_contact_clearance_gain_m": _delta(method, control, "post_contact_clearance_gain_m"),
            "ttc_recovery_gain_s": _delta(method, control, "ttc_recovery_gain_s"),
            "stable_stop_quality_event": _delta(method, control, "stable_stop_quality_event"),
            "post_contact_escape_event": _delta(method, control, "post_contact_escape_event"),
            "recontact_event": _delta(method, control, "recontact_event"),
        }
        score = (
            1.5 * _finite(terms["post_contact_terminal_clearance_m"])
            + 1.0 * _finite(terms["post_contact_free_space_auc_normalized_m"])
            + 0.75 * _finite(terms["post_contact_clearance_gain_m"])
            + 0.35 * _finite(terms["ttc_recovery_gain_s"])
            + 2.0 * _finite(terms["stable_stop_quality_event"])
            + 1.5 * _finite(terms["post_contact_escape_event"])
            - 4.0 * _finite(terms["recontact_event"])
            + 0.25 * _finite(terms["bounded_nup"])
        )
    else:
        raise ValueError(f"unsupported regime {regime}")
    intervention = _finite_or_none(method.get("intervention_rate"))
    eligible = not missing and intervention is not None and intervention > 0.0 and score > 0.0 and not _unsafe_regression(method, control)
    return float(score), terms, eligible, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-scenes", type=Path, required=True)
    ap.add_argument("--control-scenes", type=Path, required=True)
    ap.add_argument("--regime", choices=("near", "contact"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-positive", type=int, default=3)
    ap.add_argument("--num-failure", type=int, default=2)
    args = ap.parse_args()

    method = _scene_rows(args.method_scenes)
    control = _scene_rows(args.control_scenes)
    if set(method) != set(control):
        raise SystemExit(
            f"unpaired scene sets: method_only={sorted(set(method)-set(control))[:5]} "
            f"control_only={sorted(set(control)-set(method))[:5]}"
        )
    rows = []
    for key in sorted(method):
        score, terms, eligible, missing = _score(args.regime, method[key], control[key])
        rows.append({
            "target_key": key,
            "scene_id": method[key].get("scene_id"),
            "target_time_index": method[key].get("target_time_index"),
            "regime": args.regime,
            "score": score,
            "eligible_positive_example": bool(eligible),
            "missing_required_metrics": missing,
            "method_intervention_rate": _finite_or_none(method[key].get("intervention_rate")),
            "terms": terms,
        })
    positive_pool = [r for r in rows if r["eligible_positive_example"]]
    positive = sorted(positive_pool, key=lambda r: (-r["score"], str(r["target_key"])))[: max(0, args.num_positive)]
    positive_keys = {str(r["target_key"]) for r in positive}
    failure_pool = [r for r in rows if str(r["target_key"]) not in positive_keys]
    failure = sorted(failure_pool, key=lambda r: (r["score"], str(r["target_key"])))[: max(0, args.num_failure)]
    selected = []
    for category, items in (("positive_toy_example", positive), ("failure_case", failure)):
        for rank, row in enumerate(items, start=1):
            selected.append({**row, "category": category, "category_rank": rank})
    doc = {
        "event": "v48_34_1_critical_scene_selection",
        "regime": args.regime,
        "exploratory_only": True,
        "paper_claim_allowed": False,
        "selection_is_auditable_not_cherry_picked": True,
        "num_paired_scenes": len(rows),
        "num_positive_eligible_scenes": len(positive_pool),
        "positive_eligibility": "complete critical metrics, method intervention, positive composite score, and no new overlap/offroad/recontact",
        "required_metrics": sorted(REQUIRED[args.regime]),
        "selected": selected,
        "all_scene_scores": sorted(rows, key=lambda r: (-r["score"], str(r["target_key"]))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": doc["event"], "output": str(args.output), "selected": len(selected), "positive_eligible": len(positive_pool)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
