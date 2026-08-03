#!/usr/bin/env python3
"""Select auditable Near/Contact toy examples from paired closed-loop traces.

This tool never cherry-picks silently: it writes both positive examples and
failure cases, reports every scoring term, and rejects unpaired scene sets.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


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
            key = str(scene.get("target_key") or scene.get("scene_id") or envelope.get("resume_key") or "")
            if not key:
                raise ValueError(f"scene row without target/scene key in {path}")
            if key in rows:
                raise ValueError(f"duplicate scene key {key} in {path}")
            rows[key] = scene
    return rows


def _delta(method: dict[str, Any], control: dict[str, Any], name: str) -> float:
    mm = method.get("metric_summary", {}) or {}
    cm = control.get("metric_summary", {}) or {}
    return _finite(mm.get(name)) - _finite(cm.get(name))


def _unsafe_regression(method: dict[str, Any], control: dict[str, Any]) -> bool:
    bad = ("overlap_any", "offroad_any", "secondary_overlap_event", "recontact_event")
    return any(_delta(method, control, name) > 1.0e-9 for name in bad)


def _score(regime: str, method: dict[str, Any], control: dict[str, Any]) -> tuple[float, dict[str, float], bool]:
    common = {
        "bounded_nup": _finite(method.get("closed_loop_bounded_NUP")) - _finite(control.get("closed_loop_bounded_NUP")),
        "intervention_rate": _finite(method.get("intervention_rate")),
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
            1.5 * terms["ttc_p05_s"] + 0.5 * terms["terminal_ttc_s"]
            + 1.0 * terms["clearance_p05_m"] + 0.5 * terms["terminal_clearance_m"]
            - 1.5 * terms["critical_ttc_exposure_s"] - 1.0 * terms["near_zero_clearance_rate"]
            + 0.25 * terms["bounded_nup"]
        )
    elif regime == "contact":
        terms = common | {
            "post_contact_terminal_clearance_m": _delta(method, control, "post_contact_terminal_clearance_m"),
            "post_contact_free_space_auc_normalized_m": _delta(method, control, "post_contact_free_space_auc_normalized_m"),
            "clearance_recovery_gain_m": _delta(method, control, "clearance_recovery_gain_m"),
            "ttc_recovery_gain_s": _delta(method, control, "ttc_recovery_gain_s"),
            "stable_stop_quality_event": _delta(method, control, "stable_stop_quality_event"),
            "post_contact_escape_event": _delta(method, control, "post_contact_escape_event"),
            "recontact_event": _delta(method, control, "recontact_event"),
            "secondary_overlap_event": _delta(method, control, "secondary_overlap_event"),
        }
        score = (
            1.5 * terms["post_contact_terminal_clearance_m"]
            + 1.0 * terms["post_contact_free_space_auc_normalized_m"]
            + 0.75 * terms["clearance_recovery_gain_m"]
            + 0.35 * terms["ttc_recovery_gain_s"]
            + 2.0 * terms["stable_stop_quality_event"]
            + 1.5 * terms["post_contact_escape_event"]
            - 4.0 * terms["recontact_event"] - 4.0 * terms["secondary_overlap_event"]
            + 0.25 * terms["bounded_nup"]
        )
    else:
        raise ValueError(f"unsupported regime {regime}")
    eligible = terms["intervention_rate"] > 0.0 and not _unsafe_regression(method, control)
    return float(score), terms, eligible


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
        score, terms, eligible = _score(args.regime, method[key], control[key])
        rows.append({
            "target_key": key,
            "scene_id": method[key].get("scene_id"),
            "target_time_index": method[key].get("target_time_index"),
            "regime": args.regime,
            "score": score,
            "eligible_positive_example": bool(eligible),
            "method_intervention_rate": _finite(method[key].get("intervention_rate")),
            "terms": terms,
        })
    positive_pool = [r for r in rows if r["eligible_positive_example"]]
    positive = sorted(positive_pool, key=lambda r: (-r["score"], str(r["target_key"])))[: max(0, args.num_positive)]
    failure = sorted(rows, key=lambda r: (r["score"], str(r["target_key"])))[: max(0, args.num_failure)]
    selected = []
    for category, items in (("positive_toy_example", positive), ("failure_case", failure)):
        for rank, row in enumerate(items, start=1):
            selected.append({**row, "category": category, "category_rank": rank})
    doc = {
        "event": "v48_34_critical_scene_selection",
        "regime": args.regime,
        "exploratory_only": True,
        "paper_claim_allowed": False,
        "selection_is_auditable_not_cherry_picked": True,
        "num_paired_scenes": len(rows),
        "positive_eligibility": "method intervenes and introduces no new overlap/offroad/recontact",
        "selected": selected,
        "all_scene_scores": sorted(rows, key=lambda r: (-r["score"], str(r["target_key"]))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": doc["event"], "output": str(args.output), "selected": len(selected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
