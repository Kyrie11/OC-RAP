#!/usr/bin/env python3
"""Compare v48.44 ROCT A/B/C/D on development/certificate data only.

The report intentionally does not read test roots.  It extracts the exact metrics
needed for the pre-registered ROCT causal readout and recomputes component-level
frontier diagnostics from proposal rows.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

COMPONENTS = ("drs", "deployability", "gap", "hard_rule", "harm_proxy")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _f(value: Any) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else float("nan")
    except Exception:
        return float("nan")


def _auc(pos: list[float], neg: list[float]) -> float | None:
    pos = [x for x in pos if math.isfinite(x)]
    neg = [x for x in neg if math.isfinite(x)]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _median(values: list[float]) -> float | None:
    vals = [x for x in values if math.isfinite(x)]
    return None if not vals else float(median(vals))


def _component(rows: list[dict[str, Any]], idx: int, positive_gain: float) -> dict[str, Any]:
    safe = [r for r in rows if _f(r.get("teacher_adv")) > positive_gain and not bool(r.get("teacher_harmful", False))]
    harmful = [r for r in rows if bool(r.get("teacher_harmful", False))]
    def pred(r: dict[str, Any]) -> float:
        vals = r.get("predicted_component_harm")
        return _f(vals[idx]) if isinstance(vals, list) and idx < len(vals) else float("nan")
    sp = [pred(r) for r in safe]
    hp = [pred(r) for r in harmful]
    spf = [x for x in sp if math.isfinite(x)]
    hpf = [x for x in hp if math.isfinite(x)]
    return {
        "safe_positive_n": len(spf),
        "harmful_n": len(hpf),
        "safe_positive_false_veto_n": sum(x > 0.5 for x in spf),
        "safe_positive_false_veto_fraction": None if not spf else sum(x > 0.5 for x in spf) / len(spf),
        "safe_positive_harm_median": _median(spf),
        "harmful_false_safe_n": sum(x <= 0.5 for x in hpf),
        "harmful_false_safe_fraction": None if not hpf else sum(x <= 0.5 for x in hpf) / len(hpf),
        "harmful_vs_safe_positive_auc": _auc(hpf, spf),
    }


def _split(run: Path, regime: str, positive_gain: float) -> dict[str, Any]:
    cal = run / "candidates" / "precision" / "calibration"
    cert = _json(cal / f"direct_value_risk_{regime}_v48.json")
    rows = _rows(cal / f"direct_value_risk_{regime}_v48.proposal_rows.jsonl")
    summary = cert.get("summary", cert.get("verify", {}))
    if not isinstance(summary, dict):
        summary = {}
    # Current certificate JSON places the authoritative selected statistics in
    # `verify`; fall back to top-level `summary` for schema-compatible future runs.
    verify = cert.get("verify")
    if isinstance(verify, dict):
        summary = verify
    return {
        "candidate_safe_positive_auc": cert.get("candidate_safe_positive_auc"),
        "candidate_positive_auc": cert.get("candidate_positive_auc"),
        "candidate_harm_auc": cert.get("candidate_harm_auc"),
        "proposal_safe_positive_auc": cert.get("proposal_evidence_top1_safe_positive_auc"),
        "proposal_positive_auc": cert.get("proposal_evidence_top1_positive_auc"),
        "proposal_harm_auc": cert.get("proposal_evidence_top1_harm_auc"),
        "proposal_conditional_harm_auc": cert.get("proposal_evidence_top1_conditional_harm_auc"),
        "selected": summary.get("num_selected"),
        "positive_selected": summary.get("num_positive_selected"),
        "harmful_selected": summary.get("num_harmful_selected"),
        "precision": summary.get("precision"),
        "positive_recall": summary.get("positive_recall"),
        "harmful_selected_ucb90": summary.get("harmful_selected_ucb90", summary.get("harmful_selected_ucb")),
        "components": {name: _component(rows, i, positive_gain) for i, name in enumerate(COMPONENTS)},
    }


def _development_sign_geometry(run: Path, regime: str, positive_gain: float) -> dict[str, Any]:
    rows = _rows(run / "candidates" / "precision" / "calibration" / f"dev_diagnostic_{regime}_v48.proposal_rows.jsonl")
    safe = [r for r in rows if _f(r.get("teacher_adv")) > positive_gain and not bool(r.get("teacher_harmful", False))]
    harmful = [r for r in rows if bool(r.get("teacher_harmful", False))]

    def rate(subset: list[dict[str, Any]], predicate) -> float | None:
        return None if not subset else sum(bool(predicate(r)) for r in subset) / len(subset)

    safe_pred = [_f(r.get("pred_adv")) for r in safe]
    safe_opp = [_f(r.get("opportunity")) for r in safe]
    safe_harm = [_f(r.get("harm")) for r in safe]
    harm_pred = [_f(r.get("pred_adv")) for r in harmful]
    return {
        "safe_positive_n": len(safe),
        "harmful_n": len(harmful),
        "safe_positive_pred_adv_median": _median(safe_pred),
        "safe_positive_opportunity_median": _median(safe_opp),
        "safe_positive_harm_median": _median(safe_harm),
        "safe_positive_pred_adv_nonnegative_fraction": rate(safe, lambda r: _f(r.get("pred_adv")) >= 0.0),
        "safe_positive_opportunity_ge_half_fraction": rate(safe, lambda r: _f(r.get("opportunity")) >= 0.5),
        "safe_positive_harm_le_half_fraction": rate(safe, lambda r: _f(r.get("harm")) <= 0.5),
        "safe_positive_joint_semantic_eligible_fraction": rate(
            safe, lambda r: _f(r.get("pred_adv")) >= 0.0 and _f(r.get("opportunity")) >= 0.5 and _f(r.get("harm")) <= 0.5
        ),
        "harmful_pred_adv_nonnegative_fraction": rate(harmful, lambda r: _f(r.get("pred_adv")) >= 0.0),
        "harmful_pred_adv_median": _median(harm_pred),
    }


def _development(run: Path, regime: str) -> dict[str, Any]:
    decomp = _json(run / "GATE_FAILURE_DECOMPOSITION.json")
    block = decomp["variants"]["precision"][regime]
    nearest = block.get("development_nearest_rule") or {}
    stratum = nearest.get("stratum") or {}
    return {
        "proposal_oracle_feasible": block.get("proposal_oracle_feasible"),
        "proposal_safe_positive_groups": block.get("proposal_safe_positive_groups"),
        "valid": block.get("development_rule_valid"),
        "constraint_failures": nearest.get("constraint_failures"),
        "constraint_deficit": nearest.get("constraint_deficit"),
        "selected": stratum.get("num_selected"),
        "positive_selected": stratum.get("num_safe_positive_selected"),
        "harmful_selected": stratum.get("num_harmful_selected"),
        "precision": stratum.get("precision"),
        "precision_lcb90": stratum.get("precision_wilson_lcb90"),
        "positive_recall": stratum.get("positive_recall"),
        "harmful_selected_ucb90": stratum.get("harmful_selected_ucb90"),
    }


def _arm(run: Path, positive_gain: float) -> dict[str, Any]:
    status = _json(run / "AUTHORITATIVE_RUN_STATUS.json")
    return {
        "run": str(run),
        "authoritative_exit_code": status.get("authoritative_exit_code"),
        "pipeline_valid": status.get("pipeline_valid"),
        "certificate_executed": status.get("certificate_executed"),
        "gate_evaluated": status.get("gate_evaluated"),
        "near": {
            "development": _development(run, "near"),
            "development_sign_geometry": _development_sign_geometry(run, "near", positive_gain),
            "certificate": _split(run, "near", positive_gain),
        },
        "contact": {
            "development": _development(run, "contact"),
            "development_sign_geometry": _development_sign_geometry(run, "contact", positive_gain),
            "certificate": _split(run, "contact", positive_gain),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    for arm in "abcd":
        ap.add_argument(f"--{arm}", type=Path, required=True)
    ap.add_argument("--positive-gain", type=float, default=0.015)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    report = {
        "schema": "v48.44-roct-2x2-comparison-v1",
        "diagnostic_only": True,
        "test_roots_read": False,
        "positive_gain": args.positive_gain,
        "arms": {
            "A": _arm(args.a, args.positive_gain),
            "B": _arm(args.b, args.positive_gain),
            "C": _arm(args.c, args.positive_gain),
            "D": _arm(args.d, args.positive_gain),
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
