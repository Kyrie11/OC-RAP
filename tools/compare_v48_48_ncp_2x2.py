#!/usr/bin/env python3
"""Compare v48.48 NCP-DRFC A/B/C/D on development/certificate data only.

The report intentionally does not read test roots.  It extracts the exact metrics
needed for the pre-registered DS-OFR causal readout and recomputes component-level
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



def _v4848_witness(run: Path, variant: str) -> dict[str, Any]:
    root=run / "candidates" / variant
    out={"enabled":False,"stages":[]}
    for name in ("v48_47_decision_obs","v48_47_recovery_frontier"):
        p=root/name/"V48_47_WITNESS_COMPLETE.json"
        if p.is_file():
            d=_json(p); out["enabled"]=True; out["stages"].append(d)
    return out


def _safe_status(run: Path, variant: str) -> dict[str, Any]:
    path = run / "candidates" / variant / "calibration" / "SAFE_REGIME_STATUS.json"
    if not path.is_file():
        return {"available": False}
    d = _json(path)
    return {
        "available": True,
        "standard_calibration_valid": d.get("standard_calibration_valid"),
        "num_samples": d.get("num_samples"),
        "num_negative": d.get("num_negative"),
        "gamma_rec": d.get("gamma_rec"),
        "policy_natural_gate_evaluated": d.get("policy_natural_gate_evaluated"),
        "test_roots_read": d.get("test_roots_read"),
        "reason": d.get("reason"),
    }


def _factor_contract(run: Path) -> dict[str, Any]:
    return _json(run / "V48_48_FACTOR_CONTRACT.json")

def _attribution_identity(run: Path) -> dict[str, Any]:
    attempt=_json(run/"ATTEMPT_STARTED.json"); source=_json(run/"SOURCE_CHECKPOINT_CONTRACT.json"); gate=_json(run/"GATE_SPEC.json")
    checks=source.get("checks") or {}; checkpoint_sha={v:(checks.get(v) or {}).get("sha256") for v in ("balanced","precision")}
    protocol=gate.get("protocol") or {}; datasets=protocol.get("datasets") or []
    manifests={str(d.get("role")):d.get("manifest_sha256") for d in datasets if isinstance(d,dict) and d.get("role")}
    return {
      "protocol_root":attempt.get("protocol_root"),"protocol_seal_sha256":attempt.get("protocol_seal_sha256"),
      "source_run_resolved":source.get("source_run_resolved"),"source_checkpoint_sha256":checkpoint_sha,
      "gate_dataset_manifest_sha256":manifests,"gate_protocol":protocol,
    }

def _attribution_contract(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    identities={n:a["attribution_identity"] for n,a in arms.items()}; ref=identities["A"]
    missing=not all((ref.get("protocol_seal_sha256"),ref.get("source_checkpoint_sha256",{}).get("balanced"),ref.get("source_checkpoint_sha256",{}).get("precision")))
    mismatches={n:v for n,v in identities.items() if v!=ref}
    expected={
      "A":(False,False),"B":(True,False),
      "C":(False,True),"D":(True,True),
    }
    factor_errors=[]
    for name,arm in arms.items():
        f=arm.get("factor_contract") or {}; native,frontier=expected[name]
        if (f.get("arm")!=name or f.get("training_option_execution_semantics")!="observation_class"
            or f.get("evaluation_option_execution_semantics")!="observation_class"
            or bool(f.get("native_certificate_preservation"))!=native
            or bool(f.get("recovery_frontier_calibration"))!=frontier
            or bool(f.get("decision_weighted_observation"))
            or bool(f.get("root_logit_recalibration"))):
            factor_errors.append(name)
        if f.get("strategy_regime_conditioning") is not False or f.get("test_roots_read") is not False:
            factor_errors.append(name)
    valid=(not missing) and not mismatches and not factor_errors and all(a.get("pipeline_valid") is True and a.get("authoritative_exit_code") in (0,20) for a in arms.values())
    return {"valid":valid,"reference_arm":"A","missing_reference_fields":missing,"mismatched_arms":sorted(mismatches),
            "factor_contract_errors":sorted(set(factor_errors)),"identities":identities,
            "meaning":"All arms share paper-consistent observation-class training/evaluation. B-A = native OC-MERO certificate preservation; C-A = clean DRFC main effect; D-B-C+A = whether frontier calibration becomes useful when its native certificate reaches admission without a proxy-of-proxy, only when valid"}

def _arm(run: Path, positive_gain: float) -> dict[str, Any]:
    status = _json(run / "AUTHORITATIVE_RUN_STATUS.json")
    checks = status.get("checks") or {}
    return {
        "run": str(run),
        "authoritative_exit_code": status.get("authoritative_exit_code"),
        "pipeline_valid": status.get("pipeline_valid"),
        "certificate_executed": status.get("certificate_executed", checks.get("certificate_executed")),
        "gate_evaluated": status.get("gate_evaluated", checks.get("gate_evaluated")),
        "gate_passed_false": checks.get("gate_passed_false"),
        "attribution_identity": _attribution_identity(run),
        "factor_contract": _factor_contract(run),
        "witness_precision": _v4848_witness(run, "precision"),
        "witness_balanced": _v4848_witness(run, "balanced"),
        "safe_precision": _safe_status(run, "precision"),
        "safe_balanced": _safe_status(run, "balanced"),
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
    arms = {
        "A": _arm(args.a, args.positive_gain),
        "B": _arm(args.b, args.positive_gain),
        "C": _arm(args.c, args.positive_gain),
        "D": _arm(args.d, args.positive_gain),
    }
    attribution_contract = _attribution_contract(arms)
    report = {
        "schema": "v48.47-dsofr-2x2-comparison-v1",
        "diagnostic_only": True,
        "test_roots_read": False,
        "positive_gain": args.positive_gain,
        "attribution_contract": attribution_contract,
        "arms": arms,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if attribution_contract["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
