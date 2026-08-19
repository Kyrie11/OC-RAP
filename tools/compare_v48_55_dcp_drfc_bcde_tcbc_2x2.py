#!/usr/bin/env python3
"""Compare v48.55 DCP-DRFC-BCDE-TCBC A/B/C/D on development/certificate data only.

The report intentionally does not read test roots.  It extracts the exact metrics
needed for the pre-registered Coordinate-Typed Component Boundary Calibration causal readout and recomputes component-level
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



def _native_geometry(rows: list[dict[str, Any]], positive_gain: float) -> dict[str, Any]:
    safe = [r for r in rows if _f(r.get("teacher_adv")) > positive_gain and not bool(r.get("teacher_harmful", False))]
    harmful = [r for r in rows if bool(r.get("teacher_harmful", False))]

    def vals(key: str, subset: list[dict[str, Any]]) -> list[float]:
        return [_f(r.get(key)) for r in subset if math.isfinite(_f(r.get(key)))]

    def coord(subset: list[dict[str, Any]], idx: int) -> list[float]:
        out=[]
        for r in subset:
            x=r.get("predicted_native_pair_margins")
            if isinstance(x, list) and idx < len(x) and math.isfinite(_f(x[idx])):
                out.append(_f(x[idx]))
        return out

    exact=vals("native_exact_adv_margin", safe); smooth=vals("native_smooth_adv_margin", safe)
    names=("drs", "deployability", "gap_quality")
    coords={}
    for i,name in enumerate(names):
        sp=coord(safe,i); hp=coord(harmful,i)
        coords[name]={
            "safe_positive_n":len(sp),
            "safe_positive_false_veto_n":sum(x>0.0 for x in sp),
            "safe_positive_false_veto_fraction":None if not sp else sum(x>0.0 for x in sp)/len(sp),
            "harmful_n":len(hp),
            "harmful_false_safe_n":sum(x<=0.0 for x in hp),
            "harmful_false_safe_fraction":None if not hp else sum(x<=0.0 for x in hp)/len(hp),
        }
    paired=[]
    boundary_complete=[]
    for r in safe:
        a=_f(r.get("native_exact_adv_margin")); b=_f(r.get("native_smooth_adv_margin"))
        if math.isfinite(a) and math.isfinite(b):
            paired.append((a,b))
            # Stored margins are (relative advantage - positive_gain). Recover
            # the relative advantages, apply the v48.51 BC-NAP material-sign /
            # smooth-deadband rule, then return to margin space.
            exact_rel=a+positive_gain; smooth_rel=b+positive_gain
            if exact_rel >= positive_gain:
                bc_rel=max(exact_rel,smooth_rel)
            elif exact_rel <= -positive_gain:
                bc_rel=min(exact_rel,smooth_rel)
            else:
                bc_rel=smooth_rel
            boundary_complete.append(bc_rel-positive_gain)
    return {
        "available":bool(exact or smooth),
        "safe_positive_n":len(safe),
        "exact_adv_margin_median":_median(exact),
        "exact_adv_nonnegative_fraction":None if not exact else sum(x>=0.0 for x in exact)/len(exact),
        "smooth_adv_margin_median":_median(smooth),
        "smooth_adv_nonnegative_fraction":None if not smooth else sum(x>=0.0 for x in smooth)/len(smooth),
        "exact_smooth_sign_disagreement_fraction":None if not paired else sum((a>=0.0)!=(b>=0.0) for a,b in paired)/len(paired),
        "boundary_complete_adv_margin_median":_median(boundary_complete),
        "boundary_complete_adv_nonnegative_fraction":None if not boundary_complete else sum(x>=0.0 for x in boundary_complete)/len(boundary_complete),
        "coordinates":coords,
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
        "native_geometry": _native_geometry(rows, positive_gain),
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
    native_geometry = _native_geometry(rows, positive_gain)
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
        "native_geometry": native_geometry,
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



def _v4851_witness(run: Path, variant: str) -> dict[str, Any]:
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
    for name in ("V48_55_FACTOR_CONTRACT.json", "V48_53_FACTOR_CONTRACT.json", "V48_54_FACTOR_CONTRACT.json"):
        p=run/name
        if p.is_file():
            d=_json(p); d=dict(d); d["_source_contract_file"]=name; return d
    raise FileNotFoundError(f"missing v48.55/reference factor contract in {run}")

def _attribution_identity(run: Path) -> dict[str, Any]:
    source=_json(run/"SOURCE_CHECKPOINT_CONTRACT.json"); gate=_json(run/"GATE_SPEC.json")
    checks=source.get("checks") or {}; checkpoint_sha={v:(checks.get(v) or {}).get("sha256") for v in ("balanced","precision")}
    protocol=gate.get("protocol") or {}; datasets=protocol.get("datasets") or []
    manifests={str(d.get("role")):d.get("manifest_sha256") for d in datasets if isinstance(d,dict) and d.get("role")}
    return {"source_run_resolved":source.get("source_run_resolved"),"source_checkpoint_sha256":checkpoint_sha,
            "gate_dataset_manifest_sha256":manifests,"gate_protocol_sha256":gate.get("protocol_sha256"),"gate_protocol":protocol}

def _common_reference_ok(f: dict[str,Any])->bool:
    return (
      f.get("training_option_execution_semantics")=="observation_class" and f.get("evaluation_option_execution_semantics")=="observation_class"
      and f.get("native_certificate_preservation") is True and f.get("recovery_frontier_calibration") is True
      and f.get("native_margin_complete_preservation") is False and f.get("native_advantage_preservation") is True
      and f.get("native_exact_advantage_preservation") is False and f.get("native_boundary_complete_advantage_preservation") is False
      and bool(f.get("physical_teacher_sign_alignment",False)) is False and bool(f.get("physical_student_sign_alignment",False)) is False
      and bool(f.get("native_physical_student_drs",False)) is False
      and f.get("student_sign_coordinate")=="hard_qbest_ge_zero_root_mass_exact_pcd"
      and f.get("teacher_sign_coordinate")=="q_hard_proxy_drs_exact_pcd"
      and f.get("frontier_order_coordinate")=="smooth_boundary_drs_smooth_pcd"
      and f.get("strategy_regime_conditioning") is False and f.get("test_roots_read") is False
      and int(f.get("proposal_top_k",-1))==5
    )

def _attribution_contract(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids={n:a["attribution_identity"] for n,a in arms.items()}; ref=ids["A"]
    missing=not all((ref.get("gate_protocol_sha256"),ref.get("source_checkpoint_sha256",{}).get("balanced"),ref.get("source_checkpoint_sha256",{}).get("precision")))
    mismatches={n:v for n,v in ids.items() if v!=ref}
    errors=[]; expected={"A":(False,False),"B":(True,False),"C":(False,True),"D":(True,True)}
    fa=arms["A"]["factor_contract"]
    a_hist=(fa.get("version")=="v48.53-DCP-DRFC-BCDE-CSE" and fa.get("arm")=="A" and _common_reference_ok(fa))
    a_v54=(fa.get("version")=="v48.54-DCP-DRFC-BCDE-IPBD" and fa.get("arm")=="A" and not bool(fa.get("invariant_physical_boundary_distillation",False)) and _common_reference_ok(fa))
    a_v55=(fa.get("version")=="v48.55-DCP-DRFC-BCDE-TCBC" and fa.get("arm")=="A"
             and fa.get("factor_x_drs_sign_only") is False and fa.get("factor_y_continuous_component_canonicalization") is False
             and fa.get("component_margin_target_mode")=="raw" and fa.get("component_margin_regression_reliability")=="1,1,1,0,0"
             and fa.get("hard_component_veto_unchanged") is True
             and fa.get("physical_teacher_sign_alignment") is False and fa.get("physical_student_sign_alignment") is False
             and fa.get("native_physical_student_drs") is False and fa.get("invariant_physical_boundary_distillation") is False
             and fa.get("strategy_regime_conditioning") is False and fa.get("test_roots_read") is False
             and fa.get("student_sign_coordinate")=="hard_qbest_ge_zero_root_mass_exact_pcd"
             and fa.get("teacher_sign_coordinate")=="q_hard_proxy_drs_exact_pcd")
    if not (a_hist or a_v54 or a_v55): errors.append("A_reference_identity")
    for name in "BCD":
        f=arms[name]["factor_contract"]; x,y=expected[name]
        common=(f.get("version")=="v48.55-DCP-DRFC-BCDE-TCBC" and f.get("arm")==name
                and f.get("factor_x_drs_sign_only") is x and f.get("factor_y_continuous_component_canonicalization") is y
                and f.get("component_margin_target_mode")==('pooled_rms_linear' if y else 'raw')
                and f.get("component_margin_regression_reliability")==('0,1,1,0,0' if x else '1,1,1,0,0')
                and f.get("hard_component_veto_unchanged") is True
                and f.get("physical_teacher_sign_alignment") is False and f.get("physical_student_sign_alignment") is False
                and f.get("native_physical_student_drs") is False and f.get("invariant_physical_boundary_distillation") is False
                and f.get("strategy_regime_conditioning") is False and f.get("test_roots_read") is False
                and f.get("student_sign_coordinate")=="hard_qbest_ge_zero_root_mass_exact_pcd"
                and f.get("teacher_sign_coordinate")=="q_hard_proxy_drs_exact_pcd")
        if not common: errors.append(name+"_factor_contract")
    valid=(not missing) and not mismatches and not errors and all(a.get("pipeline_valid") is True and a.get("authoritative_exit_code") in (0,20) for a in arms.values())
    source="semantic-reuse-reference" if (a_hist or a_v54) else "fresh-v48.55-A"
    return {"valid":valid,"reference_arm":"A","reference_source":source,"missing_reference_fields":missing,
            "mismatched_arms":sorted(mismatches),"factor_contract_errors":errors,"identities":ids,
            "meaning":"A is the validated q-hard BC-FC + smooth-NAP raw-component reference. B-A isolates DRS sign-only supervision by removing discontinuous DRS from continuous magnitude regression while retaining component BCE. C-A isolates pooled train-only regime-free linear RMS canonicalization of continuous DEP/GAP margins. D-B-C+A tests complementarity. Hard vetoes, q-hard deployment, smooth order, source/data/top-k/gate remain fixed; no regime-conditioned policy is introduced."}

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
        "witness_precision": _v4851_witness(run, "precision"),
        "witness_balanced": _v4851_witness(run, "balanced"),
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


def _effect_readout(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Publish the pre-registered 2x2 effects for the mechanism-facing metrics."""
    paths = {
        "certificate_recall": ("certificate", "positive_recall", True),
        "certificate_harmful_ucb90": ("certificate", "harmful_selected_ucb90", False),
        "candidate_safe_positive_auc": ("certificate", "candidate_safe_positive_auc", True),
        "proposal_safe_positive_auc": ("certificate", "proposal_safe_positive_auc", True),
        "development_recall": ("development", "positive_recall", True),
        "development_precision": ("development", "precision", True),
        "development_joint_semantic_eligible_fraction": ("development_sign_geometry", "safe_positive_joint_semantic_eligible_fraction", True),
        "development_exact_adv_nonnegative_fraction": ("development_sign_geometry", "native_geometry", "exact_adv_nonnegative_fraction", True),
        "development_boundary_complete_adv_nonnegative_fraction": ("development_sign_geometry", "native_geometry", "boundary_complete_adv_nonnegative_fraction", True),
        "development_drs_safe_positive_false_veto_fraction": ("development_sign_geometry", "native_geometry", "coordinates", "drs", "safe_positive_false_veto_fraction", False),
        "development_drs_harmful_false_safe_fraction": ("development_sign_geometry", "native_geometry", "coordinates", "drs", "harmful_false_safe_fraction", False),
        "development_deployability_safe_positive_false_veto_fraction": ("development_sign_geometry", "native_geometry", "coordinates", "deployability", "safe_positive_false_veto_fraction", False),
        "development_deployability_harmful_false_safe_fraction": ("development_sign_geometry", "native_geometry", "coordinates", "deployability", "harmful_false_safe_fraction", False),
        "development_gap_safe_positive_false_veto_fraction": ("development_sign_geometry", "native_geometry", "coordinates", "gap_quality", "safe_positive_false_veto_fraction", False),
        "development_gap_harmful_false_safe_fraction": ("development_sign_geometry", "native_geometry", "coordinates", "gap_quality", "harmful_false_safe_fraction", False),
        "safe_positive_pred_adv_median": ("development_sign_geometry", "safe_positive_pred_adv_median", True),
        "safe_positive_opportunity_median": ("development_sign_geometry", "safe_positive_opportunity_median", True),
    }
    out: dict[str, Any] = {}
    for regime in ("near", "contact"):
        out[regime] = {}
        for name, spec in paths.items():
            higher = bool(spec[-1])
            keys = spec[:-1]
            vals = {}
            for arm in "ABCD":
                cur: Any = arms[arm][regime]
                for key in keys:
                    cur = cur.get(key) if isinstance(cur, dict) else None
                vals[arm] = _f(cur)
            if not all(math.isfinite(vals[a]) for a in "ABCD"):
                effects = {"B_minus_A": None, "C_minus_A": None, "interaction_D_minus_B_minus_C_plus_A": None}
            else:
                effects = {
                    "B_minus_A": vals["B"] - vals["A"],
                    "C_minus_A": vals["C"] - vals["A"],
                    "interaction_D_minus_B_minus_C_plus_A": vals["D"] - vals["B"] - vals["C"] + vals["A"],
                }
            out[regime][name] = {
                "higher_is_better": higher,
                "arms": vals,
                **effects,
            }
    return out

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
        "schema": "v48.55-dcp-drfc-bcde-tcbc-2x2-comparison-v1",
        "diagnostic_only": True,
        "test_roots_read": False,
        "positive_gain": args.positive_gain,
        "attribution_contract": attribution_contract,
        "effect_readout": _effect_readout(arms),
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
