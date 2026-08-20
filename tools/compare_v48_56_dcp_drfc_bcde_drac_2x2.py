#!/usr/bin/env python3
"""Compare v48.56 DCP-DRFC-BCDE-DRAC A/B/C/D on development/certificate data only.

The report intentionally does not read test roots.  It extracts the exact metrics
needed for the pre-registered Decision-Role Aligned Certificate causal readout and recomputes component-level
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


def _wilson_upper(k: int, n: int, z: float = 1.2815515655446004) -> float | None:
    if n <= 0:
        return None
    p = k / n
    den = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    rad = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return (center + rad) / den


def _fixed_harm_terms(row: dict[str, Any], *, dep_boundary_aligned: bool, gap_ordinal_only: bool) -> list[float] | None:
    needed = (
        "teacher_candidate_drs", "teacher_nominal_drs", "teacher_candidate_r_dep", "teacher_nominal_r_dep",
        "teacher_candidate_gap", "teacher_nominal_gap", "teacher_candidate_hard", "teacher_nominal_hard",
        "teacher_candidate_harm_proxy", "teacher_nominal_harm_proxy",
    )
    if not all(k in row for k in needed):
        return None
    cdrs, ndrs = _f(row["teacher_candidate_drs"]), _f(row["teacher_nominal_drs"])
    crd, nrd = _f(row["teacher_candidate_r_dep"]), _f(row["teacher_nominal_r_dep"])
    cg, ng = _f(row["teacher_candidate_gap"]), _f(row["teacher_nominal_gap"])
    ch, nh = _f(row["teacher_candidate_hard"]), _f(row["teacher_nominal_hard"])
    cp, np_ = _f(row["teacher_candidate_harm_proxy"]), _f(row["teacher_nominal_harm_proxy"])
    if not all(math.isfinite(x) for x in (cdrs, ndrs, crd, nrd, cg, ng, ch, nh, cp, np_)):
        return None
    cdep = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, crd))))
    ndep = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, nrd))))
    cgapq = math.exp(-max(0.0, min(cg, 20.0)))
    ngapq = math.exp(-max(0.0, min(ng, 20.0)))
    dep = (0.5 - cdep) if dep_boundary_aligned else (ndep - cdep - 0.05)
    gap = -0.05 if gap_ordinal_only else (ngapq - cgapq - 0.05)
    return [ndrs - cdrs - 0.05, dep, gap, ch - nh - 0.05, cp - np_ - 0.05]


def _fixed_semantic_readout(rows: list[dict[str, Any]], positive_gain: float, *, dep_boundary_aligned: bool, gap_ordinal_only: bool) -> dict[str, Any]:
    labeled=[]
    for r in rows:
        terms=_fixed_harm_terms(r, dep_boundary_aligned=dep_boundary_aligned, gap_ordinal_only=gap_ordinal_only)
        if terms is None:
            continue
        harmful=max(terms)>0.0
        positive=(_f(r.get("teacher_adv")) >= positive_gain) and not harmful
        labeled.append((r, terms, harmful, positive))
    if not labeled:
        return {"available": False, "reason": "proposal rows do not expose raw teacher coordinates"}
    groups={}
    for item in labeled:
        r=item[0]; groups.setdefault((str(r.get("scene")), int(r.get("time", -1))), []).append(item)
    pos_groups=sum(any(x[3] for x in xs) for xs in groups.values())
    selected_groups=sum(any(bool(x[0].get("deployed_rule_chosen", False)) for x in xs) for xs in groups.values())
    positive_selected=sum(any(bool(x[0].get("deployed_rule_chosen", False)) and x[3] for x in xs) for xs in groups.values())
    harmful_selected=sum(any(bool(x[0].get("deployed_rule_chosen", False)) and x[2] for x in xs) for xs in groups.values())
    # Candidate-level fixed-label ranking readout; score higher means recovery-safe/beneficial.
    pos_scores=[_f(r.get("pred_adv")) for r,_,_,p in labeled if p]
    neg_scores=[_f(r.get("pred_adv")) for r,_,_,p in labeled if not p]
    component_causes={"drs":0,"deployability":0,"gap":0,"hard_rule":0,"harm_proxy":0}
    overlap_positive_raw=0
    for r,terms,harmful,positive in labeled:
        if _f(r.get("teacher_adv")) >= positive_gain and harmful:
            overlap_positive_raw += 1
        if harmful:
            for i,name in enumerate(component_causes):
                component_causes[name] += int(terms[i] > 0.0)
    return {
        "available": True,
        "dep_boundary_aligned": dep_boundary_aligned,
        "gap_ordinal_only": gap_ordinal_only,
        "candidate_n": len(labeled),
        "safe_positive_candidates": sum(x[3] for x in labeled),
        "raw_benefit_and_harm_conflicts": overlap_positive_raw,
        "harm_component_positive_counts": component_causes,
        "candidate_safe_positive_auc_from_pred_adv": _auc(pos_scores, neg_scores),
        "positive_groups": pos_groups,
        "selected_groups": selected_groups,
        "positive_selected": positive_selected,
        "harmful_selected": harmful_selected,
        "positive_recall": None if pos_groups <= 0 else positive_selected / pos_groups,
        "precision": None if selected_groups <= 0 else positive_selected / selected_groups,
        "harmful_selected_fraction": None if selected_groups <= 0 else harmful_selected / selected_groups,
        "harmful_selected_ucb90": _wilson_upper(harmful_selected, selected_groups),
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
        "fixed_semantics": {
            "legacy_v48_55": _fixed_semantic_readout(rows, positive_gain, dep_boundary_aligned=False, gap_ordinal_only=False),
            "drac_v48_56": _fixed_semantic_readout(rows, positive_gain, dep_boundary_aligned=True, gap_ordinal_only=True),
        },
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
    for name in ("V48_56_FACTOR_CONTRACT.json", "V48_55_FACTOR_CONTRACT.json", "V48_53_FACTOR_CONTRACT.json", "V48_54_FACTOR_CONTRACT.json"):
        p=run/name
        if p.is_file():
            d=_json(p); d=dict(d); d["_source_contract_file"]=name; return d
    raise FileNotFoundError(f"missing v48.56/reference factor contract in {run}")

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
    for name in "ABCD":
        f=arms[name]["factor_contract"]; x,y=expected[name]
        common=(
            f.get("version")=="v48.56-DCP-DRFC-BCDE-DRAC" and f.get("arm")==name
            and f.get("factor_x_deployability_zero_boundary") is x
            and f.get("factor_y_gap_ordinal_only") is y
            and f.get("native_dep_boundary_aligned") is x
            and f.get("gap_in_teacher_pcd") is True and f.get("gap_in_native_advantage") is True
            and f.get("gap_in_hard_component_veto") is (not y)
            and f.get("component_bce_reliability")==('1,1,0,0,0' if y else '1,1,1,0,0')
            and f.get("component_margin_regression_reliability")==('1,1,0,0,0' if y else '1,1,1,0,0')
            and f.get("component_margin_target_mode")=="raw"
            and f.get("native_certificate_preservation") is True and f.get("native_advantage_preservation") is True
            and f.get("native_margin_complete_preservation") is False
            and f.get("physical_teacher_sign_alignment") is False and f.get("physical_student_sign_alignment") is False
            and f.get("invariant_physical_boundary_distillation") is False
            and f.get("root_logit_recalibration") is False
            and f.get("strategy_regime_conditioning") is False and f.get("test_roots_read") is False
            and f.get("training_option_execution_semantics")=="observation_class"
            and f.get("evaluation_option_execution_semantics")=="observation_class"
            and int(f.get("proposal_top_k",-1))==5
        )
        if not common: errors.append(name+"_factor_contract")
    fixed_available=all(
        bool(arms[a][reg]["certificate"].get("fixed_semantics",{}).get("drac_v48_56",{}).get("available"))
        for a in "ABCD" for reg in ("near","contact")
    )
    if not fixed_available: errors.append("fixed_semantic_readout_missing_raw_teacher_coordinates")
    valid=(not missing) and not mismatches and not errors and all(a.get("pipeline_valid") is True and a.get("authoritative_exit_code") in (0,20) for a in arms.values())
    return {"valid":valid,"reference_arm":"A","reference_source":"fresh-v48.56-A","missing_reference_fields":missing,
            "mismatched_arms":sorted(mismatches),"factor_contract_errors":errors,"identities":ids,
            "meaning":"Fresh A/B/C/D hold source/data/top-k/gate fixed. B-A changes only deployability from nominal-relative quality loss to the absolute teacher R_dep=0 boundary. C-A keeps GAP in PCD/order but removes it from non-compensatory hard veto. D-B-C+A tests interaction. Fixed legacy and DRAC semantic labels are recomputed from raw teacher coordinates for every arm so performance attribution does not move the evaluation target."}

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
        "fixed_drac_certificate_recall": ("certificate", "fixed_semantics", "drac_v48_56", "positive_recall", True),
        "fixed_drac_certificate_harmful_ucb90": ("certificate", "fixed_semantics", "drac_v48_56", "harmful_selected_ucb90", False),
        "fixed_drac_candidate_safe_positive_auc": ("certificate", "fixed_semantics", "drac_v48_56", "candidate_safe_positive_auc_from_pred_adv", True),
        "fixed_legacy_certificate_recall": ("certificate", "fixed_semantics", "legacy_v48_55", "positive_recall", True),
        "fixed_legacy_certificate_harmful_ucb90": ("certificate", "fixed_semantics", "legacy_v48_55", "harmful_selected_ucb90", False),
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
        "schema": "v48.56-dcp-drfc-bcde-drac-2x2-comparison-v1",
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
