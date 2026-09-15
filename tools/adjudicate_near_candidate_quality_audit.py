#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REFERENCE_STATUS = "OBSERVATION_LEGAL_NEAR_SYSTEM_AXIS_STOP"
REFERENCE_BRANCH = "keep_recovery_mechanism_frozen_audit_absolute_admitted_candidate_action_quality_no_v48_125_threshold_capacity_or_retraining_sweep"


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_equal(a: Any, b: Any, atol: float = 1.0e-9) -> bool:
    if a is None or b is None:
        return a is b
    try:
        af, bf = float(a), float(b)
    except Exception:
        return a == b
    # Replay-equivalence treats a pair of undefined diagnostic values as equal.
    # JSON NaN is used by the closed-loop writer for metrics that are not
    # applicable to a scene (e.g. post-contact fields in Near scenes).  Python
    # follows IEEE semantics where NaN != NaN, so comparing those fields with
    # ordinary equality produces a false engineering mismatch.
    if math.isnan(af) or math.isnan(bf):
        return math.isnan(af) and math.isnan(bf)
    if not (math.isfinite(af) and math.isfinite(bf)):
        return af == bf
    return abs(af - bf) <= atol


def scene_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scenes = result.get("scenes") or []
    return {str(s.get("target_key")): s for s in scenes if s.get("target_key")}


def compare_behavior(reference_full: dict[str, Any], audit_subset: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    ref = scene_map(reference_full)
    aud = scene_map(audit_subset)
    if not aud:
        return ["audit result contains no embedded scenes"]
    for key, s in aud.items():
        r = ref.get(key)
        if r is None:
            errors.append(f"audit target absent from reference base: {key}")
            continue
        for field in ("num_decisions", "macro_counts", "selection_reason_counts"):
            if s.get(field) != r.get(field):
                errors.append(f"policy behavior mismatch {key}:{field}")
        for field in ("intervention_rate", "closed_loop_bounded_NUP", "closed_loop_direct_recovery_advantage"):
            if not finite_equal(s.get(field), r.get(field)):
                errors.append(f"policy scalar mismatch {key}:{field}")
        sm = s.get("metric_summary") or {}
        rm = r.get("metric_summary") or {}
        common = sorted(set(sm).intersection(rm))
        for metric in common:
            if not finite_equal(sm.get(metric), rm.get(metric)):
                errors.append(f"metric mismatch {key}:{metric}")
                break
    return errors


def summarize_variant(result: dict[str, Any], *, eps: float = 1.0e-6) -> tuple[dict[str, Any], list[str]]:
    """Summarize the frozen relative-policy decision slice.

    Primary relative truth is teacher PCD because the frozen Stage-I direct
    value/opportunity/harm heads are explicitly supervised on
    PCD(candidate)-PCD(nominal).  Signed R_dep deltas remain a secondary
    recovery-state diagnostic; using R_dep as the primary relative target would
    compare the deployed relative head against a different object than the one
    it was trained to represent.
    """
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    scenes = result.get("scenes") or []
    for scene in scenes:
        if not scene.get("audit_intervention_only"):
            errors.append(f"{scene.get('target_key')}: audit_intervention_only is false")
        if str(scene.get("audit_candidate_scope")) != "all":
            errors.append(f"{scene.get('target_key')}: audit_candidate_scope is not all")
        if not scene.get("audit_store_candidate_records"):
            errors.append(f"{scene.get('target_key')}: candidate records were not stored")
        for rec in scene.get("candidate_quality_audit_records") or []:
            rr = dict(rec)
            rr["target_key"] = scene.get("target_key")
            records.append(rr)
    intervention_decisions = int(round(sum(float(s.get("intervention_rate", 0.0)) * int(s.get("num_decisions", 0)) for s in scenes)))
    if len(records) != intervention_decisions:
        errors.append(f"candidate audit record count {len(records)} != intervention decisions {intervention_decisions}")

    out = {
        "primary_relative_truth": "teacher_pcd_advantage_vs_nominal",
        "secondary_signed_state_truth": "teacher_r_dep_advantage_vs_nominal",
        "num_scenes": len(scenes),
        "num_intervention_decisions": intervention_decisions,
        "num_candidate_audit_records": len(records),
        "all_candidates_labeled_every_record": bool(records) and all(int(r.get("num_candidates_labeled", 0)) >= 2 for r in records),
        "teacher_pcd_better_than_nominal_anywhere": 0,
        "teacher_pcd_better_than_nominal_admitted": 0,
        "teacher_pcd_better_admitted_with_positive_relative_head": 0,
        "teacher_pcd_better_anywhere_but_none_admitted": 0,
        "selected_teacher_pcd_worse_than_nominal": 0,
        "selected_teacher_pcd_better_than_nominal": 0,
        "teacher_r_dep_better_than_nominal_anywhere": 0,
        "teacher_r_dep_better_than_nominal_admitted": 0,
        "selected_teacher_r_dep_worse_than_nominal": 0,
        "selected_teacher_r_dep_better_than_nominal": 0,
        "best_all_pcd_candidate_absolute_rejected": 0,
        "best_admitted_relative_positive": 0,
        "mean_num_absolute_admitted": None,
        "mean_selected_teacher_pcd_advantage": None,
        "mean_best_all_teacher_pcd_advantage": None,
        "mean_best_admitted_teacher_pcd_advantage": None,
        "mean_selected_teacher_r_dep_advantage": None,
        "records": [],
    }
    admitted_counts=[]; selected_pcd_adv=[]; best_all_pcd_adv=[]; best_adm_pcd_adv=[]; selected_r_adv=[]
    compact=[]
    for r in records:
        rows = r.get("candidates") or []
        nom = next((x for x in rows if int(x.get("candidate_index", -1)) == 0), None)
        sel = next((x for x in rows if x.get("selected")), None)
        if nom is None or sel is None:
            errors.append(f"{r.get('target_key')} step {r.get('step_index')}: nominal/selected row missing")
            continue
        nominal_pcd = float(nom.get("teacher_pcd"))
        nominal_r = float(nom.get("teacher_r_dep_star"))
        better_pcd = [x for x in rows if float(x.get("teacher_pcd", -1e30)) > nominal_pcd + eps]
        better_pcd_adm = [x for x in better_pcd if bool(x.get("absolute_admitted"))]
        better_pcd_adm_rel = [x for x in better_pcd_adm if (x.get("pred_direct_advantage_vs_nominal") is not None and float(x["pred_direct_advantage_vs_nominal"]) > 0.0)]
        better_r = [x for x in rows if float(x.get("teacher_r_dep_star", -1e30)) > nominal_r + eps]
        better_r_adm = [x for x in better_r if bool(x.get("absolute_admitted"))]
        admitted = [x for x in rows if bool(x.get("absolute_admitted"))]
        best_all = max(rows, key=lambda x: float(x.get("teacher_pcd", -1e30)))
        best_adm = max(admitted, key=lambda x: float(x.get("teacher_pcd", -1e30))) if admitted else None

        out["teacher_pcd_better_than_nominal_anywhere"] += int(bool(better_pcd))
        out["teacher_pcd_better_than_nominal_admitted"] += int(bool(better_pcd_adm))
        out["teacher_pcd_better_admitted_with_positive_relative_head"] += int(bool(better_pcd_adm_rel))
        out["teacher_pcd_better_anywhere_but_none_admitted"] += int(bool(better_pcd) and not bool(better_pcd_adm))
        out["teacher_r_dep_better_than_nominal_anywhere"] += int(bool(better_r))
        out["teacher_r_dep_better_than_nominal_admitted"] += int(bool(better_r_adm))

        spadv=float(sel.get("teacher_pcd"))-nominal_pcd
        sradv=float(sel.get("teacher_r_dep_star"))-nominal_r
        out["selected_teacher_pcd_worse_than_nominal"] += int(spadv < -eps)
        out["selected_teacher_pcd_better_than_nominal"] += int(spadv > eps)
        out["selected_teacher_r_dep_worse_than_nominal"] += int(sradv < -eps)
        out["selected_teacher_r_dep_better_than_nominal"] += int(sradv > eps)
        out["best_all_pcd_candidate_absolute_rejected"] += int(not bool(best_all.get("absolute_admitted")))
        if best_adm is not None:
            rel=best_adm.get("pred_direct_advantage_vs_nominal")
            out["best_admitted_relative_positive"] += int(rel is not None and float(rel) > 0.0)
        admitted_counts.append(len(admitted)); selected_pcd_adv.append(spadv); selected_r_adv.append(sradv)
        best_all_pcd_adv.append(float(best_all.get("teacher_pcd"))-nominal_pcd)
        if best_adm is not None: best_adm_pcd_adv.append(float(best_adm.get("teacher_pcd"))-nominal_pcd)
        compact.append({
            "target_key": r.get("target_key"), "step_index": r.get("step_index"),
            "selected_candidate_index": r.get("selected_candidate_index"),
            "num_absolute_admitted": len(admitted),
            "selected_teacher_pcd_advantage": spadv,
            "selected_teacher_r_dep_advantage": sradv,
            "best_all_teacher_pcd_advantage": best_all_pcd_adv[-1],
            "best_admitted_teacher_pcd_advantage": (best_adm_pcd_adv[-1] if best_adm is not None else None),
            "pcd_better_anywhere": bool(better_pcd), "pcd_better_admitted": bool(better_pcd_adm),
            "pcd_better_admitted_relative_positive": bool(better_pcd_adm_rel),
            "r_dep_better_anywhere": bool(better_r), "r_dep_better_admitted": bool(better_r_adm),
        })
    def mean(xs): return (sum(xs)/len(xs)) if xs else None
    out["mean_num_absolute_admitted"] = mean(admitted_counts)
    out["mean_selected_teacher_pcd_advantage"] = mean(selected_pcd_adv)
    out["mean_best_all_teacher_pcd_advantage"] = mean(best_all_pcd_adv)
    out["mean_best_admitted_teacher_pcd_advantage"] = mean(best_adm_pcd_adv)
    out["mean_selected_teacher_r_dep_advantage"] = mean(selected_r_adv)
    out["records"] = compact
    return out, errors

def branch_for(variants: dict[str, dict[str, Any]]) -> str:
    vals=list(variants.values())
    if vals and all(v["teacher_pcd_better_than_nominal_anywhere"] == 0 for v in vals):
        return "candidate_library_or_action_realization_bottleneck_no_better_teacher_pcd_candidate_at_intervention_states"
    if vals and all(v["teacher_pcd_better_than_nominal_admitted"] == 0 for v in vals):
        return "absolute_admission_bottleneck_better_teacher_pcd_candidates_exist_but_are_not_admitted"
    if vals and all(v["teacher_pcd_better_admitted_with_positive_relative_head"] == 0 for v in vals):
        return "relative_evidence_alignment_bottleneck_better_admitted_teacher_pcd_candidates_exist_but_relative_head_does_not_support_them"
    return "mixed_or_ranking_bottleneck_inspect_pcd_decision_records_before_any_algorithm_change"

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--reference-adjudication", required=True)
    ap.add_argument("--base-balanced", required=True)
    ap.add_argument("--base-precision", required=True)
    ap.add_argument("--audit-balanced", required=True)
    ap.add_argument("--audit-precision", required=True)
    ap.add_argument("--output", required=True)
    args=ap.parse_args()
    ref=load(args.reference_adjudication)
    errors=[]
    if not (ref.get("valid") and ref.get("attribution_ready") and ref.get("status")==REFERENCE_STATUS and ref.get("next_branch")==REFERENCE_BRANCH):
        errors.append("reference V48.124.10.2 adjudication does not authorize this audit")
    base={"balanced":load(args.base_balanced),"precision":load(args.base_precision)}
    audits={"balanced":load(args.audit_balanced),"precision":load(args.audit_precision)}
    variants={}
    for v in ("balanced","precision"):
        errors.extend([f"{v}:{e}" for e in compare_behavior(base[v],audits[v])])
        summary,errs=summarize_variant(audits[v])
        errors.extend([f"{v}:{e}" for e in errs]); variants[v]=summary
    branch=branch_for(variants) if not errors else "engineering_fix_required_before_scientific_diagnosis"
    out={
        "schema":"ocrap-v48.124-near-candidate-quality-audit-v2",
        "scientific_version":"v48.124-OC-FMSA",
        "engineering_version":"v48.124.10.3.1-CANDIDATE-QUALITY-ADJUDICATOR-ENGFIX",
        "source_experiment_engineering_version":"v48.124.10.3-CANDIDATE-QUALITY-DIAGNOSTIC",
        "valid":not errors,
        "attribution_ready":not errors,
        "errors":errors,
        "reference_status":ref.get("status"),
        "variants":variants,
        "diagnostic_branch":branch,
        "algorithm_modified":False,
        "publication_evidence":False,
        "notes":[
            "Privileged teacher labels are used only after the frozen action has been selected.",
            "This diagnostic does not alter the deployed selector, checkpoints, candidate library, recovery library, thresholds, horizon, or Waymax dynamics.",
            "No V48.125 mechanism change is authorized by this audit itself; use the branch only to localize the failed Near system axis.",
            "Primary relative truth is teacher PCD delta, matching the frozen direct-relative-head training target; R_dep delta is secondary signed-state evidence.",
            "Matching NaN diagnostic values are replay-equivalent and do not constitute an engineering mismatch.",
        ],
    }
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2,sort_keys=True))
    if errors: raise SystemExit(30)

if __name__=="__main__": main()
