#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any

from ocrap.audits.tail_boundary_crossing_flow import (
    ENGINEERING_VERSION, SCIENTIFIC_VERSION, MATCHED_DIM, BOUNDARY_GEOMETRY_DIM,
)

AUTHORITATIVE_V116_PIPELINE_SHA256 = "105d6e47cb046f5dac106bf2930004a89f032ab92faf5b3c988d1ee11b500e9e"
AUTHORITATIVE_V116_COMPARISON_SHA256 = "cca845f43b2d3e0c7a774f87ab26faad79de739a96217ef1a72242acf94d8f0f"
AUTHORITATIVE_V116_BALANCED_SHA256 = "c35ecae10f6ed9729314e19f8e46203e0c763eaad902a2cd3470f4d8c01746c3"
AUTHORITATIVE_V116_PRECISION_SHA256 = "87b9e08e11b4fb58def4c58c0577f82117e23f0b6137972ebb967cd06f77bede"
ROLES=("dev_near","dev_contact","certificate_near","certificate_contact")
SPACES=("base","cotangent_hitting","boundary_work","boundary_hitting")


def _sha(p: Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def _ok(v: Any,t:float)->bool: return v is not None and float(v)>=t
def _cross(rs:set[str],n:int)->bool: return len(rs)>=n and any("near" in r for r in rs) and any("contact" in r for r in rs)


def _action_gate(docs:dict[str,Any],space:str,metric:str)->dict[str,Any]:
    pos=[]; top=[]; roles=set(); top_roles=set()
    for variant,d in docs.items():
        for role in ROLES:
            m=d[f"{space}_cells"][role][f"{metric}_true"]
            if _ok(m.get("auc"),.65) and _ok(m.get("auc_vs_shuffled"),.05): pos.append([variant,role]); roles.add(role)
            if _ok(m.get("top1_vs_shuffled"),.10): top.append([variant,role]); top_roles.add(role)
    return {"go":bool(len(pos)>=6 and _cross(roles,3) and len(top)>=4 and _cross(top_roles,2)),
            "local_order":bool(len(top)>=4 and _cross(top_roles,2)),"positive_cells":pos,
            "top1_material_cells":top,"roles":sorted(roles),"top1_roles":sorted(top_roles)}


def _within(docs:dict[str,Any],treatment:str,control:str,metric:str,label:str)->dict[str,Any]:
    rows=[]; pos=[]; mat=[]; roles=set()
    for variant,d in docs.items():
        for role in ROLES:
            a=d[f"{treatment}_cells"][role][f"{metric}_true"].get("auc")
            b=d[f"{control}_cells"][role][f"{metric}_true"].get("auc")
            de=None if a is None or b is None else float(a)-float(b)
            rows.append({"variant":variant,"role":role,f"{treatment}_auc":a,f"{control}_auc":b,label:de})
            if de is not None and de>0: pos.append([variant,role]); roles.add(role)
            if de is not None and de>=.01: mat.append([variant,role])
    return {"go":bool(len(pos)>=6 and _cross(roles,3) and len(mat)>=4),"rows":rows,
            "positive_cells":pos,"material_cells":mat,"roles":sorted(roles)}


def _historical(docs:dict[str,Any],hist:dict[str,Any],treatment:str,old_space:str,metric:str,label:str)->dict[str,Any]:
    rows=[]; pos=[]; mat=[]; roles=set()
    for variant,d in docs.items():
        for role in ROLES:
            a=d[f"{treatment}_cells"][role][f"{metric}_true"].get("auc")
            b=hist[variant][f"{old_space}_cells"][role][f"{metric}_true"].get("auc")
            de=None if a is None or b is None else float(a)-float(b)
            rows.append({"variant":variant,"role":role,f"v48_117_{treatment}_auc":a,
                         f"v48_116_{old_space}_auc":b,label:de})
            if de is not None and de>0: pos.append([variant,role]); roles.add(role)
            if de is not None and de>=.01: mat.append([variant,role])
    return {"go":bool(len(pos)>=6 and _cross(roles,3) and len(mat)>=4),"rows":rows,
            "positive_cells":pos,"material_cells":mat,"roles":sorted(roles)}


def _activity(docs:dict[str,Any])->dict[str,Any]:
    rows=[]; ch=set(); bw=set(); bh=set(); diverse=set(); multi=set(); nontrivial=set(); two_sided=set(); reentry=set(); exact=True
    for variant,d in docs.items():
        for role in ROLES:
            e=(d.get("events") or {}).get(role) or {}; p=e.get("pair_diagnostics") or {}; b=e.get("boundary_measure_diagnostics") or {}
            row={"variant":variant,"role":role,
                 "mean_common_valid_option_count":float(p.get("mean_common_valid_option_count",0)),
                 "min_common_valid_option_count":int(p.get("min_common_valid_option_count",0)),
                 "cotangent_hitting_nonzero_fraction":float(p.get("cotangent_hitting_nonzero_fraction",0)),
                 "boundary_work_nonzero_fraction":float(p.get("boundary_work_nonzero_fraction",0)),
                 "boundary_hitting_nonzero_fraction":float(p.get("boundary_hitting_nonzero_fraction",0)),
                 "option_flow_diverse_fraction":float(p.get("option_flow_diverse_fraction",0)),
                 "reentry_set_available_fraction":float(p.get("reentry_set_available_fraction",0)),
                 "max_option_permutation_invariance_error":float(p.get("max_option_permutation_invariance_error",1)),
                 "boundary_group_count":int(b.get("group_count",0)),
                 "mean_boundary_positive_option_count":float(b.get("mean_positive_option_count",0)),
                 "min_boundary_positive_option_count":int(b.get("min_positive_option_count",0)),
                 "mean_boundary_effective_option_count":float(b.get("mean_effective_option_count",0)),
                 "mean_boundary_two_sided_root_mass":float(b.get("mean_two_sided_root_mass",0)),
                 "mean_boundary_witness_count_per_exposed_root":float(b.get("mean_witness_count_per_exposed_root",0)),
                 "max_option_weight_sum_error":float(b.get("max_option_weight_sum_error",1)),
                 "max_root_exposure_mass_error":float(b.get("max_root_exposure_mass_error",1)),
                 "all_model_padding_invalid":bool(b.get("all_model_padding_invalid",False)),
                 "all_model_physical_valid_prefix_match":bool(b.get("all_model_physical_valid_prefix_match",False)),
                 "max_padded_tail_mass":float(b.get("max_padded_tail_mass",1)),
                 "max_physical_tail_mass_error":float(b.get("max_physical_tail_mass_error",1))}
            rows.append(row)
            if row["cotangent_hitting_nonzero_fraction"]>0: ch.add(role)
            if row["boundary_work_nonzero_fraction"]>0: bw.add(role)
            if row["boundary_hitting_nonzero_fraction"]>0: bh.add(role)
            if row["option_flow_diverse_fraction"]>0: diverse.add(role)
            if row["min_common_valid_option_count"]>=2: multi.add(role)
            if row["mean_boundary_positive_option_count"]>=2: nontrivial.add(role)
            if row["mean_boundary_two_sided_root_mass"]>0: two_sided.add(role)
            if "contact" in role and row["reentry_set_available_fraction"]>0: reentry.add(role)
            if (row["max_option_permutation_invariance_error"]>1e-10 or row["max_option_weight_sum_error"]>1e-10
                or row["max_root_exposure_mass_error"]>1e-10 or not row["all_model_padding_invalid"]
                or not row["all_model_physical_valid_prefix_match"] or row["max_padded_tail_mass"]>1e-10
                or row["max_physical_tail_mass_error"]>1e-10): exact=False
    core=bool(exact and _cross(ch,3) and _cross(bw,3) and _cross(bh,3) and _cross(diverse,3) and _cross(multi,3) and len(reentry)>=1)
    measure=bool(_cross(nontrivial,3) and _cross(two_sided,3))
    return {"go":bool(core and measure),"core_activity_go":core,"nontrivial_boundary_measure_go":measure,
            "exact_boundary_contract_go":exact,"cotangent_hitting_nonzero_roles":sorted(ch),
            "boundary_work_nonzero_roles":sorted(bw),"boundary_hitting_nonzero_roles":sorted(bh),
            "physical_option_flow_diverse_roles":sorted(diverse),"multi_option_roles":sorted(multi),
            "nontrivial_boundary_option_roles":sorted(nontrivial),"two_sided_boundary_roles":sorted(two_sided),
            "reentry_contact_roles":sorted(reentry),"reentry_contact_coverage_go":bool(reentry),"rows":rows}


def _variant_identity(docs:dict[str,Any])->dict[str,Any]:
    a,b=docs["balanced"],docs["precision"]; diffs=[]
    for space in SPACES:
        for role in ROLES:
            for metric in ("support_true","support_shuffled","reserve_true","reserve_shuffled"):
                ma=a[f"{space}_cells"][role][metric]; mb=b[f"{space}_cells"][role][metric]
                for k in ("auc","top1","auc_vs_shuffled","top1_vs_shuffled","rows","positive_rows","negative_rows","powered_groups"):
                    if ma.get(k)!=mb.get(k): diffs.append(f"{space}:{role}:{metric}:{k}")
    return {"exact":not diffs,"differences":diffs,"effective_unique_roles_if_exact":4}


def _power(docs:dict[str,Any])->list[dict[str,Any]]:
    out=[]
    for variant,d in docs.items():
        for role in ROLES:
            for axis in ("support","reserve"):
                m=d["base_cells"][role][f"{axis}_true"]
                out.append({"variant":variant,"role":role,"axis":axis,"rows":m.get("rows",0),
                            "positive_rows":m.get("positive_rows",0),"negative_rows":m.get("negative_rows",0),
                            "powered_groups":m.get("powered_groups",0),
                            "underpowered":bool(int(m.get("powered_groups",0))<2 or min(int(m.get("positive_rows",0)),int(m.get("negative_rows",0)))<3)})
    return out


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--balanced",type=Path,required=True); ap.add_argument("--precision",type=Path,required=True)
    ap.add_argument("--v116-pipeline",type=Path,required=True); ap.add_argument("--v116-comparison",type=Path,required=True)
    ap.add_argument("--v116-balanced",type=Path,required=True); ap.add_argument("--v116-precision",type=Path,required=True)
    ap.add_argument("--run-id",required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    docs={"balanced":json.loads(a.balanced.read_text()),"precision":json.loads(a.precision.read_text())}
    hist={"balanced":json.loads(a.v116_balanced.read_text()),"precision":json.loads(a.v116_precision.read_text())}
    p116=json.loads(a.v116_pipeline.read_text()); c116=json.loads(a.v116_comparison.read_text()); errors=[]
    if _sha(a.v116_pipeline)!=AUTHORITATIVE_V116_PIPELINE_SHA256: errors.append("v116_pipeline_sha")
    if _sha(a.v116_comparison)!=AUTHORITATIVE_V116_COMPARISON_SHA256: errors.append("v116_comparison_sha")
    if _sha(a.v116_balanced)!=AUTHORITATIVE_V116_BALANCED_SHA256: errors.append("v116_balanced_sha")
    if _sha(a.v116_precision)!=AUTHORITATIVE_V116_PRECISION_SHA256: errors.append("v116_precision_sha")
    d116=c116.get("preregistered_decision") or {}
    if not (p116.get("valid") and p116.get("attribution_ready") and p116.get("preregistered_status")=="WEAK_ROOT_RECOVERY_SET_FLOW_STOP"
            and c116.get("valid") and c116.get("attribution_ready") and d116.get("status")=="WEAK_ROOT_RECOVERY_SET_FLOW_STOP"
            and d116.get("next_branch")=="close_first_order_nominal_ocmero_cotangent_option_pushforward_then_preregister_tail_boundary_crossing_flow_audit_no_training_capacity_regime_or_source_sweep"):
        errors.append("v116_prerequisite")
    for variant,d in docs.items():
        if not (d.get("valid") and d.get("engineering_version")==ENGINEERING_VERSION and d.get("scientific_version")==SCIENTIFIC_VERSION
                and d.get("variant")==variant and d.get("run_instance_id")==a.run_id and d.get("audit_only")
                and d.get("capacity_matched_all_boundary_families") and d.get("matched_family_dimension")==MATCHED_DIM
                and d.get("boundary_geometry_dimension")==BOUNDARY_GEOMETRY_DIM and d.get("root_decoder_parameters_trained")==0
                and d.get("source_parameters_trained")==0 and d.get("regime_conditioning") is False): errors.append(variant)
    if errors:
        fake={"go":False,"local_order":False,"rows":[]}; gates={}; activity={"go":False,"rows":[],"reentry_contact_coverage_go":False}
    else:
        gates={}
        for space in ("cotangent_hitting","boundary_work","boundary_hitting"):
            for axis in ("support","reserve"): gates[f"{space}_{axis}"]=_action_gate(docs,space,axis)
        for space,label in (("cotangent_hitting","representation_under_cotangent"),("boundary_work","measure_under_work"),("boundary_hitting","total_boundary_crossing")):
            for axis in ("support","reserve"):
                gates[f"{space}_vs_v116_{axis}"]=_historical(docs,hist,space,"tail_work",axis,f"{label}_minus_v116_tail_work")
        for treatment,control,label in (("boundary_hitting","cotangent_hitting","boundary_measure_effect_under_hitting"),
                                         ("boundary_hitting","boundary_work","hitting_effect_under_boundary_measure")):
            for axis in ("support","reserve"):
                gates[f"{label}_{axis}"]=_within(docs,treatment,control,axis,label)
        activity=_activity(docs)
    if errors:
        status="V48_117_ENGINEERING_STOP"; branch="fix_v48_117_engineering_and_rerun_same_tail_boundary_crossing_flow_audit"
    else:
        bh_core=bool(gates["boundary_hitting_support"]["go"] and gates["boundary_hitting_reserve"]["go"]
                     and gates["boundary_hitting_vs_v116_support"]["go"] and gates["boundary_hitting_vs_v116_reserve"]["go"] and activity["go"])
        bw_core=bool(gates["boundary_work_support"]["go"] and gates["boundary_work_reserve"]["go"]
                     and gates["boundary_work_vs_v116_support"]["go"] and gates["boundary_work_vs_v116_reserve"]["go"] and activity["go"])
        if bh_core and activity.get("reentry_contact_coverage_go"):
            status="TAIL_BOUNDARY_CROSSING_FLOW_GO"; branch="promote_tail_boundary_hitting_carrier_then_preregister_exactly_one_main_carrier_integration_no_source_boundary_regime_or_capacity_sweep"
        elif bw_core and activity.get("reentry_contact_coverage_go"):
            status="TAIL_BOUNDARY_MEASURE_WORK_GO"; branch="promote_tail_boundary_measure_with_signed_work_then_preregister_exactly_one_main_carrier_integration_no_source_or_regime_sweep"
        elif gates["boundary_hitting_support"]["go"] and gates["boundary_hitting_vs_v116_support"]["go"]:
            status="TAIL_BOUNDARY_CROSSING_SUPPORT_ONLY"; branch="retain_boundary_survival_support_only_then_audit_recovery_set_viability_envelope_for_reserve_no_capacity_or_regime_sweep"
        elif gates["boundary_hitting_reserve"]["go"] and gates["boundary_hitting_vs_v116_reserve"]["go"]:
            status="TAIL_BOUNDARY_CROSSING_RESERVE_ONLY"; branch="retain_boundary_reentry_reserve_only_then_audit_recovery_set_viability_envelope_for_support_no_capacity_or_regime_sweep"
        elif gates["boundary_hitting_support"].get("local_order") and gates["boundary_hitting_reserve"].get("local_order"):
            status="TAIL_BOUNDARY_CROSSING_LOCAL_ORDER_ONLY"; branch="one_convex_pairwise_audit_on_exact_boundary_hitting_features_no_feature_or_source_change"
        else:
            status="TAIL_BOUNDARY_CROSSING_FLOW_STOP"; branch="close_static_boundary_witness_hitting_flow_then_preregister_recovery_set_viability_survival_envelope_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep"
    ident=_variant_identity(docs)
    factorial_attribution_go = bool(
        not errors
        and gates.get("boundary_measure_effect_under_hitting_support", {}).get("go")
        and gates.get("boundary_measure_effect_under_hitting_reserve", {}).get("go")
        and gates.get("hitting_effect_under_boundary_measure_support", {}).get("go")
        and gates.get("hitting_effect_under_boundary_measure_reserve", {}).get("go")
    )
    decision={"status":status,"next_branch":branch,"balanced_precision_metric_identity":ident,
              "power_diagnostics":_power(docs),"tail_boundary_activity_gate":activity,
              "full_factorial_mechanism_attribution_go":factorial_attribution_go,
              "reentry_contact_coverage_go":bool(activity.get("reentry_contact_coverage_go")),
              "boundary_transport_authorized":False,"broad_encoder_training_authorized":False,
              "source_training_authorized":False,"regime_conditioned_policy_authorized":False,
              "dataset_reconstruction_authorized":False,"matched_dimension":MATCHED_DIM,
              "geometry_dimension":BOUNDARY_GEOMETRY_DIM,"constraint_names":["clearance","stopping","route","reentry"],
              "scientific_note":"V48.117 is a capacity-matched 2x2 audit. It keeps frozen weak-root observation semantics and same-option actuator-projected physics fixed, replaces the nearly singular max-option cotangent push-forward with a candidate-independent zero-margin boundary-witness measure, and separately replaces magnitude work with zero-boundary first-violation / persistent-reentry hitting channels. No regime label, learned root adapter, threshold, horizon, source or capacity sweep is introduced."}
    decision.update({k+"_gate":v for k,v in gates.items()})
    out={"schema":"ocrap-v48.117-tbcf-comparison-v1","engineering_version":ENGINEERING_VERSION,"scientific_version":SCIENTIFIC_VERSION,
         "run_instance_id":a.run_id,"valid":not errors,"attribution_ready":not errors,"errors":errors,
         "experiment_type":"audit_only_tail_boundary_crossing_flow","preregistered_decision":decision,
         "authoritative_v48_116_comparison_sha256":AUTHORITATIVE_V116_COMPARISON_SHA256,
         "v48_116_pipeline_sha256":_sha(a.v116_pipeline),"v48_116_comparison_sha256":_sha(a.v116_comparison),
         "v48_116_balanced_sha256":_sha(a.v116_balanced),"v48_116_precision_sha256":_sha(a.v116_precision),
         "stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"planner_parameters_trained":0,
         "relative_ranker_modified":False,"regime_conditioning":False,"boundary_transport":False,
         "teacher_metadata_input_to_model":False,"test_roots_read":False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"valid":out["valid"],"status":status,"errors":errors})); return 0 if out["valid"] else 30

if __name__=="__main__": raise SystemExit(main())
