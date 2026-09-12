#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import torch

from ocrap.audits.zero_boundary_viability_transition import (
    ENGINEERING_VERSION, SCIENTIFIC_VERSION, TRANSITION_GEOMETRY_DIM, MATCHED_DIM, TRANSITION_MODE_NAMES,
)

AUTHORITATIVE_V122_PIPELINE_SHA256 = "5432df0f0937969d01616f96f8a27e992932d9c297813b6866dd42831b111e8e"
AUTHORITATIVE_V122_COMPARISON_SHA256 = "6d32da4ace62ed2f01dc76bcda55a80623e339195b28267ac93c2dde92914fe2"
V122_NEXT = "close_signed_viability_rank_state_transport_then_preregister_zero_boundary_viability_state_transition_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep"
VALID_STATUSES = {
    "ZERO_BOUNDARY_VIABILITY_STATE_TRANSITION_GO",
    "EXPOSED_ZERO_BOUNDARY_VIABILITY_STATE_TRANSITION_GO",
    "ZERO_BOUNDARY_VIABILITY_STATE_TRANSITION_SUPPORT_ONLY",
    "ZERO_BOUNDARY_VIABILITY_STATE_TRANSITION_RESERVE_ONLY",
    "ZERO_BOUNDARY_VIABILITY_STATE_TRANSITION_LOCAL_ORDER_ONLY",
    "ZERO_BOUNDARY_VIABILITY_STATE_TRANSITION_STOP",
}

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser()
    for key in ("runtime","balanced","precision","balanced_state","precision_state","comparison","v48_122_pipeline","v48_122_comparison"):
        ap.add_argument("--"+key.replace("_","-"),dest=key,type=Path,required=True)
    ap.add_argument("--run-id",required=True); ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(); errors=[]; docs={}
    for key in ("runtime","balanced","precision","comparison","v48_122_pipeline","v48_122_comparison"):
        try: docs[key]=json.loads(getattr(a,key).read_text())
        except Exception as exc: errors.append(f"{key}:json:{type(exc).__name__}"); docs[key]={}

    rt=docs["runtime"]; sc=rt.get("scientific_contract") or {}
    if not (rt.get("valid") and rt.get("attribution_ready") and rt.get("engineering_version")==ENGINEERING_VERSION
        and rt.get("scientific_version")==SCIENTIFIC_VERSION and rt.get("run_instance_id")==a.run_id
        and sc.get("zero_boundary_viability_state_transition") is True and sc.get("zero_boundary_transition_decomposition") is True
        and sc.get("instantaneous_rank_transport_exactly_recoverable") is True
        and sc.get("transition_mode_names")==[str(x) for x in TRANSITION_MODE_NAMES]
        and sc.get("transition_basis")=="exact_positive_part_reserve_and_debt_repayment_x_nominal_rank"
        and sc.get("rank_coordinate")=="candidate_independent_nominal_same_option_viability_midranks"
        and sc.get("candidate_rank_sort_used_for_coordinate") is False
        and sc.get("primary_option_set")=="all_common_valid_recovery_options"
        and sc.get("boundary_support_weights_used_in_transition") is False
        and sc.get("active_option_identity_exported") is False and sc.get("pre_readout_candidate_option_selector") is False
        and sc.get("same_option_nominal_rank_correspondence") is True
        and sc.get("zero_boundary_threshold_or_window_sweep") is False
        and sc.get("matched_family_dim")==MATCHED_DIM and sc.get("transition_geometry_dim")==TRANSITION_GEOMETRY_DIM
        and sc.get("boundary_transport") is False and sc.get("rank_cut_sweep") is False and sc.get("capacity_sweep") is False
        and sc.get("horizon_sweep") is False and sc.get("threshold_sweep") is False): errors.append("runtime")

    for variant in ("balanced","precision"):
        d=docs[variant]
        if not (d.get("valid") and d.get("engineering_version")==ENGINEERING_VERSION and d.get("scientific_version")==SCIENTIFIC_VERSION
            and d.get("variant")==variant and d.get("audit_only") and d.get("convex_closed_form_ridge") and d.get("strictly_convex_unique_solution")
            and d.get("capacity_matched_all_transition_families") and d.get("matched_family_dimension")==MATCHED_DIM
            and d.get("transition_geometry_dimension")==TRANSITION_GEOMETRY_DIM
            and d.get("transition_mode_names")==[str(x) for x in TRANSITION_MODE_NAMES]
            and d.get("transition_basis")=="exact_positive_part_reserve_and_debt_repayment_x_nominal_rank"
            and d.get("rank_coordinate")=="candidate_independent_nominal_same_option_viability_midranks"
            and d.get("candidate_rank_sort_used_for_coordinate") is False and d.get("primary_option_set")=="all_common_valid_recovery_options"
            and d.get("boundary_support_weights_used_in_transition") is False and d.get("active_option_identity_exported") is False
            and d.get("zero_boundary_transition_decomposition") is True and d.get("instantaneous_rank_transport_exactly_recoverable") is True
            and d.get("teacher_npz_fields_loaded_into_feature_path")==["root_valid"] and d.get("teacher_future_fields_used") is False
            and d.get("root_decoder_parameters_trained")==0 and d.get("source_parameters_trained")==0
            and d.get("regime_conditioning") is False and d.get("boundary_transport") is False and d.get("rank_cut_sweep") is False
            and d.get("zero_boundary_threshold_or_window_sweep") is False and d.get("same_option_nominal_rank_correspondence") is True
            and d.get("run_instance_id")==a.run_id): errors.append(variant)

    cmp=docs["comparison"]; decision=cmp.get("preregistered_decision") or {}
    if not (cmp.get("valid") and cmp.get("attribution_ready") and cmp.get("engineering_version")==ENGINEERING_VERSION
        and cmp.get("scientific_version")==SCIENTIFIC_VERSION and cmp.get("run_instance_id")==a.run_id
        and decision.get("status") in VALID_STATUSES and decision.get("transition_mode_names")==[str(x) for x in TRANSITION_MODE_NAMES]
        and decision.get("transition_basis")=="exact_positive_part_reserve_and_debt_repayment_x_nominal_rank"
        and decision.get("rank_coordinate")=="candidate_independent_nominal_same_option_viability_midranks"
        and decision.get("boundary_transport_authorized") is False and decision.get("source_training_authorized") is False): errors.append("comparison")

    for variant,key in (("balanced","balanced_state"),("precision","precision_state")):
        try:
            st=torch.load(getattr(a,key),map_location="cpu",weights_only=False)
            if not (st.get("schema")=="ocrap-v48.123-zero-boundary-viability-state-transition-state-v1" and st.get("engineering_version")==ENGINEERING_VERSION
                and st.get("scientific_version")==SCIENTIFIC_VERSION and st.get("variant")==variant and st.get("run_instance_id")==a.run_id
                and st.get("convex_closed_form_ridge") is True and st.get("strictly_convex_unique_solution") is True): errors.append(key)
        except Exception as exc: errors.append(f"{key}:load:{type(exc).__name__}")

    if sha(a.v48_122_pipeline)!=AUTHORITATIVE_V122_PIPELINE_SHA256: errors.append("v122_pipeline_sha")
    if sha(a.v48_122_comparison)!=AUTHORITATIVE_V122_COMPARISON_SHA256: errors.append("v122_comparison_sha")
    d122=docs["v48_122_comparison"].get("preregistered_decision") or {}
    if not (docs["v48_122_pipeline"].get("valid") and docs["v48_122_pipeline"].get("attribution_ready")
        and docs["v48_122_pipeline"].get("preregistered_status")=="SIGNED_VIABILITY_RANK_STATE_TRANSPORT_STOP"
        and docs["v48_122_comparison"].get("valid") and docs["v48_122_comparison"].get("attribution_ready")
        and d122.get("status")=="SIGNED_VIABILITY_RANK_STATE_TRANSPORT_STOP" and d122.get("next_branch")==V122_NEXT): errors.append("v122_prerequisite")

    artifacts={}
    for key in ("runtime","balanced","precision","balanced_state","precision_state","comparison"):
        p=getattr(a,key); artifacts[key]={"path":str(p.resolve()),"sha256":sha(p)}
    out={"schema":"ocrap-v48.123-zbst-pipeline-complete-v1","engineering_version":ENGINEERING_VERSION,"scientific_version":SCIENTIFIC_VERSION,
        "run_instance_id":a.run_id,"valid":not errors,"attribution_ready":not errors,"errors":errors,
        "experiment_type":"audit_only_zero_boundary_viability_state_transition","preregistered_status":decision.get("status"),"artifacts":artifacts,
        "authoritative_v48_122_comparison_sha256":AUTHORITATIVE_V122_COMPARISON_SHA256,"v48_122_pipeline_sha256":sha(a.v48_122_pipeline),
        "v48_122_comparison_sha256":sha(a.v48_122_comparison),"dataset_reconstruction":False,"planner_parameters_trained":0,
        "stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"regime_conditioning":False,
        "boundary_transport":False,"teacher_metadata_input_to_model":False,"test_roots_read":False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"valid":out["valid"],"attribution_ready":out["attribution_ready"],"status":out["preregistered_status"],"errors":errors}))
    return 0 if out["valid"] else 30
if __name__=="__main__": raise SystemExit(main())
