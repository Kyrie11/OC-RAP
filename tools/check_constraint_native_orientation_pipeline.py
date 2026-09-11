#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from ocrap.audits.viability_order_profile import ENGINEERING_VERSION, SCIENTIFIC_VERSION, PROFILE_GEOMETRY_DIM, MATCHED_DIM, ORDER_MASSES

AUTHORITATIVE_V118_PIPELINE_SHA256='8ded1afa8a7fc002fc39a19c11da8669dc702f2704e8141e742182f241af05ee'
AUTHORITATIVE_V118_COMPARISON_SHA256='ebe8b2bc18baa33db9f62c809d22eb7b4c9d28f0fcd8f6f4140f66287e3a3184'
V118_NEXT='close_signed_joint_max_min_survival_envelope_then_preregister_recovery_set_viability_order_profile_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep'
VALID_STATUSES={'VIABILITY_ORDER_PROFILE_GO','EXPOSED_VIABILITY_ORDER_PROFILE_GO','VIABILITY_ORDER_PROFILE_SUPPORT_ONLY','VIABILITY_ORDER_PROFILE_RESERVE_ONLY','VIABILITY_ORDER_PROFILE_LOCAL_ORDER_ONLY','VIABILITY_ORDER_PROFILE_STOP'}
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
    ap=argparse.ArgumentParser()
    for key in ('runtime','balanced','precision','balanced_state','precision_state','comparison','v48_118_pipeline','v48_118_comparison'):
        ap.add_argument('--'+key.replace('_','-'),dest=key,type=Path,required=True)
    ap.add_argument('--run-id',required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    errors=[]; docs={}
    for key in ('runtime','balanced','precision','comparison','v48_118_pipeline','v48_118_comparison'):
        try: docs[key]=json.loads(getattr(a,key).read_text())
        except Exception as exc: errors.append(f'{key}:json:{type(exc).__name__}'); docs[key]={}
    rt=docs['runtime']; sc=rt.get('scientific_contract') or {}
    if not (rt.get('valid') and rt.get('attribution_ready') and rt.get('engineering_version')==ENGINEERING_VERSION and rt.get('scientific_version')==SCIENTIFIC_VERSION and rt.get('run_instance_id')==a.run_id and sc.get('viability_order_profile') is True and sc.get('order_masses')==[float(x) for x in ORDER_MASSES] and sc.get('primary_option_set')=='all_common_valid_recovery_options' and sc.get('boundary_support_weights_used_in_profile') is False and sc.get('uniform_mean_standalone_family') is False and sc.get('active_option_identity_exported') is False and sc.get('pre_readout_candidate_option_selector') is False and sc.get('matched_family_dim')==MATCHED_DIM and sc.get('profile_geometry_dim')==PROFILE_GEOMETRY_DIM and sc.get('boundary_transport') is False): errors.append('runtime')
    for variant in ('balanced','precision'):
        d=docs[variant]
        if not (d.get('valid') and d.get('engineering_version')==ENGINEERING_VERSION and d.get('scientific_version')==SCIENTIFIC_VERSION and d.get('variant')==variant and d.get('audit_only') and d.get('convex_closed_form_ridge') and d.get('strictly_convex_unique_solution') and d.get('capacity_matched_all_profile_families') and d.get('matched_family_dimension')==MATCHED_DIM and d.get('profile_geometry_dimension')==PROFILE_GEOMETRY_DIM and d.get('order_masses')==[float(x) for x in ORDER_MASSES] and d.get('primary_option_set')=='all_common_valid_recovery_options' and d.get('boundary_support_weights_used_in_profile') is False and d.get('uniform_mean_standalone_family') is False and d.get('active_option_identity_exported') is False and d.get('teacher_npz_fields_loaded_into_feature_path')==['root_valid'] and d.get('teacher_future_fields_used') is False and d.get('root_decoder_parameters_trained')==0 and d.get('source_parameters_trained')==0 and d.get('regime_conditioning') is False and d.get('boundary_transport') is False and d.get('run_instance_id')==a.run_id): errors.append(variant)
    cmp=docs['comparison']; decision=cmp.get('preregistered_decision') or {}
    if not (cmp.get('valid') and cmp.get('attribution_ready') and cmp.get('engineering_version')==ENGINEERING_VERSION and cmp.get('scientific_version')==SCIENTIFIC_VERSION and cmp.get('run_instance_id')==a.run_id and decision.get('status') in VALID_STATUSES and decision.get('order_masses')==[float(x) for x in ORDER_MASSES] and decision.get('boundary_transport_authorized') is False and decision.get('source_training_authorized') is False): errors.append('comparison')
    for variant,key in (('balanced','balanced_state'),('precision','precision_state')):
        try:
            st=torch.load(getattr(a,key),map_location='cpu',weights_only=False)
            if not (st.get('engineering_version')==ENGINEERING_VERSION and st.get('scientific_version')==SCIENTIFIC_VERSION and st.get('variant')==variant and st.get('run_instance_id')==a.run_id and st.get('convex_closed_form_ridge') is True and st.get('strictly_convex_unique_solution') is True): errors.append(key)
        except Exception as exc: errors.append(f'{key}:load:{type(exc).__name__}')
    if sha(a.v48_118_pipeline)!=AUTHORITATIVE_V118_PIPELINE_SHA256: errors.append('v118_pipeline_sha')
    if sha(a.v48_118_comparison)!=AUTHORITATIVE_V118_COMPARISON_SHA256: errors.append('v118_comparison_sha')
    d118=docs['v48_118_comparison'].get('preregistered_decision') or {}
    if not (docs['v48_118_pipeline'].get('valid') and docs['v48_118_pipeline'].get('attribution_ready') and docs['v48_118_pipeline'].get('preregistered_status')=='VIABILITY_SURVIVAL_ENVELOPE_STOP' and docs['v48_118_comparison'].get('valid') and docs['v48_118_comparison'].get('attribution_ready') and d118.get('status')=='VIABILITY_SURVIVAL_ENVELOPE_STOP' and d118.get('next_branch')==V118_NEXT): errors.append('v118_prerequisite')
    artifacts={}
    for key in ('runtime','balanced','precision','balanced_state','precision_state','comparison'):
        p=getattr(a,key); artifacts[key]={"path":str(p.resolve()),"sha256":sha(p)}
    out={"schema":"ocrap-v48.119-vop-pipeline-complete-v1","engineering_version":ENGINEERING_VERSION,"scientific_version":SCIENTIFIC_VERSION,"run_instance_id":a.run_id,"valid":not errors,"attribution_ready":not errors,"errors":errors,"experiment_type":"audit_only_recovery_set_viability_order_profile","preregistered_status":decision.get('status'),"artifacts":artifacts,"authoritative_v48_118_comparison_sha256":AUTHORITATIVE_V118_COMPARISON_SHA256,"v48_118_pipeline_sha256":sha(a.v48_118_pipeline),"v48_118_comparison_sha256":sha(a.v48_118_comparison),"dataset_reconstruction":False,"planner_parameters_trained":0,"stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"regime_conditioning":False,"boundary_transport":False,"teacher_metadata_input_to_model":False,"test_roots_read":False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({"valid":out['valid'],"attribution_ready":out['attribution_ready'],"status":out['preregistered_status'],"errors":errors})); return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
