#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any

from ocrap.v48_104_nominal_invariant_control_refinement import ENGINEERING_VERSION

ROLES=("dev_near","dev_contact","certificate_near","certificate_contact")


def _sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def _ok(v:Any,t:float)->bool:
    try: return v is not None and float(v)>=float(t)
    except Exception: return False


def _population_errors(obj,ref,variant):
    e=[]
    for role in ROLES:
        a=obj["cells"][role]; b=ref["cells"][role]
        for name,fields in (("state",("rows","drs_state_rows","dep_state_rows")),("support_true",("rows","positive_rows","negative_rows","powered_groups")),("support_shuffled",("rows","positive_rows","negative_rows","powered_groups")),("reserve_true",("rows","positive_rows","negative_rows","powered_groups")),("reserve_shuffled",("rows","positive_rows","negative_rows","powered_groups"))):
            for f in fields:
                if int((a.get(name) or {}).get(f,-1))!=int((b.get(name) or {}).get(f,-2)): e.append(f"{variant}:{role}:{name}:{f}:population_drift")
    return e


def _result_errors(obj,ref,variant):
    e=[]
    if not obj.get("valid") or obj.get("engineering_version")!=ENGINEERING_VERSION or obj.get("variant")!=variant: e.append(f"{variant}:contract")
    if int(obj.get("stage_i_last_block_parameters_trained",-1))<=0: e.append(f"{variant}:last_block_not_trainable")
    for k in ("planner_parameters_trained","stage_i_other_parameters_trained","root_decoder_parameters_trained","source_parameters_trained"):
        if int(obj.get(k,-1))!=0: e.append(f"{variant}:{k}")
    if int(obj.get("frozen_v103_readout_parameters",-1))!=1540: e.append(f"{variant}:v103_readout")
    for k in ("response_only_objective","nominal_memory_exact_identity","initial_v103_function_identity"):
        if obj.get(k) is not True: e.append(f"{variant}:{k}")
    if not (obj.get("state_metrics_exact_v103") or {}).get("valid"): e.append(f"{variant}:state_identity")
    if obj.get("boundary_transport") or obj.get("regime_conditioning") or obj.get("teacher_metadata_input_to_model"): e.append(f"{variant}:forbidden")
    # State metrics are a hard engineering identity, not merely a scientific gate.
    for role in ROLES:
        a=obj["cells"][role]["state"]; b=ref["cells"][role]["state"]
        if a.get("auc") is None or b.get("auc") is None or abs(float(a["auc"])-float(b["auc"]))>1e-7: e.append(f"{variant}:{role}:state_auc_identity")
    e+=_population_errors(obj,ref,variant)
    return e


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--balanced",type=Path,required=True); ap.add_argument("--precision",type=Path,required=True)
    ap.add_argument("--v103-balanced",type=Path,required=True); ap.add_argument("--v103-precision",type=Path,required=True); ap.add_argument("--v103-comparison",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(); docs={"balanced":json.loads(a.balanced.read_text()),"precision":json.loads(a.precision.read_text())}; refs={"balanced":json.loads(a.v103_balanced.read_text()),"precision":json.loads(a.v103_precision.read_text())}; c103=json.loads(a.v103_comparison.read_text()); d103=c103.get("preregistered_decision") or {}; errors=[]
    if not(c103.get("valid") and c103.get("attribution_ready") and d103.get("status")=="FACTORIZED_CONTROL_SUFFICIENT_STATE_STOP" and d103.get("factorized_state_representation_go") is True and d103.get("factorized_support_action_go") is False and d103.get("factorized_reserve_debt_go") is False and d103.get("next_branch")=="close_frozen_stage_i_readout_family_then_preregister_last_stage_i_block_control_sufficient_representation_objective_no_broad_encoder_or_source_sweep"):
        errors.append("v103_last_block_branch_prerequisite_missing")
    for v in ("balanced","precision"): errors+=_result_errors(docs[v],refs[v],v)
    state_cells=[]; support_cells=[]; reserve_cells=[]; support_top=[]; reserve_top=[]; state_roles=set(); support_roles=set(); reserve_roles=set(); support_top_roles=set(); reserve_top_roles=set(); deltas={"support":[],"reserve":[]}
    if not errors:
        for v in ("balanced","precision"):
            for role in ROLES:
                c=docs[v]["cells"][role]; st,su,re=c["state"],c["support_true"],c["reserve_true"]
                if _ok(st.get("auc"),.70): state_cells.append([v,role]); state_roles.add(role)
                if _ok(su.get("auc"),.65) and _ok(su.get("auc_vs_shuffled"),.05): support_cells.append([v,role]); support_roles.add(role)
                if _ok(re.get("auc"),.65) and _ok(re.get("auc_vs_shuffled"),.05): reserve_cells.append([v,role]); reserve_roles.add(role)
                if _ok(su.get("top1_vs_shuffled"),.10): support_top.append([v,role]); support_top_roles.add(role)
                if _ok(re.get("top1_vs_shuffled"),.10): reserve_top.append([v,role]); reserve_top_roles.add(role)
                for name,metric in (("support","support_true"),("reserve","reserve_true")):
                    x=c[metric].get("auc"); y=refs[v]["cells"][role][metric].get("auc")
                    if x is not None and y is not None: deltas[name].append({"variant":v,"role":role,"v104_auc":float(x),"v103_auc":float(y),"delta":float(x)-float(y)})
    def cross(rs,n): return len(rs)>=n and any("near" in x for x in rs) and any("contact" in x for x in rs)
    state_go=bool(not errors and len(state_cells)>=6 and cross(state_roles,3))
    support_go=bool(not errors and len(support_cells)>=6 and cross(support_roles,3) and len(support_top)>=4 and cross(support_top_roles,2))
    reserve_go=bool(not errors and len(reserve_cells)>=6 and cross(reserve_roles,3) and len(reserve_top)>=4 and cross(reserve_top_roles,2))
    full=bool(state_go and support_go and reserve_go)
    if errors: status="V48_104_ENGINEERING_STOP"; branch="fix_v48_104_engineering_and_rerun_same_nominal_invariant_last_block"
    elif full: status="NOMINAL_INVARIANT_CONTROL_REFINEMENT_GO"; branch="preregister_one_production_factorized_stage_i_to_root_transport_no_source_or_capacity_sweep"
    elif state_go and support_go and not reserve_go: status="NOMINAL_INVARIANT_CONTROL_REFINEMENT_PARTIAL_SUPPORT"; branch="retain_nominal_invariant_support_then_preregister_one_supported_reserve_flow_objective"
    elif state_go and reserve_go and not support_go: status="NOMINAL_INVARIANT_CONTROL_REFINEMENT_PARTIAL_RESERVE"; branch="retain_nominal_invariant_reserve_then_preregister_one_support_establishment_objective"
    else: status="NOMINAL_INVARIANT_CONTROL_REFINEMENT_STOP"; branch="close_last_stage_i_block_refinement_then_preregister_pre_last_token_action_equivariance_audit_no_broad_encoder_or_source_sweep"
    decision={"state_go":state_go,"support_go":support_go,"reserve_go":reserve_go,"nominal_invariant_control_refinement_go":full,"state_positive_cells":state_cells,"support_positive_cells":support_cells,"reserve_positive_cells":reserve_cells,"support_top1_material_cells":support_top,"reserve_top1_material_cells":reserve_top,"state_roles":sorted(state_roles),"support_roles":sorted(support_roles),"reserve_roles":sorted(reserve_roles),"auc_deltas_vs_v103":deltas,"status":status,"next_branch":branch,"source_training_authorized":False,"boundary_transport_authorized":False,"broad_encoder_training_authorized":False,"dataset_reconstruction_authorized":False,"regime_conditioned_policy_authorized":False}
    out={"schema":"ocrap-v48.104-nicr-comparison-v1","engineering_version":ENGINEERING_VERSION,"valid":not errors,"attribution_ready":not errors,"errors":errors,"experiment_type":"nominal_invariant_last_stage_i_block_control_refinement","planner_parameters_trained":0,"stage_i_other_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"dataset_reconstruction":False,"teacher_metadata_input_to_model":False,"boundary_transport":False,"relative_ranker_modified":False,"regime_conditioning":False,"v48_103_comparison_sha256":_sha(a.v103_comparison),"preregistered_decision":decision,"test_roots_read":False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps({"valid":out["valid"],"status":status,"state":state_go,"support":support_go,"reserve":reserve_go})); return 0 if out["valid"] else 30
if __name__=="__main__": raise SystemExit(main())
