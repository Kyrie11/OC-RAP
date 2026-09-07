#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

V103_ENGINEERING_VERSION="v48.103.0-OC-FCSS"
V106_ENGINEERING_VERSION="v48.106.0-OC-PEAO"
ENGINEERING_VERSION="v48.107.0-OC-FNAO"
ROLES=("dev_near","dev_contact","certificate_near","certificate_contact")


def _sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def _ok(v:Any,t:float)->bool: return v is not None and float(v)>=t
def _cross(rs:set[str],n:int)->bool:
    return len(rs)>=n and any("near" in x for x in rs) and any("contact" in x for x in rs)


def _result_errors(obj:dict[str,Any],variant:str)->list[str]:
    e=[]
    checks=[
        (obj.get("valid") is True,"valid"),(obj.get("engineering_version")==ENGINEERING_VERSION,"version"),(obj.get("variant")==variant,"variant"),
        (int(obj.get("stage_i_first_block_parameters_trained",-1))==444864,"first_block_params"),(int(obj.get("stage_i_other_parameters_trained",-1))==0,"other_stage_i"),
        (int(obj.get("frozen_v103_readout_parameters",-1))==1540,"v103_readout"),(int(obj.get("root_decoder_parameters_trained",-1))==0,"root"),
        (int(obj.get("source_parameters_trained",-1))==0,"source"),(int(obj.get("planner_parameters_trained",-1))==0,"planner"),
        (obj.get("frozen_stage_i_second_block") is True,"second_frozen"),(obj.get("ordinal_action_orientation_objective") is True,"orientation_objective"),
        (obj.get("ordinal_target_magnitude_discarded_after_sign") is True,"ordinal_sign_only"),(obj.get("nominal_first_block_exact_identity") is True,"nominal_first"),
        (obj.get("nominal_final_memory_exact_identity") is True,"nominal_final"),(obj.get("state_metrics_exact_v103",{}).get("valid") is True,"state_identity"),
        (obj.get("initial_v103_function_identity",{}).get("valid") is True,"initial_identity"),(obj.get("boundary_transport") is False,"boundary"),
        (obj.get("regime_conditioning") is False,"regime"),(obj.get("teacher_metadata_input_to_model") is False,"teacher_meta"),
    ]
    for ok,name in checks:
        if not ok:e.append(f"{variant}:{name}")
    for role in ROLES:
        if role not in obj.get("cells",{}): e.append(f"{variant}:cell:{role}")
    return e


def _action_gate(docs:dict[str,dict[str,Any]],metric:str)->dict[str,Any]:
    positive=[]; top=[]; roles=set(); top_roles=set()
    for v,d in docs.items():
        for role in ROLES:
            m=d["cells"][role][f"{metric}_true"]
            if _ok(m.get("auc"),.65) and _ok(m.get("auc_vs_shuffled"),.05):
                positive.append([v,role]);roles.add(role)
            if _ok(m.get("top1_vs_shuffled"),.10):
                top.append([v,role]);top_roles.add(role)
    go=len(positive)>=6 and _cross(roles,3) and len(top)>=4 and _cross(top_roles,2)
    return {"go":bool(go),"positive_cells":positive,"top1_material_cells":top,"roles":sorted(roles),"top1_roles":sorted(top_roles)}


def _state_gate(docs):
    positive=[];roles=set()
    for v,d in docs.items():
        for role in ROLES:
            if _ok(d["cells"][role]["state"].get("auc"),.70): positive.append([v,role]);roles.add(role)
    return {"go":bool(len(positive)>=6 and _cross(roles,3)),"positive_cells":positive,"roles":sorted(roles)}


def _auc_deltas(docs,refs):
    out={"support":[],"reserve":[],"state":[]}
    for metric in out:
        for v in ("balanced","precision"):
            for role in ROLES:
                key="state" if metric=="state" else f"{metric}_true"
                a=docs[v]["cells"][role][key].get("auc"); b=refs[v]["cells"][role][key].get("auc")
                out[metric].append({"variant":v,"role":role,"v107_auc":a,"v103_auc":b,"delta":None if a is None or b is None else float(a)-float(b)})
    return out


def main()->int:
    ap=argparse.ArgumentParser()
    for k in ("balanced","precision","v103_balanced","v103_precision","v106_pipeline","v106_comparison"): ap.add_argument("--"+k.replace("_","-"),dest=k,type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();errors=[]
    docs={v:json.loads(getattr(a,v).read_text()) for v in ("balanced","precision")}
    refs={"balanced":json.loads(a.v103_balanced.read_text()),"precision":json.loads(a.v103_precision.read_text())}
    for v in docs: errors+=_result_errors(docs[v],v)
    for v,r in refs.items():
        if not(r.get("valid") and r.get("engineering_version")==V103_ENGINEERING_VERSION and r.get("variant")==v):errors.append(f"v103:{v}")
    p106=json.loads(a.v106_pipeline.read_text());c106=json.loads(a.v106_comparison.read_text());d106=c106.get("preregistered_decision") or {}
    if not(p106.get("valid") and p106.get("attribution_ready") and p106.get("engineering_version")==V106_ENGINEERING_VERSION and p106.get("preregistered_status")=="PREENCODER_ACTION_ORIENTATION_STOP"): errors.append("v106_pipeline")
    if not(c106.get("valid") and c106.get("attribution_ready") and d106.get("status")=="PREENCODER_ACTION_ORIENTATION_STOP" and d106.get("next_branch")=="preencoder_action_orientation_insufficient_then_preregister_first_stage_i_block_nominal_invariant_action_orientation_objective_no_source_or_broad_encoder_sweep"): errors.append("v106_branch")
    state=_state_gate(docs) if not errors else {"go":False,"positive_cells":[],"roles":[]}
    support=_action_gate(docs,"support") if not errors else {"go":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    reserve=_action_gate(docs,"reserve") if not errors else {"go":False,"positive_cells":[],"top1_material_cells":[],"roles":[],"top1_roles":[]}
    full=bool(state["go"] and support["go"] and reserve["go"])
    if errors:
        status="V48_107_ENGINEERING_STOP";next_branch="fix_v48_107_engineering_and_rerun_same_first_block_orientation_experiment"
    elif full:
        status="FIRST_BLOCK_NOMINAL_INVARIANT_ACTION_ORIENTATION_GO";next_branch="first_block_orientation_full_go_then_preregister_one_production_nominal_invariant_control_transport_no_source_sweep"
    elif state["go"] and support["go"] and not reserve["go"]:
        status="FIRST_BLOCK_ACTION_ORIENTATION_PARTIAL_SUPPORT";next_branch="first_block_support_sufficient_then_preregister_one_signed_reserve_debt_orientation_objective_no_source_or_broad_encoder"
    elif state["go"] and reserve["go"] and not support["go"]:
        status="FIRST_BLOCK_ACTION_ORIENTATION_PARTIAL_RESERVE";next_branch="first_block_reserve_sufficient_then_preregister_one_support_establishment_orientation_objective_no_source_or_broad_encoder"
    else:
        status="FIRST_BLOCK_NOMINAL_INVARIANT_ACTION_ORIENTATION_STOP";next_branch="close_first_block_orientation_then_preregister_raw_to_projected_action_pathway_audit_no_broad_encoder_or_source_sweep"
    decision={
        "status":status,"next_branch":next_branch,"first_block_orientation_go":full,"state_go":state["go"],"support_go":support["go"],"reserve_go":reserve["go"],
        "state_positive_cells":state["positive_cells"],"state_roles":state["roles"],"support_positive_cells":support["positive_cells"],"support_top1_material_cells":support["top1_material_cells"],"support_roles":support["roles"],
        "reserve_positive_cells":reserve["positive_cells"],"reserve_top1_material_cells":reserve["top1_material_cells"],"reserve_roles":reserve["roles"],
        "diagnostic_auc_deltas_vs_v103":_auc_deltas(docs,refs),"source_training_authorized":False,"broad_encoder_training_authorized":False,"boundary_transport_authorized":False,"dataset_reconstruction_authorized":False,"regime_conditioned_policy_authorized":False,
    }
    out={"schema":"ocrap-v48.107-fnao-comparison-v1","engineering_version":ENGINEERING_VERSION,"valid":not errors,"attribution_ready":not errors,"errors":errors,"experiment_type":"first_stage_i_block_nominal_invariant_ordinal_action_orientation","preregistered_decision":decision,
         "stage_i_first_block_parameters_trained":444864,"stage_i_other_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"planner_parameters_trained":0,"relative_ranker_modified":False,"regime_conditioning":False,"boundary_transport":False,"teacher_metadata_input_to_model":False,"test_roots_read":False,
         "v48_106_pipeline_sha256":_sha(a.v106_pipeline),"v48_106_comparison_sha256":_sha(a.v106_comparison)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"valid":out["valid"],"status":status,"errors":errors}));return 0 if out["valid"] else 30
if __name__=="__main__": raise SystemExit(main())
