#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
KINDS=("dev_near","dev_contact","certificate_near","certificate_contact")
VARIANTS=("balanced","precision")
def load(p:Path)->dict:return json.loads(p.read_text(encoding="utf-8"))
def dep_block(d:dict):
    phase="fit" if bool(d.get("development_fit_only")) else "verify"; return phase,(d.get(phase) or {})
def metric(run:Path,v:str,k:str)->dict[str,Any]:
    fn={"dev_near":"dev_diagnostic_near_v48.json","dev_contact":"dev_diagnostic_contact_v48.json","certificate_near":"direct_value_risk_near_v48.json","certificate_contact":"direct_value_risk_contact_v48.json"}[k]
    p=run/'candidates'/v/'calibration'/fn
    if not p.is_file():return {"missing":str(p)}
    d=load(p);phase,dep=dep_block(d)
    dk=("num_groups","num_selected","selection_rate","num_positive_selected","positive_recall","precision","precision_wilson_lcb90","num_harmful_selected","harmful_selected_rate","harmful_selected_ucb90","harmful_group_exposure","harmful_group_exposure_ucb90","num_opportunities")
    qk=("candidate_safe_positive_auc","candidate_harm_auc","candidate_pred_teacher_correlation","candidate_rank_teacher_correlation","proposal_evidence_top1_correlation","proposal_evidence_top1_safe_positive_auc","proposal_evidence_top1_harm_auc","proposal_deployed_rule_abstention_rate","proposal_deployed_rule_top1_safe_positive_auc","proposal_top_k","proposal_positive_group_count","proposal_oracle_best_hit_rate_positive_groups","proposal_any_positive_hit_rate_positive_groups")
    return {"path":str(p),"phase":phase,"valid_for_deployment":d.get("valid_for_deployment"),"rejection_kind":d.get("rejection_kind"),"absolute_feasibility_mode":d.get("absolute_feasibility_mode"),"absolute_feasibility_threshold":d.get("absolute_feasibility_threshold"),"deployment":{x:dep.get(x) for x in dk if x in dep},"ranking_and_selector_diagnostics":{x:d.get(x) for x in qk if x in d},"proposal_constrained_oracle_gate":d.get("proposal_constrained_oracle_gate"),"proposal_support_curve":d.get("proposal_support_curve")}
def am(audit,arm,v,k,key):return (((audit.get('arms') or {}).get(arm) or {}).get(v) or {}).get(k,{}).get(key)
def main():
    ap=argparse.ArgumentParser(description='v48.62 OC-CWRF controlled source attribution')
    for n in ('a','b','c','d','e','f','g'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    x=ap.parse_args();audit=load(x.feasibility_audit)
    arms={'A':x.a,'B_native':x.b,'C_AFE':x.c,'D_ORFC':x.d,'E_CPHR':x.e,'F_ERWF':x.f,'G_Main_OCCWRF':x.g}
    deltas={};allpos=True;meaningful=0
    for v in VARIANTS:
        deltas[v]={}
        for k in KINDS:
            b=am(audit,'B_native',v,k,'absolute_feasibility_auc');f=am(audit,'F_ERWF',v,k,'absolute_feasibility_auc');g=am(audit,'G_OCCWRF',v,k,'absolute_feasibility_auc')
            gb=None if b is None or g is None else float(g)-float(b);gf=None if f is None or g is None else float(g)-float(f)
            deltas[v][k]={'G_minus_B_absolute_feasibility_auc':gb,'G_minus_F_absolute_feasibility_auc':gf}
            if gb is None or gb<=0:allpos=False
            if gb is not None and gb>=.01:meaningful+=1
    auc_go=bool(allpos and meaningful>=6)
    checks=[]
    for v in VARIANTS:
        for k in KINDS:
            gi=am(audit,'G_OCCWRF',v,k,'teacher_infeasible_pass_fraction');gh=am(audit,'G_OCCWRF',v,k,'harmful_pass_fraction')
            refs=[]
            for arm in ('C_AFE','D_ORFC','E_CPHR','F_ERWF'):
                z=am(audit,arm,v,k,'teacher_infeasible_pass_fraction')
                if z is not None:refs.append(float(z))
            cap=min(refs)-.05 if refs else .30
            ok=gi is not None and float(gi)<=max(0.0,cap)
            checks.append({'variant':v,'split':k,'G_teacher_infeasible_pass_fraction':gi,'G_harmful_pass_fraction':gh,'previous_best_minus_0p05_cap':max(0.0,cap),'valid':ok})
    selective_go=all(z['valid'] for z in checks);go=auc_go and selective_go
    doc={'schema':'ocrap-v48.62-occwrf-comparison-v1','arms':{n:{v:{k:metric(r,v,k) for k in KINDS} for v in VARIANTS} for n,r in arms.items()},'feasibility_role_audit':audit,'source_deltas':deltas,
      'preregistered_decision':{'status':'GO' if go else 'STOP','all_8_G_minus_B_auc_positive_and_at_least_6_ge_0p01':auc_go,'harmful_infeasible_pass_materially_below_previous_permissive_family':selective_go,'selectivity_checks':checks},
      'attribution_order':['G-B common-witness absolute-source ordering (primary)','G-F isolates non-compensatory common-witness composition from ERWF linear scalarization','G-D tests whether observation-consistent commonality retains Contact option signal without Near regression','state/variant isolation and fixed top-5 proposal contract','G-A deployment propagation only after source evidence'],
      'scientific_contract':{
        'primary_hypothesis':'recoverability requires an observation-consistent common recovery option whose executable continuation satisfies a non-compensatory finite-time physical recovery barrier',
        'GO_requires':['G-B source AUC improves in all 8 Near/Contact cells and >=0.01 in at least six','teacher-infeasible/harmful pass is materially lower than C/D/E/F rather than a permissive operating-point shift','only two shared bounded gains train; all root/option/base Stage-I state stays bitwise frozen','fixed top-5, threshold 0.5, unchanged RIFA/relative ranking, no regime id/router or teacher future','source gain precedes deployment and Safe non-interference evaluation'],
        'STOP_if':['any source AUC regresses vs B','selectivity remains in the C/D/E/F permissive range','gain is only threshold/pass movement','state/provenance fails'],
        'forbidden_next_sweeps':['threshold/LR/horizon/grid search','proposal expansion/densification','option-specific free bias','generic AFE MLP/width/class-weight sweep','candidate-only CPHR scalar correction','compensatory ERWF coordinate sum','regime routing/policy/threshold/budget','broad root/margin/encoder retraining','privileged teacher future/component distillation','relative centering/ranking changes before source GO'],
        'safe_role':'same shared mechanism; paired Safe nominal-utility non-interference only after source GO'},
    }
    x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_62_occwrf_comparison','decision':doc['preregistered_decision']['status'],'output':str(x.output)}))
if __name__=='__main__':main()
