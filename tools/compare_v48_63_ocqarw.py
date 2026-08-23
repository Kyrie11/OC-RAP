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
    ap=argparse.ArgumentParser(description='v48.63 OC-QARW controlled quantifier-alignment attribution')
    for n in ('a','b','c','d','e','f','g','h'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    x=ap.parse_args();audit=load(x.feasibility_audit)
    arms={'A':x.a,'B_native':x.b,'C_AFE':x.c,'D_ORFC':x.d,'E_CPHR':x.e,'F_ERWF':x.f,'G_OCCWRF':x.g,'H_Main_OCQARW':x.h}
    deltas={};allpos=True;meaningful=0;selectivity=[];recall=[]
    for v in VARIANTS:
        deltas[v]={}
        for k in KINDS:
            b=am(audit,'B_native',v,k,'absolute_feasibility_auc')
            f=am(audit,'F_ERWF',v,k,'absolute_feasibility_auc')
            g=am(audit,'G_OCCWRF',v,k,'absolute_feasibility_auc')
            h=am(audit,'H_OCQARW',v,k,'absolute_feasibility_auc')
            hb=None if b is None or h is None else float(h)-float(b)
            hg=None if g is None or h is None else float(h)-float(g)
            hf=None if f is None or h is None else float(h)-float(f)
            deltas[v][k]={'H_minus_B_absolute_feasibility_auc':hb,'H_minus_G_absolute_feasibility_auc':hg,'H_minus_F_absolute_feasibility_auc':hf}
            if hb is None or hb<=0:allpos=False
            if hb is not None and hb>=.01:meaningful+=1

            hi=am(audit,'H_OCQARW',v,k,'teacher_infeasible_pass_fraction')
            hh=am(audit,'H_OCQARW',v,k,'harmful_pass_fraction')
            fi=am(audit,'F_ERWF',v,k,'teacher_infeasible_pass_fraction')
            # Quantifier alignment may recover recall relative to G, but must not
            # relapse to the permissive F family.  This is intentionally not a
            # threshold search: one fixed external operating point is audited.
            cap=min(0.25, (float(fi)-0.10) if fi is not None else 0.25)
            ok=hi is not None and hh is not None and float(hi)<=cap and float(hh)<=cap
            selectivity.append({'variant':v,'split':k,'H_teacher_infeasible_pass_fraction':hi,'H_harmful_pass_fraction':hh,'fixed_selectivity_cap':cap,'valid':ok})

            hs=am(audit,'H_OCQARW',v,k,'safe_positive_pass_fraction')
            gs=am(audit,'G_OCCWRF',v,k,'safe_positive_pass_fraction')
            gain=None if hs is None or gs is None else float(hs)-float(gs)
            recall.append({'variant':v,'split':k,'H_safe_positive_pass_fraction':hs,'G_safe_positive_pass_fraction':gs,'H_minus_G_safe_positive_pass_fraction':gain,'valid':gain is not None and gain>0})
    auc_go=bool(allpos and meaningful>=6)
    selective_go=all(z['valid'] for z in selectivity)
    recall_all=all(z['valid'] for z in recall)
    recall_meaningful=sum((z['H_minus_G_safe_positive_pass_fraction'] is not None and z['H_minus_G_safe_positive_pass_fraction']>=.15) for z in recall)>=6
    recall_go=bool(recall_all and recall_meaningful)
    go=auc_go and selective_go and recall_go
    doc={'schema':'ocrap-v48.63-ocqarw-comparison-v1','arms':{n:{v:{k:metric(r,v,k) for k in KINDS} for v in VARIANTS} for n,r in arms.items()},'feasibility_role_audit':audit,'source_deltas':deltas,
      'preregistered_decision':{'status':'GO' if go else 'STOP','all_8_H_minus_B_auc_positive_and_at_least_6_ge_0p01':auc_go,
        'controlled_selectivity_without_F_permissive_relapse':selective_go,'safe_positive_recall_restored_over_G':recall_go,
        'selectivity_checks':selectivity,'safe_positive_recall_checks':recall},
      'attribution_order':['H-B quantifier-aligned absolute-source ordering (primary)','H-G isolates exists-feasibility / forall-infeasibility logic from v48.62 per-option veto','H-F checks whether quantifier alignment preserves ERWF continuation information while retaining controlled selectivity','diagnostic quantifier coverage separates witness-library coverage from logical-composition failure','state/variant isolation and fixed top-5 proposal contract','H-A deployment propagation only after source evidence'],
      'scientific_contract':{
        'primary_hypothesis':'recoverability is existential over observation-consistent executable recovery options; infeasibility may veto only when all supported common options fail',
        'GO_requires':['H-B source AUC improves in all 8 Near/Contact cells and >=0.01 in at least six','teacher-infeasible and harmful pass remain <=0.25 and at least 0.10 below F in every cell','safe-positive pass improves over G in all eight cells and by >=0.15 in at least six','only two shared bounded gains train; Stage-I/root/option/base state stays bitwise frozen','fixed top-5, threshold 0.5, unchanged RIFA/relative ranking, no regime id/router or teacher future','source gain precedes deployment and Safe non-interference evaluation'],
        'STOP_if':['any source AUC cell regresses vs B','quantifier alignment restores recall only by returning to F-like permissive false pass','safe-positive witness recall remains collapsed despite removing per-option veto','state/provenance fails'],
        'forbidden_next_sweeps':['threshold/LR/horizon/grid search','proposal expansion/densification','option-specific free bias','generic AFE MLP/width/class-weight sweep','candidate-only CPHR scalar correction','compensatory ERWF coordinate sum','per-option negative OC-CWRF veto','regime routing/policy/threshold/budget','broad root/margin/encoder retraining','privileged teacher future/component distillation','relative centering/ranking changes before source GO'],
        'if_STOP_next_question':'use quantifier-coverage diagnostics to decide recovery-option taxonomy/coverage versus observation-uncertainty/reachability representation; do not tune OC-QARW scalars',
        'safe_role':'same shared mechanism; paired Safe nominal-utility non-interference only after source GO'},
    }
    x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_63_ocqarw_comparison','decision':doc['preregistered_decision']['status'],'output':str(x.output)}))
if __name__=='__main__':main()
