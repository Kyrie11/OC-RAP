#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any

def load(p:Path)->dict:return json.loads(p.read_text(encoding='utf-8'))
def dep_block(d):
    phase='fit' if bool(d.get('development_fit_only')) else 'verify'; return phase,(d.get(phase) or {})
def metric(run:Path,v:str,kind:str)->dict[str,Any]:
    fn={'dev_near':'dev_diagnostic_near_v48.json','dev_contact':'dev_diagnostic_contact_v48.json',
        'certificate_near':'direct_value_risk_near_v48.json','certificate_contact':'direct_value_risk_contact_v48.json'}[kind]
    p=run/'candidates'/v/'calibration'/fn
    if not p.is_file():return {'missing':str(p)}
    d=load(p); phase,dep=dep_block(d)
    dk=('num_groups','num_selected','selection_rate','num_positive_selected','positive_recall','precision','precision_wilson_lcb90',
        'num_harmful_selected','harmful_selected_rate','harmful_selected_ucb90','harmful_group_exposure','harmful_group_exposure_ucb90','num_opportunities')
    qk=('candidate_safe_positive_auc','candidate_harm_auc','candidate_pred_teacher_correlation','candidate_rank_teacher_correlation',
        'proposal_evidence_top1_correlation','proposal_evidence_top1_safe_positive_auc','proposal_evidence_top1_harm_auc',
        'proposal_deployed_rule_abstention_rate','proposal_deployed_rule_top1_safe_positive_auc','proposal_top_k',
        'proposal_positive_group_count','proposal_oracle_best_hit_rate_positive_groups','proposal_any_positive_hit_rate_positive_groups')
    return {'path':str(p),'phase':phase,'valid_for_deployment':d.get('valid_for_deployment'),'rejection_kind':d.get('rejection_kind'),
            'absolute_feasibility_mode':d.get('absolute_feasibility_mode'),'absolute_feasibility_threshold':d.get('absolute_feasibility_threshold'),
            'deployment':{k:dep.get(k) for k in dk if k in dep},'ranking_and_selector_diagnostics':{k:d.get(k) for k in qk if k in d},
            'proposal_constrained_oracle_gate':d.get('proposal_constrained_oracle_gate'),'proposal_support_curve':d.get('proposal_support_curve')}

def main():
    ap=argparse.ArgumentParser();
    for n in ('a','b','c','d'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);x=ap.parse_args()
    arms={'A':x.a,'B_native':x.b,'C_AFE':x.c,'D_Main_ORFC':x.d}; doc={'schema':'ocrap-v48.59-orfc-comparison-v1','arms':{},
      'attribution_order':['reuse V48.58 B-A: structural placement/raw-source control','reuse V48.58 C-B: compressed AFE control',
                           'D-B: option-resolved absolute-source correction (primary)','D-C: does option-resolved correction beat compressed AFE?',
                           'state isolation','D-A deployment propagation'],
      'scientific_contract':{
        'primary_hypothesis':'compressed AFE loses option-resolved feasibility information; option-wise pre-OC-MERO margin correction can improve source discrimination without regime conditioning',
        'GO_requires':['Near AND Contact source discrimination improves, not just accuracy at one operating point',
                       'teacher-feasible/safe-positive rejection decreases relative to native B while teacher-infeasible/harmful pass is materially below AFE C',
                       'Stage-I is bitwise frozen and only the 24 option biases change',
                       'source gain propagates to development/certificate deployment without cross-severity risk regression'],
        'STOP_if':['AUC/order does not improve over B/C and only pass rate moves','cross-severity source tradeoff','deployment does not respond to source gain',
                   'state isolation/provenance fails'],
        'centering_authorized_only_after_GO':True,
        'if_GO_residual_question':'audit joint relative opportunity/harm/pred_adv/evidence-margin failure; do not assume pred_adv-only bias is sufficient',
        'forbidden_next_sweeps':['AFE feature-stack/width/class-weight/threshold','proposal expansion','regime router/threshold/budget','broad root/margin-head retraining']}}
    for name,r in arms.items():doc['arms'][name]={v:{k:metric(r,v,k) for k in ('dev_near','dev_contact','certificate_near','certificate_contact')} for v in ('balanced','precision')}
    doc['feasibility_role_audit']=load(x.feasibility_audit); x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_59_orfc_comparison','output':str(x.output)}))
if __name__=='__main__':main()
