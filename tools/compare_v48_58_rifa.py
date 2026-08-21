#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())
def metric(run:Path,variant:str,kind:str):
    base=run/'candidates'/variant/'calibration'
    f={'dev_near':'dev_diagnostic_near_v48.json','dev_contact':'dev_diagnostic_contact_v48.json',
       'certificate_near':'direct_value_risk_near_v48.json','certificate_contact':'direct_value_risk_contact_v48.json'}[kind]
    p=base/f
    if not p.is_file(): return {'missing':str(p)}
    d=load(p)
    keys=['positive_recall','precision','precision_lcb90','harmful_group_ucb90','harmful_selected_ucb90',
          'num_selected','proposal_safe_positive_auc','proposal_harm_auc','candidate_safe_positive_auc','candidate_harm_auc',
          'proposal_exact_eligible_safe_positive_recall','proposal_exact_eligible_precision','proposal_exact_eligible_harmful_group_rate']
    return {k:d.get(k) for k in keys if k in d} | {'exit_code':d.get('exit_code'),'gate_passed':d.get('gate_passed')}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--a',type=Path,required=True);ap.add_argument('--b',type=Path,required=True);ap.add_argument('--c',type=Path,required=True);ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    x=ap.parse_args(); arms={'A':x.a,'B':x.b,'C_Main':x.c}; d={'schema':'ocrap-v48.58-rifa-comparison-v1','arms':{},'attribution_order':['B-A: structural two-stage placement using raw native absolute feasibility','C-B: isolated absolute-feasibility source correction with Stage-I frozen','C-A: full RIFA effect']}
    for name,r in arms.items():
      d['arms'][name]={v:{k:metric(r,v,k) for k in ['dev_near','dev_contact','certificate_near','certificate_contact']} for v in ['balanced','precision']}
    d['feasibility_role_audit']=load(x.feasibility_audit)
    d['decision_contract']={
      'retain_CMRI':False,
      'retain_RIFA_only_if':['B-A demonstrates structural safety benefit without catastrophic safe-positive loss OR C recovers that loss','C-B improves absolute feasibility source geometry on Near and Contact','Stage-I state isolation is bitwise valid','certificate/dev deployment moves consistently with source geometry'],
      'authorize_centering_next_only_if':['RIFA source/native geometry forms Near+Contact Pareto improvement','remaining dominant error is final relative score sign/centering rather than feasibility source'],
      'stop_if':['learned AFE does not beat raw native source on boundary discrimination','safe-positive false veto remains dominant','improvements are regime-specific with cross-severity regression','state isolation fails']}
    x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n'); print(json.dumps({'event':'v48_58_rifa_comparison','output':str(x.output)}))
if __name__=='__main__': main()
