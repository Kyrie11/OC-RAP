#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path


def load(p:Path): return json.loads(p.read_text())

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--runtime',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--comparison',type=Path,required=True)
    ap.add_argument('--train-truth-index',type=Path,required=True)
    ap.add_argument('--eval-truth-index',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); errors=[]
    docs={}
    for name,p in [('runtime',a.runtime),('reference',a.reference),('audit',a.audit),('comparison',a.comparison)]:
        if not p.is_file(): errors.append(f'{name} missing: {p}'); continue
        try: docs[name]=load(p)
        except Exception as e: errors.append(f'{name} parse error: {e}')
    for name,d in docs.items():
        if not d.get('valid'): errors.append(f'{name} invalid')
        if d.get('errors'): errors.append(f'{name} errors: {d.get("errors")}')
    if not a.train_truth_index.is_file(): errors.append('train truth index missing')
    if not a.eval_truth_index.is_file(): errors.append('eval truth index missing')
    for v in ('balanced','precision'):
        c=a.run/'candidates'/v; best=c/'model_v48_trac_sr'/'best.pt'; iso=c/'V48_83_STAGE_I_STATE_ISOLATION.json'
        if not best.is_file(): errors.append(f'{v} best.pt missing')
        if not iso.is_file(): errors.append(f'{v} state isolation missing')
        else:
            try:
                d=load(iso)
                if not d.get('valid'): errors.append(f'{v} state isolation invalid')
            except Exception as e: errors.append(f'{v} state isolation parse error: {e}')
        # Calibration must have all four registered proposal-row artifacts.
        cal=c/'calibration'
        for fn in ('dev_diagnostic_near_v48.proposal_rows.jsonl','dev_diagnostic_contact_v48.proposal_rows.jsonl','direct_value_risk_near_v48.proposal_rows.jsonl','direct_value_risk_contact_v48.proposal_rows.jsonl'):
            if not (cal/fn).is_file(): errors.append(f'{v} calibration missing {fn}')
    comp=docs.get('comparison') or {}
    if not comp.get('attribution_ready'): errors.append('comparison attribution_ready false')
    run_rel=str(a.run)
    doc={'schema':'ocrap-v48.83-crtf-pipeline-complete-v1','algorithm_version':'v48.83-DCP-DRFC-BCDE-RIFA-OC-CRTF','engineering_version':'v48.83.0-OC-CRTF','valid':not errors,'attribution_ready':not errors,'errors':errors,'arms':{'P83_COUNTERFACTUAL_TAIL_FIELD':run_rel},'truth_contract':'V48.80 structural_interval_bounds frozen/reused','train_truth_index':str(a.train_truth_index),'eval_truth_index':str(a.eval_truth_index),'boundary_transport':False,'dataset_reconstruction':False,'test_roots_read':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n'); print(json.dumps({'valid':not errors,'errors':errors,'output':str(a.output)})); return 0 if not errors else 30

if __name__=='__main__': raise SystemExit(main())
