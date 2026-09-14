#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from ocrap.simulation.closed_loop_runner import _aggregate_with_buckets


def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser(description='Exact internal reconstruction for a monotone-subset Near selector diagnostic.')
    ap.add_argument('--baseline-full',required=True)
    ap.add_argument('--diagnostic-subset',required=True)
    ap.add_argument('--target-keys',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    base=load(args.baseline_full); sub=load(args.diagnostic_subset); keys_doc=load(args.target_keys)
    keys=set(keys_doc.get('target_keys') or [])
    bmap={str(s['target_key']):s for s in base.get('scenes') or []}
    smap={str(s['target_key']):s for s in sub.get('scenes') or []}
    if not keys or set(smap)!=keys:
        raise SystemExit(f'diagnostic subset target mismatch expected={len(keys)} got={len(smap)}')
    if not keys.issubset(bmap): raise SystemExit('subset contains target absent from baseline full population')
    historical_interventions={k for k,s in bmap.items() if float(s.get('intervention_rate',0.0) or 0.0)>0.0}
    if historical_interventions != keys:
        raise SystemExit('target keys are not exactly the historical intervention cohort; exact monotone reconstruction refused')
    unchanged=set(bmap)-keys
    if any(float(bmap[k].get('intervention_rate',0.0) or 0.0)!=0.0 for k in unchanged):
        raise SystemExit('non-cohort baseline scene contains intervention')
    merged=[smap.get(k,bmap[k]) for k in sorted(bmap)]
    out=_aggregate_with_buckets(merged, method=str(sub.get('method','ocrap')), source=str(sub.get('source','model')))
    # Retain evaluation metadata from the diagnostic subset where meaningful.
    for field in ('bucket_dataset','gamma_rec','gamma_rec_by_bucket','selector_config','closed_loop_speed_config','label_modes','source'):
        if field in sub: out[field]=sub[field]
    out['scenes']=merged
    out['scenes_embedded']=True
    out['scene_storage_detail']='metrics'
    out['exact_monotone_subset_reconstruction']={
        'internal_diagnostic_only':True,
        'baseline_full':str(Path(args.baseline_full)),
        'diagnostic_subset':str(Path(args.diagnostic_subset)),
        'historical_intervention_scene_count':len(keys),
        'reused_zero_intervention_scene_count':len(unchanged),
        'proof_contract':'new selector is a downstream restriction/re-ranker of historical absolute-admitted non-nominal candidates and preserves all historical nominal short-circuits; it cannot create a first intervention on a baseline-zero-intervention trajectory',
        'final_publication_requirement':'After selector promotion, rerun the complete 250-scene Near population fresh; do not use this reconstructed artifact as the paper main table.'
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=True)+'\n',encoding='utf-8')
    print(json.dumps({'valid':True,'num_scenes':len(merged),'reused':len(unchanged),'rerun':len(keys),'output':args.output},indent=2))

if __name__=='__main__': main()
