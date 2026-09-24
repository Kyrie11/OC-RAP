#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from ocrap.simulation.closed_loop_runner import _aggregate_with_buckets

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def main() -> None:
    ap=argparse.ArgumentParser(description='Exact 250-scene reconstruction for a trigger-gated privileged Near ceiling diagnostic.')
    ap.add_argument('--baseline-full',required=True)
    ap.add_argument('--diagnostic-subset',required=True)
    ap.add_argument('--target-keys',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    base=load(a.baseline_full); sub=load(a.diagnostic_subset); keydoc=load(a.target_keys)
    keys=set(str(x) for x in (keydoc.get('target_keys') or []))
    bmap={str(s['target_key']):s for s in (base.get('scenes') or [])}
    smap={str(s['target_key']):s for s in (sub.get('scenes') or [])}
    if len(bmap)!=250: raise SystemExit(f'baseline full population must contain 250 scenes, got {len(bmap)}')
    if not keys or set(smap)!=keys: raise SystemExit(f'diagnostic subset target mismatch expected={len(keys)} got={len(smap)}')
    historical_intervention={k for k,s in bmap.items() if float(s.get('intervention_rate',0.0) or 0.0)>0.0}
    if historical_intervention!=keys:
        raise SystemExit(f'target keys must equal the historical Base intervention cohort expected={len(historical_intervention)} got={len(keys)}')
    unchanged=set(bmap)-keys
    if any(float(bmap[k].get('intervention_rate',0.0) or 0.0)!=0.0 for k in unchanged):
        raise SystemExit('non-cohort Base scene contains intervention; exact reuse refused')
    for k,s in smap.items():
        if not bool(s.get('privileged_pcd_oracle_ceiling')):
            raise SystemExit(f'subset scene missing privileged oracle marker: {k}')
    merged=[smap.get(k,bmap[k]) for k in sorted(bmap)]
    out=_aggregate_with_buckets(merged,method=str(sub.get('method','ocrap')),source=str(sub.get('source','model')))
    for field in ('bucket_dataset','gamma_rec','gamma_rec_by_bucket','selector_config','closed_loop_speed_config','label_modes','source'):
        if field in sub: out[field]=sub[field]
    out['scenes']=merged; out['scenes_embedded']=True; out['scene_storage_detail']='metrics'
    out['privileged_pcd_oracle_ceiling']=True
    out['exact_trigger_gated_subset_reconstruction']={
        'internal_diagnostic_only':True,
        'baseline_full':str(Path(a.baseline_full)),
        'diagnostic_subset':str(Path(a.diagnostic_subset)),
        'historical_intervention_scene_count':len(keys),
        'reused_zero_intervention_scene_count':len(unchanged),
        'proof_contract':(
            'At every exact current state the privileged ceiling first executes the frozen Base selector. '
            'Teacher labels are constructed only if Base already selects a non-nominal action. Therefore a '
            'Base-zero-intervention scene cannot acquire a first intervention: both policies execute nominal at '
            'the initial state, remain in the same next state, and repeat by induction. Only the historical Base '
            'intervention-scene cohort requires fresh replay.'
        ),
        'publication_evidence':False,
        'purpose':'causal sufficiency ceiling for frozen candidate library + absolute admission + Base trigger, not a deployable method',
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=True)+'\n',encoding='utf-8')
    print(json.dumps({'valid':True,'num_scenes':len(merged),'rerun':len(keys),'reused':len(unchanged),'output':a.output},indent=2))
if __name__=='__main__': main()
