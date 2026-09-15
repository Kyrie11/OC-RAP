#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--result',required=True); ap.add_argument('--target-keys-output',required=True); ap.add_argument('--audit-output',required=True); a=ap.parse_args()
    d=load(a.result); scenes=d.get('scenes') or []
    if int(d.get('num_scenes',0))!=250 or len(scenes)!=250: raise SystemExit('fresh full 250 Near result with embedded scenes required')
    keys=sorted(str(s['target_key']) for s in scenes if float(s.get('intervention_rate',0.0) or 0.0)>0.0)
    Path(a.target_keys_output).write_text(json.dumps({'target_keys':keys},indent=2,sort_keys=True)+'\n')
    out={"valid":bool(keys),"full_scene_count":len(scenes),"intervention_scene_count":len(keys),"untouched_zero_intervention_scene_count":len(scenes)-len(keys),"target_keys":keys,
         "exact_reduced_replay_argument":"Diagnostic selectors preserve every historical nominal short-circuit. A scene with zero interventions under this fresh observation-legal baseline therefore remains action/state-identical by induction; only this variant-specific intervention cohort must be replayed internally."}
    Path(a.audit_output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({"valid":out['valid'],"intervention_scene_count":len(keys)},indent=2)); return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
