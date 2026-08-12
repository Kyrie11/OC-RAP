#!/usr/bin/env python3
"""Fail closed if the immutable v48.45 rebuilt source has unexpected geometry."""
from __future__ import annotations
import argparse,json,pathlib,time
import torch

EXPECTED={
 'direct_recovery_set_context':False,
 'direct_recovery_preference_head':False,
 'direct_recovery_preference_context':False,
 'direct_recovery_relative_features_include_absolute':False,
 'direct_recovery_set_tournament':True,
 'direct_recovery_set_tournament_hidden':48,
 'direct_recovery_set_tournament_heads':4,
 'direct_recovery_set_tournament_dropout':0.05,
 'direct_recovery_set_tournament_replace_base':True,
 'direct_recovery_delta_head':True,
 'direct_recovery_delta_regime_experts':True,
 'direct_recovery_delta_policy_features':True,
 'direct_recovery_delta_hidden':48,
 'direct_recovery_delta_dropout':0.02,
 'direct_recovery_delta_mode':'ordinal_evidence',
 'direct_recovery_evidence_calibrator':False,
 'direct_recovery_evidence_roct_benefit':False,
 'direct_recovery_evidence_roct_deployability':False,
 'direct_recovery_evidence_component_heads':False,
}
REQUIRED_PREFIXES=('encoder.','root_queries','root_cross_attn.','root_self_attn.','root_ffn.','root_logit_head.','obs_embed_head.','margin_head.')

def eq(a,b):
    if isinstance(b,float):
        try:return abs(float(a)-b)<=1e-9
        except:return False
    return a==b

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
    d=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state=d.get('model_state',{}) if isinstance(d,dict) else {}
    field_checks={k:{'expected':v,'actual':d.get(k),'match':eq(d.get(k),v)} for k,v in EXPECTED.items()}
    prefix_checks={p:any(str(k).startswith(p) for k in state) for p in REQUIRED_PREFIXES}
    checks={'metadata_exact':all(x['match'] for x in field_checks.values()),'required_witness_state_present':all(prefix_checks.values())}
    out={'event':'v48_45_sowr_source_architecture_contract','version':'v48.45.6','created_unix':time.time(),'valid':all(checks.values()),'checks':checks,'metadata':field_checks,'required_prefixes':prefix_checks,'test_roots_read':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));raise SystemExit(0 if out['valid'] else 4)
if __name__=='__main__':main()
