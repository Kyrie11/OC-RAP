#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, time
import torch


ARCH_METADATA = [
    'input_dim','num_roots','num_options','d_model','d_obs','encoder_type',
    'd_signature','d_future_signature','option_feature_dim',
    'direct_recovery_set_context','direct_recovery_set_tournament',
    'direct_recovery_set_tournament_hidden','direct_recovery_set_tournament_heads',
    'direct_recovery_set_tournament_dropout','direct_recovery_set_tournament_replace_base',
    'direct_recovery_preference_head','direct_recovery_preference_context',
    'direct_recovery_relative_features_include_absolute',
    'direct_recovery_delta_head','direct_recovery_delta_regime_experts',
    'direct_recovery_delta_policy_features','direct_recovery_delta_hidden',
    'direct_recovery_delta_dropout','direct_recovery_delta_mode',
]

DOWNSTREAM_FALSE = [
    'direct_recovery_evidence_calibrator',
    'direct_recovery_evidence_dual_interaction_bridge',
    'direct_recovery_evidence_factorized_harm_interaction',
    'direct_recovery_evidence_partial_pool_harm_residual',
    'direct_recovery_evidence_rank_benefit_skip',
    'direct_recovery_evidence_postprefix_obs_transport_benefit',
    'direct_recovery_evidence_postprefix_obs_transport_harm',
    'direct_recovery_evidence_roct_benefit',
    'direct_recovery_evidence_roct_deployability',
    'direct_recovery_evidence_unified_experts',
    'direct_recovery_evidence_component_heads',
    'direct_recovery_evidence_concord',
    'direct_recovery_evidence_admission_head',
    'direct_recovery_evidence_frontier',
    'direct_recovery_evidence_reserve_factor_alignment',
]

def load(p: pathlib.Path):
    return torch.load(p, map_location='cpu', weights_only=False)

def state(d):
    if isinstance(d, dict) and isinstance(d.get('model_state'), dict): return d['model_state']
    if isinstance(d, dict): return d
    raise TypeError('checkpoint is not a dict')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source', type=pathlib.Path, required=True)
    ap.add_argument('--checkpoint', type=pathlib.Path, required=True)
    ap.add_argument('--allowed-prefixes', required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    a=ap.parse_args()
    prefixes=tuple(x.strip() for x in a.allowed_prefixes.split(',') if x.strip())
    src=load(a.source); dst=load(a.checkpoint)
    ss, ds=state(src), state(dst)
    common=set(ss)&set(ds)
    missing=sorted(set(ss)-set(ds)); unexpected=sorted(set(ds)-set(ss))
    changed=[]; changed_disallowed=[]
    for k in sorted(common):
        x,y=ss[k],ds[k]
        same = torch.equal(x,y) if torch.is_tensor(x) and torch.is_tensor(y) else x==y
        if not same:
            changed.append(k)
            if not any(k.startswith(p) for p in prefixes): changed_disallowed.append(k)
    flags={k: bool(dst.get(k, False)) for k in DOWNSTREAM_FALSE}
    trainable=tuple(str(x) for x in (dst.get('trainable_param_prefixes') or []))
    key_drift_disallowed = [k for k in missing + unexpected if not any(k.startswith(p) for p in prefixes)]
    arch_metadata={k:{'source':src.get(k),'checkpoint':dst.get(k),'equal':src.get(k)==dst.get(k)} for k in ARCH_METADATA}
    checks={
        # Epoch 0 is deliberately evaluated by SOWR; selecting the unchanged source
        # checkpoint is a valid negative algorithm result, not an engineering error.
        'state_key_drift_only_within_allowed_prefixes': not key_drift_disallowed,
        'only_allowed_state_changed': not changed_disallowed,
        'source_architecture_metadata_preserved': all(x['equal'] for x in arch_metadata.values()),
        'checkpoint_trainable_prefixes_exact': trainable==prefixes,
        'downstream_evidence_disabled': not any(flags.values()),
    }
    doc={
      'event':'v48_45_sowr_stage_isolation_contract','version':'v48.45.6',
      'created_unix':time.time(),'valid':all(checks.values()),'checks':checks,
      'allowed_prefixes':list(prefixes),'changed_key_count':len(changed),
      'changed_keys':changed,'changed_disallowed':changed_disallowed,
      'missing_state_keys':missing,'unexpected_state_keys':unexpected,
      'key_drift_disallowed':key_drift_disallowed,
      'architecture_metadata':arch_metadata,
      'downstream_flags':flags,'checkpoint_trainable_prefixes':list(trainable),
      'test_roots_read':False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(doc,indent=2)+'\n')
    print(json.dumps(doc,indent=2))
    raise SystemExit(0 if doc['valid'] else 4)
if __name__=='__main__': main()
