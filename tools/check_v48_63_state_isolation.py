#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch

ALLOWED_KEY = 'direct_absolute_quantifier_witness_gain'
EXPECTED_NUM_WEIGHTS = 2
EXPECTED_FEATURE_SCHEMA = 1
EXPECTED_FEATURE_SOURCE = 'quantifier_aligned_common_finite_time_recovery_witness'

def load(path: Path):
    try: return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError: return torch.load(path, map_location='cpu')

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser(description='v48.63 fail-closed OC-QARW frozen-Stage-I isolation audit')
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--adapted', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a=ap.parse_args(); ra=load(a.reference); da=load(a.adapted)
    rs=ra.get('model_state',ra); ds=da.get('model_state',da)
    if not isinstance(rs,dict) or not isinstance(ds,dict): raise SystemExit('checkpoint model_state missing')
    shared=sorted(set(rs)&set(ds)); changed=[]; shape=[]
    for k in shared:
        if tuple(rs[k].shape)!=tuple(ds[k].shape): shape.append(k); continue
        if not torch.equal(rs[k].cpu(),ds[k].cpu()): changed.append(k)
    removed=sorted(set(rs)-set(ds)); added=sorted(set(ds)-set(rs))
    new_tensor=ds.get(ALLOWED_KEY)
    new_shape=(tuple(new_tensor.shape) if isinstance(new_tensor,torch.Tensor) else None)
    new_numel=(int(new_tensor.numel()) if isinstance(new_tensor,torch.Tensor) else None)
    expected=(added==[ALLOWED_KEY] and new_shape==(EXPECTED_NUM_WEIGHTS,) and new_numel==EXPECTED_NUM_WEIGHTS)
    feature_schema=int(da.get('direct_recovery_absolute_quantifier_witness_feature_schema',0) or 0)
    feature_source=str(da.get('direct_recovery_absolute_quantifier_witness_feature_source','') or '')
    feature_contract=(feature_schema==EXPECTED_FEATURE_SCHEMA and feature_source==EXPECTED_FEATURE_SOURCE)
    valid=(not removed and not shape and not changed and expected and feature_contract)
    doc={
      'schema':'ocrap-v48.63-ocqarw-state-isolation-v1','valid':valid,
      'reference':str(a.reference.resolve(strict=False)),'adapted':str(a.adapted.resolve(strict=False)),
      'reference_sha256':sha(a.reference),'adapted_sha256':sha(a.adapted),
      'allowed_trainable_key':ALLOWED_KEY,'expected_num_weights':EXPECTED_NUM_WEIGHTS,
      'shared_state_tensors':len(shared),'changed_shared_state_keys':changed,
      'added_state_keys':added,'removed_state_keys':removed,'shape_mismatch_keys':shape,
      'new_tensor_shape':new_shape,'new_tensor_numel':new_numel,
      'only_quantifier_witness_gain_added':expected,
      'quantifier_witness_feature_schema':feature_schema,
      'quantifier_witness_feature_source':feature_source,
      'quantifier_witness_feature_contract_valid':feature_contract,
      'stage_i_bitwise_identity':not changed and not removed and not shape,
      'test_roots_read':False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_63_ocqarw_state_isolation','valid':valid,'output':str(a.output)}))
    return 0 if valid else 30
if __name__=='__main__': raise SystemExit(main())
