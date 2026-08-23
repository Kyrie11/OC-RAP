#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch

ALLOWED_KEY='direct_absolute_semantic_witness_gain'
EXPECTED_FEATURE_SCHEMA=1
EXPECTED_FEATURE_SOURCE='semantics_aligned_common_executable_recovery_witness'

def load(p:Path):
    try:return torch.load(p,map_location='cpu',weights_only=False)
    except TypeError:return torch.load(p,map_location='cpu')
def sha(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def b(x:str)->bool:return str(x).strip().lower() in {'1','true','yes','on'}

def main()->int:
    ap=argparse.ArgumentParser(description='v48.65 fail-closed OC-CLRW frozen-Stage-I isolation audit')
    ap.add_argument('--reference',type=Path,required=True); ap.add_argument('--adapted',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--active-set',default='true'); ap.add_argument('--classlocal',default='true'); ap.add_argument('--path-stop',default='false')
    a=ap.parse_args(); ra=load(a.reference); da=load(a.adapted); rs=ra.get('model_state',ra); ds=da.get('model_state',da)
    if not isinstance(rs,dict) or not isinstance(ds,dict): raise SystemExit('checkpoint model_state missing')
    shared=sorted(set(rs)&set(ds)); changed=[]; shape=[]
    for k in shared:
        if tuple(rs[k].shape)!=tuple(ds[k].shape): shape.append(k); continue
        if not torch.equal(rs[k].cpu(),ds[k].cpu()): changed.append(k)
    removed=sorted(set(rs)-set(ds)); added=sorted(set(ds)-set(rs)); new=ds.get(ALLOWED_KEY)
    expected=(added==[ALLOWED_KEY] and isinstance(new,torch.Tensor) and tuple(new.shape)==(2,) and int(new.numel())==2)
    schema=int(da.get('direct_recovery_absolute_semantic_witness_feature_schema',0) or 0)
    source=str(da.get('direct_recovery_absolute_semantic_witness_feature_source','') or '')
    feature_ok=(schema==EXPECTED_FEATURE_SCHEMA and source==EXPECTED_FEATURE_SOURCE)
    flags={
      'semantic_correction':bool(da.get('direct_recovery_absolute_semantic_witness_correction',False)),
      'active_set_alignment':bool(da.get('direct_recovery_semantic_witness_active_set_alignment',False)),
      'path_stop_alignment':bool(da.get('direct_recovery_semantic_witness_path_stop_alignment',False)),
      'classlocal_transport':bool(da.get('direct_recovery_semantic_witness_classlocal_transport',False)),
    }
    flags_ok=(flags['semantic_correction'] and flags['active_set_alignment']==b(a.active_set) and flags['path_stop_alignment']==b(a.path_stop) and flags['classlocal_transport']==b(a.classlocal))
    valid=bool(not removed and not shape and not changed and expected and feature_ok and flags_ok)
    doc={'schema':'ocrap-v48.65-clrw-state-isolation-v1','valid':valid,
         'reference':str(a.reference.resolve(strict=False)),'adapted':str(a.adapted.resolve(strict=False)),
         'reference_sha256':sha(a.reference),'adapted_sha256':sha(a.adapted),'allowed_trainable_key':ALLOWED_KEY,'expected_num_weights':2,
         'shared_state_tensors':len(shared),'changed_shared_state_keys':changed,'added_state_keys':added,'removed_state_keys':removed,'shape_mismatch_keys':shape,
         'new_tensor_shape':tuple(new.shape) if isinstance(new,torch.Tensor) else None,'new_tensor_numel':int(new.numel()) if isinstance(new,torch.Tensor) else None,
         'only_semantic_witness_gain_added':expected,'semantic_witness_feature_schema':schema,'semantic_witness_feature_source':source,
         'semantic_witness_feature_contract_valid':feature_ok,'factor_flags':flags,'factor_flags_valid':flags_ok,
         'stage_i_bitwise_identity':not changed and not removed and not shape,'test_roots_read':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_65_clrw_state_isolation','valid':valid,'output':str(a.output)})); return 0 if valid else 30
if __name__=='__main__': raise SystemExit(main())
