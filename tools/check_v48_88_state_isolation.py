#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch

EXPECTED_KEYS={'direct_absolute_quotient_tail_response_weight'}
EXPECTED_SHAPE=[2,141]
EXPECTED_NUMEL=282


def load(p: Path):
    try: return torch.load(p,map_location='cpu',weights_only=False)
    except TypeError: return torch.load(p,map_location='cpu')

def sha(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(1<<20),b''): h.update(z)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--adapted',type=Path,required=True)
    ap.add_argument('--truth-index',type=Path,required=True)
    ap.add_argument('--response-index',type=Path,required=True)
    ap.add_argument('--objective',required=True,choices=['counterfactual_response_interval_huber','counterfactual_selective_response'])
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    ra,da=load(a.reference),load(a.adapted)
    rs,ds=ra.get('model_state',ra),da.get('model_state',da)
    shared=sorted(set(rs)&set(ds)); changed=[]; shape=[]
    for k in shared:
        if tuple(rs[k].shape)!=tuple(ds[k].shape): shape.append(k)
        elif not torch.equal(rs[k].cpu(),ds[k].cpu()): changed.append(k)
    removed=sorted(set(rs)-set(ds)); added=sorted(set(ds)-set(rs))
    numel=sum(int(ds[k].numel()) for k in added if k in ds)
    shapes={k:list(ds[k].shape) for k in added if k in ds}
    new_ok=set(added)==EXPECTED_KEYS and shapes.get('direct_absolute_quotient_tail_response_weight')==EXPECTED_SHAPE
    flags={
        'adapter':bool(da.get('direct_recovery_semantic_witness_action_response_adapter',False)),
        'bilinear_interaction':bool(da.get('direct_recovery_semantic_witness_action_root_bilinear_interaction',False)),
        'quotient_tail_response':bool(da.get('direct_recovery_semantic_witness_quotient_tail_response',False)),
        'root_tail_source':bool(da.get('direct_recovery_semantic_witness_root_tail_source',False)),
        'boundary_transport':bool(da.get('direct_recovery_semantic_witness_boundary_transport',False)),
    }
    expected={'adapter':False,'bilinear_interaction':False,'quotient_tail_response':True,'root_tail_source':False,'boundary_transport':False}
    tcfg=((da.get('cfg') or {}).get('training') or {})
    truth=str(tcfg.get('direct_value_absolute_feasibility_truth_contract',''))
    obj=str(tcfg.get('direct_value_absolute_feasibility_supervision_objective',''))
    idx=str(tcfg.get('direct_value_absolute_feasibility_truth_index','') or '')
    ridx=str(tcfg.get('direct_value_action_response_truth_index','') or '')
    idx_ok=bool(idx) and Path(idx).expanduser().resolve(strict=False)==a.truth_index.expanduser().resolve(strict=False)
    ridx_ok=bool(ridx) and Path(ridx).expanduser().resolve(strict=False)==a.response_index.expanduser().resolve(strict=False)
    valid=(not removed and not shape and not changed and new_ok and numel==EXPECTED_NUMEL and flags==expected
           and truth=='structural_interval_bounds' and obj==a.objective and idx_ok and ridx_ok)
    doc={
        'schema':'ocrap-v48.88-qtrr-state-isolation-v1','valid':valid,
        'reference':str(a.reference),'adapted':str(a.adapted),'reference_sha256':sha(a.reference),'adapted_sha256':sha(a.adapted),
        'shared_state_tensors':len(shared),'changed_shared_state_keys':changed,'added_state_keys':added,'removed_state_keys':removed,
        'shape_mismatch_keys':shape,'new_tensor_shapes':shapes,'new_tensor_numel':numel,
        'v48_87_barr_parameter_count':53550,'capacity_ratio_vs_barr':numel/53550.0,
        'factor_flags':flags,'factor_flags_valid':flags==expected,
        'stage_i_bitwise_identity':not changed and not removed and not shape,
        'absolute_feasibility_truth_contract':truth,'absolute_feasibility_supervision_objective':obj,
        'absolute_feasibility_truth_index_valid':idx_ok,'action_response_truth_index_valid':ridx_ok,
        'dataset_reconstruction':False,'test_roots_read':False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':valid,'added':added,'numel':numel})); return 0 if valid else 30

if __name__=='__main__': raise SystemExit(main())
