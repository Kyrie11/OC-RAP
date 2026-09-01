#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch

ALLOWED_KEY = 'direct_absolute_semantic_witness_gain'

def load(p: Path):
    try:
        return torch.load(p, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(p, map_location='cpu')

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(1<<20), b''): h.update(z)
    return h.hexdigest()

def b(x): return str(x).strip().lower() in {'1','true','yes','on'}

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--adapted', type=Path, required=True)
    ap.add_argument('--fidelity', required=True)
    ap.add_argument('--output', type=Path, required=True)
    a=ap.parse_args(); fidelity=b(a.fidelity)
    ra, da = load(a.reference), load(a.adapted)
    rs, ds = ra.get('model_state',ra), da.get('model_state',da)
    shared=sorted(set(rs)&set(ds)); changed=[]; shape=[]
    for k in shared:
        if tuple(rs[k].shape)!=tuple(ds[k].shape): shape.append(k)
        elif not torch.equal(rs[k].cpu(), ds[k].cpu()): changed.append(k)
    removed=sorted(set(rs)-set(ds)); added=sorted(set(ds)-set(rs)); new=ds.get(ALLOWED_KEY)
    new_ok=added==[ALLOWED_KEY] and isinstance(new,torch.Tensor) and tuple(new.shape)==(2,)
    schema=int(da.get('direct_recovery_absolute_semantic_witness_feature_schema',0) or 0)
    source=str(da.get('direct_recovery_absolute_semantic_witness_feature_source','') or '')
    expected_schema=4 if fidelity else 3
    expected_source='robust_trust_projected_recovery_witness' if fidelity else 'projected_boundary_common_executable_recovery_witness'
    feature_ok=(schema==expected_schema and source==expected_source)
    flags={
      'semantic_correction':bool(da.get('direct_recovery_absolute_semantic_witness_correction',False)),
      'active_set_alignment':bool(da.get('direct_recovery_semantic_witness_active_set_alignment',False)),
      'path_stop_alignment':bool(da.get('direct_recovery_semantic_witness_path_stop_alignment',False)),
      'classlocal_transport':bool(da.get('direct_recovery_semantic_witness_classlocal_transport',False)),
      'route_alignment':bool(da.get('direct_recovery_semantic_witness_route_alignment',False)),
      'reentry_alignment':bool(da.get('direct_recovery_semantic_witness_reentry_alignment',False)),
      'control_projection':bool(da.get('direct_recovery_semantic_witness_control_projection',False)),
      'boundary_transport':bool(da.get('direct_recovery_semantic_witness_boundary_transport',False)),
      'projection_fidelity':bool(da.get('direct_recovery_semantic_witness_projection_fidelity_weighting',False)),
      'demand_normalized_fidelity':bool(da.get('direct_recovery_semantic_witness_demand_normalized_fidelity',False)),
      'robust_occupancy':bool(da.get('direct_recovery_semantic_witness_robust_occupancy',False)),
      'soft_occupancy_disagreement':bool(da.get('direct_recovery_semantic_witness_soft_occupancy_disagreement',False)),
      'boundary_localized_occupancy_trust':bool(da.get('direct_recovery_semantic_witness_boundary_localized_occupancy_trust',False)),
      'history_occupancy_reachability':bool(da.get('direct_recovery_semantic_witness_history_occupancy_reachability',False)),
      'interaction_box_support':bool(da.get('direct_recovery_semantic_witness_interaction_box_support',False)),
      'interaction_hull_support':bool(da.get('direct_recovery_semantic_witness_interaction_hull_support',False)),
      'interaction_anchor_support':bool(da.get('direct_recovery_semantic_witness_interaction_anchor_support',False)),
      'interaction_response_support':bool(da.get('direct_recovery_semantic_witness_interaction_response_support',False)),
    }
    expected={
      'semantic_correction':True,'active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,
      'route_alignment':True,'reentry_alignment':True,'control_projection':True,'boundary_transport':False,
      'projection_fidelity':fidelity,'demand_normalized_fidelity':False,'robust_occupancy':False,
      'soft_occupancy_disagreement':False,'boundary_localized_occupancy_trust':False,
      'history_occupancy_reachability':False,'interaction_box_support':False,'interaction_hull_support':False,
      'interaction_anchor_support':False,'interaction_response_support':False,
    }
    flags_ok=flags==expected
    cfg=da.get('cfg') or {}; tcfg=cfg.get('training') or {}
    truth=str(tcfg.get('direct_value_absolute_feasibility_truth_contract','legacy_full'))
    truth_ok=truth=='censor_exact_0p5'
    valid=not removed and not shape and not changed and new_ok and feature_ok and flags_ok and truth_ok
    raw=[float(x) for x in new.reshape(-1)] if isinstance(new,torch.Tensor) else None
    doc={
      'schema':'ocrap-v48.75-stca-state-isolation-v1','valid':bool(valid),
      'reference':str(a.reference),'adapted':str(a.adapted),'reference_sha256':sha(a.reference),'adapted_sha256':sha(a.adapted),
      'shared_state_tensors':len(shared),'changed_shared_state_keys':changed,'added_state_keys':added,'removed_state_keys':removed,'shape_mismatch_keys':shape,
      'only_semantic_witness_gain_added':bool(new_ok),'new_tensor_numel':int(new.numel()) if isinstance(new,torch.Tensor) else None,
      'raw_semantic_witness_gain':raw,'effective_clamped_semantic_witness_gain':[min(2,max(0,x)) for x in raw] if raw else None,
      'semantic_witness_feature_schema':schema,'semantic_witness_feature_source':source,'semantic_witness_feature_contract_valid':feature_ok,
      'absolute_feasibility_truth_contract':truth,'truth_contract_valid':truth_ok,
      'factor_flags':flags,'factor_flags_valid':flags_ok,'stage_i_bitwise_identity':not changed and not removed and not shape,
      'projection_fidelity':fidelity,'dataset_reconstruction':False,'test_roots_read':False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_75_stca_state_isolation','valid':valid,'output':str(a.output)})); return 0 if valid else 30
if __name__=='__main__': raise SystemExit(main())
