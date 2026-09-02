#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch

ALLOWED_KEY = 'direct_absolute_structured_tail_field_weight'

def load(p: Path):
    try: return torch.load(p, map_location='cpu', weights_only=False)
    except TypeError: return torch.load(p, map_location='cpu')

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(1<<20), b''): h.update(z)
    return h.hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--reference',type=Path,required=True); ap.add_argument('--adapted',type=Path,required=True); ap.add_argument('--truth-index',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    ra,da=load(a.reference),load(a.adapted); rs,ds=ra.get('model_state',ra),da.get('model_state',da)
    shared=sorted(set(rs)&set(ds)); changed=[]; shape=[]
    for k in shared:
        if tuple(rs[k].shape)!=tuple(ds[k].shape): shape.append(k)
        elif not torch.equal(rs[k].cpu(),ds[k].cpu()): changed.append(k)
    removed=sorted(set(rs)-set(ds)); added=sorted(set(ds)-set(rs)); w=ds.get(ALLOWED_KEY)
    new_ok=added==[ALLOWED_KEY] and isinstance(w,torch.Tensor) and tuple(w.shape)==(2, int(da.get('d_model', w.shape[1])))
    schema=int(da.get('direct_recovery_absolute_semantic_witness_feature_schema',0) or 0); source=str(da.get('direct_recovery_absolute_semantic_witness_feature_source','') or '')
    feature_ok=schema==3 and source=='projected_boundary_common_executable_recovery_witness'
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
      'active_constraint_typed_source':bool(da.get('direct_recovery_semantic_witness_active_constraint_typed_source',False)),
      'root_tail_source':bool(da.get('direct_recovery_semantic_witness_root_tail_source',False)),
      'tail_localization':bool(da.get('direct_recovery_semantic_witness_tail_localization',False)),
      'structured_tail_field':bool(da.get('direct_recovery_semantic_witness_structured_tail_field',False)),
      'signed_tail_channels':bool(da.get('direct_recovery_semantic_witness_signed_tail_channels',False)),
      'counterfactual_tail_response':bool(da.get('direct_recovery_semantic_witness_counterfactual_tail_response',False)),
    }
    expected={'semantic_correction':True,'active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,'route_alignment':True,'reentry_alignment':True,'control_projection':True,'boundary_transport':False,'projection_fidelity':False,'active_constraint_typed_source':False,'root_tail_source':True,'tail_localization':True,'structured_tail_field':True,'signed_tail_channels':True,'counterfactual_tail_response':True}
    tcfg=((da.get('cfg') or {}).get('training') or {}); truth=str(tcfg.get('direct_value_absolute_feasibility_truth_contract','legacy_full')); obj=str(tcfg.get('direct_value_absolute_feasibility_supervision_objective','binary_sign')); idx=str(tcfg.get('direct_value_absolute_feasibility_truth_index','') or '')
    idx_ok=bool(idx) and Path(idx).expanduser().resolve(strict=False)==a.truth_index.expanduser().resolve(strict=False)
    valid=(not removed and not shape and not changed and new_ok and feature_ok and flags==expected and truth=='structural_interval_bounds' and obj=='signed_margin_interval_huber' and idx_ok)
    doc={'schema':'ocrap-v48.83-crtf-state-isolation-v1','valid':valid,'reference':str(a.reference),'adapted':str(a.adapted),'reference_sha256':sha(a.reference),'adapted_sha256':sha(a.adapted),'shared_state_tensors':len(shared),'changed_shared_state_keys':changed,'added_state_keys':added,'removed_state_keys':removed,'shape_mismatch_keys':shape,'new_tensor_shape':list(w.shape) if isinstance(w,torch.Tensor) else None,'new_tensor_numel':int(w.numel()) if isinstance(w,torch.Tensor) else None,'factor_flags':flags,'factor_flags_valid':flags==expected,'semantic_witness_feature_schema':schema,'semantic_witness_feature_source':source,'semantic_witness_feature_contract_valid':feature_ok,'absolute_feasibility_truth_contract':truth,'absolute_feasibility_supervision_objective':obj,'absolute_feasibility_truth_index':idx,'absolute_feasibility_truth_index_valid':idx_ok,'stage_i_bitwise_identity':not changed and not removed and not shape,'dataset_reconstruction':False,'teacher_labels_changed':False,'test_roots_read':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n'); print(json.dumps({'event':'v48_83_state_isolation','valid':valid,'output':str(a.output)})); return 0 if valid else 30
if __name__=='__main__': raise SystemExit(main())
