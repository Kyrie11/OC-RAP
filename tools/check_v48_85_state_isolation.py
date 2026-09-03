#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
PREFIX='direct_absolute_action_response_adapter.'
def load(p):
 try:return torch.load(p,map_location='cpu',weights_only=False)
 except TypeError:return torch.load(p,map_location='cpu')
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--adapted',type=Path,required=True);ap.add_argument('--truth-index',type=Path,required=True);ap.add_argument('--state-conditioned',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 ra,da=load(a.reference),load(a.adapted);rs,ds=ra.get('model_state',ra),da.get('model_state',da);shared=sorted(set(rs)&set(ds));changed=[];shape=[]
 for k in shared:
  if tuple(rs[k].shape)!=tuple(ds[k].shape):shape.append(k)
  elif not torch.equal(rs[k].cpu(),ds[k].cpu()):changed.append(k)
 removed=sorted(set(rs)-set(ds));added=sorted(set(ds)-set(rs));allowed=[k for k in added if k.startswith(PREFIX)];new_ok=added==allowed and len(allowed)==1 and allowed[0].endswith('action_projection')
 w=ds.get(allowed[0]) if new_ok else None;shape_ok=isinstance(w,torch.Tensor) and w.ndim==3 and w.shape[0]==2
 flags={'adapter':bool(da.get('direct_recovery_semantic_witness_action_response_adapter',False)),'state_conditioning':bool(da.get('direct_recovery_semantic_witness_action_response_state_conditioning',False)),'root_tail_source':bool(da.get('direct_recovery_semantic_witness_root_tail_source',False)),'structured_tail_field':bool(da.get('direct_recovery_semantic_witness_structured_tail_field',False)),'boundary_transport':bool(da.get('direct_recovery_semantic_witness_boundary_transport',False))}
 expected={'adapter':True,'state_conditioning':bool(a.state_conditioned),'root_tail_source':False,'structured_tail_field':False,'boundary_transport':False}
 tcfg=((da.get('cfg') or {}).get('training') or {});truth=str(tcfg.get('direct_value_absolute_feasibility_truth_contract',''));obj=str(tcfg.get('direct_value_absolute_feasibility_supervision_objective',''));idx=str(tcfg.get('direct_value_absolute_feasibility_truth_index','') or '');idx_ok=bool(idx) and Path(idx).expanduser().resolve(strict=False)==a.truth_index.expanduser().resolve(strict=False)
 valid=not removed and not shape and not changed and new_ok and shape_ok and flags==expected and truth=='structural_interval_bounds' and obj=='signed_margin_interval_huber' and idx_ok
 doc={'schema':'ocrap-v48.85-sarr-state-isolation-v1','valid':valid,'reference':str(a.reference),'adapted':str(a.adapted),'reference_sha256':sha(a.reference),'adapted_sha256':sha(a.adapted),'shared_state_tensors':len(shared),'changed_shared_state_keys':changed,'added_state_keys':added,'removed_state_keys':removed,'shape_mismatch_keys':shape,'new_tensor_shape':list(w.shape) if isinstance(w,torch.Tensor) else None,'new_tensor_numel':int(w.numel()) if isinstance(w,torch.Tensor) else None,'factor_flags':flags,'factor_flags_valid':flags==expected,'stage_i_bitwise_identity':not changed and not removed and not shape,'absolute_feasibility_truth_contract':truth,'absolute_feasibility_supervision_objective':obj,'absolute_feasibility_truth_index_valid':idx_ok,'dataset_reconstruction':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':valid,'added':added}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
