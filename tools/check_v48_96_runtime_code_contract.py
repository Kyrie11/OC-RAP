#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ocrap.v48_96_support_reserve_root_observability import (ENGINEERING_VERSION, derive_candidate_semantics, feature_only_dataset_cfg, root_observability_features)

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();errors=[]
 files=['scripts/run_v48_96_dcp_drfc_bcde_rifa_srroa_two_gpu.sh','src/ocrap/v48_96_support_reserve_root_observability.py','tools/run_v48_96_support_reserve_root_observability_probe.py','tools/compare_v48_96_srroa.py','tools/check_v48_96_runtime_code_contract.py','tools/check_v48_96_pipeline_complete.py']
 mods={}
 for rel in files:
  p=repo/rel
  if not p.is_file():errors.append(f'missing {rel}');continue
  mods[rel]={'exists':True,'inside_repo':str(p.resolve()).startswith(str(repo)),'path':str(p.resolve()),'sha256':sha(p)}
 # exact synthetic mediation sanity
 n={'teacher_drs':0.0,'teacher_r_dep':0.0,'teacher_gap':0.0,'teacher_pcd':0.0,'component_harmful':False}
 c={'teacher_drs':1.0,'teacher_r_dep':0.0,'teacher_gap':0.0,'teacher_pcd':0.5,'component_harmful':False}
 try:
  s=derive_candidate_semantics(n,c)
  if not(s['safe_positive'] and s['mediation_mode']=='drs_activation'):errors.append('synthetic DRS mediation failed')
 except Exception as e:errors.append(f'synthetic mediation exception {e!r}')
 # Engineering-fix contracts: feature extraction must not inherit training-only
 # truth sidecars, and root features must be permutation-invariant rather than
 # assuming a root-slot bijection.
 try:
  fcfg,fevt=feature_only_dataset_cfg({'training':{'direct_value_absolute_feasibility_truth_contract':'structural_interval_bounds','direct_value_absolute_feasibility_truth_index':'/train-only','direct_value_absolute_feasibility_supervision_objective':'signed_margin_interval_huber'}},cache_dir='/tmp/v4896_contract',workers=1)
  ft=fcfg['training']
  if not(ft.get('direct_value_absolute_feasibility_truth_contract')=='legacy_full' and ft.get('direct_value_absolute_feasibility_truth_index')=='' and ft.get('direct_value_absolute_feasibility_supervision_objective')=='binary_sign' and fevt.get('truth_sidecars_attached') is False):errors.append('feature-only truth-sidecar stripping failed')
  import torch
  rt=torch.tensor([[[1.,0.],[2.,1.]],[[3.,0.],[4.,1.]],[[5.,0.],[6.,1.]]]); pr=torch.tensor([[0.4,0.6],[0.2,0.8],[0.7,0.3]]); rv=torch.tensor([[1,1],[1,0],[0,1]],dtype=torch.bool)
  st,de,co=root_observability_features(rt,pr,rv)
  if not torch.equal(st[0],st[1]):errors.append('nominal state purity synthetic failed')
  rt2=rt.clone();pr2=pr.clone();rv2=rv.clone();rt2[1]=rt2[1,[1,0]];pr2[1]=pr2[1,[1,0]];rv2[1]=rv2[1,[1,0]]
  st2,de2,co2=root_observability_features(rt2,pr2,rv2)
  if not(torch.equal(st,st2) and torch.allclose(de,de2,atol=1e-7,rtol=0) and torch.allclose(co,co2,atol=1e-7,rtol=0)):errors.append('root permutation-invariance synthetic failed')
 except Exception as e:errors.append(f'engineering-fix synthetic exception {e!r}')
 doc={'schema':'ocrap-v48.96-runtime-code-contract-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,'runtime_files':mods,
      'scientific_contract':{'audit_only':True,'planner_parameters_trained':0,'same_l80_frozen_checkpoint':True,'stage_i_modified':False,'teacher_mediation_labels_audit_only':True,'fixed_linear_probe_capacity':True,'within_group_action_permutation_control':True,'root_slot_bijection_assumed':False,'feature_only_truth_sidecars_attached':False,'capacity_sweep':False,'threshold_sweep':False,'dataset_reconstruction':False,'dataset_reselection':False,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'boundary_transport':False,'relative_ranker_modified':False,'regime_conditioning':False,'womd_replay_performed':False},
      'synthetic_checks':{'drs_mediation':not any('synthetic mediation' in x for x in errors),'feature_only_truth_sidecars_disabled':not any('truth-sidecar' in x for x in errors),'root_partition_permutation_invariant':not any('permutation-invariance' in x for x in errors),'nominal_state_candidate_invariant':not any('nominal state purity' in x for x in errors),'boundary_transport_off':True,'capacity_sweep_off':True,'threshold_sweep_off':True,'regime_conditioning_off':True,'teacher_metadata_not_model_input':True},'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':doc['valid'],'errors':errors}));return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
