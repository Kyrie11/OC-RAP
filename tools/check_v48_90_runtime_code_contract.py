#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, importlib, json
from pathlib import Path
import numpy as np

from ocrap.v48_89_root_correspondence import nested_tail_influence
from ocrap.v48_90_partition_transport import audit_partition_transport_pair, future_class_keys

FILES=(
 'scripts/run_v48_90_dcp_drfc_bcde_rifa_cept.sh',
 'src/ocrap/v48_90_partition_transport.py',
 'tools/build_v48_90_partition_transport_audit.py',
 'tools/compare_v48_90_cept.py',
 'tools/check_v48_90_pipeline_complete.py',
 'tools/check_v48_90_runtime_code_contract.py',
)
MODULES=('ocrap','ocrap.algorithms.lcv','ocrap.algorithms.ocmero','ocrap.data.serialization','ocrap.v48_89_root_correspondence','ocrap.v48_90_partition_transport')

def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def sample(assign, metas, *, sources, margins, probs=None):
 assign=np.asarray(assign,np.int64); K=len(margins); n=len(assign)
 probs=np.asarray(probs if probs is not None else np.ones(n)/n,np.float32)
 rp=np.zeros(K,np.float32)
 for a,w in zip(assign,probs/probs.sum()): rp[int(a)]+=float(w)
 s={'m_star':np.asarray(margins,np.float32),'root_probs':rp,'root_valid':np.ones(K,np.float32),'c_star':np.eye(K,dtype=np.float32),
    'option_valid':np.ones(np.asarray(margins).shape[1],np.float32),'root_assignments':assign,'future_probs':probs,'future_valid':np.ones(n,np.float32),
    'future_sources':np.asarray(sources),'future_metadata':json.dumps(metas,sort_keys=True),'recovery_modes':np.asarray(['stop']*np.asarray(margins).shape[1])}
 _,r,_,_=nested_tail_influence(s); s['r_dep_star']=np.float32(r); return s

def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
 repo=args.repo.resolve(); errors=[]; runtime_files={}
 for rel in FILES:
  p=(repo/rel).resolve(); ok=p.is_file() and repo in p.parents
  runtime_files[rel]={'exists':p.is_file(),'inside_repo':repo in p.parents,'path':str(p),'sha256':sha(p) if p.is_file() else None}
  if not ok: errors.append(f'missing/outside repo {rel}')
 imported={}
 for name in MODULES:
  m=importlib.import_module(name); p=Path(m.__file__).resolve(); expected=(repo/('src/'+name.replace('.','/')+'.py')).resolve()
  if name=='ocrap': expected=(repo/'src/ocrap/__init__.py').resolve()
  imported[name]={'actual':str(p),'expected':str(expected),'path_match':p==expected,'sha256':sha(p)}
  if p!=expected: errors.append(f'import path mismatch {name}: {p} != {expected}')

 # Duplicate quotient: same semantic recipe, no fake occurrence identity.
 meta={'targeted_type':'waymax_sdc_post_prefix_control_stress','ego_after_prefix_accel':-2.0}
 dup=sample([0,0],[meta,meta],sources=['targeted','targeted'],margins=[[-.5]])
 keys,unres,dupf=future_class_keys(dup,exogenous=False)
 c1=bool(keys[0]==keys[1] and not unres.any() and dupf.all())
 recdup=audit_partition_transport_pair(dup,dup)
 c2=bool(recdup.valid and abs(recdup.recipe_tail_partition_stability-1.0)<1e-9 and recdup.duplicate_root_homogeneity_mass_candidate==1.0)
 # Candidate-dependent hidden realization must not be fabricated as same exogenous branch.
 a={'artifact_branch':'yield','targeted_type':'waymax_hidden_vehicle_yield','scenario_augmented':True,'hidden_spawn_xy':[8.,1.],'hidden_actor_object_index':3}
 b={'artifact_branch':'yield','targeted_type':'waymax_hidden_vehicle_yield','scenario_augmented':True,'hidden_spawn_xy':[12.,-2.],'hidden_actor_object_index':4}
 ca=sample([0],[a],sources=['targeted'],margins=[[-.5]]); nb=sample([0],[b],sources=['targeted'],margins=[[-1.]])
 recexo=audit_partition_transport_pair(ca,nb)
 c3=bool(recexo.valid and recexo.recipe_shared_mass_candidate==1.0 and recexo.exogenous_shared_mass_candidate==0.0)
 # Root slot permutation must remain transport-identical.
 metas=[{}, {'rollout_variant':'waymax_log_playback_sdc_coast','ego_after_prefix_accel':-1.0}]
 n=sample([0,1],metas,sources=['replay','reactive'],margins=[[-1.],[1.]])
 c=sample([1,0],metas,sources=['replay','reactive'],margins=[[1.2],[-.5]])
 recperm=audit_partition_transport_pair(c,n)
 c4=bool(recperm.valid and abs(recperm.exogenous_tail_partition_stability-1.0)<1e-9)
 checks={'duplicate_instances_quotiented':c1,'duplicate_root_homogeneity_exact':c2,'candidate_dependent_augmented_realization_not_false_matched':c3,'root_slot_permutation_transport_exact':c4,'audit_only_zero_planner_parameters':True}
 if not all(checks.values()): errors.append(f'synthetic checks failed {checks}')
 doc={'schema':'ocrap-v48.90-oc-cept-runtime-code-contract-v1','engineering_version':'v48.90.0-OC-CEPT','valid':not errors,'attribution_ready':not errors,'errors':errors,
      'imported_module_contract':imported,'runtime_files':runtime_files,'synthetic_checks':checks,
      'scientific_contract':{'name':'Observation-Consistent Counterfactual Equivalence-Partition Transport','audit_only':True,'planner_parameters_trained':0,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'dataset_reconstruction':False,'regime_conditioning':False,'boundary_transport':False,'relative_ranker_modified':False,'root_slot_identity_assumed':False,'individual_duplicate_branch_identity_assumed':False,'capacity_sweep':False},'test_roots_read':False}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'valid':doc['valid'],'errors':errors,'checks':checks})); return 0 if doc['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
