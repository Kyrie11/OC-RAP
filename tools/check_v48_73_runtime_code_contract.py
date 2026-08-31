#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import ocrap, ocrap.cli.train as tr, ocrap.models.data as data_mod, ocrap.models.inference as inference_mod, ocrap.models.ocrap as model_mod
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve()
 exp={'ocrap':(repo/'src/ocrap/__init__.py').resolve(),'train':(repo/'src/ocrap/cli/train.py').resolve(),'data':(repo/'src/ocrap/models/data.py').resolve(),'model':(repo/'src/ocrap/models/ocrap.py').resolve(),'inference':(repo/'src/ocrap/models/inference.py').resolve()};act={'ocrap':Path(ocrap.__file__).resolve(),'train':Path(tr.__file__).resolve(),'data':Path(data_mod.__file__).resolve(),'model':Path(model_mod.__file__).resolve(),'inference':Path(inference_mod.__file__).resolve()};errors=[]
 for k in exp:
  if act[k]!=exp[k]:errors.append(f'imported {k} module mismatch: {act[k]} != {exp[k]}')
 base={'direct_recovery_absolute_semantic_witness_correction':True,'direct_recovery_semantic_witness_route_alignment':True,'direct_recovery_semantic_witness_reentry_alignment':True,'direct_recovery_semantic_witness_control_projection':True,'direct_recovery_semantic_witness_projection_fidelity_weighting':True,'direct_recovery_semantic_witness_demand_normalized_fidelity':False,'direct_recovery_semantic_witness_robust_occupancy':False,'direct_recovery_semantic_witness_soft_occupancy_disagreement':False,'direct_recovery_semantic_witness_boundary_localized_occupancy_trust':False,'direct_recovery_semantic_witness_history_occupancy_reachability':False,'direct_recovery_semantic_witness_interaction_box_support':True,'direct_recovery_semantic_witness_interaction_hull_support':True}
 contracts={}
 for name,response in [('N73_ANCHORED_HULL',False),('O73_Main_OCIRRW',True)]:
  c=tr._semantic_witness_checkpoint_feature_contract({**base,'direct_recovery_semantic_witness_interaction_anchor_support':True,'direct_recovery_semantic_witness_interaction_response_support':response});contracts[name]={'schema':c[0],'source':c[1]}
  if c!=(9,'interaction_response_history_reachability_projected_recovery_witness'):errors.append(f'{name} serializer contract mismatch: {c}')
 doc={'schema':'ocrap-v48.73-runtime-code-contract-v1','valid':not errors,'repo':str(repo),'imported_modules':{k:str(v) for k,v in act.items()},'expected_modules':{k:str(v) for k,v in exp.items()},'module_sha256':{k:sha(v) for k,v in act.items()},'contracts':contracts,'python_executable':sys.executable,'errors':errors};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_73_runtime_code_contract','valid':not errors,'output':str(a.output)}));return 0 if not errors else 30
if __name__=='__main__':raise SystemExit(main())
