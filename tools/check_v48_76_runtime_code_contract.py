#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,importlib,json,sys
from pathlib import Path
import torch

def sha(p:Path):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();src=repo/'src';sys.path.insert(0,str(src));sys.path.insert(0,str(repo));errors=[];mods={}
 expected={'ocrap':src/'ocrap/__init__.py','ocrap.cli.train':src/'ocrap/cli/train.py','ocrap.models.ocrap':src/'ocrap/models/ocrap.py','ocrap.models.inference':src/'ocrap/models/inference.py','ocrap.simulation.teacher.margins':src/'ocrap/simulation/teacher/margins.py'}
 for name,ep in expected.items():
  try:
   m=importlib.import_module(name);p=Path(m.__file__).resolve();ok=p==ep.resolve() and repo in p.parents;mods[name]={'path':str(p),'expected_path':str(ep.resolve()),'exact_path':p==ep.resolve(),'inside_repo':repo in p.parents,'sha256':sha(p)}
   if not ok:errors.append(f'runtime module mismatch: {name}')
  except Exception as e:mods[name]={'error':repr(e),'inside_repo':False};errors.append(f'runtime import failed: {name}')
 from ocrap.cli.train import _absolute_feasibility_supervision_loss
 batch={'r_dep_star':torch.tensor([.5,.2,-.7,-2.0]),'is_nominal':torch.zeros(4),'bucket_id':torch.tensor([1,1,2,2]),'time_index':torch.zeros(4,dtype=torch.long)};out={'direct_recovery_absolute_feasibility_logit':torch.tensor([99.0,.1,-.2,-1.0])}
 cfg={'direct_value_absolute_feasibility_truth_contract':'censor_exact_0p5','direct_value_absolute_feasibility_supervision_objective':'signed_margin_huber'}
 got=float(_absolute_feasibility_supervision_loss(out,batch,cfg));expected_loss=float(torch.nn.functional.smooth_l1_loss(torch.tensor([.1,-.2,-1.0]),torch.tensor([.2,-.7,-2.0]),beta=1.0))
 synth={'loss':got,'expected_loss':expected_loss,'floor_prediction_ignored':True,'valid':abs(got-expected_loss)<=1e-8}
 if not synth['valid']:errors.append('signed-margin supervision synthetic check failed')
 t=(src/'ocrap/simulation/teacher/margins.py').read_text();teacher={'positive_structural_floor_0p6':'max(val, 0.6)' in t,'route_override_neg_0p8':'min(val, -0.8)' in t,'secondary_floor_0p9':'max(val, 0.9)' in t};teacher['valid']=all(teacher.values())
 valid=not errors and synth['valid'] and teacher['valid']
 doc={'schema':'ocrap-v48.76-icsm-runtime-code-contract-v1','engineering_version':'v48.76.0-OC-ICSM','valid':valid,'attribution_ready':valid,'errors':errors,'runtime_modules':mods,'supervision_contract':{'truth_contract':'censor_exact_0p5','objective':'signed_margin_huber','huber_beta':1.0,'regime_conditioned':False,'teacher_future_input':False,'floor_relabelled':False,'synthetic_check':synth},'teacher_structural_contract_present':teacher,'dataset_reconstruction':False,'uses_test_roots':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_76_runtime_contract','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
