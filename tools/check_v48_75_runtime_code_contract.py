#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib, inspect, json, os, sys
from pathlib import Path
import torch

ENGINEERING_VERSION='v48.75.0-OC-STCA'

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();errors=[];mods={}
 expected={
  'ocrap':repo/'src/ocrap/__init__.py','ocrap.cli.train':repo/'src/ocrap/cli/train.py','ocrap.models.data':repo/'src/ocrap/models/data.py',
  'ocrap.models.ocrap':repo/'src/ocrap/models/ocrap.py','ocrap.models.inference':repo/'src/ocrap/models/inference.py','ocrap.simulation.teacher.margins':repo/'src/ocrap/simulation/teacher/margins.py'}
 for name,ep in expected.items():
  try:
   m=importlib.import_module(name);p=Path(inspect.getfile(m)).resolve();exact=p==ep.resolve();inside=(repo==p or repo in p.parents);mods[name]={'path':str(p),'expected_path':str(ep.resolve()),'exact_path':exact,'inside_repo':inside,'sha256':sha(p)}
   if not (exact and inside):errors.append(f'runtime import outside repository: {name}: {p}')
  except Exception as e:
   mods[name]={'error':repr(e),'inside_repo':False,'exact_path':False};errors.append(f'import failed: {name}: {e!r}')
 truth={}
 try:
  from ocrap.cli.train import _absolute_feasibility_supervision_mask
  batch={'r_dep_star':torch.tensor([0.5,0.500001,-0.2,0.7,0.5,0.5]),'is_nominal':torch.tensor([0.,0.,0.,0.,1.,0.]),'bucket_id':torch.tensor([1,1,2,2,1,0]),'time_index':torch.arange(6)}
  lm,lt,lf=_absolute_feasibility_supervision_mask(batch,{'direct_value_absolute_feasibility_truth_contract':'legacy_full'})
  cm,ct,cf=_absolute_feasibility_supervision_mask(batch,{'direct_value_absolute_feasibility_truth_contract':'censor_exact_0p5'})
  truth={'legacy_mask':lm.tolist(),'censored_mask':cm.tolist(),'censored_floor_mask':cf.tolist(),'target_unchanged':bool(torch.equal(lt,ct)),'valid':lm.tolist()==[True,True,True,True,False,False] and cm.tolist()==[False,True,True,True,False,False] and cf.tolist()==[True,False,False,False,False,False] and bool(torch.equal(lt,ct))}
  if not truth['valid']:errors.append('truth-contract synthetic mask invalid')
 except Exception as e:
  truth={'valid':False,'error':repr(e)};errors.append(f'truth-contract synthetic check failed: {e!r}')
 teacher={}
 try:
  import ocrap.simulation.teacher.margins as mm
  src=inspect.getsource(mm.teacher_margin)
  rules={'positive_structural_floor_0p6':('max(val, 0.6)' in src or 'max(val,0.6)' in src),'route_override_neg_0p8':('min(val, -0.8)' in src or 'min(val,-0.8)' in src),'secondary_floor_0p9':('max(val, 0.9)' in src or 'max(val,0.9)' in src)}
  teacher={'rules':rules,'valid':all(rules.values())}
  if not teacher['valid']:errors.append('teacher structural rule signature changed; re-adjudicate V48.75 contract')
 except Exception as e:
  teacher={'valid':False,'error':repr(e)};errors.append(f'teacher rule audit failed: {e!r}')
 # V48.75 must not accidentally enable the V48.74 schema-10 overlay.
 overlay=str(os.getenv('OCRAP_V48_74_SIGNED_VIABILITY','')).strip().lower()
 overlay_off=overlay not in {'1','true','yes','on'}
 if not overlay_off:errors.append('OCRAP_V48_74_SIGNED_VIABILITY must be disabled for V48.75 STCA')
 valid=not errors
 doc={'schema':'ocrap-v48.75-stca-runtime-code-contract-v1','engineering_version':ENGINEERING_VERSION,'valid':valid,'attribution_ready':valid,'errors':errors,'runtime_modules':mods,'truth_contract':{'policy':'censor_exact_0p5','floor_value':0.5,'tolerance':1e-8,'relabels_floor_as_negative':False,'synthetic_check':truth},'teacher_structural_contract_present':teacher,'v48_74_signed_viability_overlay_disabled':overlay_off,'regime_conditioned':False,'uses_privileged_future':False,'uses_test_roots':False,'dataset_reconstruction':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_75_runtime_code_contract','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
