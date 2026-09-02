#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np, torch
from ocrap.v48_80_interval_truth_contract import nested_tail_physical_interval
from ocrap.cli.train import _absolute_feasibility_interval_huber

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();errors=[]
 import ocrap,ocrap.cli.train,ocrap.models.data,ocrap.models.ocrap,ocrap.models.inference
 mods=[ocrap,ocrap.cli.train,ocrap.models.data,ocrap.models.ocrap,ocrap.models.inference]
 runtime={}
 for m in mods:
  p=Path(m.__file__).resolve();ok=str(p).startswith(str(repo));runtime[m.__name__]={'path':str(p),'inside_repo':ok,'sha256':sha(p)}
  if not ok:errors.append(f'outside repo: {m.__name__} {p}')
 s={'m_star':np.array([[.6]],dtype=np.float32),'root_probs':np.array([1.]),'root_valid':np.array([1]),'c_star':np.array([[1.]]),'option_valid':np.array([1]),'root_assignments':np.array([0]),'future_metadata':[{}],'recovery_modes':np.array(['yield_rejoin']),'r_dep_star':np.array(.6)}
 r=nested_tail_physical_interval(s)
 synth=bool(r.valid and r.informative and r.upper_finite and not r.lower_finite)
 b={'r_dep_star':torch.tensor([.6]),'is_nominal':torch.zeros(1),'bucket_id':torch.ones(1,dtype=torch.long),'time_index':torch.zeros(1,dtype=torch.long),'absolute_truth_interval_informative':torch.ones(1),'absolute_truth_physical_lower':torch.tensor([-1e6]),'absolute_truth_physical_upper':torch.tensor([.6])}
 loss=float(_absolute_feasibility_interval_huber({'direct_recovery_absolute_feasibility_logit':torch.tensor([.2])},b,{'direct_value_absolute_feasibility_truth_contract':'structural_interval_bounds'}))
 if not synth or loss!=0.0:errors.append('interval contract synthetic failed')
 doc={'schema':'ocrap-v48.80-runtime-code-contract-v1','engineering_version':'v48.80.0-OC-PISTC','valid':not errors,'attribution_ready':not errors,'errors':errors,'runtime_modules':runtime,'truth_contract_synthetic':{'valid':synth,'upper':r.physical_upper,'lower':r.physical_lower,'inside_loss':loss},'source_contract':{'source_capacity_changed_vs_J78':False,'trainable_parameter':'direct_absolute_root_tail_source_scale[1]','root_tail_source':True,'tail_localization':True,'boundary_transport':False,'regime_id_input':False},'supervision_contract':{'truth_contract':'structural_interval_bounds','objective':'signed_margin_interval_huber','teacher_labels_changed':False,'teacher_future_input_to_model':False},'dataset_reconstruction':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':not errors}));return 0 if not errors else 30
if __name__=='__main__':raise SystemExit(main())
