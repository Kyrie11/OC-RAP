#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from ocrap.v48_81_switch_inverse_truth_contract import nested_tail_switch_inverse_interval

def sh(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sample(val=.7, mode='post_contact_stabilize', meta=None):
    return {'m_star':np.array([[val]],np.float32),'root_probs':np.array([1.],np.float32),'root_valid':np.array([1],np.bool_),'option_valid':np.array([1],np.bool_),'c_star':np.eye(1,dtype=np.float32),'root_assignments':np.array([0]),'future_metadata':[meta or {}],'recovery_modes':np.array([mode]),'r_dep_star':np.array(val,np.float32)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();errors=[]
 import ocrap,ocrap.cli.train,ocrap.models.data,ocrap.models.ocrap,ocrap.models.inference
 mods={'ocrap':ocrap.__file__,'ocrap.cli.train':ocrap.cli.train.__file__,'ocrap.models.data':ocrap.models.data.__file__,'ocrap.models.ocrap':ocrap.models.ocrap.__file__,'ocrap.models.inference':ocrap.models.inference.__file__}
 runtime={}
 for k,p in mods.items():
  rp=Path(p).resolve();inside=str(rp).startswith(str(repo));runtime[k]={'path':str(rp),'inside_repo':inside,'sha256':sh(rp)}
  if not inside:errors.append(f'{k} outside repo')
 # inactive floor must invert exactly; active floor remains one-sided.
 r_inactive=nested_tail_switch_inverse_interval(sample(.7))
 r_active=nested_tail_switch_inverse_interval(sample(.6))
 synth=bool(r_inactive.exact_physical and abs(r_inactive.physical_lower-.7)<1e-5 and abs(r_inactive.physical_upper-.7)<1e-5 and r_active.physical_upper<=.60001)
 if not synth:errors.append('switch-aware inverse synthetic contract failed')
 doc={'schema':'ocrap-v48.81-runtime-code-contract-v1','engineering_version':'v48.81.0-OC-SITC','valid':not errors,'attribution_ready':not errors,'errors':errors,'runtime_modules':runtime,'truth_contract_synthetic':{'valid':synth,'inactive_floor_exact':[r_inactive.physical_lower,r_inactive.physical_upper],'active_floor_interval':[r_active.physical_lower,r_active.physical_upper]},'source_contract':{'source_capacity_changed_vs_J78':False,'trainable_parameter':'direct_absolute_root_tail_source_scale[1]','root_tail_source':True,'tail_localization':True,'boundary_transport':False,'regime_id_input':False},'supervision_contract':{'truth_contract':'switch_inverse_interval_bounds','objective':'signed_margin_interval_huber','teacher_labels_changed':False,'teacher_future_input_to_model':False},'dataset_reconstruction':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':doc['valid'],'errors':errors}));return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
