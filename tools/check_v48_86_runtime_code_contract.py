#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from ocrap.cli.train import _absolute_feasibility_counterfactual_response_interval_huber,_absolute_feasibility_counterfactual_selective_response

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();errs=[]
 files=['src/ocrap/cli/train.py','src/ocrap/models/data.py','src/ocrap/models/ocrap.py','tools/build_v48_86_action_response_truth_index.py','tools/compare_v48_86_crsc.py','scripts/run_v48_86_dcp_drfc_bcde_rifa_crsc_two_gpu.sh']
 rf={}
 for rel in files:
  p=(repo/rel).resolve();ok=p.is_file() and str(p).startswith(str(repo));rf[rel]={'path':str(p),'exists':p.is_file(),'inside_repo':str(p).startswith(str(repo)),'sha256':sha(p) if p.is_file() else None}
  if not ok:errs.append(f'runtime file invalid: {rel}')
 logits=torch.tensor([0.2,0.6,-0.3,0.4],requires_grad=True)
 out={'direct_recovery_absolute_feasibility_logit':logits}
 batch={'r_dep_star':torch.zeros(4),'is_nominal':torch.tensor([1.,0.,1.,0.]),'bucket_id':torch.tensor([1,1,2,2]),'scene_hash':torch.tensor([11,11,22,22]),'time_index':torch.tensor([3,3,4,4]),'action_response_truth_informative':torch.ones(4),'action_response_truth_lower':torch.tensor([0.,0.1,0.,0.2]),'action_response_truth_upper':torch.tensor([0.,0.3,0.,0.4]),'action_response_safe_positive':torch.tensor([0.,1.,0.,0.]),'action_response_component_harmful':torch.tensor([0.,0.,0.,1.]),'action_response_deployable':torch.tensor([0.,1.,0.,1.])}
 phys=_absolute_feasibility_counterfactual_response_interval_huber(out,batch);sel=_absolute_feasibility_counterfactual_selective_response(out,batch)
 sel.backward(); grad=float(logits.grad.abs().sum())
 checks={'finite_physical_loss':bool(torch.isfinite(phys)),'finite_selective_loss':bool(torch.isfinite(sel)),'selective_includes_constraints':float(sel.detach())>=float(phys.detach())-1e-12,'nonzero_gradient':grad>1e-8,'gradient_l1':grad,'physical_loss':float(phys.detach()),'selective_loss':float(sel.detach())}
 if not all(bool(v) for k,v in checks.items() if k not in {'gradient_l1','physical_loss','selective_loss'}):errs.append('counterfactual supervision synthetic failed')
 doc={'schema':'ocrap-v48.86-crsc-runtime-code-contract-v1','engineering_version':'v48.86.0-OC-CRSC','valid':not errs,'attribution_ready':not errs,'errors':errs,'runtime_files':rf,'supervision_contract':{'action_response_adapter_capacity_changed_vs_Q85':False,'state_conditioning':False,'physical_response_interval':'[L_candidate-U_nominal, U_candidate-L_nominal]','selective_safe_signed_response':'positive pairwise ordering','selective_harmful_signed_response':'nonpositive pairwise ordering','teacher_metadata_input_to_model':False,'boundary_transport':False,'relative_ranker_modified':False,'dataset_reconstruction':False},'synthetic_checks':checks,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':doc['valid'],'errors':errs}));return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
