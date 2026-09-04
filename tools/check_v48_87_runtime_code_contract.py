#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from ocrap.models.ocrap import ObservationConsistentBilinearActionRootResponseAdapter

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();errs=[]
 files=['src/ocrap/cli/train.py','src/ocrap/models/ocrap.py','src/ocrap/models/inference.py','scripts/train_ocrap_v48_trac_sr.sh','scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh','tools/check_v48_87_state_isolation.py','tools/audit_v48_87_barr.py','tools/compare_v48_87_barr.py','scripts/run_v48_87_dcp_drfc_bcde_rifa_barr_two_gpu.sh']
 rf={}
 for rel in files:
  p=(repo/rel).resolve();ok=p.is_file() and str(p).startswith(str(repo));rf[rel]={'path':str(p),'exists':p.is_file(),'inside_repo':str(p).startswith(str(repo)),'sha256':sha(p) if p.is_file() else None}
  if not ok:errs.append(f'runtime file invalid: {rel}')
 torch.manual_seed(7);m=ObservationConsistentBilinearActionRootResponseAdapter(141,192,rank=51)
 n=sum(p.numel() for p in m.parameters()); action=torch.randn(4,141); roots=torch.randn(4,8,192); margins=torch.ones(4,8); margins[:,4:]=-1
 z=m(action,roots,margins); zero_init=float(z.detach().abs().max())==0.0
 nominal=m(torch.zeros_like(action),roots,margins); nominal_zero=float(nominal.detach().abs().max())==0.0
 target=torch.randn_like(z); loss=(z-target).pow(2).mean(); loss.backward(); grad_out=float(m.output_factor.grad.abs().sum()) if m.output_factor.grad is not None else 0.0
 with torch.no_grad(): m.output_factor.normal_(0,0.03); r1=m(action,roots,margins); roots2=roots.clone();roots2[:,0]+=2.0;r2=m(action,roots2,margins)
 root_sensitive=float((r1[:,0]-r2[:,0]).detach().abs().sum())>1e-7
 checks={'parameter_count':n,'q85_parameter_count':54144,'capacity_not_increased':n<=54144,'zero_init_exact_native':zero_init,'nominal_action_exact_zero':nominal_zero,'nonzero_first_step_output_gradient':grad_out>1e-8,'output_gradient_l1':grad_out,'root_conditioned_response_after_nonzero_output':root_sensitive,'rank':m.rank}
 if n!=53550:errs.append(f'parameter count {n} != 53550')
 for k in ['capacity_not_increased','zero_init_exact_native','nominal_action_exact_zero','nonzero_first_step_output_gradient','root_conditioned_response_after_nonzero_output']:
  if not checks[k]:errs.append(f'synthetic contract failed: {k}')
 doc={'schema':'ocrap-v48.87-barr-runtime-code-contract-v1','engineering_version':'v48.87.0-OC-BARR','valid':not errs,'attribution_ready':not errs,'errors':errs,'runtime_files':rf,'representation_contract':{'name':'Observation-Consistent Bilinear Action-Root Response','rank':51,'frozen_observation_root_tokens':True,'candidate_minus_nominal_executable_action':True,'reserve_debt_channel_from_frozen_native_root_margin':True,'state_gate':False,'regime_conditioning':False,'root_option_regime_ids':False,'generic_mlp':False,'broad_encoder_retraining':False,'boundary_transport':False,'dataset_reconstruction':False,'relative_ranker_modified':False},'synthetic_checks':checks,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':doc['valid'],'errors':errs,'checks':checks}));return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
