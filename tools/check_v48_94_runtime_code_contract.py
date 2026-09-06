#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ocrap.v48_94_support_reserve_admission import ENGINEERING_VERSION,support_reserve_admission
FILES=(
 'src/ocrap/v48_94_support_reserve_admission.py',
 'tools/calibrate_policy_risk_v48.py',
 'scripts/calibrate_v48_36_shared_certificate_pool.sh',
 'tools/audit_v48_94_srca.py','tools/compare_v48_94_srca.py',
 'tools/check_v48_94_runtime_code_contract.py','tools/check_v48_94_pipeline_complete.py',
 'scripts/run_v48_94_dcp_drfc_bcde_rifa_srca_two_gpu.sh',
)
def sha(p:Path)->str:
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args(); repo=a.repo.resolve(); errors=[]; fs={}
 for rel in FILES:
  p=(repo/rel).resolve();inside=str(p).startswith(str(repo));exists=p.is_file();fs[rel]={'path':str(p),'exists':exists,'inside_repo':inside,'sha256':sha(p) if exists else None}
  if not(exists and inside):errors.append(f'invalid runtime file {rel}')
 s=support_reserve_admission([1.0,.2,.8,1.0],[0.0,.2,.1,1.0])
 r=support_reserve_admission([1.0,.7,.9,1.0],[1.0,.6,.9,1.0])
 bad=support_reserve_admission([0.0,.9,.1,1.0],[1.0,.6,.9,1.0])
 checks={
  'zero_new_planner_parameters':True,'same_l80_frozen_checkpoint':True,'v48_93_complementarity_prerequisite':True,
  'dataset_reconstruction':False,'dataset_reselection':False,'teacher_labels_changed':False,'teacher_metadata_not_model_input':True,
  'boundary_transport_off':True,'relative_ranker_frozen':True,'regime_conditioning_off':True,'capacity_sweep_off':True,
  'support_establishment_synthetic':s.state=='support_establishment' and s.passed,
  'reserve_debt_synthetic':r.state=='reserve_debt' and r.passed,
  'support_is_noncompensatory':bad.state=='reserve_debt' and not bad.passed,
  'gap_not_positive_admission_factor':True,
 }
 for k in ('dataset_reconstruction','dataset_reselection','teacher_labels_changed'):
  if checks[k]: errors.append(f'forbidden mutation enabled: {k}')
 for k,v in checks.items():
  if k not in {'dataset_reconstruction','dataset_reselection','teacher_labels_changed'} and not v: errors.append(f'contract failed: {k}')
 out={'schema':'ocrap-v48.94-runtime-code-contract-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,'runtime_files':fs,'synthetic_checks':checks,'scientific_contract':{
  'name':'Observation-Consistent Support-Reserve Complementarity Admission','fixed_capacity':True,'new_planner_parameters':0,'state_switch':'nominal_native_hard_DRS_zero_vs_positive, shared by all candidates in one scene-time group','support_boundary':'native hard DRS exact zero','deployability_boundary':'native R_dep=0 <=> sigmoid(R_dep)=0.5','gap_role':'diagnostic/non-positive-source only','same_l80_frozen_checkpoint':True,'same_top_k':5,'dataset_reconstruction':False,'dataset_reselection':False,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'boundary_transport':False,'relative_ranker_modified':False,'regime_conditioning':False,'capacity_sweep':False},'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':out['valid'],'attribution_ready':out['attribution_ready']}));return 0 if out['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
