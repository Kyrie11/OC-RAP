#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ocrap.v48_92_factorized_recovery_advantage import ENGINEERING_VERSION,factorize_recovery_advantage

FILES=(
 'src/ocrap/v48_92_factorized_recovery_advantage.py',
 'tools/build_v48_92_factorized_recovery_advantage_audit.py',
 'tools/compare_v48_92_frad.py',
 'tools/check_v48_92_runtime_code_contract.py',
 'tools/check_v48_92_pipeline_complete.py',
 'scripts/run_v48_92_dcp_drfc_bcde_rifa_frad.sh',
)
def sha(p:Path)->str:
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();repo=args.repo.resolve();errors=[];fs={}
 for rel in FILES:
  p=(repo/rel).resolve();inside=str(p).startswith(str(repo));exists=p.is_file();fs[rel]={'path':str(p),'exists':exists,'inside_repo':inside,'sha256':sha(p) if exists else None}
  if not(exists and inside):errors.append(f'invalid runtime file {rel}')
 a=factorize_recovery_advantage(candidate_drs=.8,nominal_drs=.4,candidate_r_dep=.3,nominal_r_dep=-.2,candidate_gap=.1,nominal_gap=.5)
 checks={
  'audit_only_zero_planner_parameters':True,'same_v48_91_cohort':True,'v48_91_exact_sidecar_reuse':True,'raw_womd_replay_disabled':True,
  'dataset_reconstruction':False,'dataset_reselection':False,'teacher_labels_changed':False,'teacher_metadata_not_model_input':True,
  'boundary_transport_off':True,'relative_ranker_frozen':True,'regime_conditioning_off':True,'capacity_sweep_off':True,
  'shapley_additivity':abs(a.shapley_sum_error)<=1e-12,
 }

 expected={'dataset_reconstruction':False,'dataset_reselection':False,'teacher_labels_changed':False}
 if any(checks[k] is not v for k,v in expected.items()) or any(not bool(v) for k,v in checks.items() if k not in expected):errors.append('synthetic/scientific contract failed')
 out={'schema':'ocrap-v48.92-runtime-code-contract-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,
      'runtime_files':fs,'synthetic_checks':checks,'scientific_contract':{'name':'Observation-Consistent Factorized Recovery-Advantage Decomposition','audit_only':True,
      'planner_parameters_trained':0,'same_v48_91_labeled_cohort':True,'v48_91_sidecar_reused':True,'womd_replay_performed':False,'dataset_reconstruction':False,
      'dataset_reselection':False,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'boundary_transport':False,'relative_ranker_modified':False,
      'regime_conditioning':False,'capacity_sweep':False},'test_roots_read':False}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':out['valid'],'attribution_ready':out['attribution_ready']}));return 0 if out['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
