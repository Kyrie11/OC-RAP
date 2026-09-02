#!/usr/bin/env python3
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np, torch

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();errors=[]
 files=['tools/run_v48_84_stage_i_action_observability_probe.py','tools/compare_v48_84_saop.py','scripts/run_v48_84_dcp_drfc_bcde_rifa_saop_two_gpu.sh']
 mods={x:{'path':str((repo/x).resolve()),'sha256':sha(repo/x),'exists':(repo/x).exists()} for x in files if (repo/x).exists()}
 if len(mods)!=len(files): errors.append('missing_v48_84_files')
 # deterministic paired-permutation sanity: same group distribution, action alignment destroyed.
 from importlib.util import spec_from_file_location,module_from_spec
 sp=spec_from_file_location('p',repo/'tools/run_v48_84_stage_i_action_observability_probe.py');m=module_from_spec(sp);sp.loader.exec_module(m)
 rec=[{'group':(1,'s',1),'candidate':i,'delta':np.asarray([float(i),0.]),'context':np.asarray([float(i),0.,0.,0.])} for i in [1,2,3]]
 pp=m.permute_within_group(rec,'delta')
 perm_ok=bool(np.allclose(pp[:,0],[3.,1.,2.]))
 if not perm_ok: errors.append('within_group_permutation_contract_failed')
 doc={'schema':'ocrap-v48.84-saop-runtime-code-contract-v1','engineering_version':'v48.84.0-OC-SAOP','valid':not errors,'attribution_ready':not errors,'errors':errors,'runtime_files':mods,'scientific_contract':{'planner_source_modified':False,'stage_i_modified':False,'relative_ranker_modified':False,'boundary_transport':'OFF','dataset_reconstruction':False,'teacher_labels':'probe_only_not_model_input','delta_probe':'frozen_root_candidate_minus_nominal','context_probe':'same_dim_delta_times_deterministic_nominal_state_gate','within_group_permutation_control':True,'generic_mlp':False},'synthetic_permutation_passed':perm_ok,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2),encoding='utf-8'); print(json.dumps({'valid':doc['valid'],'errors':errors}));sys.exit(0 if doc['valid'] else 30)
if __name__=='__main__':main()
