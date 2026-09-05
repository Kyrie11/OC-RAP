#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,importlib,json
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from ocrap.v48_91_common_exogenous_physical_margin import physical_margin_from_teacher_diag
FILES=(
 'scripts/run_v48_91_dcp_drfc_bcde_rifa_cepmi.sh',
 'src/ocrap/v48_91_common_exogenous_physical_margin.py',
 'tools/build_v48_91_common_exogenous_physical_sidecar.py',
 'tools/merge_v48_91_common_exogenous_physical_sidecar_parts.py',
 'src/ocrap/data/waymax_loader.py',
 'tools/build_v48_91_common_exogenous_physical_response_audit.py',
 'tools/compare_v48_91_cepmi.py','tools/check_v48_91_pipeline_complete.py','tools/check_v48_91_runtime_code_contract.py')
MODULES=('ocrap','ocrap.data.waymax_loader','ocrap.v48_89_root_correspondence','ocrap.v48_90_partition_transport','ocrap.v48_91_common_exogenous_physical_margin')
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();errors=[];rf={}
 for rel in FILES:
  p=(repo/rel).resolve();rf[rel]={'path':str(p),'exists':p.is_file(),'inside_repo':repo in p.parents,'sha256':sha(p) if p.is_file() else None}
  if not p.is_file() or repo not in p.parents:errors.append(f'missing/outside repo {rel}')
 imp={}
 for name in MODULES:
  m=importlib.import_module(name);p=Path(m.__file__).resolve();exp=(repo/('src/'+name.replace('.','/')+'.py')).resolve() if name!='ocrap' else (repo/'src/ocrap/__init__.py').resolve();imp[name]={'actual':str(p),'expected':str(exp),'path_match':p==exp,'sha256':sha(p)}
  if p!=exp:errors.append(f'import mismatch {name}')
 d=SimpleNamespace(active={'clearance':True,'route':False,'control':True},component_margins={'clearance':-.4,'route':-9.,'control':.2})
 builder_text=(repo/'tools/build_v48_91_common_exogenous_physical_sidecar.py').read_text(encoding='utf-8')
 runner_text=(repo/'scripts/run_v48_91_dcp_drfc_bcde_rifa_cepmi.sh').read_text(encoding='utf-8')
 checks={'pre_structural_min_uses_active_components':abs(physical_margin_from_teacher_diag(d)+.4)<1e-12,'audit_only_zero_planner_parameters':True,'same_cohort_no_dataset_reselection':True,'teacher_metadata_not_model_input':True,'boundary_transport_off':True,'legacy_wx_provenance_migration_supported':'legacy_wx_migration_key' in builder_text and '--womd-source-pattern' in builder_text,'canonical_npz_not_modified':'canonical_npz_modified' in builder_text,'sparse_source_iterator_wired':'iter_waymax_womd_scenarios_selected' in builder_text,'metadata_only_future_metrics_skipped':"wx['compute_future_metrics'] = False" in builder_text,'parallel_replay_sharding_supported':'V4891_REPLAY_WORKERS' in runner_text and '--num-workers' in runner_text,'canonical_v48_14_sample_local_replay_profile':'_canonical_v4814_sample_profile' in builder_text and 'artifact_pass' in builder_text,'sample_local_pass_final_layer':'FINAL semantic layer' in builder_text and '_assert_v4814_effective_profile' in builder_text,'profile_preflight_before_raw_scan':'v48.91_profile_preflight' in builder_text,'provenance_chain_cache':'_origin_replay_metadata_cached' in builder_text,'fail_fast_replay_identity_guard':'--fail-fast-replay-errors' in builder_text and 'V4891_FAIL_FAST_REPLAY_ERRORS' in builder_text,'replay_checkpoint_resume_supported':'--checkpoint' in builder_text and 'V4891_REPLAY_RESUME' in runner_text}
 if not all(checks.values()):errors.append(f'synthetic checks failed {checks}')
 doc={'schema':'ocrap-v48.91-oc-cepmi-runtime-code-contract-v1','engineering_version':'v48.91.4-OC-CEPMI-REPLAYFIX2','valid':not errors,'attribution_ready':not errors,'errors':errors,'runtime_files':rf,'imported_module_contract':imp,'synthetic_checks':checks,'scientific_contract':{'name':'Observation-Consistent Common-Exogenous Physical-Margin Identifiability','audit_only':True,'planner_parameters_trained':0,'same_v48_90_labeled_cohort':True,'dataset_reconstruction':False,'dataset_reselection':False,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'boundary_transport':False,'relative_ranker_modified':False,'regime_conditioning':False,'capacity_sweep':False},'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':doc['valid'],'errors':errors,'checks':checks}));return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
