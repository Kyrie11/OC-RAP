#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

FROZEN_FILES=[
 'src/ocrap/planning/selector.py','src/ocrap/evaluation/baselines.py','src/ocrap/planning/prefix_generation.py',
 'src/ocrap/planning/route_lattice.py','src/ocrap/planning/utility.py','src/ocrap/data/waymax_loader.py',
 'src/ocrap/simulation/waymax_rollout.py','src/ocrap/simulation/closed_loop_runner.py',
 'src/ocrap/models/inference.py','src/ocrap/models/ocrap.py','scripts/run_ocrap_closed_loop.sh',
 'configs/v48_124_10_near_relative_delta.yaml','configs/v48_124_10_near_nested_evidence.yaml',
]
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--reference-source-manifest',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    repo=Path(a.repo).resolve(); ref=json.loads(Path(a.reference_source_manifest).read_text(encoding='utf-8')); rows=ref.get('files') or {}; errors=[]; checks={}
    for rel in FROZEN_FILES:
        p=repo/rel; exp=(rows.get(rel) or {}).get('sha256'); got=sha(p) if p.is_file() else None; ok=bool(got and exp and got==exp)
        checks[rel]={'actual_sha256':got,'expected_sha256':exp,'match':ok}
        if not ok: errors.append(f'frozen source mismatch: {rel}')
    runner=(repo/'src/ocrap/simulation/closed_loop_runner.py').read_text(encoding='utf-8')
    runsh=(repo/'scripts/run_ocrap_closed_loop.sh').read_text(encoding='utf-8')
    instrumentation={
      'coverage_mode_exists':'coverage_label_audit = label_mode in' in runner,
      'audit_after_selection':'audit_intervention_ok =' in runner and 'candidate_quality_audit_records.append' in runner,
      'all_candidate_scope':'scope in {"all", "all_candidates", "exhaustive"}' in runner,
      'records_store_gate':'audit_store_candidate_records' in runner,
      'script_exposes_intervention_only':'AUDIT_INTERVENTION_ONLY' in runsh,
      'script_exposes_candidate_scope':'AUDIT_CANDIDATE_SCOPE' in runsh,
      'script_exposes_record_store':'AUDIT_STORE_CANDIDATE_RECORDS' in runsh,
    }
    for k,v in instrumentation.items():
        if not v: errors.append(f'all-state support instrumentation contract missing: {k}')
    out={
      'schema':'ocrap-v48.124.10.5-all-state-support-runtime-contract-v1',
      'engineering_version':'v48.124.10.5-ALL-STATE-SUPPORT-LOCALIZATION-DIAGNOSTIC',
      'scientific_version':'v48.124-OC-FMSA','algorithm_modified':False,'privileged_diagnostic_only':True,
      'valid':not errors,'attribution_ready':not errors,'errors':errors,'frozen_runtime_files':checks,
      'instrumentation_contract':instrumentation,
      'causal_contract':{
        'frozen_base_policy_executes_unchanged':True,
        'teacher_labels_are_audit_only_after_action_selection':True,
        'all_24_frozen_candidates_are_labeled_at_every_base_trajectory_decision':True,
        'absolute_admission_is_observed_not_modified':True,
        'no_threshold_head_candidate_library_or_trigger_change':True,
        'cohort_is_historical_base_intervention_scenes_only':True,
        'publication_evidence':False,
      },
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'valid':out['valid'],'errors':errors,'output':a.output},indent=2)); return 0 if not errors else 30
if __name__=='__main__': raise SystemExit(main())
