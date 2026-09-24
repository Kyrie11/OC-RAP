#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

FROZEN_CORE=[
 'src/ocrap/planning/selector.py','src/ocrap/evaluation/baselines.py','src/ocrap/planning/prefix_generation.py',
 'src/ocrap/planning/route_lattice.py','src/ocrap/planning/utility.py','src/ocrap/data/waymax_loader.py',
 'src/ocrap/simulation/waymax_rollout.py','src/ocrap/models/inference.py','src/ocrap/models/ocrap.py',
 'configs/v48_124_10_near_relative_delta.yaml','configs/v48_124_10_near_nested_evidence.yaml',
]
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--reference-source-manifest',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    repo=Path(a.repo).resolve(); ref=json.loads(Path(a.reference_source_manifest).read_text()); rows=ref.get('files') or {}; errors=[]; checks={}
    for rel in FROZEN_CORE:
        p=repo/rel; exp=(rows.get(rel) or {}).get('sha256'); got=sha(p) if p.is_file() else None; ok=bool(got and exp and got==exp)
        checks[rel]={'actual_sha256':got,'expected_sha256':exp,'match':ok}
        if not ok: errors.append(f'frozen core source mismatch: {rel}')
    runner=(repo/'src/ocrap/simulation/closed_loop_runner.py').read_text(encoding='utf-8')
    runsh=(repo/'scripts/run_ocrap_closed_loop.sh').read_text(encoding='utf-8')
    instrumentation={
      'default_off':'cl_cfg.get("privileged_nonfloor_pcd_oracle_ceiling", False)' in runner and 'PRIVILEGED_NONFLOOR_PCD_ORACLE_CEILING="${PRIVILEGED_NONFLOOR_PCD_ORACLE_CEILING:-false}"' in runsh,
      'teacher_positive_required':'float(detail["teacher_pcd"]) > nominal_pcd + float(privileged_nonfloor_pcd_oracle_epsilon)' in runner,
      'nonfloor_required':'abs(r_dep_i - float(privileged_nonfloor_rdep_floor))' in runner,
      'positive_signed_rdep_required':'and r_dep_i > 0.0' in runner,
      'learned_admission_not_used_as_gate':'selected_was_observed_absolute_admitted' in runner,
      'exact_nominal_fallback':'privileged_nonfloor_pcd_oracle_nominal_no_nonfloor_positive_gain' in runner,
      'seed_start_gate':'step_idx >= privileged_nonfloor_seed_start_step' in runner,
      'nominal_before_seed_fail_closed':'Base intervened before preregistered non-floor seed start' in runner,
      'nup_recomputed_for_executed_action':'info["nup"] = float(oracle_nup["bounded_NUP"])' in runner,
      'seed_plan_exposed':'PRIVILEGED_NONFLOOR_SEED_PLAN_FILE' in runsh,
    }
    for k,v in instrumentation.items():
        if not v: errors.append(f'nonfloor screen instrumentation missing: {k}')
    out={
      'schema':'ocrap-v48.124.10.6-nonfloor-admission-screen-runtime-contract-v1',
      'engineering_version':'v48.124.10.6-NONFLOOR-ADMISSION-SEED-SCREEN',
      'scientific_version':'v48.124-OC-FMSA','algorithm_modified':False,'privileged_diagnostic_only':True,
      'valid':not errors,'attribution_ready':not errors,'errors':errors,'frozen_core_files':checks,'instrumentation_contract':instrumentation,
      'causal_contract':{
        'candidate_library_frozen':True,'deployed_model_and_selector_frozen_before_diagnostic_override':True,
        'learned_absolute_admission_is_bypassed_only_inside_privileged_screen':True,
        'structural_exact_rdep_half_plateau_is_excluded':True,'teacher_signed_rdep_must_be_positive':True,
        'teacher_pcd_must_strictly_improve_exact_nominal':True,'before_seed_start_base_must_remain_nominal':True,
        'screen_is_not_population_prevalence_evidence':True,'screen_cannot_authorize_v48_125_by_itself':True,'publication_evidence':False,
      },
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':out['valid'],'errors':errors,'output':a.output},indent=2)); return 0 if not errors else 30
if __name__=='__main__': raise SystemExit(main())
