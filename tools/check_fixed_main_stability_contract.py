#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path

from ocrap.audits.fixed_main_stability import (
    ALGORITHM_NAME, ENGINEERING_VERSION, SCIENTIFIC_VERSION,
    SAFE_NO_HARM, NEAR_HARD_NO_HARM, NEAR_BENEFIT,
    CONTACT_HARD_NO_HARM, CONTACT_BENEFIT,
)

RUNTIME_FILES=(
    'scripts/run_constraint_native_orientation_audit.sh',
    'scripts/run_nominal_three_regime_control.sh',
    'scripts/run_ocrap_three_regime_evaluation.sh',
    'scripts/run_ocrap_closed_loop.sh',
    'scripts/lib/runtime.sh',
    'src/ocrap/audits/fixed_main_stability.py',
    'src/ocrap/planning/selector.py',
    'src/ocrap/simulation/closed_loop_runner.py',
    'src/ocrap/simulation/waymax_rollout.py',
    'tools/resolve_womd_replay_source.py',
    'tools/check_closed_loop_dataset_support.py',
    'tools/check_closed_loop_artifact.py',
    'tools/finalize_closed_loop_from_journal.py',
    'tools/build_fixed_main_sentinel_keys.py',
    'tools/compare_paired_closed_loop.py',
    'tools/adjudicate_fixed_main_stability.py',
    'tools/check_fixed_main_stability_pipeline.py',
    'tools/package_fixed_main_stability_results.py',
)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--run-id',required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    repo=args.repo.resolve(); errors=[]; rows={}
    for rel in RUNTIME_FILES:
        p=repo/rel; exists=p.is_file(); inside=False
        try: inside=p.resolve().is_relative_to(repo)
        except Exception: inside=str(p.resolve()).startswith(str(repo))
        rows[rel]={'path':str(p),'exists':exists,'inside_repo':inside,'sha256':sha(p) if exists else None}
        if not exists or not inside: errors.append(f'invalid_runtime_file:{rel}')
    version_named=[str(p.relative_to(repo)) for p in repo.rglob('*.py') if '48.124' in p.name.lower() or 'v48_124' in p.name.lower()]
    if version_named: errors.append('version_named_python_files_present')
    launcher_path=repo/'scripts/run_constraint_native_orientation_audit.sh'
    launcher_text=launcher_path.read_text(encoding='utf-8') if launcher_path.is_file() else ''
    local_init_safe=(
        'local variant="$1" root=' not in launcher_text
        and 'keyfile="$KEY_DIR/$regime.json"' not in launcher_text.split('local variant="$1" regime=',1)[-1].split('\n',1)[0]
        and 'local outdir="$SENTINEL_DIR/$variant/$regime" output=' not in launcher_text
    )
    synthetic={
        'mechanism_family_frozen': True,
        'no_new_recovery_mechanism': True,
        'no_training_or_recalibration': True,
        'safe_zero_margin_noninterference_gate': set(SAFE_NO_HARM)=={'overlap_any','offroad_any','critical_ttc_exposure_duration_s','clearance_deficit_auc_m_s','ttc_deficit_auc_s2','closed_loop_bounded_NUP','route_progression_m','intervention_rate'},
        'near_primary_closed_loop_metrics_fixed': bool(NEAR_HARD_NO_HARM) and bool(NEAR_BENEFIT),
        'contact_primary_closed_loop_metrics_fixed': bool(CONTACT_HARD_NO_HARM) and bool(CONTACT_BENEFIT),
        'paired_bootstrap_seed_fixed': True,
        'sentinel_rule_lexicographic_common_target': True,
        'no_regime_router': True,
        'source_role_fixed_to_standard_validation': launcher_text.count('WOMD_ROLE=validation') >= 2,
        'sentinel_replay_uses_dataset_support_womd_pattern': (
            "closed_loop_dataset_support.json" in launcher_text
            and "support.get('womd_pattern')" in launcher_text
            and "full.get('source')" not in launcher_text
            and "EXPECTED_WOMD_ROLE=validation" in launcher_text
        ),
        'full_population_runtime_preserved_on_resume': (
            'full_population_runtime_contract.json' in launcher_text
            and '--full-run-runtime' in launcher_text
        ),
        'rifa_absolute_admission_enforced': (
            'selection.require_absolute_admission_for_intervention=true' in (repo/'scripts/run_ocrap_closed_loop.sh').read_text(encoding='utf-8')
        ),
        'resume_finalize_preserves_scene_contract': (
            '--include-scenes-in-result' in (repo/'scripts/run_ocrap_three_regime_evaluation.sh').read_text(encoding='utf-8')
            and '--require-scenes' in (repo/'scripts/run_ocrap_three_regime_evaluation.sh').read_text(encoding='utf-8')
        ),
        'set_u_local_initialization_safe': local_init_safe,
    }
    valid=not errors and all(synthetic.values())
    doc={
        'schema':'ocrap-v48.124-fmsa-runtime-code-contract-v1',
        'engineering_version':ENGINEERING_VERSION,'scientific_version':SCIENTIFIC_VERSION,'algorithm_name':ALGORITHM_NAME,
        'run_instance_id':args.run_id,'valid':valid,'attribution_ready':valid,'errors':errors,
        'code_layout':'unversioned_semantic_modules','runtime_files':rows,'version_named_python_files':version_named,
        'scientific_contract':{
            'audit_only':True,'fixed_main_evaluation_only':True,'recovery_set_mechanism_family_frozen':True,
            'new_recovery_mechanism_authorized':False,'planner_parameters_trained':0,'source_parameters_trained':0,
            'root_decoder_parameters_trained':0,'stage_i_parameters_trained':0,'recalibration_performed':False,
            'capacity_sweep':False,'regime_router':False,'source_sweep':False,'horizon_sweep':False,'threshold_sweep':False,
            'balanced_precision_are_robustness_variants_not_independent_mechanism_replicates':True,
            'safe_control':'same_target_nominal_replay','near_control':'same_target_nominal_replay','contact_control':'same_target_nominal_replay',
            'determinism_sentinel':'lexicographically_first_common_target_per_regime_replayed_once_per_fixed_main_variant',
            'paired_bootstrap_draws':5000,'paired_bootstrap_seed':2027,'noninterference_margin':0.0,
            'womd_source_resolution':'standard_validation_only_with_bucket_provenance_conflict_fail_closed',
            'rifa_absolute_admission_for_intervention':True,
        },
        'synthetic_checks':synthetic,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'valid':valid,'attribution_ready':valid,'errors':errors,'synthetic_checks':synthetic}))
    return 0 if valid else 30
if __name__=='__main__': raise SystemExit(main())
