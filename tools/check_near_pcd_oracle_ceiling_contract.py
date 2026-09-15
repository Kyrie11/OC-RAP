#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

# These files define the frozen deployable candidate generation / scoring / admission
# semantics inherited from the attribution-ready V48.124.10.3 runtime.  The
# closed-loop runner itself is intentionally instrumented by this diagnostic and
# is therefore checked by behavioral/flag contracts rather than byte identity.
FROZEN_FILES = [
    "src/ocrap/planning/selector.py",
    "src/ocrap/evaluation/baselines.py",
    "src/ocrap/planning/prefix_generation.py",
    "src/ocrap/planning/route_lattice.py",
    "src/ocrap/planning/utility.py",
    "src/ocrap/data/waymax_loader.py",
    "src/ocrap/simulation/waymax_rollout.py",
    "src/ocrap/models/inference.py",
    "src/ocrap/models/ocrap.py",
    "configs/v48_124_10_near_relative_delta.yaml",
    "configs/v48_124_10_near_nested_evidence.yaml",
]

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',required=True)
    ap.add_argument('--reference-source-manifest',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    repo=Path(a.repo).resolve(); ref=json.loads(Path(a.reference_source_manifest).read_text(encoding='utf-8'))
    rows=ref.get('files') or {}; errors=[]; checks={}
    for rel in FROZEN_FILES:
        p=repo/rel; meta=rows.get(rel) or {}
        actual=sha(p) if p.is_file() else None; expected=meta.get('sha256')
        ok=bool(actual and expected and actual==expected)
        checks[rel]={'actual_sha256':actual,'expected_sha256':expected,'match':ok}
        if not ok: errors.append(f'frozen source mismatch: {rel}')
    runner=repo/'src/ocrap/simulation/closed_loop_runner.py'
    runsh=repo/'scripts/run_ocrap_closed_loop.sh'
    runner_text=runner.read_text(encoding='utf-8') if runner.is_file() else ''
    runsh_text=runsh.read_text(encoding='utf-8') if runsh.is_file() else ''
    instrumentation={
        'default_off_in_runner':'cl_cfg.get("privileged_pcd_oracle_ceiling", False)' in runner_text,
        'base_select_before_teacher':'base_sel_idx = int(sel_idx)' in runner_text and 'if privileged_pcd_oracle_ceiling' in runner_text,
        'absolute_admission_reused':'bool(admitted[pos])' in runner_text,
        'positive_teacher_pcd_required':'best_pcd > nominal_pcd + privileged_pcd_oracle_epsilon' in runner_text,
        'nominal_fallback':'privileged_pcd_oracle_nominal_no_positive_admitted_gain' in runner_text,
        'script_default_off':'PRIVILEGED_PCD_ORACLE_CEILING="${PRIVILEGED_PCD_ORACLE_CEILING:-false}"' in runsh_text,
    }
    for k,v in instrumentation.items():
        if not v: errors.append(f'oracle instrumentation contract missing: {k}')
    out={
        'schema':'ocrap-v48.124.10.4-pcd-oracle-runtime-contract-v1',
        'engineering_version':'v48.124.10.4-PCD-ORACLE-CEILING-DIAGNOSTIC',
        'scientific_version':'v48.124-OC-FMSA',
        'algorithm_modified':False,
        'privileged_diagnostic_only':True,
        'valid':not errors,'attribution_ready':not errors,'errors':errors,
        'frozen_runtime_files':checks,'instrumentation_contract':instrumentation,
        'causal_contract':{
            'frozen_base_selector_is_trigger':True,
            'teacher_labels_only_after_base_non_nominal_trigger':True,
            'absolute_rejected_candidates_can_never_be_rescued':True,
            'teacher_pcd_gain_must_be_positive_vs_exact_nominal':True,
            'no_positive_admitted_candidate_means_exact_nominal':True,
            'zero_intervention_base_scenes_are_reusable_by_induction':True,
            'publication_evidence':False,
        },
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'valid':out['valid'],'errors':errors,'output':a.output},indent=2))
    return 0 if not errors else 30
if __name__=='__main__': raise SystemExit(main())
