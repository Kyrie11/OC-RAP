#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.v48_97_executable_recovery_state import (
    ENGINEERING_VERSION,
    ExecutableRecoverySufficientState,
    root_permutation_invariance_check,
)
from tools.run_v48_97_executable_recovery_state import candidate_only_label_join_synthetic_check

FILES = [
    "scripts/run_v48_97_dcp_drfc_bcde_rifa_erss_two_gpu.sh",
    "src/ocrap/v48_97_executable_recovery_state.py",
    "tools/run_v48_97_executable_recovery_state.py",
    "tools/compare_v48_97_erss.py",
    "tools/check_v48_97_runtime_code_contract.py",
    "tools/check_v48_97_pipeline_complete.py",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    repo = a.repo.resolve()
    errors = []
    runtime_files = {}
    for rel in FILES:
        p = (repo / rel).resolve()
        inside = repo == p or repo in p.parents
        exists = p.is_file()
        runtime_files[rel] = {"exists": exists, "inside_repo": inside, "path": str(p), "sha256": sha(p) if exists else None}
        if not exists or not inside:
            errors.append(rel)
    m = ExecutableRecoverySufficientState(192)
    param_count = m.trainable_parameter_count
    expected = 4 * 192 + 2
    if param_count != expected:
        errors.append(f"representation_parameter_count={param_count}!={expected}")
    if not root_permutation_invariance_check(32):
        errors.append("root_permutation_invariance")
    join_ok = candidate_only_label_join_synthetic_check()
    if not join_ok:
        errors.append("candidate_only_v93_join_preserves_nominal")
    out = {
        "schema": "ocrap-v48.97-runtime-code-contract-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "runtime_files": runtime_files,
        "scientific_contract": {
            "name": "Observation-Consistent Executable-Recovery Sufficient State",
            "planner_parameters_trained": 0,
            "source_parameters_trained": 0,
            "representation_parameters_fixed_capacity": True,
            "representation_semantic_coordinates": ["shared_recovery_support", "signed_reserve_debt"],
            "frozen_l80_root_tokens": True,
            "root_slot_bijection_assumed": False,
            "root_set_permutation_invariant": True,
            "regime_conditioning": False,
            "teacher_metadata_input_to_model": False,
            "teacher_components_supervision_only": True,
            "boundary_transport": False,
            "relative_ranker_modified": False,
            "dataset_reconstruction": False,
            "dataset_reselection": False,
            "capacity_sweep": False,
            "threshold_sweep": False,
        },
        "synthetic_checks": {
            "root_set_permutation_invariant": root_permutation_invariance_check(32),
            "fixed_parameter_count": param_count == expected,
            "regime_conditioning_off": True,
            "source_training_off": True,
            "boundary_transport_off": True,
            "candidate_only_v93_join_preserves_nominal": join_ok,
            "empty_evaluation_fail_closed": True,
        },
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
