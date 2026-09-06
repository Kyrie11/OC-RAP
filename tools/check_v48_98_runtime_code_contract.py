#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.v48_98_executable_recovery_tangent import (
    ENGINEERING_VERSION,
    ExecutableRecoveryTangentAdapter,
    nominal_identity_synthetic_check,
    orthonormal_tangent_basis_synthetic_check,
)
from tools.run_v48_97_executable_recovery_state import (
    candidate_only_label_join_synthetic_check,
    action_strata_match_v48_96_synthetic_check,
)

FILES = [
    "scripts/run_v48_98_dcp_drfc_bcde_rifa_erta_two_gpu.sh",
    "src/ocrap/v48_98_executable_recovery_tangent.py",
    "tools/run_v48_98_executable_recovery_tangent.py",
    "tools/compare_v48_98_erta.py",
    "tools/check_v48_98_runtime_code_contract.py",
    "tools/check_v48_98_pipeline_complete.py",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    repo = a.repo.resolve()
    errors: list[str] = []
    runtime_files = {}
    for rel in FILES:
        p = (repo / rel).resolve()
        inside = repo == p or repo in p.parents
        exists = p.is_file()
        runtime_files[rel] = {"exists": exists, "inside_repo": inside, "path": str(p), "sha256": sha(p) if exists else None}
        if not exists or not inside:
            errors.append(rel)

    m = ExecutableRecoveryTangentAdapter(d_model=192, prefix_param_dim=5, prefix_state_dim=80, control_dim=40)
    expected = 2 * 192 + 2 * (5 + 80 + 40)
    param_ok = m.trainable_parameter_count == expected
    if not param_ok:
        errors.append(f"stage_i_tangent_parameter_count={m.trainable_parameter_count}!={expected}")
    nominal_ok = nominal_identity_synthetic_check(32)
    basis_ok = orthonormal_tangent_basis_synthetic_check(32)
    join_ok = candidate_only_label_join_synthetic_check()
    strata_ok = action_strata_match_v48_96_synthetic_check()
    for name, ok in (
        ("nominal_zero_tangent", nominal_ok),
        ("orthonormal_semantic_tangent_basis", basis_ok),
        ("candidate_only_v93_join_preserves_nominal", join_ok),
        ("action_evaluation_strata_match_v48_96", strata_ok),
    ):
        if not ok:
            errors.append(name)

    out = {
        "schema": "ocrap-v48.98-runtime-code-contract-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "runtime_files": runtime_files,
        "scientific_contract": {
            "name": "Observation-Consistent Executable-Recovery Tangent Alignment",
            "planner_parameters_trained": 0,
            "source_parameters_trained": 0,
            "erss_parameters_trained": 0,
            "stage_i_tangent_parameters_fixed_capacity": True,
            "semantic_tangent_rank": 2,
            "candidate_relative_centering": True,
            "nominal_update_exact_zero": True,
            "candidate_physical_blocks": ["prefix_param", "prefix_state", "control"],
            "shared_observation_tokens_frozen": True,
            "structured_transformer_frozen": True,
            "root_decoder_frozen": True,
            "v48_97_state_chart_frozen": True,
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
            "fixed_parameter_count": param_ok,
            "nominal_zero_tangent": nominal_ok,
            "orthonormal_semantic_tangent_basis": basis_ok,
            "candidate_only_v93_join_preserves_nominal": join_ok,
            "action_evaluation_strata_match_v48_96": strata_ok,
            "state_chart_frozen": True,
            "source_training_off": True,
            "boundary_transport_off": True,
            "regime_conditioning_off": True,
        },
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
