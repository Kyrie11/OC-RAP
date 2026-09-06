#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.v48_99_recovery_jacobian import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    ObservationConditionedRecoveryJacobian,
    action_magnitude_linearity_synthetic_check,
    coordinate_scale_invariance_synthetic_check,
    nominal_identity_synthetic_check,
    observation_conditioning_synthetic_check,
    root_permutation_equivariance_synthetic_check,
    zero_init_nonzero_gradient_synthetic_check,
)
from tools.run_v48_97_executable_recovery_state import (
    action_strata_match_v48_96_synthetic_check,
    candidate_only_label_join_synthetic_check,
)

FILES = [
    "scripts/run_v48_99_dcp_drfc_bcde_rifa_ocrj_two_gpu.sh",
    "src/ocrap/v48_99_recovery_jacobian.py",
    "tools/run_v48_99_recovery_jacobian.py",
    "tools/compare_v48_99_ocrj.py",
    "tools/check_v48_99_runtime_code_contract.py",
    "tools/check_v48_99_pipeline_complete.py",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(); repo = a.repo.resolve(); errors: list[str] = []; runtime_files = {}
    for rel in FILES:
        p = (repo / rel).resolve(); inside = repo == p or repo in p.parents; exists = p.is_file()
        runtime_files[rel] = {"exists": exists, "inside_repo": inside, "path": str(p), "sha256": sha(p) if exists else None}
        if not exists or not inside:
            errors.append(rel)

    d_model = 192; action_dim = 5 + 80 + 40
    m = ObservationConditionedRecoveryJacobian(d_model=d_model, action_dim=action_dim)
    expected = d_model * 2 + 2 * action_dim + 2 * (1 + 2 * d_model)
    checks = {
        "fixed_parameter_count": m.trainable_parameter_count == expected,
        "nominal_zero_root_update": nominal_identity_synthetic_check(32, 17),
        "root_permutation_equivariance": root_permutation_equivariance_synthetic_check(32, 17),
        "observation_conditioned_jacobian": observation_conditioning_synthetic_check(32, 17),
        "zero_init_nonzero_gradient": zero_init_nonzero_gradient_synthetic_check(32, 17),
        "action_magnitude_preserved": action_magnitude_linearity_synthetic_check(32, 17),
        "coordinate_scale_invariant_metric": coordinate_scale_invariance_synthetic_check(),
        "candidate_only_v93_join_preserves_nominal": candidate_only_label_join_synthetic_check(),
        "action_evaluation_strata_match_v48_96": action_strata_match_v48_96_synthetic_check(),
    }
    for name, ok in checks.items():
        if not ok:
            errors.append(name)
    if m.trainable_parameter_count != expected:
        errors.append(f"parameter_count={m.trainable_parameter_count}!={expected}")

    out = {
        "schema": "ocrap-v48.99-runtime-code-contract-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "runtime_files": runtime_files,
        "scientific_contract": {
            "name": ALGORITHM_NAME,
            "planner_parameters_trained": 0,
            "source_parameters_trained": 0,
            "erss_parameters_trained": 0,
            "stage_i_parameters_trained": 0,
            "root_jacobian_parameters_fixed_capacity": True,
            "root_jacobian_parameter_count": expected,
            "semantic_rank": 2,
            "candidate_relative_centering": True,
            "nominal_update_exact_zero": True,
            "state_conditioned_control_affine": True,
            "nominal_context": ["v48.97_support_state", "v48.97_reserve_state"],
            "candidate_physical_blocks": ["prefix_param", "prefix_state", "control"],
            "root_set_permutation_equivariant": True,
            "root_slot_bijection_assumed": False,
            "coordinate_scale_invariant_semantic_metric": True,
            "structured_encoder_frozen": True,
            "structured_transformer_frozen": True,
            "root_decoder_frozen": True,
            "v48_97_state_chart_frozen": True,
            "teacher_components_supervision_only": True,
            "teacher_metadata_input_to_model": False,
            "regime_conditioning": False,
            "boundary_transport": False,
            "relative_ranker_modified": False,
            "dataset_reconstruction": False,
            "dataset_reselection": False,
            "capacity_sweep": False,
            "threshold_sweep": False,
        },
        "synthetic_checks": checks,
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "errors": errors, "parameters": expected}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
