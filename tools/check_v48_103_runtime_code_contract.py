#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.v48_103_factorized_control_sufficient_state import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    expected_parameter_count,
    nominal_response_zero_check,
    state_response_parameter_disjoint_check,
    token_permutation_invariance_check,
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    repo = a.repo.resolve(); errors: list[str] = []
    files = [
        "scripts/run_v48_103_dcp_drfc_bcde_rifa_fcss_two_gpu.sh",
        "src/ocrap/v48_103_factorized_control_sufficient_state.py",
        "tools/run_v48_103_factorized_control_sufficient_state.py",
        "tools/compare_v48_103_fcss.py",
        "tools/check_v48_103_runtime_code_contract.py",
        "tools/check_v48_103_pipeline_complete.py",
    ]
    mods = {}
    for rel in files:
        p = repo / rel
        if not p.is_file():
            errors.append(f"missing {rel}"); continue
        inside = str(p.resolve()).startswith(str(repo))
        if not inside: errors.append(f"outside repo {rel}")
        mods[rel] = {"exists": True, "inside_repo": inside, "path": str(p.resolve()), "sha256": sha(p)}

    if expected_parameter_count(192) != 1540:
        errors.append("representation_parameter_count")
    if not nominal_response_zero_check(): errors.append("nominal_response_zero_check")
    if not token_permutation_invariance_check(): errors.append("token_permutation_invariance_check")
    if not state_response_parameter_disjoint_check(): errors.append("state_response_parameter_disjoint_check")

    imported = {}
    try:
        import ocrap.models.data as data_mod
        import ocrap.models.inference as inference_mod
        import ocrap.models.ocrap as model_mod
        for name, mod in (("ocrap.models.data", data_mod), ("ocrap.models.inference", inference_mod), ("ocrap.models.ocrap", model_mod)):
            p = Path(mod.__file__).resolve(); inside = str(p).startswith(str(repo))
            if not inside: errors.append(f"import outside repo {name}:{p}")
            imported[name] = {"path": str(p), "inside_repo": inside, "sha256": sha(p)}
    except Exception as exc:
        errors.append(f"import_provenance_exception:{exc!r}")

    doc = {
        "schema": "ocrap-v48.103-runtime-code-contract-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "runtime_files": mods,
        "imported_module_provenance": imported,
        "scientific_contract": {
            "minimal_stage_i_recovery_representation_objective": True,
            "full_stage_i_memory_consumed": True,
            "factorized_nominal_state_and_action_response": True,
            "nominal_response_exact_zero": True,
            "state_response_learned_mixing": False,
            "fixed_representation_parameters": 1540,
            "hidden_mlp": False,
            "rank_or_width_sweep": False,
            "planner_parameters_trained": 0,
            "stage_i_parameters_trained": 0,
            "root_decoder_parameters_trained": 0,
            "source_parameters_trained": 0,
            "same_v48_100_four_term_dimensionless_semantic_objective": True,
            "same_v48_100_semantic_metric_scales_reused": True,
            "same_v48_93_target_specific_evaluation_semantics": True,
            "dataset_reconstruction": False,
            "teacher_metadata_input_to_model": False,
            "boundary_transport": False,
            "relative_ranker_modified": False,
            "regime_conditioning": False,
        },
        "synthetic_checks": {
            "representation_parameter_count_1540": expected_parameter_count(192) == 1540,
            "nominal_response_exact_zero": not any("nominal_response" in x for x in errors),
            "token_pool_permutation_invariant": not any("token_permutation" in x for x in errors),
            "state_response_parameter_channels_disjoint": not any("state_response" in x for x in errors),
        },
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": doc["valid"], "errors": errors}))
    return 0 if doc["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
