#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ACTIVE = [
    "scripts/run_constraint_native_orientation_audit.sh",
    "src/ocrap/audits/constraint_native_orientation.py",
    "src/ocrap/audits/heterogeneous_constraint_normal_cone.py",
    "src/ocrap/audits/executable_constraint_jacobian.py",
    "src/ocrap/audits/common_option_constraint_work.py",
    "src/ocrap/audits/recovery_set_constraint_flow.py",
    "src/ocrap/audits/weak_root_recovery_set_flow.py",
    "tools/run_constraint_native_recovery_orientation_audit.py",
    "tools/compare_constraint_native_recovery_orientation.py",
    "tools/check_constraint_native_orientation_contract.py",
    "tools/check_constraint_native_orientation_pipeline.py",
    "tools/package_constraint_native_orientation_results.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    repo = a.repo.resolve()
    errors: list[str] = []
    files: dict[str, dict] = {}

    src = str((repo / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    import ocrap
    import ocrap.audits.constraint_native_orientation as base_primitives
    import ocrap.audits.heterogeneous_constraint_normal_cone as hcnc
    import ocrap.audits.executable_constraint_jacobian as ecj
    import ocrap.audits.common_option_constraint_work as ccw
    import ocrap.audits.recovery_set_constraint_flow as rscf
    import ocrap.audits.weak_root_recovery_set_flow as wrcf

    for rel in ACTIVE:
        p = (repo / rel).resolve()
        ok = p.is_file() and str(p).startswith(str(repo))
        files[rel] = {
            "exists": p.is_file(), "inside_repo": str(p).startswith(str(repo)),
            "path": str(p), "sha256": sha(p) if p.is_file() else None,
        }
        if not ok:
            errors.append(f"runtime_file:{rel}")

    imported = {
        "ocrap": str(Path(ocrap.__file__).resolve()),
        "constraint_native_orientation": str(Path(base_primitives.__file__).resolve()),
        "heterogeneous_constraint_normal_cone": str(Path(hcnc.__file__).resolve()),
        "executable_constraint_jacobian": str(Path(ecj.__file__).resolve()),
        "common_option_constraint_work": str(Path(ccw.__file__).resolve()),
        "recovery_set_constraint_flow": str(Path(rscf.__file__).resolve()),
        "weak_root_recovery_set_flow": str(Path(wrcf.__file__).resolve()),
    }
    expected = {
        "ocrap": str((repo / "src/ocrap/__init__.py").resolve()),
        "constraint_native_orientation": str((repo / "src/ocrap/audits/constraint_native_orientation.py").resolve()),
        "heterogeneous_constraint_normal_cone": str((repo / "src/ocrap/audits/heterogeneous_constraint_normal_cone.py").resolve()),
        "executable_constraint_jacobian": str((repo / "src/ocrap/audits/executable_constraint_jacobian.py").resolve()),
        "common_option_constraint_work": str((repo / "src/ocrap/audits/common_option_constraint_work.py").resolve()),
        "recovery_set_constraint_flow": str((repo / "src/ocrap/audits/recovery_set_constraint_flow.py").resolve()),
        "weak_root_recovery_set_flow": str((repo / "src/ocrap/audits/weak_root_recovery_set_flow.py").resolve()),
    }
    for k, v in imported.items():
        if v != expected[k]:
            errors.append(f"import_path:{k}:{v}")

    checks = wrcf.contract_checks()
    for k, v in checks.items():
        if not v:
            errors.append(f"synthetic:{k}")

    out = {
        "schema": "ocrap-v48.116-wrcf-runtime-code-contract-v1",
        "engineering_version": wrcf.ENGINEERING_VERSION,
        "scientific_version": wrcf.SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "runtime_files": files,
        "imported_modules": imported,
        "expected_imported_modules": expected,
        "code_layout": "unversioned_semantic_modules",
        "historical_code_dependency": False,
        "scientific_contract": {
            "audit_only": True,
            "weak_root_cotangent_recovery_set_flow": True,
            "tail_measure_source": "frozen_nominal_native_model_root_logits_margins_and_observation_compatibility",
            "tail_measure_candidate_independent": True,
            "tail_measure_teacher_value_free": True,
            "frozen_root_validity_mask_used": True,
            "frozen_root_decoder_read_only": True,
            "frozen_margin_head_read_only": True,
            "root_decoder_parameters_trained": 0,
            "constraint_names": ["clearance", "stopping", "route", "reentry"],
            "same_option_inside_each_weighted_summand": True,
            "actuator_projection": True,
            "existing_recovery_horizon_only": True,
            "work_bins": 8,
            "work_bins_cover_full_horizon": True,
            "integral_channels": ["bin_mean_delta_h", "bin_mean_delta_h_times_h0"],
            "constraint_work_channels": ["positive_reserve_work", "negative_debt_repayment_work"],
            "work_conservation_identity": "tail_weighted_reserve_work_plus_debt_work_equals_tail_weighted_bin_delta_h",
            "option_aggregation": "nominal_ocmero_nested_lcvar_cotangent_pushforward_over_recovery_options",
            "model_physical_option_alignment": "raw_physical_prefix_plus_invalid_checkpoint_padding_only",
            "padded_model_options_must_be_invalid_and_zero_tail_mass": True,
            "pre_readout_candidate_option_selector": False,
            "downstream_ocmero_option_selection_unchanged": True,
            "option_permutation_invariant": True,
            "tail_geometry_dim": wrcf.TAIL_GEOMETRY_DIM,
            "matched_family_dim": wrcf.MATCHED_DIM,
            "capacity_matched_all_families": True,
            "candidate_identity_shuffle": "whole_feature_row_cyclic_permutation_within_scene_time_group",
            "convex_closed_form_ridge": True,
            "strictly_convex_unique_solution": True,
            "ridge_lambda_rule": "1_over_axis_train_rows",
            "teacher_npz_fields_loaded_into_feature_path": ["root_valid"],
            "teacher_margin_probability_compatibility_fields_used": False,
            "teacher_future_fields_used": False,
            "teacher_metadata_input_to_model": False,
            "iterative_optimizer_used": False,
            "posthoc_feature_selection": False,
            "lr_or_epoch_sweep": False,
            "threshold_sweep": False,
            "capacity_sweep": False,
            "horizon_sweep": False,
            "option_count_sweep": False,
            "planner_parameters_trained": 0,
            "stage_i_parameters_trained": 0,
            "source_parameters_trained": 0,
            "relative_ranker_modified": False,
            "boundary_transport": False,
            "regime_conditioning": False,
        },
        "synthetic_checks": checks,
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "attribution_ready": out["attribution_ready"], "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
