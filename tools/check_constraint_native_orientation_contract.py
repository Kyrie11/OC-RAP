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
    "src/ocrap/audits/tail_boundary_crossing_flow.py",
    "src/ocrap/audits/viability_survival_envelope.py",
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
    import ocrap.audits.tail_boundary_crossing_flow as tbcf
    import ocrap.audits.viability_survival_envelope as vse

    for rel in ACTIVE:
        p = (repo / rel).resolve()
        inside = str(p).startswith(str(repo))
        ok = p.is_file() and inside
        files[rel] = {"exists": p.is_file(), "inside_repo": inside, "path": str(p), "sha256": sha(p) if p.is_file() else None}
        if not ok:
            errors.append(f"runtime_file:{rel}")

    modules = {
        "ocrap": ocrap,
        "constraint_native_orientation": base_primitives,
        "heterogeneous_constraint_normal_cone": hcnc,
        "executable_constraint_jacobian": ecj,
        "common_option_constraint_work": ccw,
        "recovery_set_constraint_flow": rscf,
        "weak_root_recovery_set_flow": wrcf,
        "tail_boundary_crossing_flow": tbcf,
        "viability_survival_envelope": vse,
    }
    imported = {k: str(Path(v.__file__).resolve()) for k, v in modules.items()}
    expected = {
        "ocrap": str((repo / "src/ocrap/__init__.py").resolve()),
        "constraint_native_orientation": str((repo / "src/ocrap/audits/constraint_native_orientation.py").resolve()),
        "heterogeneous_constraint_normal_cone": str((repo / "src/ocrap/audits/heterogeneous_constraint_normal_cone.py").resolve()),
        "executable_constraint_jacobian": str((repo / "src/ocrap/audits/executable_constraint_jacobian.py").resolve()),
        "common_option_constraint_work": str((repo / "src/ocrap/audits/common_option_constraint_work.py").resolve()),
        "recovery_set_constraint_flow": str((repo / "src/ocrap/audits/recovery_set_constraint_flow.py").resolve()),
        "weak_root_recovery_set_flow": str((repo / "src/ocrap/audits/weak_root_recovery_set_flow.py").resolve()),
        "tail_boundary_crossing_flow": str((repo / "src/ocrap/audits/tail_boundary_crossing_flow.py").resolve()),
        "viability_survival_envelope": str((repo / "src/ocrap/audits/viability_survival_envelope.py").resolve()),
    }
    for k, v in imported.items():
        if v != expected[k]:
            errors.append(f"import_path:{k}:{v}")

    checks = vse.contract_checks()
    for k, v in checks.items():
        if not v:
            errors.append(f"synthetic:{k}")

    out = {
        "schema": "ocrap-v48.118-vse-runtime-code-contract-v1",
        "engineering_version": vse.ENGINEERING_VERSION,
        "scientific_version": vse.SCIENTIFIC_VERSION,
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
            "viability_survival_envelope": True,
            "primary_option_set": "all_common_valid_recovery_options",
            "control_option_set": "support_of_frozen_v48_117_weak_root_zero_boundary_witnesses",
            "boundary_support_weights_used": False,
            "envelope_definition": "max_option_min_constraint_running_signed_margin",
            "envelope_channels": ["signed_joint_prefix_viability_envelope", "signed_joint_suffix_persistent_reentry_envelope"],
            "set_envelope_option_identity_may_switch_over_time": True,
            "active_option_identity_exported": False,
            "zero_boundary_threshold": 0.0,
            "zero_boundary_threshold_sweep": False,
            "frozen_root_validity_mask_used": True,
            "frozen_root_decoder_read_only": True,
            "frozen_margin_head_read_only": True,
            "root_decoder_parameters_trained": 0,
            "constraint_names": ["clearance", "stopping", "route", "reentry"],
            "actuator_projection": True,
            "existing_recovery_horizon_only": True,
            "work_bins": 8,
            "work_bins_cover_full_horizon": True,
            "option_aggregation": "permutation_invariant_max_over_common_valid_recovery_options_after_joint_constraint_min",
            "model_physical_option_alignment": "raw_physical_prefix_plus_invalid_checkpoint_padding_only",
            "padded_model_options_must_be_invalid_and_zero_boundary_witness_mass": True,
            "pre_readout_candidate_option_selector": False,
            "downstream_ocmero_option_selection_unchanged": True,
            "option_permutation_invariant": True,
            "envelope_geometry_dim": vse.ENVELOPE_GEOMETRY_DIM,
            "matched_family_dim": vse.MATCHED_DIM,
            "capacity_matched_all_families": True,
            "families": ["base", "exposed_envelope", "full_envelope"],
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
