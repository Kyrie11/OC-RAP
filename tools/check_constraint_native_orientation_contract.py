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

    for rel in ACTIVE:
        p = (repo / rel).resolve()
        ok = p.is_file() and str(p).startswith(str(repo))
        files[rel] = {
            "exists": p.is_file(),
            "inside_repo": str(p).startswith(str(repo)),
            "path": str(p),
            "sha256": sha(p) if p.is_file() else None,
        }
        if not ok:
            errors.append(f"runtime_file:{rel}")

    imported = {
        "ocrap": str(Path(ocrap.__file__).resolve()),
        "constraint_native_orientation": str(Path(base_primitives.__file__).resolve()),
        "heterogeneous_constraint_normal_cone": str(Path(hcnc.__file__).resolve()),
        "executable_constraint_jacobian": str(Path(ecj.__file__).resolve()),
    }
    expected = {
        "ocrap": str((repo / "src/ocrap/__init__.py").resolve()),
        "constraint_native_orientation": str((repo / "src/ocrap/audits/constraint_native_orientation.py").resolve()),
        "heterogeneous_constraint_normal_cone": str((repo / "src/ocrap/audits/heterogeneous_constraint_normal_cone.py").resolve()),
        "executable_constraint_jacobian": str((repo / "src/ocrap/audits/executable_constraint_jacobian.py").resolve()),
    }
    for k, v in imported.items():
        if v != expected[k]:
            errors.append(f"import_path:{k}:{v}")

    checks = ecj.contract_checks()
    for k, v in checks.items():
        if not v:
            errors.append(f"synthetic:{k}")

    out = {
        "schema": "ocrap-v48.113-ecj-runtime-code-contract-v1",
        "engineering_version": ecj.ENGINEERING_VERSION,
        "scientific_version": ecj.SCIENTIFIC_VERSION,
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
            "candidate_option_executable_constraint_jacobian": True,
            "constraint_names": ["clearance", "stopping", "route", "reentry"],
            "constraint_response": "same_option_actuator_projected_candidate_minus_nominal_signed_constraint_path",
            "actuator_projection": True,
            "recovery_knots": ecj.RECOVERY_KNOTS,
            "existing_recovery_horizon_only": True,
            "nominal_option_selector": "nominal_maximin_over_full_executable_recovery_constraint_path",
            "candidate_option_selector": "candidate_maximin_over_full_executable_recovery_constraint_path",
            "jacobian_geometry_dim": ecj.JACOBIAN_GEOMETRY_DIM,
            "matched_family_dim": ecj.MATCHED_DIM,
            "capacity_matched_nominal_vs_candidate_option": True,
            "candidate_identity_shuffle": "whole_feature_row_cyclic_permutation_within_scene_time_group",
            "convex_closed_form_ridge": True,
            "strictly_convex_unique_solution": True,
            "ridge_lambda_rule": "1_over_axis_train_rows",
            "teacher_npz_fields_loaded_into_feature_path": False,
            "teacher_metadata_input_to_model": False,
            "iterative_optimizer_used": False,
            "posthoc_feature_selection": False,
            "lr_or_epoch_sweep": False,
            "threshold_sweep": False,
            "capacity_sweep": False,
            "planner_parameters_trained": 0,
            "stage_i_parameters_trained": 0,
            "root_decoder_parameters_trained": 0,
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
