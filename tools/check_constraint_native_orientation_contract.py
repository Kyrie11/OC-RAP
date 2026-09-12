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
    "src/ocrap/audits/viability_rank_transport.py",
    "tools/run_constraint_native_recovery_orientation_audit.py",
    "tools/compare_constraint_native_recovery_orientation.py",
    "tools/check_constraint_native_orientation_contract.py",
    "tools/check_constraint_native_orientation_pipeline.py",
    "tools/package_constraint_native_orientation_results.py",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    repo = a.repo.resolve()
    errors: list[str] = []
    files: dict[str, object] = {}
    src = str((repo / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)

    import ocrap
    import ocrap.audits.constraint_native_orientation as base
    import ocrap.audits.heterogeneous_constraint_normal_cone as hcnc
    import ocrap.audits.executable_constraint_jacobian as ecj
    import ocrap.audits.common_option_constraint_work as ccw
    import ocrap.audits.recovery_set_constraint_flow as rscf
    import ocrap.audits.weak_root_recovery_set_flow as wrcf
    import ocrap.audits.tail_boundary_crossing_flow as tbcf
    import ocrap.audits.viability_rank_transport as vrt

    for rel in ACTIVE:
        p = (repo / rel).resolve()
        inside = str(p).startswith(str(repo))
        ok = p.is_file() and inside
        files[rel] = {
            "exists": p.is_file(), "inside_repo": inside, "path": str(p),
            "sha256": sha(p) if p.is_file() else None,
        }
        if not ok:
            errors.append(f"runtime_file:{rel}")

    modules = {
        "ocrap": ocrap,
        "constraint_native_orientation": base,
        "heterogeneous_constraint_normal_cone": hcnc,
        "executable_constraint_jacobian": ecj,
        "common_option_constraint_work": ccw,
        "recovery_set_constraint_flow": rscf,
        "weak_root_recovery_set_flow": wrcf,
        "tail_boundary_crossing_flow": tbcf,
        "viability_rank_transport": vrt,
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
        "viability_rank_transport": str((repo / "src/ocrap/audits/viability_rank_transport.py").resolve()),
    }
    for k, want in expected.items():
        if imported.get(k) != want:
            errors.append(f"import_path:{k}")

    checks = vrt.contract_checks()
    if not checks or not all(bool(v) for v in checks.values()):
        errors.append("synthetic_contract")

    sc = {
        "audit_only": True,
        "viability_rank_transport": True,
        "boundary_transport": False,
        "candidate_identity_shuffle": "whole_feature_row_cyclic_permutation_within_scene_time_group",
        "capacity_matched_all_families": True,
        "capacity_sweep": False,
        "horizon_sweep": False,
        "threshold_sweep": False,
        "option_count_sweep": False,
        "rank_cut_sweep": False,
        "lr_or_epoch_sweep": False,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "transport_channels": ["joint_prefix_viability_rank_transport", "joint_suffix_persistent_reentry_rank_transport"],
        "transport_mode_degrees": [int(x) for x in vrt.TRANSPORT_MODE_DEGREES],
        "transport_basis": "exact_shifted_legendre_degree_0_to_3_on_nominal_rank_intervals",
        "rank_coordinate": "candidate_independent_nominal_same_option_viability_order",
        "candidate_rank_sort_used_for_coordinate": False,
        "option_aggregation": "same_option_signed_margin_displacement_projected_on_candidate_independent_nominal_rank_basis",
        "primary_option_set": "all_common_valid_recovery_options",
        "control_option_set": "support_of_frozen_v48_117_weak_root_zero_boundary_witnesses",
        "boundary_support_weights_used_in_transport": False,
        "active_option_identity_exported": False,
        "pre_readout_candidate_option_selector": False,
        "frozen_root_decoder_read_only": True,
        "frozen_margin_head_read_only": True,
        "frozen_root_validity_mask_used": True,
        "teacher_future_fields_used": False,
        "teacher_margin_probability_compatibility_fields_used": False,
        "teacher_metadata_input_to_model": False,
        "teacher_npz_fields_loaded_into_feature_path": ["root_valid"],
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "source_parameters_trained": 0,
        "posthoc_feature_selection": False,
        "regime_conditioning": False,
        "relative_ranker_modified": False,
        "convex_closed_form_ridge": True,
        "strictly_convex_unique_solution": True,
        "iterative_optimizer_used": False,
        "ridge_lambda_rule": "1_over_axis_train_rows",
        "matched_family_dim": vrt.MATCHED_DIM,
        "transport_geometry_dim": vrt.TRANSPORT_GEOMETRY_DIM,
        "work_bins": 8,
        "work_bins_cover_full_horizon": True,
        "existing_recovery_horizon_only": True,
        "zero_boundary_threshold": 0.0,
        "zero_boundary_threshold_sweep": False,
        "option_permutation_invariant": True,
        "same_option_joint_constraint_viability": True,
        "same_option_nominal_rank_correspondence": True,
        "downstream_ocmero_option_selection_unchanged": True,
    }
    out = {
        "schema": "ocrap-v48.120-vrt-runtime-code-contract-v1",
        "engineering_version": vrt.ENGINEERING_VERSION,
        "scientific_version": vrt.SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "code_layout": "unversioned_semantic_modules",
        "historical_code_dependency": False,
        "runtime_files": files,
        "imported_modules": imported,
        "expected_imported_modules": expected,
        "scientific_contract": sc,
        "synthetic_checks": checks,
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "attribution_ready": out["attribution_ready"], "errors": errors, "synthetic_checks": checks}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
