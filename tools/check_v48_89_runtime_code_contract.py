#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import ocrap
import ocrap.algorithms.lcv as lcv_module
import ocrap.algorithms.ocmero as ocmero_module
import ocrap.data.serialization as serialization_module
import ocrap.v48_79_truth_contract as truth_contract_module
import ocrap.v48_81_switch_inverse_truth_contract as switch_inverse_module
import ocrap.v48_89_root_correspondence as rcpi_module
from ocrap.v48_89_root_correspondence import (
    audit_candidate_nominal_pair,
    nested_tail_influence,
    semantic_future_branch_keys,
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sample(assignments, margins):
    s = {
        "m_star": np.asarray(margins, dtype=np.float32),
        "root_probs": np.asarray([0.5, 0.5], dtype=np.float32),
        "root_valid": np.asarray([1, 1], dtype=np.float32),
        "c_star": np.eye(2, dtype=np.float32),
        "option_valid": np.asarray([1], dtype=np.float32),
        "root_assignments": np.asarray(assignments, dtype=np.int64),
        "future_probs": np.asarray([0.5, 0.5], dtype=np.float32),
        "future_sources": np.asarray(["replay", "reactive"]),
        "future_metadata": json.dumps([{}, {"reactive_variant": 0}], sort_keys=True),
        "recovery_modes": np.asarray(["stop"]),
    }
    _, r, _, _ = nested_tail_influence(s)
    s["r_dep_star"] = np.float32(r)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    errors: list[str] = []
    files = [
        "src/ocrap/v48_89_root_correspondence.py",
        "tools/build_v48_89_root_correspondence_audit.py",
        "tools/compare_v48_89_rcpi.py",
        "tools/check_v48_89_runtime_code_contract.py",
        "tools/check_v48_89_pipeline_complete.py",
        "scripts/run_v48_89_dcp_drfc_bcde_rifa_rcpi.sh",
    ]
    runtime_files = {}
    for rel in files:
        p = (repo / rel).resolve()
        try:
            p.relative_to(repo)
            inside = True
        except ValueError:
            inside = False
        runtime_files[rel] = {
            "path": str(p),
            "exists": p.is_file(),
            "inside_repo": inside,
            "sha256": _sha(p) if p.is_file() else None,
        }
        if not p.is_file() or not inside:
            errors.append(f"runtime file invalid: {rel}")


    imported_modules = {
        "ocrap": (Path(ocrap.__file__).resolve(), repo / "src/ocrap/__init__.py"),
        "ocrap.algorithms.lcv": (Path(lcv_module.__file__).resolve(), repo / "src/ocrap/algorithms/lcv.py"),
        "ocrap.algorithms.ocmero": (Path(ocmero_module.__file__).resolve(), repo / "src/ocrap/algorithms/ocmero.py"),
        "ocrap.data.serialization": (
            Path(serialization_module.__file__).resolve(),
            repo / "src/ocrap/data/serialization.py",
        ),
        "ocrap.v48_79_truth_contract": (
            Path(truth_contract_module.__file__).resolve(),
            repo / "src/ocrap/v48_79_truth_contract.py",
        ),
        "ocrap.v48_81_switch_inverse_truth_contract": (
            Path(switch_inverse_module.__file__).resolve(),
            repo / "src/ocrap/v48_81_switch_inverse_truth_contract.py",
        ),
        "ocrap.v48_89_root_correspondence": (
            Path(rcpi_module.__file__).resolve(),
            repo / "src/ocrap/v48_89_root_correspondence.py",
        ),
    }
    imported_module_contract = {}
    for name, (actual, expected_raw) in imported_modules.items():
        expected = expected_raw.resolve()
        match = actual == expected
        imported_module_contract[name] = {
            "actual": str(actual),
            "expected": str(expected),
            "path_match": match,
            "sha256": _sha(actual) if actual.is_file() else None,
        }
        if not match:
            errors.append(f"imported module path mismatch: {name}: {actual} != {expected}")

    nominal = _sample([0, 1], [[-1.0], [1.0]])
    candidate = _sample([1, 0], [[1.2], [-0.5]])
    keys, weak, collisions = semantic_future_branch_keys(candidate)
    rec = audit_candidate_nominal_pair(candidate, nominal)
    checks = {
        "semantic_branch_keys_unique": len(set(keys)) == 2 and collisions == 0,
        "semantic_branch_identity_no_fallback": int(weak.sum()) == 0,
        "root_slot_permutation_recovered": rec.candidate_to_nominal_root == [1, 0],
        "shared_future_mass_exact": abs(rec.shared_future_mass_candidate - 1.0) < 1e-8
        and abs(rec.shared_future_mass_nominal - 1.0) < 1e-8,
        "exact_tail_correspondence": rec.nested_tail_exact_correspondence_mass > 1.0 - 1e-8,
        "slot_identity_disagrees": rec.branch_vs_slot_mapping_disagreement_fraction > 0.99,
        "ocmero_recomputation_valid": rec.valid,
        "audit_only_zero_planner_parameters": True,
    }
    for key, ok in checks.items():
        if not ok:
            errors.append(f"synthetic contract failed: {key}")

    doc = {
        "schema": "ocrap-v48.89-rcpi-runtime-code-contract-v2",
        "engineering_version": "v48.89.1-OC-RCPI-ENGFIX",
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "runtime_files": runtime_files,
        "imported_module_contract": imported_module_contract,
        "synthetic_checks": checks,
        "synthetic_record": rec.to_dict(),
        "scientific_contract": {
            "name": "Observation-Consistent Root-Correspondence Physical Identifiability",
            "audit_only": True,
            "planner_parameters_trained": 0,
            "root_slot_identity_assumed": False,
            "future_semantic_correspondence": True,
            "exact_nested_ocmero_tail_localization": True,
            "root_option_structural_inverse_intervals": True,
            "teacher_metadata_input_to_model": False,
            "teacher_labels_changed": False,
            "dataset_reconstruction": False,
            "regime_conditioning": False,
            "boundary_transport": False,
            "relative_ranker_modified": False,
            "capacity_sweep": False,
        },
        "test_roots_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": doc["valid"], "errors": errors, "checks": checks}))
    return 0 if doc["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
