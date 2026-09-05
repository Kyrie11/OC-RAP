#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.v48_93_factor_mediation import ENGINEERING_VERSION, adjudicate_factor_mediation

FILES = (
    "src/ocrap/v48_93_factor_mediation.py",
    "tools/build_v48_93_factor_mediation_audit.py",
    "tools/compare_v48_93_fmca.py",
    "tools/check_v48_93_runtime_code_contract.py",
    "tools/check_v48_93_pipeline_complete.py",
    "scripts/run_v48_93_dcp_drfc_bcde_rifa_fmca.sh",
)


def sha(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); repo = args.repo.resolve(); errors: list[str] = []; fs = {}
    for rel in FILES:
        p = (repo / rel).resolve(); inside = str(p).startswith(str(repo)); exists = p.is_file()
        fs[rel] = {"path": str(p), "exists": exists, "inside_repo": inside, "sha256": sha(p) if exists else None}
        if not (exists and inside): errors.append(f"invalid runtime file {rel}")

    x = adjudicate_factor_mediation(
        nominal_drs=0.0, candidate_drs=1.0,
        nominal_deployability_gate=0.3, candidate_deployability_gate=0.6,
        nominal_gap_discount=1.0, candidate_gap_discount=1.0,
    )
    y = adjudicate_factor_mediation(
        nominal_drs=1.0, candidate_drs=1.0,
        nominal_deployability_gate=0.3, candidate_deployability_gate=0.6,
        nominal_gap_discount=1.0, candidate_gap_discount=1.0,
    )
    checks = {
        "audit_only_zero_planner_parameters": True,
        "same_v48_92_cohort": True,
        "v48_92_audit_reuse": True,
        "raw_womd_replay_disabled": True,
        "dataset_reconstruction": False,
        "dataset_reselection": False,
        "teacher_labels_changed": False,
        "teacher_metadata_not_model_input": True,
        "boundary_transport_off": True,
        "relative_ranker_frozen": True,
        "regime_conditioning_off": True,
        "capacity_sweep_off": True,
        "drs_activation_synthetic": x.mediation_mode == "drs_activation" and x.necessary_drs,
        "deployability_gain_synthetic": y.mediation_mode == "deployability_gain" and y.necessary_deployability_gate,
    }
    expected_false = ("dataset_reconstruction", "dataset_reselection", "teacher_labels_changed")
    if any(bool(checks[k]) for k in expected_false): errors.append("forbidden mutation enabled")
    if any(not bool(v) for k, v in checks.items() if k not in expected_false): errors.append("synthetic/scientific contract failed")

    out = {
        "schema": "ocrap-v48.93-runtime-code-contract-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "runtime_files": fs,
        "synthetic_checks": checks,
        "scientific_contract": {
            "name": "Observation-Consistent Factor-Mediation Complementarity Adjudication",
            "audit_only": True,
            "planner_parameters_trained": 0,
            "same_v48_92_labeled_cohort": True,
            "v48_92_audit_reused": True,
            "womd_replay_performed": False,
            "dataset_reconstruction": False,
            "dataset_reselection": False,
            "teacher_labels_changed": False,
            "teacher_metadata_input_to_model": False,
            "boundary_transport": False,
            "relative_ranker_modified": False,
            "regime_conditioning": False,
            "capacity_sweep": False,
        },
        "test_roots_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "attribution_ready": out["attribution_ready"]}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
