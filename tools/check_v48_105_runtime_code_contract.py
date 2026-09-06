#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.v48_105_prelast_action_equivariance_localization import (
    ENGINEERING_VERSION,
    action_interaction_dimension_check,
    agent_permutation_invariance_check,
    candidate_zero_delta_check,
    summary_partition_check,
)

ACTIVE = [
    "scripts/run_v48_105_dcp_drfc_bcde_rifa_pael_two_gpu.sh",
    "src/ocrap/v48_105_prelast_action_equivariance_localization.py",
    "tools/run_v48_105_prelast_action_equivariance_localization_audit.py",
    "tools/compare_v48_105_pael.py",
    "tools/check_v48_105_runtime_code_contract.py",
    "tools/check_v48_105_pipeline_complete.py",
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
    files = {}
    for rel in ACTIVE:
        p = (repo / rel).resolve()
        inside = str(p).startswith(str(repo))
        files[rel] = {"exists": p.is_file(), "inside_repo": inside, "path": str(p), "sha256": sha(p) if p.is_file() else None}
        if not (p.is_file() and inside):
            errors.append(f"runtime_file:{rel}")

    checks = {
        "summary_partition_15d": summary_partition_check(16),
        "candidate_nominal_zero_delta": candidate_zero_delta_check(16),
        "agent_set_permutation_invariant": agent_permutation_invariance_check(16),
        "action_interaction_dimension_1920_at_d192": action_interaction_dimension_check(192),
    }
    for k, v in checks.items():
        if not v:
            errors.append(k)

    out = {
        "schema": "ocrap-v48.105-pael-runtime-code-contract-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "runtime_files": files,
        "scientific_contract": {
            "audit_only": True,
            "prelast_stage_i_tokens_only": True,
            "same_v48_102_summary_operator": True,
            "same_v48_102_linear_probe_recipe": True,
            "semantic_token_positions_preserved": True,
            "agent_set_summary_permutation_invariant": True,
            "action_interaction_subspace_fixed_a_priori": True,
            "action_interaction_definition": "control_plus_scene_context_plus_agent_set_moments_excluding_cls_and_ego_history",
            "token_localization_groups_fixed_a_priori": ["cls", "ego_history", "control", "scene_context", "agents"],
            "within_group_action_permutation_control": True,
            "stage_i_parameters_trained": 0,
            "root_decoder_parameters_trained": 0,
            "source_parameters_trained": 0,
            "planner_parameters_trained": 0,
            "boundary_transport": False,
            "broad_encoder_training": False,
            "regime_conditioning": False,
            "teacher_metadata_input_to_model": False,
            "capacity_sweep": False,
            "threshold_sweep": False,
        },
        "synthetic_checks": checks,
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
