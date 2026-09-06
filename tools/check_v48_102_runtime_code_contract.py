#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.v48_102_action_information_transport_sufficiency import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    agent_permutation_invariance_check,
    candidate_zero_delta_check,
    semantic_position_sensitivity_check,
    stage_i_memory_summary,
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    repo = a.repo.resolve()
    errors: list[str] = []
    files = [
        "scripts/run_v48_102_dcp_drfc_bcde_rifa_aits_two_gpu.sh",
        "src/ocrap/v48_102_action_information_transport_sufficiency.py",
        "tools/run_v48_102_stage_i_action_information_transport_audit.py",
        "tools/compare_v48_102_aits.py",
        "tools/check_v48_102_runtime_code_contract.py",
        "tools/check_v48_102_pipeline_complete.py",
    ]
    mods = {}
    for rel in files:
        p = repo / rel
        if not p.is_file():
            errors.append(f"missing {rel}")
            continue
        inside = str(p.resolve()).startswith(str(repo))
        if not inside:
            errors.append(f"outside repo {rel}")
        mods[rel] = {"exists": True, "inside_repo": inside, "path": str(p.resolve()), "sha256": sha(p)}

    try:
        import torch
        x = torch.randn(4, 43, 192)
        dim = int(stage_i_memory_summary(x, semantic_token_count=11).shape[-1])
        if dim != 2880:
            errors.append(f"stage_i_summary_dim:{dim}")
    except Exception as exc:
        errors.append(f"stage_i_summary_exception:{exc!r}")
    if not candidate_zero_delta_check():
        errors.append("candidate_zero_delta_check")
    if not agent_permutation_invariance_check():
        errors.append("agent_permutation_invariance_check")
    if not semantic_position_sensitivity_check():
        errors.append("semantic_position_sensitivity_check")

    # Resolve imported model modules and prove the audit is using this repository.
    imported = {}
    try:
        import ocrap.models.data as data_mod
        import ocrap.models.inference as inference_mod
        import ocrap.models.ocrap as model_mod
        for name, mod in (("ocrap.models.data", data_mod), ("ocrap.models.inference", inference_mod), ("ocrap.models.ocrap", model_mod)):
            p = Path(mod.__file__).resolve()
            inside = str(p).startswith(str(repo))
            if not inside:
                errors.append(f"import outside repo {name}:{p}")
            imported[name] = {"path": str(p), "inside_repo": inside, "sha256": sha(p)}
    except Exception as exc:
        errors.append(f"import_provenance_exception:{exc!r}")

    doc = {
        "schema": "ocrap-v48.102-runtime-code-contract-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "runtime_files": mods,
        "imported_module_provenance": imported,
        "scientific_contract": {
            "audit_only": True,
            "planner_parameters_trained": 0,
            "stage_i_parameters_trained": 0,
            "root_decoder_parameters_trained": 0,
            "source_parameters_trained": 0,
            "same_v48_93_target_specific_semantics": True,
            "fixed_linear_probe_capacity": True,
            "within_group_action_permutation_control": True,
            "stage_i_summary_fixed_semantic_positions": True,
            "agent_set_summary_permutation_invariant": True,
            "capacity_sweep": False,
            "threshold_sweep": False,
            "dataset_reconstruction": False,
            "dataset_reselection": False,
            "teacher_metadata_input_to_model": False,
            "boundary_transport": False,
            "relative_ranker_modified": False,
            "regime_conditioning": False,
        },
        "synthetic_checks": {
            "stage_i_summary_dim_2880": not any(x.startswith("stage_i_summary") for x in errors),
            "candidate_nominal_zero_delta": not any("candidate_zero_delta" in x for x in errors),
            "agent_set_permutation_invariant": not any("agent_permutation" in x for x in errors),
            "semantic_token_positions_preserved": not any("semantic_position" in x for x in errors),
        },
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": doc["valid"], "errors": errors}))
    return 0 if doc["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
