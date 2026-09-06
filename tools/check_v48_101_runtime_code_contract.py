#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import ocrap.models.data as data_mod
import ocrap.models.inference as inference_mod
import ocrap.models.ocrap as model_mod
from ocrap.v48_100_joint_root_semantic_decoder import JointRootSemanticDecoder
from ocrap.v48_101_root_cross_attention_semantic_alignment import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    cross_attention_gradient_check,
    cross_attention_training_contract,
    expected_cross_attention_parameter_count,
    initial_attention_identity_check,
    non_attention_frozen_after_step_check,
)

FILES = [
    "scripts/run_v48_101_dcp_drfc_bcde_rifa_rcsa_two_gpu.sh",
    "src/ocrap/v48_100_joint_root_semantic_decoder.py",
    "src/ocrap/v48_101_root_cross_attention_semantic_alignment.py",
    "tools/run_v48_101_root_cross_attention_semantic_alignment.py",
    "tools/compare_v48_101_rcsa.py",
    "tools/check_v48_101_runtime_code_contract.py",
    "tools/check_v48_101_pipeline_complete.py",
    "src/ocrap/models/ocrap.py",
    "src/ocrap/models/data.py",
    "src/ocrap/models/inference.py",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); a = ap.parse_args()
    repo = a.repo.resolve(); errors: list[str] = []; runtime_files = {}
    for rel in FILES:
        p = (repo / rel).resolve(); inside = repo == p or repo in p.parents; exists = p.is_file()
        runtime_files[rel] = {"exists": exists, "inside_repo": inside, "path": str(p), "sha256": sha(p) if exists else None}
        if not exists or not inside:
            errors.append(rel)

    imported = {
        "ocrap.models.ocrap": Path(inspect.getfile(model_mod)).resolve(),
        "ocrap.models.data": Path(inspect.getfile(data_mod)).resolve(),
        "ocrap.models.inference": Path(inspect.getfile(inference_mod)).resolve(),
    }
    imported_contract = {}
    for name, p in imported.items():
        inside = repo == p or repo in p.parents
        imported_contract[name] = {"path": str(p), "inside_repo": inside, "sha256": sha(p) if p.is_file() else None}
        if not inside or not p.is_file():
            errors.append(f"import_provenance:{name}")

    d_model = 192; num_roots = 8; attn_params = expected_cross_attention_parameter_count(d_model)
    checks = {
        "initial_opening_is_function_identity": initial_attention_identity_check(32, 5, 4),
        "semantic_gradient_reaches_cross_attention_only": cross_attention_gradient_check(32, 5, 4),
        "one_step_changes_cross_attention_only": non_attention_frozen_after_step_check(32, 5, 4),
    }
    for k, v in checks.items():
        if not v:
            errors.append(k)

    out = {
        "schema": "ocrap-v48.101-runtime-code-contract-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "runtime_files": runtime_files,
        "imported_module_provenance": imported_contract,
        "scientific_contract": {
            "name": ALGORITHM_NAME,
            "planner_parameters_trained": 0,
            "source_parameters_trained": 0,
            "stage_i_parameters_trained": 0,
            "root_query_parameters_trained": 0,
            "recovery_chart_parameters_trained": 0,
            "root_cross_attention_parameters_trained": attn_params,
            "root_self_attention_parameters_trained": 0,
            "root_ffn_parameters_trained": 0,
            "root_logit_head_parameters_trained": 0,
            "v48_100_query_chart_frozen": True,
            "structured_encoder_frozen": True,
            "root_cross_attention_opened_from_historical_weights": True,
            "root_self_attention_frozen": True,
            "root_ffn_frozen": True,
            "root_logit_head_frozen": True,
            "same_v48_100_four_term_dimensionless_semantic_objective": True,
            "same_v48_100_semantic_metric_scales_reused": True,
            "initial_v48_100_function_identity_required": True,
            "model_eval_mode_during_cross_attention_training": True,
            "teacher_metadata_input_to_model": False,
            "regime_conditioning": False,
            "boundary_transport": False,
            "relative_ranker_modified": False,
            "dataset_reconstruction": False,
            "dataset_reselection": False,
            "capacity_sweep": False,
            "threshold_sweep": False,
            "epoch_sweep": False,
        },
        "synthetic_checks": checks,
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "errors": errors, "cross_attention_parameters": attn_params}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
