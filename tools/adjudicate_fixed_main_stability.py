#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.audits.fixed_main_stability import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    SCIENTIFIC_VERSION,
    adjudicate,
)

AUTHORITATIVE_V123_PIPELINE_SHA256 = "cd963f76b508f19dbb5140d36d94f76ab815669ad36ec3bbae8431a7e097fe81"
AUTHORITATIVE_V123_COMPARISON_SHA256 = "cb850cbf24fd1c27f00f56805bcb919e2c15267a365ac045dccebed7ddcba067"
AUTHORITATIVE_V123_BALANCED_SHA256 = "0d4f1281f4278334f2b63fe6893a61e3445a0385baba5c62d755de48a90c9837"
AUTHORITATIVE_V123_PRECISION_SHA256 = "8e76846b56f90bafff02110cb31e58ca4f4eacf5076a7edeadd37e0914b47538"
V123_STATUS = "ZERO_BOUNDARY_VIABILITY_STATE_TRANSITION_STOP"
V123_NEXT = (
    "close_zero_boundary_viability_state_transition_then_freeze_recovery_set_mechanism_family_"
    "and_preregister_fixed_main_stability_noninterference_adjudication_"
    "no_new_recovery_mechanism_capacity_regime_source_horizon_or_threshold_sweep"
)

FROZEN_FULL_RUN_CORE_SHA256 = {
    "src/ocrap/planning/selector.py": "a8c035b4ae3d182620755c00013cb42814c150b601893ea0ec13c9d486a6f38c",
    "src/ocrap/simulation/closed_loop_runner.py": "a95285c176c0edec79930fa353fd9939286dfaa65310cc1309b5b791060787f9",
    "src/ocrap/simulation/waymax_rollout.py": "cb1b0b3693a883f740df838e63f811929805fdc3c2337b1de9cae3c9ecea7648",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha(path), "size": path.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser(description="V48.124 fixed-Main stability/non-interference adjudication.")
    for regime in ("safe", "near", "contact"):
        for variant in ("nominal", "balanced", "precision"):
            ap.add_argument(f"--{variant}-{regime}", type=Path, required=True)
        for variant in ("balanced", "precision"):
            ap.add_argument(f"--{variant}-{regime}-comparison", type=Path, required=True)
            ap.add_argument(f"--{variant}-{regime}-sentinel", type=Path, required=True)
    ap.add_argument("--sentinel-index", type=Path, required=True)
    ap.add_argument("--balanced-checkpoint", type=Path, required=True)
    ap.add_argument("--precision-checkpoint", type=Path, required=True)
    ap.add_argument("--balanced-calibration", type=Path, required=True)
    ap.add_argument("--precision-calibration", type=Path, required=True)
    ap.add_argument("--v123-pipeline", type=Path, required=True)
    ap.add_argument("--v123-comparison", type=Path, required=True)
    ap.add_argument("--v123-balanced", type=Path, required=True)
    ap.add_argument("--v123-precision", type=Path, required=True)
    ap.add_argument("--full-run-runtime", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    errors: list[str] = []
    # Frozen prerequisite is checked here as well as in the launcher/pipeline so
    # an adjudication JSON can never be detached from the decision that licensed it.
    expected = {
        a.v123_pipeline: AUTHORITATIVE_V123_PIPELINE_SHA256,
        a.v123_comparison: AUTHORITATIVE_V123_COMPARISON_SHA256,
        a.v123_balanced: AUTHORITATIVE_V123_BALANCED_SHA256,
        a.v123_precision: AUTHORITATIVE_V123_PRECISION_SHA256,
    }
    for p, want in expected.items():
        if not p.is_file():
            errors.append(f"missing_v123_prerequisite:{p.name}")
        elif sha(p) != want:
            errors.append(f"v123_sha_mismatch:{p.name}")
    try:
        p123, c123 = load(a.v123_pipeline), load(a.v123_comparison)
        d123 = c123.get("preregistered_decision") or {}
        if not (
            p123.get("valid") and p123.get("attribution_ready")
            and p123.get("preregistered_status") == V123_STATUS
            and c123.get("valid") and c123.get("attribution_ready")
            and d123.get("status") == V123_STATUS and d123.get("next_branch") == V123_NEXT
        ):
            errors.append("v123_freeze_branch_not_authorized")
    except Exception as exc:
        errors.append(f"v123_prerequisite_parse:{type(exc).__name__}")

    for p in (a.balanced_checkpoint, a.precision_checkpoint, a.balanced_calibration, a.precision_calibration):
        if not p.is_file():
            errors.append(f"missing_frozen_main_artifact:{p}")

    full_run_runtime: dict[str, Any] = {}
    try:
        full_run_runtime = load(a.full_run_runtime)
        rows = full_run_runtime.get("runtime_files") or {}
        sc_full = full_run_runtime.get("scientific_contract") or {}
        if not (
            full_run_runtime.get("valid") and full_run_runtime.get("attribution_ready")
            and full_run_runtime.get("scientific_version") == SCIENTIFIC_VERSION
            and full_run_runtime.get("engineering_version") == ENGINEERING_VERSION
            and sc_full.get("fixed_main_evaluation_only") is True
            and sc_full.get("recovery_set_mechanism_family_frozen") is True
            and sc_full.get("new_recovery_mechanism_authorized") is False
            and sc_full.get("planner_parameters_trained") == 0
            and sc_full.get("womd_source_resolution") == "standard_validation_only_with_bucket_provenance_conflict_fail_closed"
            and sc_full.get("rifa_absolute_admission_for_intervention") is True
        ):
            errors.append("full_run_runtime_contract")
        for rel, want in FROZEN_FULL_RUN_CORE_SHA256.items():
            if (rows.get(rel) or {}).get("sha256") != want:
                errors.append(f"full_run_core_sha:{rel}")
    except Exception as exc:
        errors.append(f"full_run_runtime_parse:{type(exc).__name__}")

    results: dict[str, dict[str, dict[str, Any]]] = {v: {} for v in ("nominal", "balanced", "precision")}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {v: {} for v in ("balanced", "precision")}
    sentinels: dict[str, dict[str, dict[str, Any]]] = {v: {} for v in ("balanced", "precision")}
    support_docs: dict[str, dict[str, dict[str, Any]]] = {v: {} for v in ("nominal", "balanced", "precision")}
    sentinel_support_docs: dict[str, dict[str, dict[str, Any]]] = {v: {} for v in ("balanced", "precision")}
    artifacts: dict[str, Any] = {
        "full_run_runtime": artifact_record(a.full_run_runtime) if a.full_run_runtime.is_file() else {"path": str(a.full_run_runtime.resolve()), "sha256": None, "size": 0}
    }
    try:
        for regime in ("safe", "near", "contact"):
            for variant in ("nominal", "balanced", "precision"):
                p = getattr(a, f"{variant}_{regime}")
                results[variant][regime] = load(p)
                artifacts[f"{variant}_{regime}"] = artifact_record(p)
                support_path = p.parent / "closed_loop_dataset_support.json"
                support_docs[variant][regime] = load(support_path)
                artifacts[f"{variant}_{regime}_support"] = artifact_record(support_path)
                if variant != "nominal":
                    selector_cfg = results[variant][regime].get("selector_config") or {}
                    if selector_cfg.get("require_absolute_admission_for_intervention") is not True:
                        errors.append(f"rifa_absolute_admission_not_enforced:{variant}:{regime}")
            for variant in ("balanced", "precision"):
                cp = getattr(a, f"{variant}_{regime}_comparison")
                sp = getattr(a, f"{variant}_{regime}_sentinel")
                comparisons[variant][regime] = load(cp)
                sentinels[variant][regime] = load(sp)
                artifacts[f"{variant}_{regime}_comparison"] = artifact_record(cp)
                artifacts[f"{variant}_{regime}_sentinel"] = artifact_record(sp)
                sentinel_support_path = sp.parent / "closed_loop_dataset_support.json"
                sentinel_support_docs[variant][regime] = load(sentinel_support_path)
                artifacts[f"{variant}_{regime}_sentinel_support"] = artifact_record(sentinel_support_path)
    except Exception as exc:
        errors.append(f"evaluation_artifact_parse:{type(exc).__name__}:{exc}")

    sentinel_index: dict[str, Any] = {}
    try:
        sentinel_index = load(a.sentinel_index)
        artifacts["sentinel_index"] = artifact_record(a.sentinel_index)
        if not sentinel_index.get("valid"):
            errors.append("sentinel_index_invalid")
    except Exception as exc:
        errors.append(f"sentinel_index_parse:{type(exc).__name__}")

    # Enforce exact paired comparison coverage and the fixed bootstrap contract.
    for variant in ("balanced", "precision"):
        for regime in ("safe", "near", "contact"):
            row = comparisons.get(variant, {}).get(regime, {})
            n_expected = len(results.get("nominal", {}).get(regime, {}).get("scenes") or [])
            if int(row.get("num_paired_scenes") or -1) != n_expected or n_expected <= 0:
                errors.append(f"paired_coverage_mismatch:{variant}:{regime}")
            if int(row.get("bootstrap_draws") or 0) != 5000 or int(row.get("bootstrap_seed") or -1) != 2027:
                errors.append(f"bootstrap_contract:{variant}:{regime}")

    # Dataset collection identity is owned by the support artifacts, not by
    # result["source"] (which names the policy/result source such as model).
    for variant in ("nominal", "balanced", "precision"):
        for regime in ("safe", "near", "contact"):
            support = support_docs.get(variant, {}).get(regime, {})
            if not (support.get("schema_supports_closed_loop") and support.get("raw_source_role") == "validation"):
                errors.append(f"full_support_provenance:{variant}:{regime}")
    for variant in ("balanced", "precision"):
        for regime in ("safe", "near", "contact"):
            support = sentinel_support_docs.get(variant, {}).get(regime, {})
            if not (
                support.get("schema_supports_closed_loop")
                and support.get("raw_source_role") == "validation"
                and support.get("target_keys_valid") is True
                and int(support.get("num_requested_target_keys") or 0) == 1
                and int(support.get("num_matching_requested_target_keys") or 0) == 1
            ):
                errors.append(f"sentinel_support_provenance:{variant}:{regime}")

    decision = adjudicate(
        comparisons=comparisons, results=results, sentinel_results=sentinels, support_docs=support_docs
    ) if not errors else {
        "status": "FIXED_MAIN_COVERAGE_STOP", "go": False,
        "next_branch": "keep_recovery_mechanism_family_frozen_and_diagnose_only_failed_stability_or_closed_loop_axis_no_new_recovery_mechanism_capacity_regime_source_horizon_or_threshold_sweep",
    }

    # Source and frozen-Main provenance are reportable evidence, not tunable inputs.
    provenance = {
        "authoritative_v48_123_pipeline_sha256": AUTHORITATIVE_V123_PIPELINE_SHA256,
        "authoritative_v48_123_comparison_sha256": AUTHORITATIVE_V123_COMPARISON_SHA256,
        "v48_123_balanced_sha256": AUTHORITATIVE_V123_BALANCED_SHA256,
        "v48_123_precision_sha256": AUTHORITATIVE_V123_PRECISION_SHA256,
        "balanced_checkpoint": artifact_record(a.balanced_checkpoint) if a.balanced_checkpoint.is_file() else None,
        "precision_checkpoint": artifact_record(a.precision_checkpoint) if a.precision_checkpoint.is_file() else None,
        "balanced_calibration": artifact_record(a.balanced_calibration) if a.balanced_calibration.is_file() else None,
        "precision_calibration": artifact_record(a.precision_calibration) if a.precision_calibration.is_file() else None,
        "full_population_runtime": {
            "artifact": artifact_record(a.full_run_runtime) if a.full_run_runtime.is_file() else None,
            "engineering_version": full_run_runtime.get("engineering_version"),
            "scientific_version": full_run_runtime.get("scientific_version"),
            "run_instance_id": full_run_runtime.get("run_instance_id"),
            "frozen_core_sha256": FROZEN_FULL_RUN_CORE_SHA256,
        },
        "womd_sources": {
            regime: {
                variant: {
                    "role": support_docs.get(variant, {}).get(regime, {}).get("raw_source_role"),
                    "pattern": support_docs.get(variant, {}).get(regime, {}).get("womd_pattern"),
                }
                for variant in ("nominal", "balanced", "precision")
            }
            for regime in ("safe", "near", "contact")
        },
        "bucket_datasets": {
            regime: {
                variant: results.get(variant, {}).get(regime, {}).get("bucket_dataset")
                for variant in ("nominal", "balanced", "precision")
            }
            for regime in ("safe", "near", "contact")
        },
        "gamma_rec": {
            regime: {
                variant: results.get(variant, {}).get(regime, {}).get("gamma_rec")
                for variant in ("balanced", "precision")
            }
            for regime in ("safe", "near", "contact")
        },
    }

    doc = {
        "schema": "ocrap-v48.124-fixed-main-stability-adjudication-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "experiment_type": "fixed_main_stability_noninterference_closed_loop_adjudication",
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "audit_only": True,
        "recovery_set_mechanism_family_frozen": True,
        "new_recovery_mechanism_authorized": False,
        "main_modified": False,
        "planner_parameters_trained": 0,
        "source_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "recalibration_performed": False,
        "balanced_precision_are_robustness_variants": True,
        "bootstrap_draws": 5000,
        "bootstrap_seed": 2027,
        "noninterference_margin": 0.0,
        "sentinel_rule": "lexicographically_first_common_target_per_regime_replayed_once_per_fixed_main_variant",
        "sentinel_index": sentinel_index,
        "provenance": provenance,
        "artifacts": artifacts,
        "preregistered_decision": decision,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"valid": doc["valid"], "attribution_ready": doc["attribution_ready"], "status": decision.get("status"), "errors": errors}))
    return 0 if doc["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
