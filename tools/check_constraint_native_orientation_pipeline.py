#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ocrap.audits.recovery_set_constraint_flow import ENGINEERING_VERSION, SCIENTIFIC_VERSION, MATCHED_DIM, SET_GEOMETRY_DIM

AUTHORITATIVE_V114_COMPARISON_SHA256 = "7ce7d09809da348fb3229c0d89338858fa05fe461301088a9a0e6392bf8c0319"
AUTHORITATIVE_V114_PIPELINE_SHA256 = "4887300ad1af2232c36ce4d8101ca3526ce7eeae056be846b114f91b14e43ece"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for key in ("runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison", "v48_114_pipeline", "v48_114_comparison"):
        ap.add_argument("--" + key.replace("_", "-"), dest=key, type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []

    docs: dict[str, dict] = {}
    for key in ("runtime", "balanced", "precision", "comparison", "v48_114_pipeline", "v48_114_comparison"):
        p = getattr(a, key)
        try:
            docs[key] = json.loads(p.read_text())
        except Exception as exc:
            errors.append(f"{key}:json:{type(exc).__name__}")
            docs[key] = {}

    rt = docs["runtime"]
    sc = rt.get("scientific_contract") or {}
    if not (
        rt.get("valid") and rt.get("attribution_ready")
        and rt.get("engineering_version") == ENGINEERING_VERSION
        and rt.get("scientific_version") == SCIENTIFIC_VERSION
        and rt.get("run_instance_id") == a.run_id
        and sc.get("selector_free_recovery_set_constraint_flow") is True
        and sc.get("pre_readout_hard_option_selector") is False
        and sc.get("same_option_inside_each_set_summand") is True
        and sc.get("matched_family_dim") == MATCHED_DIM
        and sc.get("set_geometry_dim") == SET_GEOMETRY_DIM
    ):
        errors.append("runtime")

    for variant in ("balanced", "precision"):
        d = docs[variant]
        if not (
            d.get("valid")
            and d.get("engineering_version") == ENGINEERING_VERSION
            and d.get("scientific_version") == SCIENTIFIC_VERSION
            and d.get("variant") == variant
            and d.get("audit_only")
            and d.get("convex_closed_form_ridge")
            and d.get("strictly_convex_unique_solution")
            and d.get("capacity_matched_all_set_families")
            and d.get("matched_family_dimension") == MATCHED_DIM
            and d.get("set_geometry_dimension") == SET_GEOMETRY_DIM
            and d.get("selector_free_feature_path") is True
            and d.get("same_option_inside_each_set_summand") is True
            and d.get("teacher_npz_fields_loaded_into_feature_path") is False
            and d.get("run_instance_id") == a.run_id
        ):
            errors.append(variant)

    cmp = docs["comparison"]
    decision = cmp.get("preregistered_decision") or {}
    if not (
        cmp.get("valid") and cmp.get("attribution_ready")
        and cmp.get("engineering_version") == ENGINEERING_VERSION
        and cmp.get("scientific_version") == SCIENTIFIC_VERSION
        and cmp.get("run_instance_id") == a.run_id
        and decision.get("status") != "V48_115_ENGINEERING_STOP"
    ):
        errors.append("comparison")

    for variant, key in (("balanced", "balanced_state"), ("precision", "precision_state")):
        try:
            st = torch.load(getattr(a, key), map_location="cpu", weights_only=False)
            if not (
                st.get("engineering_version") == ENGINEERING_VERSION
                and st.get("scientific_version") == SCIENTIFIC_VERSION
                and st.get("variant") == variant
                and st.get("run_instance_id") == a.run_id
                and st.get("convex_closed_form_ridge") is True
                and st.get("strictly_convex_unique_solution") is True
            ):
                errors.append(key)
        except Exception as exc:
            errors.append(f"{key}:load:{type(exc).__name__}")

    if sha(a.v48_114_pipeline) != AUTHORITATIVE_V114_PIPELINE_SHA256:
        errors.append("v114_pipeline_sha")
    if sha(a.v48_114_comparison) != AUTHORITATIVE_V114_COMPARISON_SHA256:
        errors.append("v114_comparison_sha")
    d114 = docs["v48_114_comparison"].get("preregistered_decision") or {}
    if not (
        docs["v48_114_pipeline"].get("valid")
        and docs["v48_114_pipeline"].get("attribution_ready")
        and docs["v48_114_pipeline"].get("preregistered_status") == "COMMON_OPTION_CONSTRAINT_WORK_STOP"
        and docs["v48_114_comparison"].get("valid")
        and docs["v48_114_comparison"].get("attribution_ready")
        and d114.get("status") == "COMMON_OPTION_CONSTRAINT_WORK_STOP"
        and d114.get("next_branch") == "close_selected_option_fixed_bin_work_family_then_preregister_selector_free_recovery_set_constraint_flow_audit_no_capacity_or_regime_sweep"
    ):
        errors.append("v114_prerequisite")

    artifacts = {}
    for key in ("runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison"):
        p = getattr(a, key)
        artifacts[key] = {"path": str(p.resolve()), "sha256": sha(p)}

    out = {
        "schema": "ocrap-v48.115-rscf-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_selector_free_recovery_set_constraint_flow",
        "preregistered_status": decision.get("status"),
        "artifacts": artifacts,
        "authoritative_v48_114_comparison_sha256": AUTHORITATIVE_V114_COMPARISON_SHA256,
        "v48_114_pipeline_sha256": sha(a.v48_114_pipeline),
        "v48_114_comparison_sha256": sha(a.v48_114_comparison),
        "dataset_reconstruction": False,
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "source_parameters_trained": 0,
        "regime_conditioning": False,
        "boundary_transport": False,
        "teacher_metadata_input_to_model": False,
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "attribution_ready": out["attribution_ready"], "status": out["preregistered_status"], "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
