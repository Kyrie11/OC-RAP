#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ocrap.audits.tail_boundary_crossing_flow import (
    BOUNDARY_GEOMETRY_DIM, ENGINEERING_VERSION, MATCHED_DIM, SCIENTIFIC_VERSION,
)

AUTHORITATIVE_V116_COMPARISON_SHA256 = "cca845f43b2d3e0c7a774f87ab26faad79de739a96217ef1a72242acf94d8f0f"
AUTHORITATIVE_V116_PIPELINE_SHA256 = "105d6e47cb046f5dac106bf2930004a89f032ab92faf5b3c988d1ee11b500e9e"
V116_NEXT = "close_first_order_nominal_ocmero_cotangent_option_pushforward_then_preregister_tail_boundary_crossing_flow_audit_no_training_capacity_regime_or_source_sweep"
VALID_STATUSES = {
    "TAIL_BOUNDARY_CROSSING_FLOW_GO",
    "TAIL_BOUNDARY_MEASURE_WORK_GO",
    "TAIL_BOUNDARY_CROSSING_SUPPORT_ONLY",
    "TAIL_BOUNDARY_CROSSING_RESERVE_ONLY",
    "TAIL_BOUNDARY_CROSSING_LOCAL_ORDER_ONLY",
    "TAIL_BOUNDARY_CROSSING_FLOW_STOP",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for key in ("runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison", "v48_116_pipeline", "v48_116_comparison"):
        ap.add_argument("--" + key.replace("_", "-"), dest=key, type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []

    docs: dict[str, dict] = {}
    for key in ("runtime", "balanced", "precision", "comparison", "v48_116_pipeline", "v48_116_comparison"):
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
        and sc.get("tail_boundary_crossing_flow") is True
        and sc.get("boundary_measure_candidate_independent") is True
        and sc.get("boundary_measure_teacher_value_free") is True
        and sc.get("zero_boundary_threshold") == 0.0
        and sc.get("zero_boundary_threshold_sweep") is False
        and sc.get("frozen_root_validity_mask_used") is True
        and sc.get("frozen_root_decoder_read_only") is True
        and sc.get("pre_readout_candidate_option_selector") is False
        and sc.get("same_option_inside_each_weighted_summand") is True
        and sc.get("matched_family_dim") == MATCHED_DIM
        and sc.get("boundary_geometry_dim") == BOUNDARY_GEOMETRY_DIM
        and sc.get("boundary_transport") is False
    ):
        errors.append("runtime")

    for variant in ("balanced", "precision"):
        d = docs[variant]
        if not (
            d.get("valid") and d.get("engineering_version") == ENGINEERING_VERSION
            and d.get("scientific_version") == SCIENTIFIC_VERSION and d.get("variant") == variant
            and d.get("audit_only") and d.get("convex_closed_form_ridge")
            and d.get("strictly_convex_unique_solution")
            and d.get("capacity_matched_all_boundary_families")
            and d.get("matched_family_dimension") == MATCHED_DIM
            and d.get("boundary_geometry_dimension") == BOUNDARY_GEOMETRY_DIM
            and d.get("candidate_independent_boundary_measure") is True
            and d.get("frozen_root_decoder_read_only") is True
            and d.get("frozen_margin_head_read_only") is True
            and d.get("same_option_inside_each_weighted_summand") is True
            and d.get("teacher_npz_fields_loaded_into_feature_path") == ["root_valid"]
            and d.get("teacher_margin_probability_compatibility_fields_used") is False
            and d.get("teacher_future_fields_used") is False
            and d.get("root_decoder_parameters_trained") == 0
            and d.get("source_parameters_trained") == 0
            and d.get("boundary_transport") is False
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
        and decision.get("status") in VALID_STATUSES
        and decision.get("boundary_transport_authorized") is False
        and decision.get("source_training_authorized") is False
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

    if sha(a.v48_116_pipeline) != AUTHORITATIVE_V116_PIPELINE_SHA256:
        errors.append("v116_pipeline_sha")
    if sha(a.v48_116_comparison) != AUTHORITATIVE_V116_COMPARISON_SHA256:
        errors.append("v116_comparison_sha")
    d116 = docs["v48_116_comparison"].get("preregistered_decision") or {}
    if not (
        docs["v48_116_pipeline"].get("valid")
        and docs["v48_116_pipeline"].get("attribution_ready")
        and docs["v48_116_pipeline"].get("preregistered_status") == "WEAK_ROOT_RECOVERY_SET_FLOW_STOP"
        and docs["v48_116_comparison"].get("valid")
        and docs["v48_116_comparison"].get("attribution_ready")
        and d116.get("status") == "WEAK_ROOT_RECOVERY_SET_FLOW_STOP"
        and d116.get("next_branch") == V116_NEXT
    ):
        errors.append("v116_prerequisite")

    artifacts = {}
    for key in ("runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison"):
        p = getattr(a, key)
        artifacts[key] = {"path": str(p.resolve()), "sha256": sha(p)}

    out = {
        "schema": "ocrap-v48.117-tbcf-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_tail_boundary_crossing_flow",
        "preregistered_status": decision.get("status"),
        "artifacts": artifacts,
        "authoritative_v48_116_comparison_sha256": AUTHORITATIVE_V116_COMPARISON_SHA256,
        "v48_116_pipeline_sha256": sha(a.v48_116_pipeline),
        "v48_116_comparison_sha256": sha(a.v48_116_comparison),
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
