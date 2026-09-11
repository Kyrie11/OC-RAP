#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ocrap.audits.viability_survival_envelope import (
    ENGINEERING_VERSION, ENVELOPE_GEOMETRY_DIM, MATCHED_DIM, SCIENTIFIC_VERSION,
)

AUTHORITATIVE_V117_PIPELINE_SHA256 = "674902c61b68c05387798f011d5e0b9639efd1eff5f3e1b08b3222d819f7c399"
AUTHORITATIVE_V117_COMPARISON_SHA256 = "8a4e54a71dfc77821842041a053002ef8b4b10c9b6a778270384b7da219fc99b"
V117_NEXT = "close_static_boundary_witness_hitting_flow_then_preregister_recovery_set_viability_survival_envelope_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep"
VALID_STATUSES = {
    "VIABILITY_SURVIVAL_ENVELOPE_GO",
    "WEAK_EXPOSED_VIABILITY_ENVELOPE_GO",
    "VIABILITY_SURVIVAL_ENVELOPE_SUPPORT_ONLY",
    "VIABILITY_SURVIVAL_ENVELOPE_RESERVE_ONLY",
    "VIABILITY_SURVIVAL_ENVELOPE_LOCAL_ORDER_ONLY",
    "VIABILITY_SURVIVAL_ENVELOPE_STOP",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for key in ("runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison", "v48_117_pipeline", "v48_117_comparison"):
        ap.add_argument("--" + key.replace("_", "-"), dest=key, type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []
    docs: dict[str, dict] = {}
    for key in ("runtime", "balanced", "precision", "comparison", "v48_117_pipeline", "v48_117_comparison"):
        try:
            docs[key] = json.loads(getattr(a, key).read_text())
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
        and sc.get("viability_survival_envelope") is True
        and sc.get("primary_option_set") == "all_common_valid_recovery_options"
        and sc.get("boundary_support_weights_used") is False
        and sc.get("active_option_identity_exported") is False
        and sc.get("frozen_root_decoder_read_only") is True
        and sc.get("pre_readout_candidate_option_selector") is False
        and sc.get("matched_family_dim") == MATCHED_DIM
        and sc.get("envelope_geometry_dim") == ENVELOPE_GEOMETRY_DIM
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
            and d.get("capacity_matched_all_envelope_families")
            and d.get("matched_family_dimension") == MATCHED_DIM
            and d.get("envelope_geometry_dimension") == ENVELOPE_GEOMETRY_DIM
            and d.get("primary_option_set") == "all_common_valid_recovery_options"
            and d.get("boundary_support_weights_used") is False
            and d.get("active_option_identity_exported") is False
            and d.get("frozen_root_decoder_read_only") is True
            and d.get("frozen_margin_head_read_only") is True
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
                and st.get("variant") == variant and st.get("run_instance_id") == a.run_id
                and st.get("convex_closed_form_ridge") is True
                and st.get("strictly_convex_unique_solution") is True
            ):
                errors.append(key)
        except Exception as exc:
            errors.append(f"{key}:load:{type(exc).__name__}")

    if sha(a.v48_117_pipeline) != AUTHORITATIVE_V117_PIPELINE_SHA256:
        errors.append("v117_pipeline_sha")
    if sha(a.v48_117_comparison) != AUTHORITATIVE_V117_COMPARISON_SHA256:
        errors.append("v117_comparison_sha")
    d117 = docs["v48_117_comparison"].get("preregistered_decision") or {}
    if not (
        docs["v48_117_pipeline"].get("valid") and docs["v48_117_pipeline"].get("attribution_ready")
        and docs["v48_117_pipeline"].get("preregistered_status") == "TAIL_BOUNDARY_CROSSING_FLOW_STOP"
        and docs["v48_117_comparison"].get("valid") and docs["v48_117_comparison"].get("attribution_ready")
        and d117.get("status") == "TAIL_BOUNDARY_CROSSING_FLOW_STOP"
        and d117.get("next_branch") == V117_NEXT
    ):
        errors.append("v117_prerequisite")

    artifacts = {}
    for key in ("runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison"):
        p = getattr(a, key)
        artifacts[key] = {"path": str(p.resolve()), "sha256": sha(p)}
    out = {
        "schema": "ocrap-v48.118-vse-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_recovery_set_viability_survival_envelope",
        "preregistered_status": decision.get("status"),
        "artifacts": artifacts,
        "authoritative_v48_117_comparison_sha256": AUTHORITATIVE_V117_COMPARISON_SHA256,
        "v48_117_pipeline_sha256": sha(a.v48_117_pipeline),
        "v48_117_comparison_sha256": sha(a.v48_117_comparison),
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
