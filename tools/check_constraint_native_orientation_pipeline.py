#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ocrap.audits.signed_viability_rank_state import (
    ENGINEERING_VERSION, SCIENTIFIC_VERSION, STATE_GEOMETRY_DIM, MATCHED_DIM, STATE_MODE_NAMES,
)

AUTHORITATIVE_V121_PIPELINE_SHA256 = "b2e08fbe1b1b1e2b676a3d9bd0e9b84082fd35844b8fa6dd0cb7fa193d22e6e6"
AUTHORITATIVE_V121_COMPARISON_SHA256 = "7557a98d23f79d146c620c51deabe41ade063295f906fec02c07836e8af1513f"
V121_NEXT = "close_nominal_rank_persistence_coupling_then_preregister_signed_viability_rank_state_transport_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep"
VALID_STATUSES = {
    "SIGNED_VIABILITY_RANK_STATE_TRANSPORT_GO",
    "EXPOSED_SIGNED_VIABILITY_RANK_STATE_TRANSPORT_GO",
    "SIGNED_VIABILITY_RANK_STATE_TRANSPORT_SUPPORT_ONLY",
    "SIGNED_VIABILITY_RANK_STATE_TRANSPORT_RESERVE_ONLY",
    "SIGNED_VIABILITY_RANK_STATE_TRANSPORT_LOCAL_ORDER_ONLY",
    "SIGNED_VIABILITY_RANK_STATE_TRANSPORT_STOP",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for key in (
        "runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison",
        "v48_121_pipeline", "v48_121_comparison",
    ):
        ap.add_argument("--" + key.replace("_", "-"), dest=key, type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    errors: list[str] = []
    docs: dict[str, dict] = {}
    for key in ("runtime", "balanced", "precision", "comparison", "v48_121_pipeline", "v48_121_comparison"):
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
        and sc.get("signed_viability_rank_state_transport") is True
        and sc.get("absolute_signed_nominal_viability_state") is True
        and sc.get("state_mode_names") == [str(x) for x in STATE_MODE_NAMES]
        and sc.get("signed_state_basis") == "fixed_global_rank_signed_state_and_rank_x_signed_state_modes"
        and sc.get("rank_coordinate") == "candidate_independent_nominal_same_option_viability_midranks"
        and sc.get("candidate_rank_sort_used_for_coordinate") is False
        and sc.get("primary_option_set") == "all_common_valid_recovery_options"
        and sc.get("boundary_support_weights_used_in_signed_state") is False
        and sc.get("active_option_identity_exported") is False
        and sc.get("pre_readout_candidate_option_selector") is False
        and sc.get("same_option_nominal_rank_correspondence") is True
        and sc.get("signed_state_threshold_or_scale_sweep") is False
        and sc.get("matched_family_dim") == MATCHED_DIM
        and sc.get("state_geometry_dim") == STATE_GEOMETRY_DIM
        and sc.get("boundary_transport") is False
        and sc.get("rank_cut_sweep") is False
        and sc.get("capacity_sweep") is False
        and sc.get("horizon_sweep") is False
        and sc.get("threshold_sweep") is False
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
            and d.get("capacity_matched_all_signed_state_families")
            and d.get("matched_family_dimension") == MATCHED_DIM
            and d.get("state_geometry_dimension") == STATE_GEOMETRY_DIM
            and d.get("state_mode_names") == [str(x) for x in STATE_MODE_NAMES]
            and d.get("signed_state_basis") == "fixed_global_rank_signed_state_and_rank_x_signed_state_modes"
            and d.get("rank_coordinate") == "candidate_independent_nominal_same_option_viability_midranks"
            and d.get("candidate_rank_sort_used_for_coordinate") is False
            and d.get("primary_option_set") == "all_common_valid_recovery_options"
            and d.get("boundary_support_weights_used_in_signed_state") is False
            and d.get("active_option_identity_exported") is False
            and d.get("absolute_signed_nominal_viability_state") is True
            and d.get("teacher_npz_fields_loaded_into_feature_path") == ["root_valid"]
            and d.get("teacher_future_fields_used") is False
            and d.get("root_decoder_parameters_trained") == 0
            and d.get("source_parameters_trained") == 0
            and d.get("regime_conditioning") is False
            and d.get("boundary_transport") is False
            and d.get("rank_cut_sweep") is False
            and d.get("signed_state_threshold_or_scale_sweep") is False
            and d.get("same_option_nominal_rank_correspondence") is True
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
        and decision.get("state_mode_names") == [str(x) for x in STATE_MODE_NAMES]
        and decision.get("signed_state_basis") == "fixed_global_rank_signed_state_and_rank_x_signed_state_modes"
        and decision.get("rank_coordinate") == "candidate_independent_nominal_same_option_viability_midranks"
        and decision.get("boundary_transport_authorized") is False
        and decision.get("source_training_authorized") is False
    ):
        errors.append("comparison")

    for variant, key in (("balanced", "balanced_state"), ("precision", "precision_state")):
        try:
            st = torch.load(getattr(a, key), map_location="cpu", weights_only=False)
            if not (
                st.get("schema") == "ocrap-v48.122-signed-viability-rank-state-transport-state-v1"
                and st.get("engineering_version") == ENGINEERING_VERSION
                and st.get("scientific_version") == SCIENTIFIC_VERSION
                and st.get("variant") == variant
                and st.get("run_instance_id") == a.run_id
                and st.get("convex_closed_form_ridge") is True
                and st.get("strictly_convex_unique_solution") is True
            ):
                errors.append(key)
        except Exception as exc:
            errors.append(f"{key}:load:{type(exc).__name__}")

    if sha(a.v48_121_pipeline) != AUTHORITATIVE_V121_PIPELINE_SHA256:
        errors.append("v121_pipeline_sha")
    if sha(a.v48_121_comparison) != AUTHORITATIVE_V121_COMPARISON_SHA256:
        errors.append("v121_comparison_sha")
    d121 = docs["v48_121_comparison"].get("preregistered_decision") or {}
    if not (
        docs["v48_121_pipeline"].get("valid")
        and docs["v48_121_pipeline"].get("attribution_ready")
        and docs["v48_121_pipeline"].get("preregistered_status") == "VIABILITY_RANK_PERSISTENCE_COUPLING_STOP"
        and docs["v48_121_comparison"].get("valid")
        and docs["v48_121_comparison"].get("attribution_ready")
        and d121.get("status") == "VIABILITY_RANK_PERSISTENCE_COUPLING_STOP"
        and d121.get("next_branch") == V121_NEXT
    ):
        errors.append("v121_prerequisite")

    artifacts = {}
    for key in ("runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison"):
        p = getattr(a, key)
        artifacts[key] = {"path": str(p.resolve()), "sha256": sha(p)}

    out = {
        "schema": "ocrap-v48.122-svrt-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_signed_viability_rank_state_transport",
        "preregistered_status": decision.get("status"),
        "artifacts": artifacts,
        "authoritative_v48_121_comparison_sha256": AUTHORITATIVE_V121_COMPARISON_SHA256,
        "v48_121_pipeline_sha256": sha(a.v48_121_pipeline),
        "v48_121_comparison_sha256": sha(a.v48_121_comparison),
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
