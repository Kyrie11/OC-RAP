#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ocrap.audits.viability_rank_persistence import (
    ENGINEERING_VERSION, SCIENTIFIC_VERSION, COUPLING_GEOMETRY_DIM, MATCHED_DIM, COUPLING_MODE_NAMES,
)

AUTHORITATIVE_V120_PIPELINE_SHA256 = "2839be06885f1b05cf6064b934fbe6ed55eda9ff0db6ffe46bf50e89c21cb471"
AUTHORITATIVE_V120_COMPARISON_SHA256 = "9a7827d769b3abe4cda0802fa4aa95b879f38c0ffe8392df65c21bc017cae3c4"
V120_NEXT = "close_nominal_rank_viability_transport_then_preregister_recovery_set_rank_persistence_coupling_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep"
VALID_STATUSES = {
    "VIABILITY_RANK_PERSISTENCE_COUPLING_GO",
    "EXPOSED_VIABILITY_RANK_PERSISTENCE_COUPLING_GO",
    "VIABILITY_RANK_PERSISTENCE_COUPLING_SUPPORT_ONLY",
    "VIABILITY_RANK_PERSISTENCE_COUPLING_RESERVE_ONLY",
    "VIABILITY_RANK_PERSISTENCE_COUPLING_LOCAL_ORDER_ONLY",
    "VIABILITY_RANK_PERSISTENCE_COUPLING_STOP",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for key in (
        "runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison",
        "v48_120_pipeline", "v48_120_comparison",
    ):
        ap.add_argument("--" + key.replace("_", "-"), dest=key, type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    errors: list[str] = []
    docs: dict[str, dict] = {}
    for key in ("runtime", "balanced", "precision", "comparison", "v48_120_pipeline", "v48_120_comparison"):
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
        and sc.get("viability_rank_persistence_coupling") is True
        and sc.get("coupling_mode_names") == [str(x) for x in COUPLING_MODE_NAMES]
        and sc.get("persistence_basis") == "fixed_global_rank_persistence_and_rank_x_persistence_modes"
        and sc.get("rank_coordinate") == "candidate_independent_nominal_same_option_viability_midranks"
        and sc.get("candidate_rank_sort_used_for_coordinate") is False
        and sc.get("primary_option_set") == "all_common_valid_recovery_options"
        and sc.get("boundary_support_weights_used_in_persistence") is False
        and sc.get("active_option_identity_exported") is False
        and sc.get("pre_readout_candidate_option_selector") is False
        and sc.get("same_option_nominal_rank_correspondence") is True
        and sc.get("same_option_nominal_rank_persistence") is True
        and sc.get("persistence_decay_or_window_sweep") is False
        and sc.get("matched_family_dim") == MATCHED_DIM
        and sc.get("coupling_geometry_dim") == COUPLING_GEOMETRY_DIM
        and sc.get("boundary_transport") is False
        and sc.get("rank_cut_sweep") is False
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
            and d.get("capacity_matched_all_persistence_families")
            and d.get("matched_family_dimension") == MATCHED_DIM
            and d.get("coupling_geometry_dimension") == COUPLING_GEOMETRY_DIM
            and d.get("coupling_mode_names") == [str(x) for x in COUPLING_MODE_NAMES]
            and d.get("persistence_basis") == "fixed_global_rank_persistence_and_rank_x_persistence_modes"
            and d.get("rank_coordinate") == "candidate_independent_nominal_same_option_viability_midranks"
            and d.get("candidate_rank_sort_used_for_coordinate") is False
            and d.get("primary_option_set") == "all_common_valid_recovery_options"
            and d.get("boundary_support_weights_used_in_persistence") is False
            and d.get("active_option_identity_exported") is False
            and d.get("teacher_npz_fields_loaded_into_feature_path") == ["root_valid"]
            and d.get("teacher_future_fields_used") is False
            and d.get("root_decoder_parameters_trained") == 0
            and d.get("source_parameters_trained") == 0
            and d.get("regime_conditioning") is False
            and d.get("boundary_transport") is False
            and d.get("rank_cut_sweep") is False
            and d.get("persistence_decay_or_window_sweep") is False
            and d.get("same_option_nominal_rank_correspondence") is True
            and d.get("same_option_nominal_rank_persistence") is True
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
        and decision.get("coupling_mode_names") == [str(x) for x in COUPLING_MODE_NAMES]
        and decision.get("persistence_basis") == "fixed_global_rank_persistence_and_rank_x_persistence_modes"
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

    if sha(a.v48_120_pipeline) != AUTHORITATIVE_V120_PIPELINE_SHA256:
        errors.append("v120_pipeline_sha")
    if sha(a.v48_120_comparison) != AUTHORITATIVE_V120_COMPARISON_SHA256:
        errors.append("v120_comparison_sha")
    d119 = docs["v48_120_comparison"].get("preregistered_decision") or {}
    if not (
        docs["v48_120_pipeline"].get("valid")
        and docs["v48_120_pipeline"].get("attribution_ready")
        and docs["v48_120_pipeline"].get("preregistered_status") == "VIABILITY_RANK_TRANSPORT_STOP"
        and docs["v48_120_comparison"].get("valid")
        and docs["v48_120_comparison"].get("attribution_ready")
        and d119.get("status") == "VIABILITY_RANK_TRANSPORT_STOP"
        and d119.get("next_branch") == V120_NEXT
    ):
        errors.append("v120_prerequisite")

    artifacts = {}
    for key in ("runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison"):
        p = getattr(a, key)
        artifacts[key] = {"path": str(p.resolve()), "sha256": sha(p)}

    out = {
        "schema": "ocrap-v48.121-vrpc-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_recovery_set_viability_rank_persistence_coupling",
        "preregistered_status": decision.get("status"),
        "artifacts": artifacts,
        "authoritative_v48_120_comparison_sha256": AUTHORITATIVE_V120_COMPARISON_SHA256,
        "v48_120_pipeline_sha256": sha(a.v48_120_pipeline),
        "v48_120_comparison_sha256": sha(a.v48_120_comparison),
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
