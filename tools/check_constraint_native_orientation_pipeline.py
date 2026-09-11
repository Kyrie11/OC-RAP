#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ocrap.audits.common_option_constraint_work import ENGINEERING_VERSION, SCIENTIFIC_VERSION

AUTHORITATIVE_V113_COMPARISON_SHA256 = "6ca24f95aaca4198eb565547abdcdd4d7d37aebc2bdf9758b6b003653411cc00"
AUTHORITATIVE_V113_PIPELINE_SHA256 = "141164bc3881f734ec64963cf32c1f1b07ac0180e4835cc27b078b02b923930a"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for key in ("runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison", "v48_113_pipeline", "v48_113_comparison"):
        ap.add_argument("--" + key.replace("_", "-"), dest=key, type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []

    docs: dict[str, dict] = {}
    for key in ("runtime", "balanced", "precision", "comparison", "v48_113_pipeline", "v48_113_comparison"):
        p = getattr(a, key)
        try:
            docs[key] = json.loads(p.read_text())
        except Exception as exc:
            errors.append(f"{key}:json:{type(exc).__name__}")
            docs[key] = {}

    rt = docs["runtime"]
    if not (
        rt.get("valid") and rt.get("attribution_ready")
        and rt.get("engineering_version") == ENGINEERING_VERSION
        and rt.get("scientific_version") == SCIENTIFIC_VERSION
        and rt.get("run_instance_id") == a.run_id
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
            and d.get("capacity_matched_all_work_families")
            and d.get("actuator_projection")
            and d.get("teacher_npz_fields_loaded_into_feature_path") is False
            and d.get("run_instance_id") == a.run_id
        ):
            errors.append(variant)

    cmp = docs["comparison"]
    if not (
        cmp.get("valid") and cmp.get("attribution_ready")
        and cmp.get("engineering_version") == ENGINEERING_VERSION
        and cmp.get("scientific_version") == SCIENTIFIC_VERSION
        and cmp.get("run_instance_id") == a.run_id
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
            ):
                errors.append(key)
        except Exception as exc:
            errors.append(f"{key}:load:{type(exc).__name__}")

    if sha(a.v48_113_pipeline) != AUTHORITATIVE_V113_PIPELINE_SHA256:
        errors.append("v113_pipeline_sha")
    if sha(a.v48_113_comparison) != AUTHORITATIVE_V113_COMPARISON_SHA256:
        errors.append("v113_comparison_sha")
    d113 = docs["v48_113_comparison"].get("preregistered_decision") or {}
    if not (
        docs["v48_113_pipeline"].get("valid")
        and docs["v48_113_pipeline"].get("attribution_ready")
        and docs["v48_113_pipeline"].get("preregistered_status") == "EXECUTABLE_CONSTRAINT_JACOBIAN_STOP"
    ):
        errors.append("v113_pipeline")
    if not (
        d113.get("status") == "EXECUTABLE_CONSTRAINT_JACOBIAN_STOP"
        and d113.get("next_branch")
        == "close_selected_option_first_order_jacobian_then_preregister_common_option_constraint_work_audit_no_capacity_or_regime_sweep"
    ):
        errors.append("v113_branch")

    artifacts: dict[str, dict[str, str]] = {}
    for key in ("balanced", "precision", "balanced_state", "precision_state", "comparison", "runtime"):
        p = getattr(a, key)
        artifacts[key] = {"path": str(p.resolve()), "sha256": sha(p)}
    status = (cmp.get("preregistered_decision") or {}).get("status")
    out = {
        "schema": "ocrap-v48.114-ccw-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_common_option_full_horizon_constraint_work",
        "artifacts": artifacts,
        "preregistered_status": status,
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "source_parameters_trained": 0,
        "boundary_transport": False,
        "dataset_reconstruction": False,
        "regime_conditioning": False,
        "teacher_metadata_input_to_model": False,
        "test_roots_read": False,
        "v48_113_pipeline_sha256": sha(a.v48_113_pipeline),
        "v48_113_comparison_sha256": sha(a.v48_113_comparison),
        "authoritative_v48_113_comparison_sha256": AUTHORITATIVE_V113_COMPARISON_SHA256,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
