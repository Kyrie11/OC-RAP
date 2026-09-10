#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ocrap.audits.heterogeneous_constraint_normal_cone import ENGINEERING_VERSION, SCIENTIFIC_VERSION

AUTHORITATIVE_V111_COMPARISON_SHA256 = "ee2a3f13f2793dd8d0a4a1bdf73192a188d21bfac31c1549b3d6d0ae63cb8373"
AUTHORITATIVE_V111_PIPELINE_SHA256 = "c155ac8277b2fe690be030eaaf4031e75873e1e22d2521ce56d10aabebb56187"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for key in (
        "runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison",
        "v48_111_pipeline", "v48_111_comparison",
    ):
        ap.add_argument("--" + key.replace("_", "-"), dest=key, type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []

    docs = {
        k: json.loads(getattr(a, k).read_text())
        for k in ("runtime", "balanced", "precision", "comparison", "v48_111_pipeline", "v48_111_comparison")
    }
    if not (
        docs["runtime"].get("valid")
        and docs["runtime"].get("attribution_ready")
        and docs["runtime"].get("engineering_version") == ENGINEERING_VERSION
        and docs["runtime"].get("scientific_version") == SCIENTIFIC_VERSION
        and docs["runtime"].get("run_instance_id") == a.run_id
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
            and d.get("capacity_matched_nominal_vs_candidate_cone")
            and d.get("run_instance_id") == a.run_id
        ):
            errors.append(variant)

    if not (
        docs["comparison"].get("valid")
        and docs["comparison"].get("attribution_ready")
        and docs["comparison"].get("engineering_version") == ENGINEERING_VERSION
        and docs["comparison"].get("scientific_version") == SCIENTIFIC_VERSION
        and docs["comparison"].get("run_instance_id") == a.run_id
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

    if sha(a.v48_111_pipeline) != AUTHORITATIVE_V111_PIPELINE_SHA256:
        errors.append("v111_pipeline_sha")
    if sha(a.v48_111_comparison) != AUTHORITATIVE_V111_COMPARISON_SHA256:
        errors.append("v111_comparison_sha")
    d111 = docs["v48_111_comparison"].get("preregistered_decision") or {}
    if not (
        docs["v48_111_pipeline"].get("valid")
        and docs["v48_111_pipeline"].get("attribution_ready")
        and docs["v48_111_pipeline"].get("preregistered_status") == "CONSTRAINT_NATIVE_ACTIVE_GEOMETRY_STOP"
    ):
        errors.append("v111_pipeline")
    if not (
        d111.get("status") == "CONSTRAINT_NATIVE_ACTIVE_GEOMETRY_STOP"
        and d111.get("next_branch")
        == "close_fixed_cv_circle_agent_geometry_then_preregister_heterogeneous_active_constraint_normal_cone_audit_no_training_or_source_sweep"
    ):
        errors.append("v111_branch")

    artifacts: dict[str, dict[str, str]] = {}
    for key in ("balanced", "precision", "balanced_state", "precision_state", "comparison", "runtime"):
        p = getattr(a, key)
        artifacts[key] = {"path": str(p.resolve()), "sha256": sha(p)}
    status = (docs["comparison"].get("preregistered_decision") or {}).get("status")
    out = {
        "schema": "ocrap-v48.112-hcnc-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_heterogeneous_active_constraint_normal_cone",
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
        "v48_111_pipeline_sha256": sha(a.v48_111_pipeline),
        "v48_111_comparison_sha256": sha(a.v48_111_comparison),
        "authoritative_v48_111_comparison_sha256": AUTHORITATIVE_V111_COMPARISON_SHA256,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
