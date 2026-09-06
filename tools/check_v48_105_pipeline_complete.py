#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ENGINEERING_VERSION = "v48.105.0-OC-PAEL"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in (
        "runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison",
        "v48_104_pipeline", "v48_104_comparison", "v48_102_comparison",
    ):
        ap.add_argument("--" + k.replace("_", "-"), dest=k, type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []
    docs = {k: json.loads(getattr(a, k).read_text()) for k in ("runtime", "balanced", "precision", "comparison", "v48_104_pipeline", "v48_104_comparison", "v48_102_comparison")}

    if not docs["runtime"].get("valid") or not docs["runtime"].get("attribution_ready"):
        errors.append("runtime")
    for v in ("balanced", "precision"):
        d = docs[v]
        if not d.get("valid") or d.get("engineering_version") != ENGINEERING_VERSION or d.get("variant") != v or not d.get("audit_only"):
            errors.append(v)
        s = torch.load(getattr(a, f"{v}_state"), map_location="cpu", weights_only=False)
        if s.get("engineering_version") != ENGINEERING_VERSION or s.get("variant") != v or s.get("schema") != "ocrap-v48.105-pael-probe-state-v1":
            errors.append(f"{v}_state")
        if "adapted_last_block_state" in s or "state_dict" in s:
            errors.append(f"{v}_state_contains_model_parameters")
    if not docs["comparison"].get("valid") or not docs["comparison"].get("attribution_ready"):
        errors.append("comparison")

    p104 = docs["v48_104_pipeline"]
    d104 = docs["v48_104_comparison"].get("preregistered_decision") or {}
    if not (p104.get("valid") and p104.get("attribution_ready") and p104.get("preregistered_status") == "NOMINAL_INVARIANT_CONTROL_REFINEMENT_STOP"):
        errors.append("v104_pipeline")
    if not (
        d104.get("state_go") is True and d104.get("support_go") is False and d104.get("reserve_go") is False
        and d104.get("next_branch") == "close_last_stage_i_block_refinement_then_preregister_pre_last_token_action_equivariance_audit_no_broad_encoder_or_source_sweep"
    ):
        errors.append("v104_branch")
    d102 = docs["v48_102_comparison"].get("preregistered_decision") or {}
    if d102.get("status") != "STAGE_I_ACTION_INFORMATION_SUFFICIENCY_STOP":
        errors.append("v102_reference")

    artifacts = {}
    for k in ("balanced", "precision", "balanced_state", "precision_state", "comparison", "runtime"):
        p = getattr(a, k)
        artifacts[k] = {"path": str(p.resolve()), "sha256": sha(p)}
    status = (docs["comparison"].get("preregistered_decision") or {}).get("status")
    out = {
        "schema": "ocrap-v48.105-pael-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_pre_last_stage_i_action_equivariance_localization",
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
        "v48_104_pipeline_sha256": sha(a.v48_104_pipeline),
        "v48_104_comparison_sha256": sha(a.v48_104_comparison),
        "v48_102_comparison_sha256": sha(a.v48_102_comparison),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
