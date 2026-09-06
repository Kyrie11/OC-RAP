#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ocrap.v48_103_factorized_control_sufficient_state import ENGINEERING_VERSION, expected_parameter_count


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for x in ("runtime", "balanced", "precision", "balanced-state", "precision-state", "comparison", "v48-102-pipeline", "v48-102-comparison"):
        ap.add_argument("--" + x, type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(); errors: list[str] = []
    attrs = {k: getattr(a, k.replace("-", "_")) for k in ("runtime", "balanced", "precision", "comparison")}
    docs = {k: json.loads(p.read_text()) for k, p in attrs.items()}
    for k, d in docs.items():
        if not d.get("valid"): errors.append(f"{k}:invalid")
        if d.get("engineering_version") != ENGINEERING_VERSION: errors.append(f"{k}:version")
    p102 = json.loads(a.v48_102_pipeline.read_text()); c102 = json.loads(a.v48_102_comparison.read_text()); d102 = c102.get("preregistered_decision") or {}
    if not (
        p102.get("valid") and p102.get("attribution_ready") and p102.get("engineering_version") == "v48.102.0-OC-AITS"
        and p102.get("preregistered_status") == "STAGE_I_ACTION_INFORMATION_SUFFICIENCY_STOP"
        and c102.get("valid") and c102.get("attribution_ready")
        and d102.get("stage_i_state_observability_go") is False
        and d102.get("stage_i_support_action_observability_go") is False
        and d102.get("stage_i_reserve_action_observability_go") is False
        and d102.get("next_branch") == "stage_i_action_information_insufficient_then_preregister_minimal_stage_i_recovery_representation_objective_no_source_or_broad_encoder_sweep"
    ):
        errors.append("v48_102_all_stop_prerequisite")
    for p, v in ((a.balanced_state, "balanced"), (a.precision_state, "precision")):
        if not p.is_file():
            errors.append(f"missing representation state {p}"); continue
        try:
            o = torch.load(p, map_location="cpu", weights_only=False)
            if o.get("engineering_version") != ENGINEERING_VERSION or o.get("variant") != v:
                errors.append(f"state contract {v}")
            if int(o.get("representation_parameter_count", -1)) != expected_parameter_count(192):
                errors.append(f"state parameter count {v}")
        except Exception as exc:
            errors.append(f"state load {v}:{exc!r}")
    status = (docs["comparison"].get("preregistered_decision") or {}).get("status")
    out = {
        "schema": "ocrap-v48.103-fcss-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "minimal_frozen_stage_i_factorized_control_sufficient_representation",
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "representation_parameters_trained": expected_parameter_count(192),
        "source_parameters_trained": 0,
        "dataset_reconstruction": False,
        "teacher_metadata_input_to_model": False,
        "boundary_transport": False,
        "relative_ranker_modified": False,
        "regime_conditioning": False,
        "test_roots_read": False,
        "preregistered_status": status,
        "v48_102_pipeline_sha256": sha(a.v48_102_pipeline),
        "v48_102_comparison_sha256": sha(a.v48_102_comparison),
        "artifacts": {
            "runtime": {"path": str(a.runtime.resolve()), "sha256": sha(a.runtime)},
            "balanced": {"path": str(a.balanced.resolve()), "sha256": sha(a.balanced)},
            "balanced_state": {"path": str(a.balanced_state.resolve()), "sha256": sha(a.balanced_state)},
            "precision": {"path": str(a.precision.resolve()), "sha256": sha(a.precision)},
            "precision_state": {"path": str(a.precision_state.resolve()), "sha256": sha(a.precision_state)},
            "comparison": {"path": str(a.comparison.resolve()), "sha256": sha(a.comparison)},
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
