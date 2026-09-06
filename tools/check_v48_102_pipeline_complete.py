#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.v48_102_action_information_transport_sufficiency import ENGINEERING_VERSION


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for x in ("runtime", "balanced", "precision", "balanced-state", "precision-state", "comparison", "v48-101-pipeline", "v48-101-comparison"):
        ap.add_argument("--" + x, type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []
    attrs = {k: getattr(a, k.replace("-", "_")) for k in ("runtime", "balanced", "precision", "comparison")}
    docs = {k: json.loads(p.read_text()) for k, p in attrs.items()}
    for k, d in docs.items():
        if not d.get("valid"):
            errors.append(f"{k}:invalid")
        if d.get("engineering_version") != ENGINEERING_VERSION:
            errors.append(f"{k}:version")
    p101 = json.loads(a.v48_101_pipeline.read_text())
    c101 = json.loads(a.v48_101_comparison.read_text())
    d101 = c101.get("preregistered_decision") or {}
    if not (
        p101.get("valid") and p101.get("attribution_ready")
        and p101.get("engineering_version") == "v48.101.0-OC-RCSA"
        and p101.get("preregistered_status") == "ROOT_CROSS_ATTENTION_SEMANTIC_ALIGNMENT_STOP"
        and c101.get("valid") and c101.get("attribution_ready")
        and d101.get("next_branch") == "close_root_decoder_semantic_family_then_preregister_stage_i_action_information_transport_audit_no_capacity_or_source_sweep"
    ):
        errors.append("v48_101_stop_prerequisite")
    for p in (a.balanced_state, a.precision_state):
        if not p.is_file():
            errors.append(f"missing probe state {p}")
    status = (docs["comparison"].get("preregistered_decision") or {}).get("status")
    out = {
        "schema": "ocrap-v48.102-aits-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_stage_i_action_information_transport_sufficiency",
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "source_parameters_trained": 0,
        "dataset_reconstruction": False,
        "dataset_reselection": False,
        "teacher_metadata_input_to_model": False,
        "boundary_transport": False,
        "relative_ranker_modified": False,
        "regime_conditioning": False,
        "test_roots_read": False,
        "preregistered_status": status,
        "v48_101_pipeline_sha256": sha(a.v48_101_pipeline),
        "v48_101_comparison_sha256": sha(a.v48_101_comparison),
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
