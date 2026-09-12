#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.audits.fixed_main_stability import (
    ENGINEERING_VERSION,
    SCIENTIFIC_VERSION,
    STATUS_GO,
    STATUS_COVERAGE_STOP,
    STATUS_DETERMINISM_STOP,
    STATUS_SAFE_STOP,
    STATUS_NEAR_STOP,
    STATUS_CONTACT_STOP,
    GO_NEXT_BRANCH,
    STOP_NEXT_BRANCH,
)

AUTHORITATIVE_V123_PIPELINE_SHA256 = "cd963f76b508f19dbb5140d36d94f76ab815669ad36ec3bbae8431a7e097fe81"
AUTHORITATIVE_V123_COMPARISON_SHA256 = "cb850cbf24fd1c27f00f56805bcb919e2c15267a365ac045dccebed7ddcba067"
V123_STATUS = "ZERO_BOUNDARY_VIABILITY_STATE_TRANSITION_STOP"
V123_NEXT = (
    "close_zero_boundary_viability_state_transition_then_freeze_recovery_set_mechanism_family_"
    "and_preregister_fixed_main_stability_noninterference_adjudication_"
    "no_new_recovery_mechanism_capacity_regime_source_horizon_or_threshold_sweep"
)
VALID_STATUSES = {STATUS_GO, STATUS_COVERAGE_STOP, STATUS_DETERMINISM_STOP, STATUS_SAFE_STOP, STATUS_NEAR_STOP, STATUS_CONTACT_STOP}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--adjudication", type=Path, required=True)
    ap.add_argument("--sentinel-index", type=Path, required=True)
    ap.add_argument("--v123-pipeline", type=Path, required=True)
    ap.add_argument("--v123-comparison", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []
    docs = {}
    for name in ("runtime", "adjudication", "sentinel_index", "v123_pipeline", "v123_comparison"):
        try:
            docs[name] = load(getattr(a, name))
        except Exception as exc:
            errors.append(f"{name}:parse:{type(exc).__name__}")
            docs[name] = {}

    rt = docs["runtime"]
    sc = rt.get("scientific_contract") or {}
    if not (
        rt.get("valid") and rt.get("attribution_ready")
        and rt.get("engineering_version") == ENGINEERING_VERSION
        and rt.get("scientific_version") == SCIENTIFIC_VERSION
        and rt.get("run_instance_id") == a.run_id
        and sc.get("fixed_main_evaluation_only") is True
        and sc.get("recovery_set_mechanism_family_frozen") is True
        and sc.get("new_recovery_mechanism_authorized") is False
        and sc.get("planner_parameters_trained") == 0
        and sc.get("recalibration_performed") is False
        and sc.get("paired_bootstrap_draws") == 5000
        and sc.get("paired_bootstrap_seed") == 2027
        and float(sc.get("noninterference_margin", 1.0)) == 0.0
    ):
        errors.append("runtime_contract")

    ad = docs["adjudication"]
    decision = ad.get("preregistered_decision") or {}
    if not (
        ad.get("valid") and ad.get("attribution_ready")
        and ad.get("engineering_version") == ENGINEERING_VERSION
        and ad.get("scientific_version") == SCIENTIFIC_VERSION
        and ad.get("run_instance_id") == a.run_id
        and ad.get("recovery_set_mechanism_family_frozen") is True
        and ad.get("new_recovery_mechanism_authorized") is False
        and ad.get("main_modified") is False
        and ad.get("planner_parameters_trained") == 0
        and decision.get("status") in VALID_STATUSES
        and decision.get("next_branch") == (GO_NEXT_BRANCH if decision.get("status") == STATUS_GO else STOP_NEXT_BRANCH)
    ):
        errors.append("adjudication_contract")

    si = docs["sentinel_index"]
    if not (si.get("valid") and all((si.get("regimes") or {}).get(r, {}).get("sentinel_target_key") for r in ("safe", "near", "contact"))):
        errors.append("sentinel_index_contract")

    if sha(a.v123_pipeline) != AUTHORITATIVE_V123_PIPELINE_SHA256:
        errors.append("v123_pipeline_sha")
    if sha(a.v123_comparison) != AUTHORITATIVE_V123_COMPARISON_SHA256:
        errors.append("v123_comparison_sha")
    p123, c123 = docs["v123_pipeline"], docs["v123_comparison"]
    d123 = c123.get("preregistered_decision") or {}
    if not (
        p123.get("valid") and p123.get("attribution_ready") and p123.get("preregistered_status") == V123_STATUS
        and c123.get("valid") and c123.get("attribution_ready")
        and d123.get("status") == V123_STATUS and d123.get("next_branch") == V123_NEXT
    ):
        errors.append("v123_prerequisite")

    artifacts = {
        "runtime": {"path": str(a.runtime.resolve()), "sha256": sha(a.runtime)},
        "adjudication": {"path": str(a.adjudication.resolve()), "sha256": sha(a.adjudication)},
        "sentinel_index": {"path": str(a.sentinel_index.resolve()), "sha256": sha(a.sentinel_index)},
    }
    # Include all underlying evidence with its adjudicator-verified SHA. This lets
    # the packager close every paired/sentinel/full-run byte without re-inference.
    for key, rec in (ad.get("artifacts") or {}).items():
        artifacts[key] = rec

    out = {
        "schema": "ocrap-v48.124-fmsa-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "fixed_main_stability_noninterference_closed_loop_adjudication",
        "preregistered_status": decision.get("status"),
        "next_branch": decision.get("next_branch"),
        "artifacts": artifacts,
        "authoritative_v48_123_pipeline_sha256": AUTHORITATIVE_V123_PIPELINE_SHA256,
        "authoritative_v48_123_comparison_sha256": AUTHORITATIVE_V123_COMPARISON_SHA256,
        "v48_123_pipeline_sha256": sha(a.v123_pipeline),
        "v48_123_comparison_sha256": sha(a.v123_comparison),
        "recovery_set_mechanism_family_frozen": True,
        "new_recovery_mechanism_authorized": False,
        "main_modified": False,
        "dataset_reconstruction": False,
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "source_parameters_trained": 0,
        "regime_conditioning": False,
        "test_roots_read_for_training": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": out["valid"], "attribution_ready": out["attribution_ready"], "status": out["preregistered_status"], "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
