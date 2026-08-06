#!/usr/bin/env python3
"""Fail-closed v48.36 checkpoint/inference contract audit.

This checker is intentionally version-specific so new admission-prior modes
cannot be passed to an older parser without an explicit test and release bump.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ocrap.models.inference import load_model_bundle


PRIOR_MODES = ("risk_centered", "benefit_only", "safety_slack", "barrier_gated_slack", "frontier_capped_slack")


def _parse_csv(value: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in value.split(",") if x.strip())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fail-closed v48.36 checkpoint/inference contract audit")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--support-contract", type=Path, required=True)
    ap.add_argument("--expect-frontier", choices=("true", "false"), default="true")
    ap.add_argument("--expect-admission-bounded", choices=("true", "false"), default="false")
    ap.add_argument("--expect-context-source", choices=("relative", "tournament", "physical_relative", "physical_interaction"), default="physical_interaction")
    ap.add_argument("--expect-context-enabled", choices=("true", "false"), default="true")
    ap.add_argument("--expect-frontier-cap-temperature", type=float, default=0.10)
    ap.add_argument("--expect-component-prior-logit", type=float, default=-2.0)
    ap.add_argument("--expect-component-count", type=int, default=5)
    ap.add_argument("--expect-component-scale", type=float, default=6.0)
    ap.add_argument("--expect-admission-prior-detach", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-admission-prior-mode", choices=PRIOR_MODES, default="frontier_capped_slack")
    ap.add_argument("--expect-slack-temperature", type=float, default=0.025)
    ap.add_argument("--expect-slack-penalty", type=float, default=1.0)
    ap.add_argument("--expect-interaction-hidden", type=int, default=64)
    ap.add_argument("--expect-consensus-prior-scale", type=float, default=0.50)
    ap.add_argument("--reliability-override", default="", help="Ablation override; empty uses support contract")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    if not args.support_contract.is_file():
        raise SystemExit(f"support contract not found: {args.support_contract}")
    support = json.loads(args.support_contract.read_text(encoding="utf-8"))
    support_reliability = tuple(float(x) for x in support.get("reliability", []))
    expected_reliability = _parse_csv(args.reliability_override) if args.reliability_override else support_reliability
    if len(expected_reliability) != args.expect_component_count:
        raise SystemExit(f"expected {args.expect_component_count} reliability values, got {expected_reliability}")

    bundle = load_model_bundle(args.checkpoint)
    if bundle is None:
        raise SystemExit(f"unable to load checkpoint: {args.checkpoint}")
    model = bundle.model
    actual_reliability = tuple(float(x) for x in model.direct_recovery_evidence_component_reliability)
    actual = {
        "direct_recovery_evidence_frontier": bool(model.direct_recovery_evidence_frontier),
        "direct_recovery_evidence_calibrator_context": bool(model.direct_recovery_evidence_calibrator_context),
        "direct_recovery_evidence_calibrator_context_source": str(model.direct_recovery_evidence_calibrator_context_source),
        "direct_recovery_evidence_frontier_cap_temperature": float(model.direct_recovery_evidence_frontier_cap_temperature),
        "direct_recovery_evidence_admission_bounded": bool(model.direct_recovery_evidence_admission_bounded),
        "direct_recovery_evidence_admission_prior_detach": bool(model.direct_recovery_evidence_admission_prior_detach),
        "direct_recovery_evidence_admission_prior_mode": str(model.direct_recovery_evidence_admission_prior_mode),
        "direct_recovery_evidence_component_prior_logit": float(model.direct_recovery_evidence_component_prior_logit),
        "direct_recovery_evidence_component_count": int(model.direct_recovery_evidence_component_count),
        "direct_recovery_evidence_component_scale": float(model.direct_recovery_evidence_component_scale),
        "direct_recovery_evidence_slack_temperature": float(model.direct_recovery_evidence_slack_temperature),
        "direct_recovery_evidence_slack_penalty": float(model.direct_recovery_evidence_slack_penalty),
        "direct_recovery_evidence_component_reliability": list(actual_reliability),
        "inference_evidence_contract_verified": bool((bundle.cfg.get("model", {}) or {}).get("inference_evidence_contract_verified", False)),
        "direct_recovery_evidence_interaction_hidden": int(model.direct_recovery_evidence_interaction_hidden),
        "direct_recovery_evidence_consensus_prior_scale": float(model.direct_recovery_evidence_consensus_prior_scale),
        "interaction_bridge_present": model.direct_evidence_interaction_bridge is not None,
    }
    expected = {
        "direct_recovery_evidence_frontier": args.expect_frontier == "true",
        "direct_recovery_evidence_calibrator_context": args.expect_context_enabled == "true",
        "direct_recovery_evidence_calibrator_context_source": args.expect_context_source,
        "direct_recovery_evidence_frontier_cap_temperature": float(args.expect_frontier_cap_temperature),
        "direct_recovery_evidence_admission_bounded": args.expect_admission_bounded == "true",
        "direct_recovery_evidence_admission_prior_detach": (
            None if args.expect_admission_prior_detach == "any" else args.expect_admission_prior_detach == "true"
        ),
        "direct_recovery_evidence_admission_prior_mode": args.expect_admission_prior_mode,
        "direct_recovery_evidence_component_prior_logit": float(args.expect_component_prior_logit),
        "direct_recovery_evidence_component_count": int(args.expect_component_count),
        "direct_recovery_evidence_component_scale": float(args.expect_component_scale),
        "direct_recovery_evidence_slack_temperature": float(args.expect_slack_temperature),
        "direct_recovery_evidence_slack_penalty": float(args.expect_slack_penalty),
        "direct_recovery_evidence_component_reliability": list(expected_reliability),
        "inference_evidence_contract_verified": True,
        "direct_recovery_evidence_interaction_hidden": int(args.expect_interaction_hidden),
        "direct_recovery_evidence_consensus_prior_scale": float(args.expect_consensus_prior_scale),
        "interaction_bridge_present": args.expect_context_source == "physical_interaction",
    }
    mismatches: dict[str, dict] = {}
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if expected_value is None:
            ok = True
        elif key == "direct_recovery_evidence_component_reliability":
            ok = len(actual_value) == len(expected_value) and all(
                math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-7)
                for a, b in zip(actual_value, expected_value)
            )
        elif isinstance(expected_value, float):
            ok = math.isclose(float(actual_value), expected_value, rel_tol=0.0, abs_tol=1.0e-7)
        else:
            ok = actual_value == expected_value
        if not ok:
            mismatches[key] = {"expected": expected_value, "actual": actual_value}

    doc = {
        "event": "v48_36_checkpoint_inference_contract_audit",
        "version": "v48.36-OCAF",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "support_contract": str(args.support_contract),
        "support_contract_sha256": hashlib.sha256(args.support_contract.read_bytes()).hexdigest(),
        "expected": expected,
        "actual": actual,
        "valid": not mismatches,
        "mismatches": mismatches,
        "regime_routing": False,
        "shared_deployment_rule_required": True,
        "noncompensatory_frontier_cap": args.expect_admission_prior_mode == "frontier_capped_slack",
        "observation_conditioned_action_frontier": args.expect_context_source == "physical_interaction",
        "zero_action_no_scene_shortcut": args.expect_context_source == "physical_interaction",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if not mismatches else 4


if __name__ == "__main__":
    raise SystemExit(main())
