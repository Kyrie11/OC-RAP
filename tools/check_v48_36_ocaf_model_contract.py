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


PRIOR_MODES = ("risk_centered", "benefit_only", "safety_slack", "barrier_gated_slack", "frontier_capped_slack", "joint_reserve")


def _parse_csv(value: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in value.split(",") if x.strip())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fail-closed v48.36 checkpoint/inference contract audit")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--support-contract", type=Path, required=True)
    ap.add_argument("--expect-frontier", choices=("true", "false"), default="true")
    ap.add_argument("--expect-value-regime-conditioning", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-admission-bounded", choices=("true", "false"), default="false")
    ap.add_argument("--expect-context-source", choices=("relative", "tournament", "physical_relative", "physical_interaction"), default="physical_interaction")
    ap.add_argument("--expect-context-enabled", choices=("true", "false"), default="true")
    ap.add_argument("--expect-frontier-cap-temperature", type=float, default=0.10)
    ap.add_argument("--expect-admission-head", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-benefit-margin-temperature", type=float, default=0.025)
    ap.add_argument("--expect-joint-reserve-temperature", type=float, default=0.025)
    ap.add_argument("--expect-component-prior-logit", type=float, default=-2.0)
    ap.add_argument("--expect-component-count", type=int, default=5)
    ap.add_argument("--expect-component-scale", type=float, default=6.0)
    ap.add_argument("--expect-benefit-residual-scale", type=float, default=None)
    ap.add_argument("--expect-unbounded-benefit-factor", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-unbounded-harm-factors", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-reserve-factor-alignment", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-admission-prior-detach", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-admission-prior-mode", choices=PRIOR_MODES, default="frontier_capped_slack")
    ap.add_argument("--expect-slack-temperature", type=float, default=0.025)
    ap.add_argument("--expect-slack-penalty", type=float, default=1.0)
    ap.add_argument("--expect-interaction-hidden", type=int, default=64)
    ap.add_argument("--expect-dual-interaction-bridge", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-factorized-harm-interaction", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-partial-pool-harm-residual", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-partial-pool-harm-residual-scale", type=float, default=None)
    ap.add_argument("--expect-rank-benefit-skip", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-rank-benefit-gain-init", type=float, default=None)
    ap.add_argument("--expect-postprefix-obs-transport-benefit", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-postprefix-obs-transport-harm", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-postprefix-obs-transport-scale", type=float, default=None)
    ap.add_argument("--expect-roct-benefit", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-roct-deployability", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-roct-scale", type=float, default=None)
    ap.add_argument("--expect-roct-alpha", type=float, default=None)
    ap.add_argument("--expect-roct-beta", type=float, default=None)
    ap.add_argument("--expect-roct-top-m", type=int, default=None)
    ap.add_argument("--expect-roct-option-temperature", type=float, default=None)
    ap.add_argument("--expect-common-measure-root-mass", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-absolute-feasibility-head", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-native-certificate-preservation", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-native-drs-tolerance", type=float, default=None)
    ap.add_argument("--expect-native-deployability-tolerance", type=float, default=None)
    ap.add_argument("--expect-native-margin-complete-preservation", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-native-advantage-preservation", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-native-exact-advantage-preservation", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-native-boundary-complete-advantage-preservation", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-native-physical-student-drs", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-native-gap-tolerance", type=float, default=None)
    ap.add_argument("--expect-native-positive-gain", type=float, default=None)
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
        "direct_recovery_value_regime_conditioning": bool(model.direct_recovery_value_regime_conditioning),
        "direct_recovery_evidence_frontier": bool(model.direct_recovery_evidence_frontier),
        "direct_recovery_evidence_calibrator_context": bool(model.direct_recovery_evidence_calibrator_context),
        "direct_recovery_evidence_calibrator_context_source": str(model.direct_recovery_evidence_calibrator_context_source),
        "direct_recovery_evidence_frontier_cap_temperature": float(model.direct_recovery_evidence_frontier_cap_temperature),
        "direct_recovery_evidence_admission_head": bool(model.direct_recovery_evidence_admission_head),
        "direct_recovery_evidence_benefit_margin_temperature": float(model.direct_recovery_evidence_benefit_margin_temperature),
        "direct_recovery_evidence_joint_reserve_temperature": float(model.direct_recovery_evidence_joint_reserve_temperature),
        "direct_recovery_evidence_admission_bounded": bool(model.direct_recovery_evidence_admission_bounded),
        "direct_recovery_evidence_admission_prior_detach": bool(model.direct_recovery_evidence_admission_prior_detach),
        "direct_recovery_evidence_admission_prior_mode": str(model.direct_recovery_evidence_admission_prior_mode),
        "direct_recovery_evidence_component_prior_logit": float(model.direct_recovery_evidence_component_prior_logit),
        "direct_recovery_evidence_component_count": int(model.direct_recovery_evidence_component_count),
        "direct_recovery_evidence_component_scale": float(model.direct_recovery_evidence_component_scale),
        "direct_recovery_evidence_benefit_residual_scale": float(model.direct_recovery_evidence_benefit_residual_scale),
        "direct_recovery_evidence_unbounded_benefit_factor": bool(model.direct_recovery_evidence_unbounded_benefit_factor),
        "direct_recovery_evidence_unbounded_harm_factors": bool(model.direct_recovery_evidence_unbounded_harm_factors),
        "direct_recovery_evidence_reserve_factor_alignment": bool(model.direct_recovery_evidence_reserve_factor_alignment),
        "direct_recovery_evidence_slack_temperature": float(model.direct_recovery_evidence_slack_temperature),
        "direct_recovery_evidence_slack_penalty": float(model.direct_recovery_evidence_slack_penalty),
        "direct_recovery_evidence_component_reliability": list(actual_reliability),
        "inference_evidence_contract_verified": bool((bundle.cfg.get("model", {}) or {}).get("inference_evidence_contract_verified", False)),
        "direct_recovery_evidence_interaction_hidden": int(model.direct_recovery_evidence_interaction_hidden),
        "direct_recovery_evidence_dual_interaction_bridge": bool(
            model.direct_recovery_evidence_dual_interaction_bridge
        ),
        "direct_recovery_evidence_factorized_harm_interaction": bool(
            model.direct_recovery_evidence_factorized_harm_interaction
        ),
        "direct_recovery_evidence_partial_pool_harm_residual": bool(
            model.direct_recovery_evidence_partial_pool_harm_residual
        ),
        "direct_recovery_evidence_partial_pool_harm_residual_scale": float(
            model.direct_recovery_evidence_partial_pool_harm_residual_scale
        ),
        "direct_recovery_evidence_rank_benefit_skip": bool(
            model.direct_recovery_evidence_rank_benefit_skip
        ),
        "direct_recovery_evidence_rank_benefit_gain_init": float(
            model.direct_recovery_evidence_rank_benefit_gain_init
        ),
        "direct_recovery_evidence_postprefix_obs_transport_benefit": bool(
            model.direct_recovery_evidence_postprefix_obs_transport_benefit
        ),
        "direct_recovery_evidence_postprefix_obs_transport_harm": bool(
            model.direct_recovery_evidence_postprefix_obs_transport_harm
        ),
        "direct_recovery_evidence_postprefix_obs_transport_scale": float(
            model.direct_recovery_evidence_postprefix_obs_transport_scale
        ),
        "direct_recovery_evidence_roct_benefit": bool(model.direct_recovery_evidence_roct_benefit),
        "direct_recovery_evidence_roct_deployability": bool(model.direct_recovery_evidence_roct_deployability),
        "direct_recovery_evidence_roct_scale": float(model.direct_recovery_evidence_roct_scale),
        "direct_recovery_evidence_roct_alpha": float(model.direct_recovery_evidence_roct_alpha),
        "direct_recovery_evidence_roct_beta": float(model.direct_recovery_evidence_roct_beta),
        "direct_recovery_evidence_roct_top_m": int(model.direct_recovery_evidence_roct_top_m),
        "direct_recovery_evidence_roct_option_temperature": float(model.direct_recovery_evidence_roct_option_temperature),
        "direct_recovery_evidence_common_measure_root_mass": bool(model.direct_recovery_evidence_common_measure_root_mass),
        "direct_recovery_absolute_feasibility_head": bool(model.direct_recovery_absolute_feasibility_head),
        "direct_recovery_evidence_native_certificate_preservation": bool(model.direct_recovery_evidence_native_certificate_preservation),
        "direct_recovery_evidence_native_drs_tolerance": float(model.direct_recovery_evidence_native_drs_tolerance),
        "direct_recovery_evidence_native_deployability_tolerance": float(model.direct_recovery_evidence_native_deployability_tolerance),
        "direct_recovery_evidence_native_margin_complete_preservation": bool(model.direct_recovery_evidence_native_margin_complete_preservation),
        "direct_recovery_evidence_native_advantage_preservation": bool(model.direct_recovery_evidence_native_advantage_preservation),
        "direct_recovery_evidence_native_exact_advantage_preservation": bool(model.direct_recovery_evidence_native_exact_advantage_preservation),
        "direct_recovery_evidence_native_boundary_complete_advantage_preservation": bool(model.direct_recovery_evidence_native_boundary_complete_advantage_preservation),
        "direct_recovery_evidence_physical_student_drs": bool(model.direct_recovery_evidence_physical_student_drs),
        "direct_recovery_evidence_native_gap_tolerance": float(model.direct_recovery_evidence_native_gap_tolerance),
        "direct_recovery_evidence_native_positive_gain": float(model.direct_recovery_evidence_native_positive_gain),
        "direct_recovery_evidence_consensus_prior_scale": float(model.direct_recovery_evidence_consensus_prior_scale),
        "interaction_bridge_present": model.direct_evidence_interaction_bridge is not None,
    }
    expected = {
        "direct_recovery_value_regime_conditioning": (
            None if args.expect_value_regime_conditioning == "any"
            else args.expect_value_regime_conditioning == "true"
        ),
        "direct_recovery_evidence_frontier": args.expect_frontier == "true",
        "direct_recovery_evidence_calibrator_context": args.expect_context_enabled == "true",
        "direct_recovery_evidence_calibrator_context_source": args.expect_context_source,
        "direct_recovery_evidence_frontier_cap_temperature": float(args.expect_frontier_cap_temperature),
        "direct_recovery_evidence_admission_head": (
            None if args.expect_admission_head == "any" else args.expect_admission_head == "true"
        ),
        "direct_recovery_evidence_benefit_margin_temperature": float(args.expect_benefit_margin_temperature),
        "direct_recovery_evidence_joint_reserve_temperature": float(args.expect_joint_reserve_temperature),
        "direct_recovery_evidence_admission_bounded": args.expect_admission_bounded == "true",
        "direct_recovery_evidence_admission_prior_detach": (
            None if args.expect_admission_prior_detach == "any" else args.expect_admission_prior_detach == "true"
        ),
        "direct_recovery_evidence_admission_prior_mode": args.expect_admission_prior_mode,
        "direct_recovery_evidence_component_prior_logit": float(args.expect_component_prior_logit),
        "direct_recovery_evidence_component_count": int(args.expect_component_count),
        "direct_recovery_evidence_component_scale": float(args.expect_component_scale),
        "direct_recovery_evidence_benefit_residual_scale": args.expect_benefit_residual_scale,
        "direct_recovery_evidence_unbounded_benefit_factor": (
            None if args.expect_unbounded_benefit_factor == "any" else args.expect_unbounded_benefit_factor == "true"
        ),
        "direct_recovery_evidence_unbounded_harm_factors": (
            None if args.expect_unbounded_harm_factors == "any" else args.expect_unbounded_harm_factors == "true"
        ),
        "direct_recovery_evidence_reserve_factor_alignment": (
            None if args.expect_reserve_factor_alignment == "any" else args.expect_reserve_factor_alignment == "true"
        ),
        "direct_recovery_evidence_slack_temperature": float(args.expect_slack_temperature),
        "direct_recovery_evidence_slack_penalty": float(args.expect_slack_penalty),
        "direct_recovery_evidence_component_reliability": list(expected_reliability),
        "inference_evidence_contract_verified": True,
        "direct_recovery_evidence_interaction_hidden": int(args.expect_interaction_hidden),
        "direct_recovery_evidence_dual_interaction_bridge": (
            None if args.expect_dual_interaction_bridge == "any"
            else args.expect_dual_interaction_bridge == "true"
        ),
        "direct_recovery_evidence_factorized_harm_interaction": (
            None if args.expect_factorized_harm_interaction == "any"
            else args.expect_factorized_harm_interaction == "true"
        ),
        "direct_recovery_evidence_partial_pool_harm_residual": (
            None if args.expect_partial_pool_harm_residual == "any"
            else args.expect_partial_pool_harm_residual == "true"
        ),
        "direct_recovery_evidence_partial_pool_harm_residual_scale": args.expect_partial_pool_harm_residual_scale,
        "direct_recovery_evidence_rank_benefit_skip": (
            None if args.expect_rank_benefit_skip == "any"
            else args.expect_rank_benefit_skip == "true"
        ),
        "direct_recovery_evidence_rank_benefit_gain_init": args.expect_rank_benefit_gain_init,
        "direct_recovery_evidence_postprefix_obs_transport_benefit": (
            None if args.expect_postprefix_obs_transport_benefit == "any"
            else args.expect_postprefix_obs_transport_benefit == "true"
        ),
        "direct_recovery_evidence_postprefix_obs_transport_harm": (
            None if args.expect_postprefix_obs_transport_harm == "any"
            else args.expect_postprefix_obs_transport_harm == "true"
        ),
        "direct_recovery_evidence_postprefix_obs_transport_scale": args.expect_postprefix_obs_transport_scale,
        "direct_recovery_evidence_roct_benefit": (None if args.expect_roct_benefit == "any" else args.expect_roct_benefit == "true"),
        "direct_recovery_evidence_roct_deployability": (None if args.expect_roct_deployability == "any" else args.expect_roct_deployability == "true"),
        "direct_recovery_evidence_roct_scale": args.expect_roct_scale,
        "direct_recovery_evidence_roct_alpha": args.expect_roct_alpha,
        "direct_recovery_evidence_roct_beta": args.expect_roct_beta,
        "direct_recovery_evidence_roct_top_m": args.expect_roct_top_m,
        "direct_recovery_evidence_roct_option_temperature": args.expect_roct_option_temperature,
        "direct_recovery_evidence_common_measure_root_mass": (
            None if args.expect_common_measure_root_mass == "any"
            else args.expect_common_measure_root_mass == "true"
        ),
        "direct_recovery_absolute_feasibility_head": (
            None if args.expect_absolute_feasibility_head == "any"
            else args.expect_absolute_feasibility_head == "true"
        ),
        "direct_recovery_evidence_native_certificate_preservation": (
            None if args.expect_native_certificate_preservation == "any"
            else args.expect_native_certificate_preservation == "true"
        ),
        "direct_recovery_evidence_native_drs_tolerance": args.expect_native_drs_tolerance,
        "direct_recovery_evidence_native_deployability_tolerance": args.expect_native_deployability_tolerance,
        "direct_recovery_evidence_native_margin_complete_preservation": (
            None if args.expect_native_margin_complete_preservation == "any"
            else args.expect_native_margin_complete_preservation == "true"
        ),
        "direct_recovery_evidence_native_advantage_preservation": (
            None if args.expect_native_advantage_preservation == "any"
            else args.expect_native_advantage_preservation == "true"
        ),
        "direct_recovery_evidence_native_exact_advantage_preservation": (
            None if args.expect_native_exact_advantage_preservation == "any"
            else args.expect_native_exact_advantage_preservation == "true"
        ),
        "direct_recovery_evidence_native_boundary_complete_advantage_preservation": (
            None if args.expect_native_boundary_complete_advantage_preservation == "any"
            else args.expect_native_boundary_complete_advantage_preservation == "true"
        ),
        "direct_recovery_evidence_physical_student_drs": (
            None if args.expect_native_physical_student_drs == "any"
            else args.expect_native_physical_student_drs == "true"
        ),
        "direct_recovery_evidence_native_gap_tolerance": args.expect_native_gap_tolerance,
        "direct_recovery_evidence_native_positive_gain": args.expect_native_positive_gain,
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
        "regime_routing": bool(actual["direct_recovery_value_regime_conditioning"]),
        "shared_deployment_rule_required": True,
        "noncompensatory_frontier_cap": args.expect_admission_prior_mode in {"frontier_capped_slack", "joint_reserve"},
        "deterministic_joint_reserve": args.expect_admission_prior_mode == "joint_reserve",
        "observation_conditioned_action_frontier": args.expect_context_source == "physical_interaction",
        "zero_action_no_scene_shortcut": args.expect_context_source == "physical_interaction",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if not mismatches else 4


if __name__ == "__main__":
    raise SystemExit(main())
