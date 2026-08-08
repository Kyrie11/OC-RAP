#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def _history_metric(summary: dict[str, Any]) -> tuple[str, list[float]]:
    metric = str(summary.get("best_metric", ""))
    value_key = "direct_contract_safe_rank_risk" if metric == "direct_contract_lexicographic" else metric
    values: list[float] = []
    initial = summary.get("initial_checkpoint")
    if isinstance(initial, dict) and initial.get(value_key) is not None:
        value = float(initial[value_key])
        if math.isfinite(value):
            values.append(value)
    for row in summary.get("history", []):
        if not isinstance(row, dict) or not isinstance(row.get("val"), dict):
            continue
        raw = row["val"].get(value_key)
        if raw is None:
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return metric, values


def _natural(arch: dict[str, Any]) -> bool:
    return arch.get("group_batch_stratified") is False and arch.get("group_batching_replacement") is False


def _trainable(arch: dict[str, Any]) -> tuple[str, ...]:
    values = arch.get("trainable") or [""]
    raw = str(values[0])
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _torch_load_trusted_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load a locally produced OC-RAP checkpoint including its config payload."""
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected checkpoint mapping in {path}")
    return payload


def _checkpoint_exact_eligibility(path: Path) -> dict[str, Any]:
    payload = _torch_load_trusted_checkpoint(path)
    cfg = payload.get("cfg")
    if not isinstance(cfg, Mapping):
        raise TypeError(f"checkpoint cfg missing or not a mapping in {path}")
    training = cfg.get("training")
    if not isinstance(training, Mapping):
        raise TypeError(f"checkpoint training cfg missing or not a mapping in {path}")
    return {
        "checkpoint": str(path),
        "direct_policy_metric_exact_eligibility": training.get("direct_policy_metric_exact_eligibility"),
        "direct_policy_metric_risk_source": training.get("direct_policy_metric_risk_source"),
        "direct_policy_metric_proposal_top_k": training.get("direct_policy_metric_proposal_top_k"),
        "direct_policy_metric_evidence_rerank_top_k": training.get("direct_policy_metric_evidence_rerank_top_k"),
    }


def _exact_eligibility_contract(
    architectures: Sequence[Mapping[str, Any]], checkpoint_paths: Sequence[Path]
) -> dict[str, Any]:
    if len(architectures) != len(checkpoint_paths):
        raise ValueError("architectures/checkpoints length mismatch")
    checkpoint_records = [_checkpoint_exact_eligibility(path) for path in checkpoint_paths]
    metadata_exact_values = [a.get("exact_deployment_eligibility_metric") for a in architectures]
    metadata_exact = [value is True for value in metadata_exact_values]
    metadata_exact_absent = ["exact_deployment_eligibility_metric" not in a for a in architectures]
    metadata_legacy = [a.get("semantic_frontier_eligibility_metric") is True for a in architectures]
    checkpoint_exact = [r.get("direct_policy_metric_exact_eligibility") is True for r in checkpoint_records]
    # Legacy repair is allowed only when the new key is absent. An explicitly false
    # new key is contradictory metadata and must never be hidden by the legacy bit.
    metadata_supported = [
        new or (absent and legacy)
        for new, absent, legacy in zip(metadata_exact, metadata_exact_absent, metadata_legacy)
    ]
    return {
        "valid": all(checkpoint_exact) and all(metadata_supported),
        "checkpoint_exact_all_stages": all(checkpoint_exact),
        "metadata_exact_all_stages": all(metadata_exact),
        "metadata_exact_absent_all_stages": all(metadata_exact_absent),
        "metadata_legacy_semantic_all_stages": all(metadata_legacy),
        "metadata_exact_or_legacy_all_stages": all(metadata_supported),
        "metadata_contradiction_present": any(value is False for value in metadata_exact_values),
        "legacy_metadata_repair_used": any(metadata_exact_absent) and all(metadata_supported),
        "stages": [
            {
                **record,
                "architecture_exact_deployment_eligibility_metric": exact,
                "architecture_semantic_frontier_eligibility_metric": legacy,
            }
            for record, exact, legacy in zip(checkpoint_records, metadata_exact, metadata_legacy)
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed v48.36 continuous-frontier training contract audit")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect-identity-all", choices=("true", "false"), default="true")
    ap.add_argument("--expect-factor-preserving", choices=("true", "false"), default="false")
    ap.add_argument("--expect-reserve-only", choices=("true", "false"), default="false")
    ap.add_argument("--expect-benefit-margin-regression", type=float, default=0.0)
    ap.add_argument("--expect-benefit-margin-temperature", type=float, default=0.025)
    ap.add_argument("--expect-component-underestimation", type=float, default=0.0)
    ap.add_argument("--expect-safe-positive-component-overestimation", type=float, default=0.0)
    ap.add_argument("--expect-joint-reserve-regression", type=float, default=0.0)
    ap.add_argument("--expect-benefit-residual-scale", type=float, default=None)
    ap.add_argument("--expect-unbounded-benefit-factor", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-unbounded-harm-factors", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-reserve-factor-alignment", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-dual-interaction-bridge", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-factorized-harm-interaction", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-rank-benefit-skip", choices=("true", "false", "any"), default="any")
    ap.add_argument("--expect-rank-benefit-gain-init", type=float, default=None)
    ap.add_argument("--expect-component-margin-target-mode", default="any")
    ap.add_argument("--expect-component-margin-target-scale", type=float, default=None)
    ap.add_argument("--expect-algorithm-variant", default="")
    ap.add_argument("--expect-prior-coupled", choices=("true", "false"), default="true")
    ap.add_argument("--expect-adaptive-margin", choices=("true", "false"), default="false")
    ap.add_argument("--expect-final-enabled", choices=("true", "false"), default="false")
    ap.add_argument("--expect-eligible-policy", choices=("true", "false"), default="true")
    ap.add_argument("--expect-boundary", choices=("true", "false"), default="true")
    ap.add_argument("--expect-prior-mode", default="frontier_capped_slack")
    ap.add_argument("--expect-context-source", default="physical_interaction")
    ap.add_argument("--expect-best-metric", default="direct_contract_lexicographic")
    ap.add_argument("--expect-proposal-top-k", type=int, default=5)
    ap.add_argument("--expect-opportunity-threshold", type=float, default=0.50)
    ap.add_argument("--expect-harm-threshold", type=float, default=0.50)
    ap.add_argument("--reliability-override", default="")
    args = ap.parse_args()

    factor_arch = _load(args.run / "factor_stage" / "STAGE_ARCHITECTURE.json")
    identity_arch = _load(args.run / "identity_stage" / "STAGE_ARCHITECTURE.json")
    final_arch = _load(args.run / "STAGE_ARCHITECTURE.json")
    factor_summary = _load(args.run / "factor_stage" / "model_v48_trac_sr" / "train_summary.json")
    identity_summary = _load(args.run / "identity_stage" / "model_v48_trac_sr" / "train_summary.json")
    final_summary = _load(args.run / "model_v48_trac_sr" / "train_summary.json")
    factor_complete = _load(args.run / "factor_stage" / "TRAINING_COMPLETE.json")
    identity_complete = _load(args.run / "identity_stage" / "TRAINING_COMPLETE.json")
    final_complete = _load(args.run / "TRAINING_COMPLETE.json")
    transfer = _load(args.run / "STAGE_TRANSFER_INTEGRITY.json")
    three_stage = _load(args.run / "THREE_STAGE_TRAINING_COMPLETE.json")
    support = _load(args.run / "FACTOR_SUPPORT_CONTRACT.json")
    factor_cache = _load(args.run / "factor_stage" / "FACTOR_CACHE_CONTRACT.json")
    factor_cache_validation = _load(args.run / "FACTOR_CACHE_VALIDATION.json")

    factor_metric, factor_values = _history_metric(factor_summary)
    identity_metric, identity_values = _history_metric(identity_summary)
    final_metric, final_values = _history_metric(final_summary)
    reliability = [float(x) for x in support.get("reliability", [])]
    override = [float(x) for x in args.reliability_override.split(",") if x.strip()]
    expected_rel = override or reliability
    final_rel = [float(x) for x in str(final_arch.get("component_reliability", "")).split(",") if x.strip()]

    expect_identity_all = args.expect_identity_all == "true"
    expect_factor_preserving = args.expect_factor_preserving == "true"
    expect_reserve_only = args.expect_reserve_only == "true"
    expect_coupled = args.expect_prior_coupled == "true"
    expect_adaptive = args.expect_adaptive_margin == "true"
    expect_final = args.expect_final_enabled == "true"
    expect_eligible = args.expect_eligible_policy == "true"
    expect_boundary = args.expect_boundary == "true"
    expected_topk = int(args.expect_proposal_top_k)
    expected_unbounded_benefit = None if args.expect_unbounded_benefit_factor == "any" else args.expect_unbounded_benefit_factor == "true"
    expected_unbounded_harm = None if args.expect_unbounded_harm_factors == "any" else args.expect_unbounded_harm_factors == "true"
    expected_reserve_alignment = None if args.expect_reserve_factor_alignment == "any" else args.expect_reserve_factor_alignment == "true"
    expected_dual_bridge = None if args.expect_dual_interaction_bridge == "any" else args.expect_dual_interaction_bridge == "true"
    expected_factorized_harm = None if args.expect_factorized_harm_interaction == "any" else args.expect_factorized_harm_interaction == "true"
    expected_rank_benefit_skip = None if args.expect_rank_benefit_skip == "any" else args.expect_rank_benefit_skip == "true"
    expected_rank_benefit_gain_init = args.expect_rank_benefit_gain_init
    identity_trainable = _trainable(identity_arch)
    final_trainable = _trainable(final_arch)
    expected_ocaf = args.expect_context_source == "physical_interaction"
    expected_noncompensatory_cap = args.expect_prior_mode == "frontier_capped_slack"
    if args.expect_prior_mode == "joint_reserve":
        expected_noncompensatory_cap = True
    all_prefixes = (
        "direct_evidence_concord_benefit_calibrator",
        "direct_evidence_concord_harm_calibrator",
        "direct_evidence_concord_admission_calibrator",
    ) + (("direct_evidence_interaction_bridge",) if expected_ocaf else ())
    reference_prefixes = (
        "direct_evidence_concord_admission_calibrator",
    ) + (("direct_evidence_interaction_bridge",) if expected_ocaf else ())
    factor_preserving_prefixes = ("direct_evidence_concord_admission_calibrator",)
    expected_identity_prefixes = (
        set()
        if expect_reserve_only
        else set(
            all_prefixes
            if expect_identity_all
            else (factor_preserving_prefixes if expect_factor_preserving else reference_prefixes)
        )
    )
    exact_eligibility = _exact_eligibility_contract(
        (factor_arch, identity_arch, final_arch),
        (
            args.run / "factor_stage" / "model_v48_trac_sr" / "best.pt",
            args.run / "identity_stage" / "model_v48_trac_sr" / "best.pt",
            args.run / "model_v48_trac_sr" / "best.pt",
        ),
    )

    expected_algorithm_variant = str(args.expect_algorithm_variant).strip()

    checks = {
        "algorithm_variant_provenance": (
            not expected_algorithm_variant
            or all(
                str(a.get("algorithm_variant", "")) == expected_algorithm_variant
                for a in (factor_arch, identity_arch, final_arch)
            )
        ),
        "no_regime_routing_all_stages": all(
            a.get("regime_id_exposed_to_evidence_model") is False
            for a in (factor_arch, identity_arch, final_arch)
        ),
        "natural_stage1_sampling": _natural(factor_arch),
        "natural_stage2_sampling": _natural(identity_arch),
        "natural_stage3_sampling": _natural(final_arch),
        # v48.36 originally wrote only semantic_frontier_eligibility_metric even
        # though the actual checkpoint config enabled exact deployment eligibility.
        # New runs carry the exact metadata key; old v48.36 runs are repairable only
        # when every trusted checkpoint independently proves the exact config bit.
        "checkpoint_exact_eligibility_all_stages": exact_eligibility["checkpoint_exact_all_stages"],
        "exact_eligibility_metadata_supported_all_stages": exact_eligibility["metadata_exact_or_legacy_all_stages"],
        "exact_eligibility_all_stages": exact_eligibility["valid"],
        "factor_metric": factor_metric == "direct_factor_supervised_risk",
        "identity_contract_metric": identity_metric == (
            "direct_factor_supervised_risk" if expect_reserve_only else args.expect_best_metric
        ),
        "final_contract_metric": final_metric == (
            "direct_factor_supervised_risk" if expect_reserve_only else args.expect_best_metric
        ),
        "finite_factor_metric": bool(factor_values) and all(math.isfinite(x) for x in factor_values),
        "finite_identity_metric": bool(identity_values) and all(math.isfinite(x) for x in identity_values),
        "finite_final_metric": bool(final_values) and all(math.isfinite(x) for x in final_values),
        "factor_margin_regression_enabled": float(factor_arch.get("component_margin_regression_weight", 0.0)) > 0.0,
        "factor_benefit_margin_regression_contract": math.isclose(
            float(factor_arch.get("benefit_margin_regression_weight", 0.0)),
            float(args.expect_benefit_margin_regression), rel_tol=0.0, abs_tol=1.0e-9,
        ),
        "factor_benefit_margin_temperature_contract": math.isclose(
            float(factor_arch.get("benefit_margin_temperature", 0.025)),
            float(args.expect_benefit_margin_temperature), rel_tol=0.0, abs_tol=1.0e-9,
        ),
        "factor_component_underestimation_contract": math.isclose(
            float(factor_arch.get("component_underestimation_weight", 0.0)),
            float(args.expect_component_underestimation), rel_tol=0.0, abs_tol=1.0e-9,
        ),
        "factor_safe_positive_component_overestimation_contract": math.isclose(
            float(factor_arch.get("safe_positive_component_overestimation_weight", 0.0)),
            float(args.expect_safe_positive_component_overestimation), rel_tol=0.0, abs_tol=1.0e-9,
        ),
        "factor_joint_reserve_regression_contract": math.isclose(
            float(factor_arch.get("joint_reserve_regression_weight", 0.0)),
            float(args.expect_joint_reserve_regression), rel_tol=0.0, abs_tol=1.0e-9,
        ),
        "factor_benefit_residual_scale_contract": (
            args.expect_benefit_residual_scale is None
            or math.isclose(float(factor_arch.get("benefit_residual_scale", -1.0)), float(args.expect_benefit_residual_scale), rel_tol=0.0, abs_tol=1.0e-9)
        ),
        "unbounded_benefit_factor_contract": (
            expected_unbounded_benefit is None
            or all(bool(a.get("unbounded_benefit_factor", False)) is expected_unbounded_benefit for a in (factor_arch, identity_arch, final_arch))
        ),
        "unbounded_harm_factors_contract": (
            expected_unbounded_harm is None
            or all(bool(a.get("unbounded_harm_factors", False)) is expected_unbounded_harm for a in (factor_arch, identity_arch, final_arch))
        ),
        "reserve_factor_alignment_contract": (
            expected_reserve_alignment is None
            or all(bool(a.get("reserve_factor_alignment", False)) is expected_reserve_alignment for a in (factor_arch, identity_arch, final_arch))
        ),
        "dual_interaction_bridge_contract": (
            expected_dual_bridge is None
            or all(bool(a.get("dual_interaction_bridge", False)) is expected_dual_bridge for a in (factor_arch, identity_arch, final_arch))
        ),
        "factorized_harm_interaction_contract": (
            expected_factorized_harm is None
            or all(bool(a.get("factorized_harm_interaction", False)) is expected_factorized_harm for a in (factor_arch, identity_arch, final_arch))
        ),
        "rank_benefit_skip_contract": (
            expected_rank_benefit_skip is None
            or all(bool(a.get("rank_benefit_skip", False)) is expected_rank_benefit_skip for a in (factor_arch, identity_arch, final_arch))
        ),
        "rank_benefit_gain_init_contract": (
            expected_rank_benefit_gain_init is None
            or all(
                math.isclose(float(a.get("rank_benefit_gain_init", -1.0)), float(expected_rank_benefit_gain_init), rel_tol=0.0, abs_tol=1.0e-9)
                for a in (factor_arch, identity_arch, final_arch)
            )
        ),
        "component_margin_target_mode_contract": (
            args.expect_component_margin_target_mode == "any"
            or str(factor_arch.get("component_margin_target_mode", "raw")) == args.expect_component_margin_target_mode
        ),
        "component_margin_target_scale_contract": (
            args.expect_component_margin_target_scale is None
            or math.isclose(float(factor_arch.get("component_margin_target_scale", -1.0)), float(args.expect_component_margin_target_scale), rel_tol=0.0, abs_tol=1.0e-9)
        ),
        "reserve_only_architecture_contract": (
            all(
                a.get("identity_stage_skipped") is True
                and a.get("deterministic_joint_reserve") is True
                and a.get("learned_admission_residual") is False
                for a in (identity_arch, final_arch)
            )
            if expect_reserve_only
            else True
        ),
        "factor_preserving_bridge_contract": (
            identity_arch.get("interaction_bridge_trainable_this_stage") is False
            if expect_factor_preserving and expected_ocaf
            else True
        ),
        "identity_trainable_contract": set(identity_trainable) == expected_identity_prefixes,
        "identity_prior_gradient_contract": bool(identity_arch.get("admission_prior_detach", True)) is (not expect_coupled),
        "identity_adaptive_margin_contract": (
            float(identity_arch.get("safe_hard_negative_teacher_scale", 0.0)) > 0.0
        ) is expect_adaptive,
        "unified_proposal_top_k": all(
            int(a.get("proposal_top_k", 0)) == expected_topk
            for a in (factor_arch, identity_arch, final_arch)
        ),
        "eligible_set_policy_contract": (
            float(identity_arch.get("eligible_set_policy_weight", 0.0)) > 0.0
        ) is expect_eligible,
        "hard_boundary_continuation_enabled": (
            float(identity_arch.get("eligibility_boundary_weight", 0.0)) > 0.0
        ) is expect_boundary,
        "scene_invariant_evidence_context": all(
            a.get("context_source") == args.expect_context_source
            for a in (factor_arch, identity_arch, final_arch)
        ),
        "candidate_context_family_contract": all(
            a.get("context_source") == args.expect_context_source
            and (a.get("observation_conditioned_action_frontier") is True) is expected_ocaf
            and (a.get("interaction_zero_action_exact_zero") is True) is expected_ocaf
            and (a.get("interaction_scene_only_shortcut_forbidden") is True) is expected_ocaf
            for a in (factor_arch, identity_arch, final_arch)
        ),
        "noncompensatory_frontier_cap": all(
            (a.get("noncompensatory_frontier_cap") is True) is expected_noncompensatory_cap
            for a in (factor_arch, identity_arch, final_arch)
        ),
        "single_shared_rule_required": all(
            a.get("shared_deployment_rule_required") is True
            for a in (factor_arch, identity_arch, final_arch)
        ),
        "eligible_set_policy_runtime_order": all(
            a.get("selection_semantics") == "rank_topk_then_filter_then_evidence_rerank"
            for a in (factor_arch, identity_arch, final_arch)
        ),
        "eligibility_thresholds_match_runtime": (
            math.isclose(
                float(identity_arch.get("eligible_opportunity_threshold", -1.0)),
                float(args.expect_opportunity_threshold), abs_tol=1e-9,
            )
            and math.isclose(
                float(identity_arch.get("eligible_harm_threshold", -1.0)),
                float(args.expect_harm_threshold), abs_tol=1e-9,
            )
        ),
        "final_admission_only_or_disabled": (
            set(final_trainable) == {"direct_evidence_concord_admission_calibrator"}
            if expect_final
            else set(final_trainable) == set(identity_trainable)
        ),
        "unbounded_residual_with_frontier_cap": (
            identity_arch.get("admission_head") is False
            and final_arch.get("admission_head") is False
            and identity_arch.get("deterministic_joint_reserve") is True
            and final_arch.get("deterministic_joint_reserve") is True
            if expect_reserve_only
            else identity_arch.get("admission_residual_bounded") is False
            and final_arch.get("admission_residual_bounded") is False
        ),
        "admission_prior_mode": identity_arch.get("admission_prior_mode") == args.expect_prior_mode and final_arch.get("admission_prior_mode") == args.expect_prior_mode,
        "five_harm_factors": all(int(a.get("component_harm_count", 0)) == 5 for a in (factor_arch, identity_arch, final_arch)),
        "support_contract_has_five_coordinates": len(reliability) == 5,
        "support_contract_preserves_measured_hard_veto": support.get("independent_measured_hard_veto_preserved") is True,
        "component_reliability_propagated": len(final_rel) == 5 and len(expected_rel) == 5 and all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-7) for a, b in zip(final_rel, expected_rel)
        ),
        "stage_transfer_valid": bool(transfer.get("valid", False)),
        "no_op_identity_is_accepted": transfer.get("no_op_identity_selection_is_valid") is True,
        "no_op_final_is_accepted": transfer.get("no_op_final_selection_is_valid") is True,
        "factor_cache_contract_registered": factor_cache.get("version") == "v48.32-IDENTITY-UTILITY-BRIDGE",
        "factor_cache_validation_passed": factor_cache_validation.get("valid") is True,
        "factor_checkpoint_not_epoch_zero": int(factor_complete.get("best_epoch", 0)) > 0,
        "identity_checkpoint_registered": int(identity_complete.get("best_epoch", -1)) >= 0,
        "final_checkpoint_registered": int(final_complete.get("best_epoch", -1)) >= 0,
        "final_enabled_metadata": bool(three_stage.get("final_calibration_enabled", False)) is expect_final,
        "reserve_only_metadata": bool(three_stage.get("reserve_only", False)) is expect_reserve_only,
        "coupled_metadata": bool(three_stage.get("deployment_safe_utility_gradient_coupled", False)) is expect_coupled,
        "adaptive_metadata": bool(three_stage.get("adaptive_teacher_gap_margin", False)) is expect_adaptive,
        "legacy_noisy_or_disabled": all(a.get("noisy_or_group_objective_disabled") is True for a in (factor_arch, identity_arch, final_arch)),
        "test_roots_sealed": all(d.get("test_roots_read") is False for d in (factor_arch, identity_arch, final_arch, three_stage)),
    }
    doc = {
        "event": "v48_36_training_contract_audit",
        "run": str(args.run),
        "valid": all(checks.values()),
        "checks": checks,
        "metrics": {
            "factor": {"name": factor_metric, "values": factor_values, "best_epoch": factor_summary.get("best_epoch")},
            "identity": {"name": identity_metric, "values": identity_values, "best_epoch": identity_summary.get("best_epoch")},
            "final": {"name": final_metric, "values": final_values, "best_epoch": final_summary.get("best_epoch")},
        },
        "support_reliability": reliability,
        "expected_runtime_reliability": expected_rel,
        "exact_eligibility_provenance": exact_eligibility,
        "expected": {
            "identity_all": expect_identity_all,
            "factor_preserving_identity": expect_factor_preserving,
            "reserve_only": expect_reserve_only,
            "benefit_margin_regression_weight": float(args.expect_benefit_margin_regression),
            "benefit_margin_temperature": float(args.expect_benefit_margin_temperature),
            "component_underestimation_weight": float(args.expect_component_underestimation),
            "safe_positive_component_overestimation_weight": float(args.expect_safe_positive_component_overestimation),
            "joint_reserve_regression_weight": float(args.expect_joint_reserve_regression),
            "benefit_residual_scale": args.expect_benefit_residual_scale,
            "unbounded_benefit_factor": expected_unbounded_benefit,
            "unbounded_harm_factors": expected_unbounded_harm,
            "reserve_factor_alignment": expected_reserve_alignment,
            "algorithm_variant": expected_algorithm_variant,
            "identity_trainable_prefixes": sorted(expected_identity_prefixes),
            "prior_coupled": expect_coupled,
            "adaptive_margin": expect_adaptive,
            "final_enabled": expect_final,
            "eligible_policy": expect_eligible,
            "boundary": expect_boundary,
            "prior_mode": args.expect_prior_mode,
            "context_source": args.expect_context_source,
            "best_metric": args.expect_best_metric,
            "proposal_top_k": expected_topk,
            "opportunity_threshold": float(args.expect_opportunity_threshold),
            "harm_threshold": float(args.expect_harm_threshold),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if doc["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
