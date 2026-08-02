#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def _history_metric(summary: dict[str, Any]) -> tuple[str, list[float]]:
    metric = str(summary.get("best_metric", ""))
    values: list[float] = []
    initial = summary.get("initial_checkpoint")
    if isinstance(initial, dict) and initial.get(metric) is not None:
        value = float(initial[metric])
        if math.isfinite(value):
            values.append(value)
    for row in summary.get("history", []):
        if not isinstance(row, dict) or not isinstance(row.get("val"), dict):
            continue
        raw = row["val"].get(metric)
        if raw is None:
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return metric, values


def _natural(arch: dict[str, Any]) -> bool:
    return arch.get("group_batch_stratified") is False and arch.get("group_batching_replacement") is False


def _trainable(arch: dict[str, Any]) -> str:
    values = arch.get("trainable") or [""]
    return str(values[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed v48.32 identity-utility training contract audit")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect-identity-all", choices=("true", "false"), default="true")
    ap.add_argument("--expect-prior-coupled", choices=("true", "false"), default="true")
    ap.add_argument("--expect-adaptive-margin", choices=("true", "false"), default="true")
    ap.add_argument("--expect-final-enabled", choices=("true", "false"), default="true")
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
    expect_coupled = args.expect_prior_coupled == "true"
    expect_adaptive = args.expect_adaptive_margin == "true"
    expect_final = args.expect_final_enabled == "true"
    identity_trainable = _trainable(identity_arch)
    final_trainable = _trainable(final_arch)
    all_prefixes = (
        "direct_evidence_concord_benefit_calibrator",
        "direct_evidence_concord_harm_calibrator",
        "direct_evidence_concord_admission_calibrator",
    )

    checks = {
        "no_regime_routing_all_stages": all(
            a.get("regime_id_exposed_to_evidence_model") is False
            for a in (factor_arch, identity_arch, final_arch)
        ),
        "natural_stage1_sampling": _natural(factor_arch),
        "natural_stage2_sampling": _natural(identity_arch),
        "natural_stage3_sampling": _natural(final_arch),
        "exact_eligibility_all_stages": all(
            a.get("exact_deployment_eligibility_metric") is True
            for a in (factor_arch, identity_arch, final_arch)
        ),
        "factor_metric": factor_metric == "direct_factor_supervised_risk",
        "identity_contract_metric": identity_metric == "direct_contract_safe_rank_risk",
        "final_contract_metric": final_metric == "direct_contract_safe_rank_risk",
        "finite_factor_metric": bool(factor_values) and all(math.isfinite(x) for x in factor_values),
        "finite_identity_metric": bool(identity_values) and all(math.isfinite(x) for x in identity_values),
        "finite_final_metric": bool(final_values) and all(math.isfinite(x) for x in final_values),
        "factor_margin_regression_enabled": float(factor_arch.get("component_margin_regression_weight", 0.0)) > 0.0,
        "identity_trainable_contract": (
            all(prefix in identity_trainable for prefix in all_prefixes)
            if expect_identity_all
            else identity_trainable == "direct_evidence_concord_admission_calibrator"
        ),
        "identity_prior_gradient_contract": bool(identity_arch.get("admission_prior_detach", True)) is (not expect_coupled),
        "identity_adaptive_margin_contract": (
            float(identity_arch.get("safe_hard_negative_teacher_scale", 0.0)) > 0.0
        ) is expect_adaptive,
        "final_admission_only_or_disabled": (
            final_trainable == "direct_evidence_concord_admission_calibrator"
            if expect_final
            else final_trainable == identity_trainable
        ),
        "bounded_admission": identity_arch.get("admission_residual_bounded") is True and final_arch.get("admission_residual_bounded") is True,
        "safety_slack_prior": identity_arch.get("admission_prior_mode") == "safety_slack" and final_arch.get("admission_prior_mode") == "safety_slack",
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
        "coupled_metadata": bool(three_stage.get("deployment_safe_utility_gradient_coupled", False)) is expect_coupled,
        "adaptive_metadata": bool(three_stage.get("adaptive_teacher_gap_margin", False)) is expect_adaptive,
        "legacy_noisy_or_disabled": all(a.get("noisy_or_group_objective_disabled") is True for a in (factor_arch, identity_arch, final_arch)),
        "test_roots_sealed": all(d.get("test_roots_read") is False for d in (factor_arch, identity_arch, final_arch, three_stage)),
    }
    doc = {
        "event": "v48_32_training_contract_audit",
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
        "expected": {
            "identity_all": expect_identity_all,
            "prior_coupled": expect_coupled,
            "adaptive_margin": expect_adaptive,
            "final_enabled": expect_final,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if doc["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
