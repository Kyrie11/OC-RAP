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


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed v48.31 three-stage training contract audit")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allow-no-joint", action="store_true")
    ap.add_argument("--reliability-override", default="")
    args = ap.parse_args()

    factor_arch = _load(args.run / "factor_stage" / "STAGE_ARCHITECTURE.json")
    admission_arch = _load(args.run / "admission_stage" / "STAGE_ARCHITECTURE.json")
    final_arch = _load(args.run / "STAGE_ARCHITECTURE.json")
    factor_summary = _load(args.run / "factor_stage" / "model_v48_trac_sr" / "train_summary.json")
    admission_summary = _load(args.run / "admission_stage" / "model_v48_trac_sr" / "train_summary.json")
    final_summary = _load(args.run / "model_v48_trac_sr" / "train_summary.json")
    factor_complete = _load(args.run / "factor_stage" / "TRAINING_COMPLETE.json")
    admission_complete = _load(args.run / "admission_stage" / "TRAINING_COMPLETE.json")
    final_complete = _load(args.run / "TRAINING_COMPLETE.json")
    stage_transfer = _load(args.run / "STAGE_TRANSFER_INTEGRITY.json")
    three_stage = _load(args.run / "THREE_STAGE_TRAINING_COMPLETE.json")
    support = _load(args.run / "FACTOR_SUPPORT_CONTRACT.json")

    factor_metric, factor_values = _history_metric(factor_summary)
    admission_metric, admission_values = _history_metric(admission_summary)
    final_metric, final_values = _history_metric(final_summary)
    reliability = [float(x) for x in support.get("reliability", [])]
    override = [float(x) for x in args.reliability_override.split(",") if x.strip()]
    final_rel = [float(x) for x in str(final_arch.get("component_reliability", "")).split(",") if x.strip()]
    expected_rel = override or reliability

    final_trainable = str((final_arch.get("trainable") or [""])[0])
    joint_enabled = bool(three_stage.get("joint_refinement_enabled", False))
    checks = {
        "no_regime_routing_all_stages": all(a.get("regime_id_exposed_to_evidence_model") is False for a in (factor_arch, admission_arch, final_arch)),
        "natural_stage1_sampling": _natural(factor_arch),
        "natural_stage2_sampling": _natural(admission_arch),
        "natural_stage3_sampling": _natural(final_arch),
        "exact_eligibility_all_stages": all(a.get("exact_deployment_eligibility_metric") is True for a in (factor_arch, admission_arch, final_arch)),
        "factor_metric": factor_metric == "direct_factor_supervised_risk",
        "admission_contract_metric": admission_metric == "direct_contract_safe_rank_risk",
        "final_contract_metric": final_metric == "direct_contract_safe_rank_risk",
        "finite_factor_metric": bool(factor_values) and all(math.isfinite(x) for x in factor_values),
        "finite_admission_metric": bool(admission_values) and all(math.isfinite(x) for x in admission_values),
        "finite_final_metric": bool(final_values) and all(math.isfinite(x) for x in final_values),
        "factor_margin_regression_enabled": float(factor_arch.get("component_margin_regression_weight", 0.0)) > 0.0,
        "stage2_admission_only": str((admission_arch.get("trainable") or [""])[0]) == "direct_evidence_concord_admission_calibrator",
        "stage3_joint_or_registered_ablation": (
            all(x in final_trainable for x in ("direct_evidence_concord_benefit_calibrator", "direct_evidence_concord_harm_calibrator", "direct_evidence_concord_admission_calibrator"))
            if joint_enabled else args.allow_no_joint
        ),
        "bounded_admission": admission_arch.get("admission_residual_bounded") is True and final_arch.get("admission_residual_bounded") is True,
        "safety_slack_prior": admission_arch.get("admission_prior_mode") == "safety_slack" and final_arch.get("admission_prior_mode") == "safety_slack",
        "five_harm_factors": all(int(a.get("component_harm_count", 0)) == 5 for a in (factor_arch, admission_arch, final_arch)),
        "support_contract_has_five_coordinates": len(reliability) == 5,
        "support_contract_preserves_measured_hard_veto": support.get("independent_measured_hard_veto_preserved") is True,
        "component_reliability_propagated": len(final_rel) == 5 and len(expected_rel) == 5 and all(math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-7) for a, b in zip(final_rel, expected_rel)),
        "stage_transfer_valid": bool(stage_transfer.get("valid", False)),
        "factor_checkpoint_not_epoch_zero": int(factor_complete.get("best_epoch", 0)) > 0,
        "admission_checkpoint_not_epoch_zero": int(admission_complete.get("best_epoch", 0)) > 0,
        "final_checkpoint_registered": int(final_complete.get("best_epoch", 0)) >= 0,
        "legacy_noisy_or_disabled": all(a.get("noisy_or_group_objective_disabled") is True for a in (factor_arch, admission_arch, final_arch)),
        "test_roots_sealed": all(d.get("test_roots_read") is False for d in (factor_arch, admission_arch, final_arch, three_stage)),
    }
    doc = {
        "event": "v48_31_training_contract_audit",
        "run": str(args.run),
        "valid": all(checks.values()),
        "checks": checks,
        "metrics": {
            "factor": {"name": factor_metric, "values": factor_values, "best_epoch": factor_summary.get("best_epoch")},
            "admission": {"name": admission_metric, "values": admission_values, "best_epoch": admission_summary.get("best_epoch")},
            "final": {"name": final_metric, "values": final_values, "best_epoch": final_summary.get("best_epoch")},
        },
        "support_reliability": reliability,
        "expected_runtime_reliability": expected_rel,
        "joint_refinement_enabled": joint_enabled,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if doc["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
