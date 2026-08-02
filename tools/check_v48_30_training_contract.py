#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed v48.30 two-stage training contract audit")
    ap.add_argument("--run", type=Path, required=True, help="candidate variant run directory")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    final_arch = _load(args.run / "STAGE_ARCHITECTURE.json")
    factor_arch = _load(args.run / "factor_stage" / "STAGE_ARCHITECTURE.json")
    summary = _load(args.run / "model_v48_trac_sr" / "train_summary.json")
    factor_complete = _load(args.run / "factor_stage" / "TRAINING_COMPLETE.json")
    transfer = _load(args.run / "FACTOR_TRANSFER_INTEGRITY.json")

    best_metric = str(summary.get("best_metric", ""))
    metric_values = [
        float((row.get("val", {}) or {}).get(best_metric))
        for row in summary.get("history", [])
        if isinstance(row, dict)
        and isinstance(row.get("val", {}), dict)
        and (row.get("val", {}) or {}).get(best_metric) is not None
    ]

    checks = {
        "no_regime_routing": final_arch.get("regime_id_exposed_to_evidence_model") is False,
        "natural_stage2_sampling": final_arch.get("group_batch_stratified") is False,
        "stage2_without_replacement": final_arch.get("group_batching_replacement") is False,
        "safety_slack_prior": final_arch.get("admission_prior_mode") == "safety_slack",
        "bounded_admission": final_arch.get("admission_residual_bounded") is True,
        "population_checkpoint_metric": best_metric == "direct_population_safe_rank_risk",
        "finite_checkpoint_metric": bool(metric_values) and all(math.isfinite(v) for v in metric_values),
        "checkpoint_metric_varies": len({round(v, 10) for v in metric_values}) > 1,
        "factor_margin_regression_enabled": float(
            factor_arch.get("component_margin_regression_weight", 0.0)
        ) > 0.0,
        "factor_checkpoint_not_epoch_zero": int(factor_complete.get("best_epoch", 0)) > 0,
        "factor_transfer_valid": bool(transfer.get("valid", False)),
        "five_harm_factors": int(final_arch.get("component_harm_count", 0)) == 5,
        "legacy_noisy_or_disabled": final_arch.get("noisy_or_group_objective_disabled") is True,
    }
    doc = {
        "event": "v48_30_training_contract_audit",
        "run": str(args.run),
        "valid": all(checks.values()),
        "checks": checks,
        "best_epoch": summary.get("best_epoch"),
        "best_metric": best_metric,
        "best_metric_values": metric_values,
        "factor_best_epoch": factor_complete.get("best_epoch"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if doc["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
