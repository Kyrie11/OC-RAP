#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON is not an object")
    return data


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _nearest(doc: dict[str, Any]) -> dict[str, Any] | None:
    rows = doc.get("near_miss_frontier") or []
    if not rows:
        return None
    return min(rows, key=lambda row: float(row.get("constraint_deficit", 1e9)))


def _development_summary(calibration: Path, regime: str) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    shared = calibration / "dev_frozen_shared_rule_v48.json"
    if shared.is_file():
        doc = _read(shared)
        fit = doc.get("fit") if isinstance(doc.get("fit"), dict) else {}
        by_stratum = fit.get("by_stratum") if isinstance(fit.get("by_stratum"), dict) else {}
        diagnostic = {
            "shared_rule": True,
            "rule": doc.get("rule") or doc.get("diagnostic_fit_rule") or fit.get("rule"),
            "valid": bool(doc.get("valid_for_deployment", doc.get("valid", False))),
            "constraint_failures": fit.get("constraint_failures"),
            "constraint_deficit": fit.get("constraint_deficit"),
            "stratum": by_stratum.get(regime),
            "pooled": fit.get("pooled"),
        }
        return doc, "shared", diagnostic

    legacy = calibration / f"dev_frozen_rule_{regime}_v48.json"
    if legacy.is_file():
        doc = _read(legacy)
        return doc, "legacy_regime_specific", _nearest(doc)
    raise FileNotFoundError(f"missing shared and legacy development rule under {calibration}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    variants: dict[str, Any] = {}
    overall: list[str] = []
    errors: list[str] = []
    shared_rule_modes: set[str] = set()
    for variant in ("balanced", "precision"):
        calibration = args.run / "candidates" / variant / "calibration"
        if not calibration.exists():
            continue
        regimes: dict[str, Any] = {}
        for regime in ("near", "contact"):
            try:
                certificate = _read(calibration / f"direct_value_risk_{regime}_v48.json")
                development, rule_mode, development_nearest = _development_summary(calibration, regime)
            except Exception as exc:
                errors.append(f"{variant}/{regime}: {exc!r}")
                regimes[regime] = {"artifact_error": repr(exc)}
                overall.append("artifact_or_protocol")
                continue
            shared_rule_modes.add(rule_mode)
            oracle = ((certificate.get("proposal_constrained_oracle_gate") or {}).get("verify") or {})
            development_valid = bool(development.get("valid_for_deployment", development.get("valid", False)))
            layer = (
                "proposal_or_contract"
                if not bool(oracle.get("feasible", False))
                else "development_rule_fit"
                if not development_valid
                else "certificate_generalization"
                if not bool(certificate.get("valid_for_deployment", False))
                else "passed"
            )
            regimes[regime] = {
                "failure_layer": layer,
                "rejection_kind": str(certificate.get("rejection_kind") or ""),
                "development_rule_mode": rule_mode,
                "proposal_oracle_feasible": bool(oracle.get("feasible", False)),
                "proposal_safe_positive_groups": oracle.get("proposal_safe_positive_groups"),
                "development_rule_valid": development_valid,
                "development_nearest_rule": development_nearest,
                "certificate_verify": certificate.get("verify"),
                "candidate_positive_auc": certificate.get("candidate_positive_auc"),
                "legacy_evidence_only_safe_positive_auc": certificate.get(
                    "legacy_evidence_only_top1_safe_positive_auc",
                    certificate.get("proposal_evidence_top1_safe_positive_auc"),
                ),
                "legacy_evidence_only_harm_auc": certificate.get("proposal_evidence_top1_harm_auc"),
                "legacy_evidence_only_correlation": certificate.get(
                    "legacy_evidence_only_top1_correlation",
                    certificate.get("proposal_evidence_top1_correlation"),
                ),
                "exact_eligible_safe_positive_auc": certificate.get("proposal_exact_eligible_top1_safe_positive_auc"),
                "exact_eligible_harm_auc": certificate.get("proposal_exact_eligible_top1_harm_auc"),
                "exact_eligible_correlation": certificate.get("proposal_exact_eligible_top1_correlation"),
                "exact_eligible_selected_count": certificate.get("proposal_exact_eligible_selected_count"),
                "exact_eligible_abstention_rate": certificate.get("proposal_exact_eligible_abstention_rate"),
            }
            overall.append(layer)
        variants[variant] = regimes

    artifact_valid = bool(overall) and "artifact_or_protocol" not in overall and not errors
    result = {
        "event": "v48_35_gate_failure_decomposition",
        "version": "v48.35.2-ENGINEERING-INTEGRITY",
        "created_unix": time.time(),
        "run": str(args.run),
        "artifact_valid": artifact_valid,
        "gate_passed": artifact_valid and all(layer == "passed" for layer in overall),
        "dominant_failure_layer": (
            "artifact_or_protocol"
            if "artifact_or_protocol" in overall or errors
            else "proposal_or_contract"
            if "proposal_or_contract" in overall
            else "development_rule_fit"
            if "development_rule_fit" in overall
            else "certificate_generalization"
            if "certificate_generalization" in overall
            else "passed"
            if overall
            else "missing_results"
        ),
        "development_rule_modes": sorted(shared_rule_modes),
        "variants": variants,
        "errors": errors,
        "test_roots_read": False,
    }
    _atomic_write(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if artifact_valid else 4


if __name__ == "__main__":
    raise SystemExit(main())
