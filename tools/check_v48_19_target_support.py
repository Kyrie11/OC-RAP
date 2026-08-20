#!/usr/bin/env python3
"""Validate FACET teacher-index identity and binary target support.

The contract check prevents a stale teacher index from being reused after the
training roots, manifest contents, gain threshold, macro set, or component-veto
tolerances change.  The support check is a protocol/data-contract check, not a
performance gate: it rejects only mathematically unidentifiable binary tails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


def _audit_bucket(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    support = (summary.get("factorized_harm_support_by_bucket") or {}).get(name) or {}
    total = int(support.get("deployable_candidates", 0) or 0)
    beneficial = int(support.get("beneficial_candidates", 0) or 0)
    harmful = int(support.get("component_harmful_candidates", 0) or 0)
    overlap = int(support.get("overlap_candidates", 0) or 0)
    safe_beneficial_raw = support.get("safe_beneficial_candidates")
    safe_beneficial = int(safe_beneficial_raw or 0) if safe_beneficial_raw is not None else None
    failures: list[str] = []
    if total <= 0:
        failures.append("no deployable recovery candidates")
    if beneficial <= 0:
        failures.append("benefit tail has no positive examples")
    if total > 0 and beneficial >= total:
        failures.append("benefit tail has no negative examples")
    if harmful <= 0:
        failures.append("harm tail has no positive examples")
    if total > 0 and harmful >= total:
        failures.append("harm tail has no negative examples")
    warnings: list[str] = []
    if overlap <= 0:
        warnings.append("no observed benefit/harm overlap; independent-tail novelty is not exercised in this regime")
    if safe_beneficial is not None and safe_beneficial <= 0:
        failures.append("safe-benefit admission target has no positive examples")
    return {
        "regime": name,
        "deployable_candidates": total,
        "beneficial_candidates": beneficial,
        "component_harmful_candidates": harmful,
        "overlap_candidates": overlap,
        "safe_beneficial_candidates": safe_beneficial,
        "safe_beneficial_groups": int(support.get("safe_beneficial_groups", 0) or 0) if safe_beneficial is not None else None,
        "safe_beneficial_scenes": int(support.get("safe_beneficial_scenes", 0) or 0) if safe_beneficial is not None else None,
        "benefit_prevalence": beneficial / total if total else None,
        "safe_benefit_prevalence": (safe_beneficial / total) if total and safe_beneficial is not None else None,
        "harm_prevalence": harmful / total if total else None,
        "overlap_prevalence": overlap / total if total else None,
        "component_harmful_groups": int(support.get("component_harmful_groups", 0) or 0),
        "overlap_groups": int(support.get("beneficial_and_component_harmful_groups", 0) or 0),
        "learnable": not failures,
        "failures": failures,
        "warnings": warnings,
    }


def _manifest_record(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    manifest = resolved / "manifest.csv"
    return {
        "root": str(resolved),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
    }


def _float_equal(a: Any, b: Any) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-12)
    except (TypeError, ValueError):
        return False


def _contract_audit(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    actual = summary.get("index_contract") or {}
    roots = [Path(x.strip()) for x in str(args.expected_dataset or "").split(",") if x.strip()]
    expected = {
        "dataset_roots": [str(x.resolve()) for x in roots],
        "dataset_manifests": [_manifest_record(x) for x in roots],
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "top_m": int(args.top_m),
        "positive_gain": float(args.positive_gain),
        "deployable_macro_ids": sorted(int(x.strip()) for x in str(args.deployable_macro_ids).split(",") if x.strip()),
        "component_harm_tolerances": {
            "drs": float(args.component_harm_drs_tolerance),
            "deployability_gate": float(args.component_harm_dep_tolerance),
            "gap_discount": float(args.component_harm_gap_tolerance),
            "hard_violation": float(args.component_harm_hard_tolerance),
            "harm_proxy": float(args.component_harm_proxy_tolerance),
            "deployability_boundary_aligned": bool(args.dep_boundary_aligned),
            "gap_ordinal_only": bool(args.gap_ordinal_only),
        },
    }
    failures: list[str] = []
    if not actual:
        failures.append("teacher-index summary has no index_contract")
    for key in ("dataset_roots", "dataset_manifests", "deployable_macro_ids"):
        if actual.get(key) != expected[key]:
            failures.append(f"index contract mismatch: {key}")
    for key in ("alpha", "beta", "positive_gain"):
        if not _float_equal(actual.get(key), expected[key]):
            failures.append(f"index contract mismatch: {key}")
    if int(actual.get("top_m", -1) or -1) != expected["top_m"]:
        failures.append("index contract mismatch: top_m")
    actual_tol = actual.get("component_harm_tolerances") or {}
    for key, value in expected["component_harm_tolerances"].items():
        if isinstance(value, bool):
            if bool(actual_tol.get(key, False)) != value:
                failures.append(f"index contract mismatch: component_harm_tolerances.{key}")
        elif not _float_equal(actual_tol.get(key), value):
            failures.append(f"index contract mismatch: component_harm_tolerances.{key}")
    return {"valid": not failures, "expected": expected, "actual": actual, "failures": failures}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--mode", choices=["contract", "support", "all"], default="all")
    ap.add_argument("--expected-dataset", default="")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--top-m", type=int, default=8)
    ap.add_argument("--positive-gain", type=float, default=0.015)
    ap.add_argument("--deployable-macro-ids", default="2,3,5,6,7")
    ap.add_argument("--component-harm-drs-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-dep-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-gap-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-hard-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-proxy-tolerance", type=float, default=0.05)
    ap.add_argument("--dep-boundary-aligned", action="store_true")
    ap.add_argument("--gap-ordinal-only", action="store_true")
    args = ap.parse_args()
    try:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"cannot read teacher-index summary {args.summary}: {exc}")

    contract = _contract_audit(args, summary) if args.mode in {"contract", "all"} else None
    buckets = (
        {name: _audit_bucket(name, summary) for name in ("near", "contact")}
        if args.mode in {"support", "all"} else {}
    )
    contract_valid = True if contract is None else bool(contract["valid"])
    support_valid = all(bool(row["learnable"]) for row in buckets.values()) if buckets else True
    valid = contract_valid and support_valid
    doc = {
        "event": "v48_19_factorized_target_support_audit",
        "created_unix": time.time(),
        "summary": str(args.summary),
        "mode": args.mode,
        "contract": contract,
        "valid_for_training": valid,
        "buckets": buckets,
        "test_roots_read": False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False), flush=True)
    if not contract_valid:
        return 5
    return 0 if support_valid else 4


if __name__ == "__main__":
    raise SystemExit(main())
