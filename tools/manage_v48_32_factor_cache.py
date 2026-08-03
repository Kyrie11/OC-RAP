#!/usr/bin/env python3
"""Create and verify an exact v48.32 Stage-1 factor-cache contract.

The cache is valid only when every input that can change optimization or
checkpoint selection is identical.  Paths are recorded for diagnosis, while
file SHA256 digests and normalized hyperparameters define the contract hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalise(value: str) -> Any:
    text = value.strip()
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if any(c in text for c in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _payload(args: argparse.Namespace) -> dict[str, Any]:
    support = json.loads(args.support_contract.read_text(encoding="utf-8"))
    if not isinstance(support, dict):
        raise TypeError("support contract must be a JSON object")
    settings: dict[str, Any] = {}
    for item in args.setting:
        if "=" not in item:
            raise ValueError(f"invalid --setting {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        settings[key] = _normalise(value)
    support_identity = {
        key: support.get(key)
        for key in (
            "version", "semantic", "num_rows", "num_groups",
            "num_eligible_candidates", "component_tolerances", "eligibility",
            "components", "component_order", "reliability",
            "independent_measured_hard_veto_preserved", "skipped",
        )
    }
    support_encoded = json.dumps(
        support_identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    identity = {
        "version": "v48.32-IDENTITY-UTILITY-BRIDGE",
        "source_checkpoint_sha256": _sha(args.source_checkpoint),
        "group_index_sha256": _sha(args.group_index),
        "validation_group_index_sha256": _sha(args.validation_group_index),
        "support_contract_semantic_sha256": hashlib.sha256(support_encoded).hexdigest(),
        "support_reliability": support.get("reliability"),
        "train_mix": args.train_mix,
        "validation_mix": args.validation_mix,
        "variant": args.variant,
        "settings": dict(sorted(settings.items())),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        **identity,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "group_index": str(args.group_index.resolve()),
        "validation_group_index": str(args.validation_group_index.resolve()),
        "support_contract": str(args.support_contract.resolve()),
        "support_contract_file_sha256": _sha(args.support_contract),
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("create", "verify", "verify-reuse"), required=True)
    ap.add_argument("--source-checkpoint", type=Path, required=True)
    ap.add_argument("--group-index", type=Path, required=True)
    ap.add_argument("--validation-group-index", type=Path, required=True)
    ap.add_argument("--support-contract", type=Path, required=True)
    ap.add_argument("--train-mix", required=True)
    ap.add_argument("--validation-mix", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--setting", action="append", default=[])
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    expected = _payload(args)
    report: dict[str, Any] = {
        "event": "v48_32_factor_cache_contract",
        "mode": args.mode,
        "valid": True,
        "expected_contract_sha256": expected["contract_sha256"],
        "contract": str(args.contract),
    }
    if args.mode == "create":
        args.contract.parent.mkdir(parents=True, exist_ok=True)
        args.contract.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        try:
            actual = json.loads(args.contract.read_text(encoding="utf-8"))
            report["actual_contract_sha256"] = actual.get("contract_sha256")
            if args.mode == "verify":
                report["valid"] = actual.get("contract_sha256") == expected["contract_sha256"]
                if not report["valid"]:
                    report["reason"] = "factor cache inputs or hyperparameters changed"
            else:
                # v48.34: a frozen Stage-1 artifact owns its Stage-1 optimization
                # settings.  A Stage-2 ablation must not invalidate that artifact
                # merely because the caller has different batch-size/defaults.
                # Verify the source contract against its own recorded settings,
                # then compare only inputs that semantically determine which
                # examples/labels/support contract the frozen factor saw.
                identity_keys = (
                    "version", "source_checkpoint_sha256", "group_index_sha256",
                    "validation_group_index_sha256", "support_contract_semantic_sha256",
                    "support_reliability", "train_mix", "validation_mix", "variant", "settings",
                )
                actual_identity = {key: actual.get(key) for key in identity_keys}
                actual_encoded = json.dumps(
                    actual_identity, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                self_hash = hashlib.sha256(actual_encoded).hexdigest()
                report["source_contract_self_hash_valid"] = self_hash == actual.get("contract_sha256")
                semantic_keys = (
                    "source_checkpoint_sha256", "group_index_sha256",
                    "validation_group_index_sha256", "support_contract_semantic_sha256",
                    "support_reliability", "train_mix", "validation_mix", "variant",
                )
                mismatches = {
                    key: {"source": actual.get(key), "current": expected.get(key)}
                    for key in semantic_keys
                    if actual.get(key) != expected.get(key)
                }
                report["semantic_mismatches"] = mismatches
                report["source_settings"] = actual.get("settings", {})
                report["caller_settings_ignored_for_reuse"] = expected.get("settings", {})
                report["valid"] = bool(report["source_contract_self_hash_valid"] and not mismatches)
                if not report["valid"]:
                    report["reason"] = (
                        "factor cache source contract is corrupt or semantic inputs changed"
                    )
        except Exception as exc:
            report["valid"] = False
            report["reason"] = f"unreadable factor cache contract: {exc}"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
