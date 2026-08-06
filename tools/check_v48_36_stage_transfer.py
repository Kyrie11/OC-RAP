#!/usr/bin/env python3
"""Verify the v48.36 OCAF factor -> identity -> final stage transfer.

Unlike the legacy v48.32 checker, this contract accepts the OCAF interaction
bridge only when the registered stage architecture and the controller-provided
trainable-prefix contract both authorize it.  The allowed set is therefore
explicit, versioned, and cannot be inferred from the changed checkpoint itself.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

BASE_ALGORITHM_VERSION = "v48.36-OCAF"
IMPLEMENTATION_VERSION = "v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX"
APPROVED_PREFIXES = {
    "direct_evidence_concord_benefit_calibrator",
    "direct_evidence_concord_harm_calibrator",
    "direct_evidence_concord_admission_calibrator",
    "direct_evidence_interaction_bridge",
}
ADMISSION_PREFIX = "direct_evidence_concord_admission_calibrator"
INTERACTION_PREFIX = "direct_evidence_interaction_bridge"


def _atomic_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint model_state missing: {path}")
    tensors: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"non-tensor state entry in {path}: {key!r}")
        tensors[key] = value
    return tensors


def _parse_prefixes(raw: str, *, field: str) -> tuple[str, ...]:
    values: list[str] = []
    for token in raw.split(","):
        prefix = token.strip().rstrip(".")
        if not prefix:
            continue
        if prefix not in APPROVED_PREFIXES:
            raise ValueError(f"{field} contains unapproved prefix: {prefix}")
        if prefix in values:
            raise ValueError(f"{field} contains duplicate prefix: {prefix}")
        values.append(prefix)
    if not values:
        raise ValueError(f"{field} must contain at least one approved prefix")
    return tuple(values)


def _arch_trainable(architecture: Mapping[str, Any]) -> tuple[str, ...]:
    raw = architecture.get("trainable")
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
        text = raw[0]
    else:
        raise TypeError("STAGE_ARCHITECTURE.trainable must be a string or one-item string list")
    return _parse_prefixes(text, field="STAGE_ARCHITECTURE.trainable")


def _starts_with_any(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key.startswith(prefix + ".") for prefix in prefixes)


def _compare(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    allowed: tuple[str, ...],
) -> dict[str, Any]:
    changed_allowed: list[dict[str, Any]] = []
    changed_disallowed: list[dict[str, Any]] = []
    missing_before: list[str] = []
    missing_after: list[str] = []
    for key in sorted(set(before) | set(after)):
        if key not in before:
            missing_before.append(key)
            target = changed_allowed if _starts_with_any(key, allowed) else changed_disallowed
            target.append({"name": key, "max_abs_diff": None, "reason": "added_parameter"})
            continue
        if key not in after:
            missing_after.append(key)
            continue
        left = before[key].detach().cpu()
        right = after[key].detach().cpu()
        if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
            diff = None
            changed = True
            reason = "shape_or_dtype_changed"
        else:
            if left.numel() == 0:
                changed = False
                diff = 0.0
            else:
                diff = float((left.float() - right.float()).abs().max().item())
                changed = diff > 0.0
            reason = "value_changed"
        if not changed:
            continue
        target = changed_allowed if _starts_with_any(key, allowed) else changed_disallowed
        target.append({"name": key, "max_abs_diff": diff, "reason": reason})
    return {
        "allowed_changed": changed_allowed,
        "disallowed_changed": changed_disallowed,
        "missing_from_before": missing_before,
        "missing_from_after": missing_after,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed v48.36 OCAF stage-transfer contract")
    ap.add_argument("--factor", type=Path, required=True)
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--identity-architecture", type=Path, required=True)
    ap.add_argument("--final-architecture", type=Path, required=True)
    ap.add_argument("--identity-allowed-prefixes", required=True)
    ap.add_argument("--final-allowed-prefixes", default=ADMISSION_PREFIX)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--final-stage-disabled", action="store_true")
    ap.add_argument("--implementation-version", default=IMPLEMENTATION_VERSION)
    args = ap.parse_args()

    failures: list[str] = []
    try:
        identity_allowed = _parse_prefixes(
            args.identity_allowed_prefixes, field="identity_allowed_prefixes"
        )
        final_allowed = _parse_prefixes(
            args.final_allowed_prefixes, field="final_allowed_prefixes"
        )
        factor = _load_state(args.factor)
        identity = _load_state(args.identity)
        final = _load_state(args.final)
        identity_arch = _json(args.identity_architecture)
        final_arch = _json(args.final_architecture)
        identity_arch_trainable = _arch_trainable(identity_arch)
        final_arch_trainable = _arch_trainable(final_arch)
    except Exception as exc:
        doc = {
            "event": "v48_36_stage_transfer_integrity",
            "version": BASE_ALGORITHM_VERSION,
            "implementation_version": args.implementation_version,
            "created_unix": time.time(),
            "valid": False,
            "failure_reasons": [f"contract input error: {type(exc).__name__}: {exc}"],
            "test_roots_read": False,
        }
        _atomic_json(args.output, doc)
        print(json.dumps(doc, ensure_ascii=False))
        return 31

    expected_final_arch_trainable = identity_allowed if args.final_stage_disabled else final_allowed
    if set(identity_arch_trainable) != set(identity_allowed):
        failures.append(
            "identity architecture/trainable-prefix mismatch: "
            f"architecture={list(identity_arch_trainable)} expected={list(identity_allowed)}"
        )
    if set(final_arch_trainable) != set(expected_final_arch_trainable):
        failures.append(
            "final architecture/trainable-prefix mismatch: "
            f"architecture={list(final_arch_trainable)} expected={list(expected_final_arch_trainable)}"
        )

    identity_context = str(identity_arch.get("context_source", ""))
    final_context = str(final_arch.get("context_source", ""))
    interaction_expected = INTERACTION_PREFIX in identity_allowed
    if interaction_expected:
        if identity_context != "physical_interaction" or final_context != "physical_interaction":
            failures.append("interaction bridge authorized without physical_interaction context metadata")
        if identity_arch.get("observation_conditioned_action_frontier") is not True:
            failures.append("identity architecture does not register OCAF while interaction bridge is trainable")
    elif identity_context == "physical_interaction":
        # v48.37 HAF explicitly permits the OCAF bridge to remain frozen during
        # admission refinement.  This is fail-closed: legacy architectures that
        # merely omit the bridge are still rejected unless they register the
        # factor-preserving contract and the bridge is verified byte-identical
        # by the ordinary disallowed-drift comparison below.
        factor_preserving_bridge = (
            identity_arch.get("interaction_bridge_trainable_this_stage") is False
            and identity_arch.get("observation_conditioned_action_frontier") is True
        )
        if not factor_preserving_bridge:
            failures.append(
                "physical_interaction identity stage omitted the interaction bridge from the allowed set "
                "without an explicit frozen-bridge contract"
            )

    identity_diff = _compare(factor, identity, identity_allowed)
    final_compare_allowed: tuple[str, ...] = () if args.final_stage_disabled else final_allowed
    final_diff = _compare(identity, final, final_compare_allowed)

    if identity_diff["missing_from_after"]:
        failures.append(
            "identity checkpoint missing parameters: "
            + ", ".join(identity_diff["missing_from_after"][:10])
        )
    if identity_diff["disallowed_changed"]:
        failures.append(
            "identity stage changed frozen parameters: "
            + ", ".join(
                f"{item['name']}={item['max_abs_diff']}"
                for item in identity_diff["disallowed_changed"][:10]
            )
        )
    if final_diff["missing_from_after"]:
        failures.append(
            "final checkpoint missing parameters: "
            + ", ".join(final_diff["missing_from_after"][:10])
        )
    if final_diff["disallowed_changed"]:
        failures.append(
            "final stage changed frozen parameters: "
            + ", ".join(
                f"{item['name']}={item['max_abs_diff']}"
                for item in final_diff["disallowed_changed"][:10]
            )
        )
    if args.final_stage_disabled and final_diff["allowed_changed"]:
        failures.append("final stage was disabled but final checkpoint changed")

    identity_selected_initial = not (
        identity_diff["allowed_changed"]
        or identity_diff["disallowed_changed"]
        or identity_diff["missing_from_before"]
        or identity_diff["missing_from_after"]
    )
    final_selected_initial = not (
        final_diff["allowed_changed"]
        or final_diff["disallowed_changed"]
        or final_diff["missing_from_before"]
        or final_diff["missing_from_after"]
    )

    doc = {
        "event": "v48_36_stage_transfer_integrity",
        "version": BASE_ALGORITHM_VERSION,
        "implementation_version": args.implementation_version,
        "created_unix": time.time(),
        "valid": not failures,
        "factor": str(args.factor),
        "identity": str(args.identity),
        "final": str(args.final),
        "identity_architecture": str(args.identity_architecture),
        "final_architecture": str(args.final_architecture),
        "identity_allowed_prefixes": list(identity_allowed),
        "final_allowed_prefixes": list(final_allowed),
        "identity_architecture_trainable_prefixes": list(identity_arch_trainable),
        "final_architecture_trainable_prefixes": list(final_arch_trainable),
        "final_stage_disabled": bool(args.final_stage_disabled),
        "identity_allowed_changed_parameter_count": len(identity_diff["allowed_changed"]),
        "identity_disallowed_changed_parameter_count": len(identity_diff["disallowed_changed"]),
        "identity_selected_initial_checkpoint": identity_selected_initial,
        "no_op_identity_selection_is_valid": True,
        "final_allowed_changed_parameter_count": len(final_diff["allowed_changed"]),
        "final_disallowed_changed_parameter_count": len(final_diff["disallowed_changed"]),
        "final_selected_initial_checkpoint": final_selected_initial,
        "no_op_final_selection_is_valid": True,
        "identity_diff": identity_diff,
        "final_diff": final_diff,
        "failure_reasons": failures,
        "test_roots_read": False,
    }
    _atomic_json(args.output, doc)
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if not failures else 31


if __name__ == "__main__":
    raise SystemExit(main())
