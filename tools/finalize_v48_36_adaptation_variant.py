#!/usr/bin/env python3
"""Finalize one v48.36 OCAF adaptation variant after transfer validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BASE_ALGORITHM_VERSION = "v48.36-OCAF"
IMPLEMENTATION_VERSION = "v48.36.2-STAGE-TRANSFER-HOTFIX"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _atomic_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _bool(text: str) -> bool:
    value = text.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {text}")


def _trainable(architecture: Mapping[str, Any]) -> list[str]:
    raw = architecture.get("trainable")
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
        text = raw[0]
    else:
        raise TypeError("STAGE_ARCHITECTURE.trainable is malformed")
    return [item.strip() for item in text.split(",") if item.strip()]


def _verify_completion(checkpoint: Path, completion: Path) -> dict[str, Any]:
    doc = _json(completion)
    digest = _sha256(checkpoint)
    expected = doc.get("checkpoint_sha256")
    if expected != digest:
        raise RuntimeError(
            f"checkpoint/completion hash mismatch: checkpoint={checkpoint} expected={expected} actual={digest}"
        )
    return {"checkpoint": str(checkpoint), "sha256": digest, "completion": str(completion)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Finalize a validated v48.36 OCAF adaptation variant")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--factor", type=Path, required=True)
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--support", type=Path, required=True)
    ap.add_argument("--support-reliability-enabled", type=_bool, required=True)
    ap.add_argument("--identity-train-all", type=_bool, required=True)
    ap.add_argument("--prior-coupled", type=_bool, required=True)
    ap.add_argument("--adaptive-margin", type=_bool, required=True)
    ap.add_argument("--final-enabled", type=_bool, required=True)
    ap.add_argument("--context-source", required=True)
    ap.add_argument("--consensus-prior-scale", type=float, required=True)
    ap.add_argument("--interaction-hidden", type=int, required=True)
    ap.add_argument("--interaction-dropout", type=float, required=True)
    ap.add_argument("--admission-prior-mode", required=True)
    ap.add_argument("--implementation-version", default=IMPLEMENTATION_VERSION)
    args = ap.parse_args()

    run = args.run
    required = [
        args.source,
        args.factor,
        args.identity,
        args.final,
        args.support,
        run / "STAGE_TRANSFER_INTEGRITY.json",
        run / "factor_stage" / "STAGE_ARCHITECTURE.json",
        run / "identity_stage" / "STAGE_ARCHITECTURE.json",
        run / "STAGE_ARCHITECTURE.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing v48.36 adaptation artifact(s): " + ", ".join(missing))

    transfer = _json(run / "STAGE_TRANSFER_INTEGRITY.json")
    if transfer.get("valid") is not True:
        raise SystemExit("stage-transfer contract is not valid")
    if transfer.get("version") != BASE_ALGORITHM_VERSION:
        raise SystemExit(f"unexpected stage-transfer algorithm version: {transfer.get('version')}")

    stage_records = {
        "factor": _verify_completion(
            args.factor, run / "factor_stage" / "TRAINING_COMPLETE.json"
        ),
        "identity": _verify_completion(
            args.identity, run / "identity_stage" / "TRAINING_COMPLETE.json"
        ),
        "final": _verify_completion(args.final, run / "TRAINING_COMPLETE.json"),
    }
    architectures = {
        "factor": _json(run / "factor_stage" / "STAGE_ARCHITECTURE.json"),
        "identity": _json(run / "identity_stage" / "STAGE_ARCHITECTURE.json"),
        "final": _json(run / "STAGE_ARCHITECTURE.json"),
    }
    for name, architecture in architectures.items():
        if architecture.get("regime_id_exposed_to_evidence_model") is not False:
            raise SystemExit(f"{name} stage exposes regime routing")
        if architecture.get("shared_deployment_rule_required") is not True:
            raise SystemExit(f"{name} stage does not require the shared rule")
        if architecture.get("test_roots_read") is not False:
            raise SystemExit(f"{name} stage reports test-root access")
        if architecture.get("context_source") != args.context_source:
            raise SystemExit(f"{name} context-source mismatch")

    source_sha = _sha256(args.source)
    support_sha = _sha256(args.support)
    identity_trainable = _trainable(architectures["identity"])
    final_trainable = _trainable(architectures["final"])
    doc = {
        "event": "v48_36_ocaf_complete",
        "version": BASE_ALGORITHM_VERSION,
        "implementation_version": args.implementation_version,
        "created_unix": time.time(),
        "source_checkpoint": str(args.source),
        "source_sha256": source_sha,
        "factor_checkpoint": str(args.factor),
        "factor_sha256": stage_records["factor"]["sha256"],
        "identity_checkpoint": str(args.identity),
        "identity_sha256": stage_records["identity"]["sha256"],
        "final_checkpoint": str(args.final),
        "final_sha256": stage_records["final"]["sha256"],
        "factor_support_contract": str(args.support),
        "factor_support_sha256": support_sha,
        "stage1_population": "natural_without_replacement",
        "stage2_population": "natural_without_replacement",
        "stage3_population": "natural_without_replacement" if args.final_enabled else "disabled",
        "stage2_trainable": "all_compact_evidence_calibrators_with_ocaf_interaction"
        if args.identity_train_all
        else "admission_and_ocaf_interaction_reference",
        "stage2_trainable_prefixes": identity_trainable,
        "stage3_trainable_prefixes": final_trainable if args.final_enabled else [],
        "deployment_safe_utility_gradient_coupled": args.prior_coupled,
        "adaptive_teacher_gap_margin": args.adaptive_margin,
        "stage2_selected_initial_checkpoint": bool(
            transfer.get("identity_selected_initial_checkpoint", False)
        ),
        "stage3_selected_initial_checkpoint": bool(
            transfer.get("final_selected_initial_checkpoint", False)
        ),
        "stage_transfer_contract_event": transfer.get("event"),
        "stage_transfer_contract_implementation_version": transfer.get(
            "implementation_version"
        ),
        "model_regime_routing": False,
        "shared_deployment_rule_required": True,
        "audit_strata_only": ["near", "contact"],
        "evidence_context_source": args.context_source,
        "continuous_unified_semantics": (
            "top5 proposal plus observation-conditioned executable-action margins and noncompensatory frontier cap"
            if args.admission_prior_mode == "frontier_capped_slack"
            else "top5 proposal plus continuous candidate context and compensatory safety slack"
        ),
        "independent_measured_hard_veto": True,
        "checkpoint_metric": "direct_contract_lexicographic",
        "semantic_frontier_eligibility_metric": True,
        "exact_deployment_eligibility_metric": True,
        "observation_conditioned_action_frontier": args.context_source
        == "physical_interaction",
        "source_consensus_prior_scale": args.consensus_prior_scale,
        "interaction_hidden": args.interaction_hidden,
        "interaction_dropout": args.interaction_dropout,
        "admission_prior_mode": args.admission_prior_mode,
        "noncompensatory_frontier_cap": args.admission_prior_mode
        == "frontier_capped_slack",
        "final_thresholds_fit_by_single_shared_rule": True,
        "train_metric_uses_final_fitted_thresholds": False,
        "selection_semantics": "rank_topk_then_filter_then_evidence_rerank",
        "support_reliability_enabled": args.support_reliability_enabled,
        "final_calibration_enabled": args.final_enabled,
        "checkpoint_completion_records": stage_records,
        "test_roots_read": False,
    }
    _atomic_json(run / "THREE_STAGE_TRAINING_COMPLETE.json", doc)
    print(json.dumps(doc, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
