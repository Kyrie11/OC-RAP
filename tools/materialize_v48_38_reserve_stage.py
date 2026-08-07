#!/usr/bin/env python3
"""Materialize a no-learning RFR stage from the factor checkpoint.

The v48.38 Robust Frontier Reserve has no separately learned admission residual.
Identity/final stage directories are retained only because the audited v48.36
controller and certificate contracts expect the three-stage artifact layout.
This tool copies checkpoint bytes but rewrites stage metadata to state explicitly
that zero optimizer steps and zero parameter updates occurred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor-stage", type=Path, required=True)
    ap.add_argument("--destination", type=Path, required=True)
    ap.add_argument("--role", choices=("identity", "final"), required=True)
    ap.add_argument("--implementation-version", default="v48.38-RFR")
    args = ap.parse_args()

    factor = args.factor_stage
    dest = args.destination
    required = [
        factor / "model_v48_trac_sr" / "best.pt",
        factor / "model_v48_trac_sr" / "train_summary.json",
        factor / "STAGE_ARCHITECTURE.json",
        factor / "TRAINING_COMPLETE.json",
        factor / "EVIDENCE_CORRECTION_COMPLETE.json",
        factor / "POLICY_CONTRACT.env",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing factor-stage artifact(s): " + ", ".join(missing))

    factor_arch = _load(factor / "STAGE_ARCHITECTURE.json")
    if factor_arch.get("admission_prior_mode") != "joint_reserve":
        raise SystemExit("RFR reserve materialization requires admission_prior_mode=joint_reserve")
    if factor_arch.get("admission_head") is not False:
        raise SystemExit("RFR reserve materialization requires admission_head=false")
    if factor_arch.get("deterministic_joint_reserve") is not True:
        raise SystemExit("factor architecture does not register deterministic_joint_reserve=true")
    if factor_arch.get("regime_id_exposed_to_evidence_model") is not False:
        raise SystemExit("RFR factor stage unexpectedly exposes a regime identifier")

    dest.mkdir(parents=True, exist_ok=True)
    src_model = factor / "model_v48_trac_sr"
    dst_model = dest / "model_v48_trac_sr"
    if dst_model.exists():
        shutil.rmtree(dst_model)
    dst_model.mkdir(parents=True, exist_ok=True)
    shutil.copy2(factor / "POLICY_CONTRACT.env", dest / "POLICY_CONTRACT.env")

    factor_checkpoint = src_model / "best.pt"
    dest_checkpoint = dst_model / "best.pt"
    # Reserve-only stages do not need factor ``latest.pt`` or historical epoch
    # checkpoints.  Copying the entire model directory duplicated large files and
    # falsely suggested identity/final training had produced those checkpoints.
    # A same-filesystem hard link is immutable-by-contract and avoids that I/O;
    # fall back to a byte copy when linking is unavailable.
    try:
        os.link(factor_checkpoint, dest_checkpoint)
        checkpoint_materialization = "hardlink"
    except OSError:
        shutil.copy2(factor_checkpoint, dest_checkpoint)
        checkpoint_materialization = "copy"
    factor_sha = _sha256(factor_checkpoint)
    dest_sha = _sha256(dest_checkpoint)
    if dest_sha != factor_sha:
        raise SystemExit("checkpoint bytes changed during RFR materialization")

    factor_complete = _load(factor / "TRAINING_COMPLETE.json")
    factor_summary = _load(src_model / "train_summary.json")
    factor_correction = _load(factor / "EVIDENCE_CORRECTION_COMPLETE.json")
    best_metric = str(factor_summary.get("best_metric") or factor_complete.get("best_metric") or "direct_factor_supervised_risk")
    best_value = factor_summary.get("best_metric_value")
    initial_checkpoint: dict[str, Any] = {}
    if best_value is not None:
        initial_checkpoint[best_metric] = best_value

    summary = {
        "event": "v48_38_rfr_no_training_stage_summary",
        "created_unix": time.time(),
        "materialized_role": args.role,
        "best_epoch": 0,
        "epochs_completed": 0,
        "total_train_steps": 0,
        "elapsed_seconds": 0.0,
        "best_metric": best_metric,
        "best_metric_value": best_value,
        "best_val_loss": factor_summary.get("best_val_loss"),
        "initial_checkpoint": initial_checkpoint,
        "history": [],
        "checkpoint": str(dest_checkpoint),
        "checkpoint_dir": str(dst_model),
        "init_checkpoint": str(factor_checkpoint),
        "trainable_param_prefixes": [],
        "freeze_param_prefixes": ["*"],
        "materialized_without_training": True,
        "parameter_update_performed": False,
        "factor_checkpoint_reused_without_parameter_update": True,
        "checkpoint_materialization": checkpoint_materialization,
        "source_factor_checkpoint": str(factor_checkpoint),
        "source_factor_checkpoint_sha256": factor_sha,
        "source_factor_best_epoch": factor_summary.get("best_epoch"),
        "source_factor_epochs_completed": factor_summary.get("epochs_completed"),
        "test_roots_read": False,
    }
    _atomic_json(dst_model / "train_summary.json", summary)

    completion = {
        "event": "v48_38_rfr_stage_materialized_complete",
        "created_unix": time.time(),
        "checkpoint": str(dest_checkpoint),
        "checkpoint_sha256": dest_sha,
        "best_epoch": 0,
        "epochs_completed": 0,
        "best_metric": best_metric,
        "materialized_role": args.role,
        "materialized_without_training": True,
        "parameter_update_performed": False,
        "source_factor_checkpoint": str(factor_checkpoint),
        "source_factor_checkpoint_sha256": factor_sha,
        "checkpoint_materialization": checkpoint_materialization,
        "test_roots_read": False,
    }
    _atomic_json(dest / "TRAINING_COMPLETE.json", completion)

    arch = dict(factor_arch)
    arch.update({
        "version": "v48.38-RFR",
        "implementation_version": args.implementation_version,
        "training_role": "deterministic_joint_physical_reserve",
        "trainable": [],
        "identity_stage_skipped": True,
        "learned_admission_residual": False,
        "deterministic_joint_reserve": True,
        "factor_checkpoint_reused_without_parameter_update": True,
        "materialized_role": args.role,
        "interaction_bridge_trainable_this_stage": False,
        "group_batch_stratified": False,
        "group_batching_replacement": False,
        "test_roots_read": False,
    })
    _atomic_json(dest / "STAGE_ARCHITECTURE.json", arch)

    correction = dict(factor_correction)
    correction.update({
        "event": "v48_38_rfr_evidence_materialized_without_update",
        "created_unix": time.time(),
        "checkpoint": str(dest_checkpoint),
        "checkpoint_sha256": dest_sha,
        "trainable_prefixes": [],
        "matched_state_keys": {},
        "trainable_state_params": 0,
        "admission_head": False,
        "interaction_bridge_trainable_this_stage": False,
        "checkpoint_metric": best_metric,
        "materialized_role": args.role,
        "materialized_without_training": True,
        "parameter_update_performed": False,
        "source_factor_checkpoint": str(factor_checkpoint),
        "source_factor_checkpoint_sha256": factor_sha,
        "test_roots_read": False,
    })
    _atomic_json(dest / "EVIDENCE_CORRECTION_COMPLETE.json", correction)

    _atomic_json(dest / "V48_38_RESERVE_STAGE.json", {
        "event": "v48_38_reserve_stage_materialized",
        "created_unix": time.time(),
        "role": args.role,
        "source_factor_stage": str(factor),
        "source_factor_checkpoint": str(factor_checkpoint),
        "source_factor_checkpoint_sha256": factor_sha,
        "destination": str(dest),
        "destination_checkpoint": str(dest_checkpoint),
        "destination_checkpoint_sha256": dest_sha,
        "checkpoint_materialization": checkpoint_materialization,
        "parameter_update_performed": False,
        "optimizer_steps": 0,
        "deterministic_joint_reserve": True,
        "regime_conditioning": False,
        "test_roots_read": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
