#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.v48_98_executable_recovery_tangent import ENGINEERING_VERSION

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _result_errors(obj: dict, variant: str) -> list[str]:
    errors: list[str] = []
    if obj.get("engineering_version") != ENGINEERING_VERSION or not obj.get("valid"):
        errors.append(f"{variant}_contract")
    if int(obj.get("planner_parameters_trained", -1)) != 0 or int(obj.get("source_parameters_trained", -1)) != 0 or int(obj.get("erss_parameters_trained", -1)) != 0:
        errors.append(f"{variant}_unexpected_non_tangent_training")
    if int(obj.get("stage_i_tangent_parameters_trained", 0)) <= 0:
        errors.append(f"{variant}_no_stage_i_tangent_training")
    identity = obj.get("nominal_identity") or {}
    for split in ("dev", "certificate"):
        d = identity.get(split) or {}
        if max(float(d.get("support_max_abs_error", 1.0)), float(d.get("reserve_max_abs_error", 1.0))) > 1.0e-7:
            errors.append(f"{variant}_{split}_nominal_identity")
    contracts = obj.get("evaluation_contracts") or {}
    cells = obj.get("cells") or {}
    for role in ROLES:
        if not (isinstance(contracts.get(role), dict) and contracts[role].get("valid")):
            errors.append(f"{variant}_{role}_evaluation_contract")
        c = cells.get(role) or {}
        for name in ("state", "support_true", "reserve_true"):
            m = c.get(name) or {}
            if int(m.get("rows", 0)) <= 0 or m.get("auc") is None:
                errors.append(f"{variant}_{role}_{name}_empty_or_null")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--balanced", type=Path, required=True)
    ap.add_argument("--precision", type=Path, required=True)
    ap.add_argument("--balanced-state", type=Path, required=True)
    ap.add_argument("--precision-state", type=Path, required=True)
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--v48-97-pipeline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []
    objs: dict[str, dict] = {}
    for name, p in (("runtime", a.runtime), ("balanced", a.balanced), ("precision", a.precision), ("comparison", a.comparison)):
        if not p.is_file():
            errors.append(f"missing_{name}")
            continue
        try:
            objs[name] = json.loads(p.read_text())
        except Exception as exc:
            errors.append(f"invalid_{name}:{exc}")
    runtime = objs.get("runtime", {})
    if runtime.get("engineering_version") != ENGINEERING_VERSION or not runtime.get("valid") or not runtime.get("attribution_ready"):
        errors.append("runtime_contract")
    errors += _result_errors(objs.get("balanced", {}), "balanced")
    errors += _result_errors(objs.get("precision", {}), "precision")
    comp = objs.get("comparison", {})
    if not comp.get("valid") or not comp.get("attribution_ready") or comp.get("engineering_version") != ENGINEERING_VERSION:
        errors.append("comparison_contract")
    if not a.balanced_state.is_file() or not a.precision_state.is_file():
        errors.append("missing_tangent_state")
    if not a.v48_97_pipeline.is_file():
        errors.append("missing_v48_97_pipeline")
    else:
        v97 = json.loads(a.v48_97_pipeline.read_text())
        if not (v97.get("valid") and v97.get("attribution_ready") and v97.get("preregistered_status") == "EXECUTABLE_RECOVERY_SUFFICIENT_STATE_STOP"):
            errors.append("v48_97_stop_prerequisite")
        expected = ((v97.get("artifacts") or {}).get("comparison") or {}).get("sha256")
        actual = comp.get("v48_97_comparison_sha256")
        if not expected or actual != expected:
            errors.append("v48_97_provenance_mismatch")
    artifacts = {}
    for name, p in (
        ("runtime", a.runtime), ("balanced", a.balanced), ("precision", a.precision),
        ("balanced_state", a.balanced_state), ("precision_state", a.precision_state),
        ("comparison", a.comparison),
    ):
        if p.is_file():
            artifacts[name] = {"path": str(p.resolve()), "sha256": sha(p)}
    status = (comp.get("preregistered_decision") or {}).get("status")
    out = {
        "schema": "ocrap-v48.98-erta-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "centered_rank2_stage_i_recovery_tangent_learning",
        "planner_parameters_trained": 0,
        "source_parameters_trained": 0,
        "erss_parameters_trained": 0,
        "regime_conditioning": False,
        "relative_ranker_modified": False,
        "boundary_transport": False,
        "dataset_reconstruction": False,
        "dataset_reselection": False,
        "teacher_metadata_input_to_model": False,
        "test_roots_read": False,
        "preregistered_status": status,
        "artifacts": artifacts,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
