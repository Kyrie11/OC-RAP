#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.v48_97_executable_recovery_state import ENGINEERING_VERSION

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _evaluation_errors(obj: dict, variant: str) -> list[str]:
    errors: list[str] = []
    contracts = obj.get("evaluation_contracts") or {}
    cells = obj.get("cells") or {}
    for role in ROLES:
        contract = contracts.get(role)
        if not isinstance(contract, dict) or not contract.get("valid"):
            errors.append(f"{variant}_{role}_evaluation_contract")
        c = cells.get(role)
        if not isinstance(c, dict):
            errors.append(f"{variant}_{role}_missing_cell")
            continue
        state = c.get("state") or {}
        if int(state.get("rows", 0)) <= 0 or state.get("auc") is None:
            errors.append(f"{variant}_{role}_state_empty_or_auc_null")
        for metric_name in ("support_true", "reserve_true"):
            m = c.get(metric_name) or {}
            if (
                int(m.get("positive_rows", 0)) <= 0
                or int(m.get("negative_rows", 0)) <= 0
                or m.get("auc") is None
            ):
                errors.append(f"{variant}_{role}_{metric_name}_empty_or_auc_null")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--balanced", type=Path, required=True)
    ap.add_argument("--precision", type=Path, required=True)
    ap.add_argument("--balanced-state", type=Path, required=True)
    ap.add_argument("--precision-state", type=Path, required=True)
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--v48-96-pipeline", type=Path, required=True)
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
    for name in ("balanced", "precision"):
        o = objs.get(name, {})
        if o.get("engineering_version") != ENGINEERING_VERSION or not o.get("valid"):
            errors.append(f"{name}_contract")
        if int(o.get("planner_parameters_trained", -1)) != 0 or int(o.get("source_parameters_trained", -1)) != 0:
            errors.append(f"{name}_unexpected_planner_or_source_training")
        errors.extend(_evaluation_errors(o, name))
    comp = objs.get("comparison", {})
    if not comp.get("valid") or not comp.get("attribution_ready") or comp.get("engineering_version") != ENGINEERING_VERSION:
        errors.append("comparison_contract")
    if not a.balanced_state.is_file() or not a.precision_state.is_file():
        errors.append("missing_representation_state")
    if not a.v48_96_pipeline.is_file():
        errors.append("missing_v48_96_pipeline")
    else:
        v96 = json.loads(a.v48_96_pipeline.read_text())
        if not (
            v96.get("valid") and v96.get("attribution_ready")
            and v96.get("preregistered_status") == "FROZEN_ROOT_SUPPORT_RESERVE_OBSERVABILITY_STOP"
        ):
            errors.append("v48_96_stop_prerequisite")
        expected_v96_comp = ((v96.get("artifacts") or {}).get("comparison") or {}).get("sha256")
        actual_v96_comp = comp.get("v48_96_comparison_sha256")
        if not expected_v96_comp or actual_v96_comp != expected_v96_comp:
            errors.append("v48_96_evaluation_population_provenance")
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
        "schema": "ocrap-v48.97-erss-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "learned_two_coordinate_executable_recovery_state_representation",
        "planner_parameters_trained": 0,
        "source_parameters_trained": 0,
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
