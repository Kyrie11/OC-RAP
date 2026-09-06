#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocrap.v48_97_executable_recovery_state import ENGINEERING_VERSION

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")


def _ok(v: Any, thresh: float) -> bool:
    try:
        return v is not None and float(v) >= float(thresh)
    except Exception:
        return False


def _evaluation_errors(obj: dict[str, Any], variant: str) -> list[str]:
    errors: list[str] = []
    if obj.get("engineering_version") != ENGINEERING_VERSION:
        errors.append(f"{variant}:engineering_version")
    contracts = obj.get("evaluation_contracts") or {}
    cells = obj.get("cells") or {}
    for role in ROLES:
        contract = contracts.get(role)
        if not isinstance(contract, dict) or not contract.get("valid"):
            errors.append(f"{variant}:{role}:evaluation_contract")
        c = cells.get(role)
        if not isinstance(c, dict):
            errors.append(f"{variant}:{role}:missing_cell")
            continue
        state = c.get("state") or {}
        if int(state.get("rows", 0)) <= 0 or state.get("auc") is None:
            errors.append(f"{variant}:{role}:state_empty_or_auc_null")
        for name in ("support_true", "reserve_true"):
            m = c.get(name) or {}
            if (
                int(m.get("positive_rows", 0)) <= 0
                or int(m.get("negative_rows", 0)) <= 0
                or m.get("auc") is None
            ):
                errors.append(f"{variant}:{role}:{name}_empty_or_auc_null")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", type=Path, required=True)
    ap.add_argument("--precision", type=Path, required=True)
    ap.add_argument("--v48-96-comparison", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    b = json.loads(a.balanced.read_text())
    p = json.loads(a.precision.read_text())
    v96 = json.loads(a.v48_96_comparison.read_text())
    errors: list[str] = []
    if not (b.get("valid") and p.get("valid")):
        errors.append("invalid_v48_97_result")
    if (v96.get("preregistered_decision") or {}).get("status") != "FROZEN_ROOT_SUPPORT_RESERVE_OBSERVABILITY_STOP":
        errors.append("v48_96_stop_prerequisite_missing")
    errors.extend(_evaluation_errors(b, "balanced"))
    errors.extend(_evaluation_errors(p, "precision"))

    state_cells: list[list[str]] = []
    support_cells: list[list[str]] = []
    support_top1_cells: list[list[str]] = []
    reserve_cells: list[list[str]] = []
    reserve_top1_cells: list[list[str]] = []
    state_roles: set[str] = set()
    support_roles: set[str] = set()
    support_top1_roles: set[str] = set()
    reserve_roles: set[str] = set()
    reserve_top1_roles: set[str] = set()
    dense: dict[str, Any] = {}

    if not errors:
        for variant, obj in (("balanced", b), ("precision", p)):
            dense[variant] = obj.get("dense_metrics")
            for role in ROLES:
                c = obj["cells"][role]
                if _ok(c["state"].get("auc"), 0.70):
                    state_cells.append([variant, role]); state_roles.add(role)
                st = c["support_true"]
                if _ok(st.get("auc"), 0.65) and _ok(st.get("auc_vs_shuffled"), 0.05):
                    support_cells.append([variant, role]); support_roles.add(role)
                if _ok(st.get("top1_vs_shuffled"), 0.10):
                    support_top1_cells.append([variant, role]); support_top1_roles.add(role)
                rt = c["reserve_true"]
                if _ok(rt.get("auc"), 0.65) and _ok(rt.get("auc_vs_shuffled"), 0.05):
                    reserve_cells.append([variant, role]); reserve_roles.add(role)
                if _ok(rt.get("top1_vs_shuffled"), 0.10):
                    reserve_top1_cells.append([variant, role]); reserve_top1_roles.add(role)

    def cross_regime(roles: set[str]) -> bool:
        return any("near" in r for r in roles) and any("contact" in r for r in roles)

    state_go = bool(not errors and len(state_cells) >= 6 and len(state_roles) >= 3 and cross_regime(state_roles))
    support_go = bool(
        not errors
        and len(support_cells) >= 6 and len(support_roles) >= 3 and cross_regime(support_roles)
        and len(support_top1_cells) >= 4 and cross_regime(support_top1_roles)
    )
    reserve_go = bool(
        not errors
        and len(reserve_cells) >= 6 and len(reserve_roles) >= 3 and cross_regime(reserve_roles)
        and len(reserve_top1_cells) >= 4 and cross_regime(reserve_top1_roles)
    )
    full_go = bool(state_go and support_go and reserve_go)

    if errors:
        status = "V48_97_EVALUATION_ENGINEERING_STOP"
        next_branch = "fix_evaluation_contract_and_rerun_same_v48_97_scientific_design"
    else:
        status = "EXECUTABLE_RECOVERY_SUFFICIENT_STATE_GO" if full_go else "EXECUTABLE_RECOVERY_SUFFICIENT_STATE_STOP"
        next_branch = (
            "one_final_fixed_capacity_observation_aligned_absolute_source_then_freeze_if_source_gate_passes"
            if full_go else
            "close_frozen_root_readout_and_low_capacity_sufficient_state_extractor_then_adjudicate_narrow_root_representation_learning_no_source_sweep"
        )

    decision = {
        "state_representation_go": state_go,
        "support_action_representation_go": support_go,
        "reserve_debt_representation_go": reserve_go,
        "executable_recovery_sufficient_state_go": full_go,
        "state_positive_cells": state_cells,
        "state_roles": sorted(state_roles),
        "support_positive_cells": support_cells,
        "support_roles": sorted(support_roles),
        "support_top1_material_cells": support_top1_cells,
        "reserve_positive_cells": reserve_cells,
        "reserve_roles": sorted(reserve_roles),
        "reserve_top1_material_cells": reserve_top1_cells,
        "final_source_experiment_authorized": bool(full_go and not errors),
        "boundary_transport_authorized": False,
        "regime_conditioned_policy_authorized": False,
        "dataset_reconstruction_authorized": False,
        "status": status,
        "next_branch": next_branch,
    }
    out = {
        "schema": "ocrap-v48.97-erss-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "learned_two_coordinate_executable_recovery_state_representation",
        "representation_parameters_trained": int(b.get("representation_parameters_trained", 0)),
        "planner_parameters_trained": 0,
        "source_parameters_trained": 0,
        "boundary_transport": False,
        "regime_conditioning": False,
        "relative_ranker_modified": False,
        "teacher_metadata_input_to_model": False,
        "dataset_reconstruction": False,
        "dataset_reselection": False,
        "dense_metrics": dense,
        "preregistered_decision": decision,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": decision["status"], "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
