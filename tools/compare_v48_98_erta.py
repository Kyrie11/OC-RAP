#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.v48_98_executable_recovery_tangent import ENGINEERING_VERSION

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")


def _ok(v: Any, thresh: float) -> bool:
    try:
        return v is not None and float(v) >= float(thresh)
    except Exception:
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _population_errors(obj: dict[str, Any], ref: dict[str, Any], variant: str) -> list[str]:
    errors: list[str] = []
    for role in ROLES:
        a = (obj.get("cells") or {}).get(role) or {}
        b = (ref.get("cells") or {}).get(role) or {}
        for metric_name, fields in (
            ("state", ("rows", "drs_state_rows", "dep_state_rows")),
            ("support_true", ("rows", "positive_rows", "negative_rows", "powered_groups")),
            ("support_shuffled", ("rows", "positive_rows", "negative_rows", "powered_groups")),
            ("reserve_true", ("rows", "positive_rows", "negative_rows", "powered_groups")),
            ("reserve_shuffled", ("rows", "positive_rows", "negative_rows", "powered_groups")),
        ):
            aa = a.get(metric_name) or {}
            bb = b.get(metric_name) or {}
            for field in fields:
                if int(aa.get(field, -1)) != int(bb.get(field, -2)):
                    errors.append(f"{variant}:{role}:{metric_name}:{field}:v98={aa.get(field)}!=v97={bb.get(field)}")
    return errors


def _state_identity_errors(obj: dict[str, Any], ref: dict[str, Any], variant: str) -> list[str]:
    errors: list[str] = []
    for role in ROLES:
        a = ((obj.get("cells") or {}).get(role) or {}).get("state") or {}
        b = ((ref.get("cells") or {}).get(role) or {}).get("state") or {}
        if int(a.get("rows", -1)) != int(b.get("rows", -2)):
            errors.append(f"{variant}:{role}:state_rows")
        aa, bb = a.get("auc"), b.get("auc")
        if aa is None or bb is None or abs(float(aa) - float(bb)) > 1.0e-12:
            errors.append(f"{variant}:{role}:state_auc:v98={aa}!=v97={bb}")
    identity = obj.get("nominal_identity") or {}
    for split in ("dev", "certificate"):
        d = identity.get(split) or {}
        if max(float(d.get("support_max_abs_error", 1.0)), float(d.get("reserve_max_abs_error", 1.0))) > 1.0e-7:
            errors.append(f"{variant}:{split}:nominal_identity")
    return errors


def _result_errors(obj: dict[str, Any], variant: str) -> list[str]:
    errors: list[str] = []
    if not obj.get("valid"):
        errors.append(f"{variant}:invalid")
    if obj.get("engineering_version") != ENGINEERING_VERSION:
        errors.append(f"{variant}:engineering_version")
    if int(obj.get("stage_i_tangent_parameters_trained", -1)) <= 0:
        errors.append(f"{variant}:no_stage_i_tangent_parameters")
    if int(obj.get("erss_parameters_trained", -1)) != 0 or int(obj.get("source_parameters_trained", -1)) != 0:
        errors.append(f"{variant}:unexpected_erss_or_source_training")
    if int(obj.get("semantic_tangent_rank", -1)) != 2:
        errors.append(f"{variant}:semantic_tangent_rank")
    for role in ROLES:
        c = (obj.get("cells") or {}).get(role) or {}
        for name in ("state", "support_true", "reserve_true"):
            m = c.get(name) or {}
            if int(m.get("rows", 0)) <= 0 or m.get("auc") is None:
                errors.append(f"{variant}:{role}:{name}:empty_or_null")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", type=Path, required=True)
    ap.add_argument("--precision", type=Path, required=True)
    ap.add_argument("--v48-97-balanced", type=Path, required=True)
    ap.add_argument("--v48-97-precision", type=Path, required=True)
    ap.add_argument("--v48-97-comparison", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    b = json.loads(a.balanced.read_text())
    p = json.loads(a.precision.read_text())
    rb = json.loads(a.v48_97_balanced.read_text())
    rp = json.loads(a.v48_97_precision.read_text())
    rcmp = json.loads(a.v48_97_comparison.read_text())
    errors: list[str] = []
    rd = rcmp.get("preregistered_decision") or {}
    if not (rcmp.get("valid") and rcmp.get("attribution_ready") and rd.get("status") == "EXECUTABLE_RECOVERY_SUFFICIENT_STATE_STOP"):
        errors.append("v48_97_stop_prerequisite_missing")
    if not (rd.get("state_representation_go") is True and rd.get("support_action_representation_go") is False and rd.get("reserve_debt_representation_go") is False):
        errors.append("v48_97_branch_shape_mismatch")
    errors += _result_errors(b, "balanced") + _result_errors(p, "precision")
    errors += _population_errors(b, rb, "balanced") + _population_errors(p, rp, "precision")
    errors += _state_identity_errors(b, rb, "balanced") + _state_identity_errors(p, rp, "precision")

    support_cells: list[list[str]] = []
    reserve_cells: list[list[str]] = []
    support_top1: list[list[str]] = []
    reserve_top1: list[list[str]] = []
    support_roles: set[str] = set()
    reserve_roles: set[str] = set()
    support_top1_roles: set[str] = set()
    reserve_top1_roles: set[str] = set()
    if not errors:
        for variant, obj in (("balanced", b), ("precision", p)):
            for role in ROLES:
                c = obj["cells"][role]
                s = c["support_true"]
                r = c["reserve_true"]
                if _ok(s.get("auc"), 0.65) and _ok(s.get("auc_vs_shuffled"), 0.05):
                    support_cells.append([variant, role]); support_roles.add(role)
                if _ok(s.get("top1_vs_shuffled"), 0.10):
                    support_top1.append([variant, role]); support_top1_roles.add(role)
                if _ok(r.get("auc"), 0.65) and _ok(r.get("auc_vs_shuffled"), 0.05):
                    reserve_cells.append([variant, role]); reserve_roles.add(role)
                if _ok(r.get("top1_vs_shuffled"), 0.10):
                    reserve_top1.append([variant, role]); reserve_top1_roles.add(role)

    def cross_regime(roles: set[str]) -> bool:
        return any("near" in x for x in roles) and any("contact" in x for x in roles)

    support_go = bool(not errors and len(support_cells) >= 6 and len(support_roles) >= 3 and cross_regime(support_roles)
                      and len(support_top1) >= 4 and cross_regime(support_top1_roles))
    reserve_go = bool(not errors and len(reserve_cells) >= 6 and len(reserve_roles) >= 3 and cross_regime(reserve_roles)
                      and len(reserve_top1) >= 4 and cross_regime(reserve_top1_roles))
    state_preserved = bool(not errors)
    full_go = bool(state_preserved and support_go and reserve_go)
    if errors:
        status = "V48_98_ENGINEERING_STOP"
        next_branch = "fix_v48_98_engineering_and_rerun_same_tangent_experiment"
    else:
        status = "EXECUTABLE_RECOVERY_TANGENT_ALIGNMENT_GO" if full_go else "EXECUTABLE_RECOVERY_TANGENT_ALIGNMENT_STOP"
        next_branch = (
            "one_final_fixed_capacity_observation_aligned_source_then_freeze_if_source_gate_passes"
            if full_go else
            "close_rank2_stage_i_recovery_tangent_then_preregister_deeper_root_representation_objective_no_source_sweep"
        )
    decision = {
        "state_chart_preserved": state_preserved,
        "support_tangent_go": support_go,
        "reserve_debt_tangent_go": reserve_go,
        "executable_recovery_tangent_alignment_go": full_go,
        "support_positive_cells": support_cells,
        "support_roles": sorted(support_roles),
        "support_top1_material_cells": support_top1,
        "reserve_positive_cells": reserve_cells,
        "reserve_roles": sorted(reserve_roles),
        "reserve_top1_material_cells": reserve_top1,
        "final_source_experiment_authorized": bool(full_go and not errors),
        "boundary_transport_authorized": False,
        "regime_conditioned_policy_authorized": False,
        "dataset_reconstruction_authorized": False,
        "status": status,
        "next_branch": next_branch,
    }
    out = {
        "schema": "ocrap-v48.98-erta-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "centered_rank2_stage_i_recovery_tangent_learning",
        "stage_i_tangent_parameters_trained": int(b.get("stage_i_tangent_parameters_trained", 0)),
        "planner_parameters_trained": 0,
        "erss_parameters_trained": 0,
        "source_parameters_trained": 0,
        "semantic_tangent_rank": 2,
        "boundary_transport": False,
        "regime_conditioning": False,
        "relative_ranker_modified": False,
        "teacher_metadata_input_to_model": False,
        "dataset_reconstruction": False,
        "dataset_reselection": False,
        "v48_97_comparison_sha256": _sha256(a.v48_97_comparison),
        "v48_97_balanced_sha256": _sha256(a.v48_97_balanced),
        "v48_97_precision_sha256": _sha256(a.v48_97_precision),
        "evaluation_population_contract": "exact_v48_97_2_equals_v48_96_strata",
        "preregistered_decision": decision,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
