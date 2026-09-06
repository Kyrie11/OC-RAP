#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.v48_103_factorized_control_sufficient_state import ENGINEERING_VERSION, expected_parameter_count

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _ok(v: Any, t: float) -> bool:
    try:
        return v is not None and float(v) >= float(t)
    except Exception:
        return False


def _result_errors(obj: dict[str, Any], variant: str) -> list[str]:
    e: list[str] = []
    if not obj.get("valid") or obj.get("engineering_version") != ENGINEERING_VERSION:
        e.append(f"{variant}:contract")
    for k in ("planner_parameters_trained", "stage_i_parameters_trained", "root_decoder_parameters_trained", "source_parameters_trained"):
        if int(obj.get(k, -1)) != 0:
            e.append(f"{variant}:{k}")
    if int(obj.get("representation_parameters_trained", -1)) != expected_parameter_count(192):
        e.append(f"{variant}:representation_parameter_count")
    if obj.get("boundary_transport") or obj.get("regime_conditioning") or obj.get("teacher_metadata_input_to_model"):
        e.append(f"{variant}:forbidden_input_or_transport")
    if obj.get("state_response_learned_mixing") or obj.get("nominal_response_exact_zero") is not True:
        e.append(f"{variant}:factorization_contract")
    for role in ROLES:
        c = (obj.get("cells") or {}).get(role) or {}
        for name in ("state", "support_true", "reserve_true"):
            m = c.get(name) or {}
            if int(m.get("rows", 0)) <= 0 or m.get("auc") is None:
                e.append(f"{variant}:{role}:{name}:empty_or_null")
    return e


def _population_errors(obj: dict[str, Any], ref: dict[str, Any], variant: str) -> list[str]:
    e: list[str] = []
    for role in ROLES:
        a = (obj.get("cells") or {}).get(role) or {}
        b = (ref.get("cells") or {}).get(role) or {}
        for name, fields in (
            ("state", ("rows", "drs_state_rows", "dep_state_rows")),
            ("support_true", ("rows", "positive_rows", "negative_rows", "powered_groups")),
            ("support_shuffled", ("rows", "positive_rows", "negative_rows", "powered_groups")),
            ("reserve_true", ("rows", "positive_rows", "negative_rows", "powered_groups")),
            ("reserve_shuffled", ("rows", "positive_rows", "negative_rows", "powered_groups")),
        ):
            for f in fields:
                if int((a.get(name) or {}).get(f, -1)) != int((b.get(name) or {}).get(f, -2)):
                    e.append(f"{variant}:{role}:{name}:{f}:population_drift")
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", type=Path, required=True)
    ap.add_argument("--precision", type=Path, required=True)
    ap.add_argument("--v102-balanced", type=Path, required=True)
    ap.add_argument("--v102-precision", type=Path, required=True)
    ap.add_argument("--v102-comparison", type=Path, required=True)
    ap.add_argument("--v101-balanced", type=Path, required=True)
    ap.add_argument("--v101-precision", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    docs = {"balanced": json.loads(a.balanced.read_text()), "precision": json.loads(a.precision.read_text())}
    refs102 = {"balanced": json.loads(a.v102_balanced.read_text()), "precision": json.loads(a.v102_precision.read_text())}
    refs101 = {"balanced": json.loads(a.v101_balanced.read_text()), "precision": json.loads(a.v101_precision.read_text())}
    c102 = json.loads(a.v102_comparison.read_text())
    d102 = c102.get("preregistered_decision") or {}
    errors: list[str] = []
    if not (
        c102.get("valid") and c102.get("attribution_ready")
        and d102.get("status") == "STAGE_I_ACTION_INFORMATION_SUFFICIENCY_STOP"
        and d102.get("stage_i_state_observability_go") is False
        and d102.get("stage_i_support_action_observability_go") is False
        and d102.get("stage_i_reserve_action_observability_go") is False
        and d102.get("next_branch") == "stage_i_action_information_insufficient_then_preregister_minimal_stage_i_recovery_representation_objective_no_source_or_broad_encoder_sweep"
    ):
        errors.append("v48_102_all_stop_branch_prerequisite_missing")
    for v in ("balanced", "precision"):
        errors += _result_errors(docs[v], v)
        errors += _population_errors(docs[v], refs102[v], v)

    state_cells: list[list[str]] = []
    support_cells: list[list[str]] = []
    reserve_cells: list[list[str]] = []
    support_top: list[list[str]] = []
    reserve_top: list[list[str]] = []
    state_roles: set[str] = set(); support_roles: set[str] = set(); reserve_roles: set[str] = set()
    support_top_roles: set[str] = set(); reserve_top_roles: set[str] = set()
    deltas = {"vs_v102": {"state": [], "support": [], "reserve": []}, "vs_v101": {"state": [], "support": [], "reserve": []}}

    if not errors:
        for v in ("balanced", "precision"):
            obj = docs[v]
            for role in ROLES:
                c = obj["cells"][role]
                st, su, re = c["state"], c["support_true"], c["reserve_true"]
                if _ok(st.get("auc"), 0.70):
                    state_cells.append([v, role]); state_roles.add(role)
                if _ok(su.get("auc"), 0.65) and _ok(su.get("auc_vs_shuffled"), 0.05):
                    support_cells.append([v, role]); support_roles.add(role)
                if _ok(re.get("auc"), 0.65) and _ok(re.get("auc_vs_shuffled"), 0.05):
                    reserve_cells.append([v, role]); reserve_roles.add(role)
                if _ok(su.get("top1_vs_shuffled"), 0.10):
                    support_top.append([v, role]); support_top_roles.add(role)
                if _ok(re.get("top1_vs_shuffled"), 0.10):
                    reserve_top.append([v, role]); reserve_top_roles.add(role)
                for ref_name, refs in (("vs_v102", refs102), ("vs_v101", refs101)):
                    for name, metric in (("state", "state"), ("support", "support_true"), ("reserve", "reserve_true")):
                        x = (c.get(metric) or {}).get("auc")
                        y = ((refs[v].get("cells") or {}).get(role) or {}).get(metric, {}).get("auc")
                        if x is not None and y is not None:
                            deltas[ref_name][name].append({
                                "variant": v, "role": role, "v103_auc": float(x),
                                "reference_auc": float(y), "delta": float(x) - float(y),
                            })

    def cross(rs: set[str], n: int) -> bool:
        return len(rs) >= n and any("near" in x for x in rs) and any("contact" in x for x in rs)

    state_go = bool(not errors and len(state_cells) >= 6 and cross(state_roles, 3))
    support_go = bool(not errors and len(support_cells) >= 6 and cross(support_roles, 3) and len(support_top) >= 4 and cross(support_top_roles, 2))
    reserve_go = bool(not errors and len(reserve_cells) >= 6 and cross(reserve_roles, 3) and len(reserve_top) >= 4 and cross(reserve_top_roles, 2))
    full_go = bool(state_go and support_go and reserve_go)

    if errors:
        status = "V48_103_ENGINEERING_STOP"
        next_branch = "fix_v48_103_engineering_and_rerun_same_factorized_representation"
    elif full_go:
        status = "FACTORIZED_CONTROL_SUFFICIENT_STATE_GO"
        next_branch = "preregister_one_production_stage_i_semantic_token_transport_from_fcss_no_source_or_capacity_sweep"
    elif state_go and support_go and not reserve_go:
        status = "FACTORIZED_CONTROL_SUFFICIENT_STATE_PARTIAL_SUPPORT"
        next_branch = "retain_state_support_factorization_then_preregister_one_supported_reserve_flow_objective_no_encoder_or_source_sweep"
    elif state_go and reserve_go and not support_go:
        status = "FACTORIZED_CONTROL_SUFFICIENT_STATE_PARTIAL_RESERVE"
        next_branch = "retain_state_reserve_factorization_then_preregister_one_support_establishment_objective_no_encoder_or_source_sweep"
    else:
        status = "FACTORIZED_CONTROL_SUFFICIENT_STATE_STOP"
        next_branch = "close_frozen_stage_i_readout_family_then_preregister_last_stage_i_block_control_sufficient_representation_objective_no_broad_encoder_or_source_sweep"

    decision = {
        "factorized_state_representation_go": state_go,
        "factorized_support_action_go": support_go,
        "factorized_reserve_debt_go": reserve_go,
        "factorized_control_sufficient_state_go": full_go,
        "state_positive_cells": state_cells,
        "support_positive_cells": support_cells,
        "reserve_positive_cells": reserve_cells,
        "support_top1_material_cells": support_top,
        "reserve_top1_material_cells": reserve_top,
        "state_roles": sorted(state_roles),
        "support_roles": sorted(support_roles),
        "reserve_roles": sorted(reserve_roles),
        "diagnostic_auc_deltas": deltas,
        "status": status,
        "next_branch": next_branch,
        "source_training_authorized": False,
        "boundary_transport_authorized": False,
        "dataset_reconstruction_authorized": False,
        "broad_encoder_training_authorized": False,
        "regime_conditioned_policy_authorized": False,
    }
    out = {
        "schema": "ocrap-v48.103-fcss-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "minimal_frozen_stage_i_factorized_control_sufficient_representation",
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "representation_parameters_trained": expected_parameter_count(192),
        "source_parameters_trained": 0,
        "dataset_reconstruction": False,
        "teacher_metadata_input_to_model": False,
        "boundary_transport": False,
        "relative_ranker_modified": False,
        "regime_conditioning": False,
        "v48_102_comparison_sha256": _sha(a.v102_comparison),
        "preregistered_decision": decision,
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "state": state_go, "support": support_go, "reserve": reserve_go}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
