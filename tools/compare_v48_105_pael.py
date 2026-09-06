#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.v48_105_prelast_action_equivariance_localization import ENGINEERING_VERSION, TOKEN_GROUP_ORDER

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
V102_ENGINEERING_VERSION = "v48.102.0-OC-AITS"
V104_ENGINEERING_VERSION = "v48.104.0-OC-NICR"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _ok(v: Any, t: float) -> bool:
    try:
        return v is not None and float(v) >= float(t)
    except Exception:
        return False


def _result_errors(obj: dict[str, Any], variant: str) -> list[str]:
    e: list[str] = []
    if not obj.get("valid") or obj.get("engineering_version") != ENGINEERING_VERSION or obj.get("variant") != variant:
        e.append(f"{variant}:contract")
    if not obj.get("audit_only") or not obj.get("prelast_only"):
        e.append(f"{variant}:not_prelast_audit")
    for k in ("planner_parameters_trained", "stage_i_parameters_trained", "root_decoder_parameters_trained", "source_parameters_trained"):
        if int(obj.get(k, -1)) != 0:
            e.append(f"{variant}:{k}")
    if obj.get("boundary_transport") or obj.get("regime_conditioning") or obj.get("teacher_metadata_input_to_model") or obj.get("test_roots_read"):
        e.append(f"{variant}:forbidden_runtime_input_or_transport")
    if obj.get("same_v48_102_summary_operator") is not True or obj.get("same_v48_102_linear_probe_recipe") is not True:
        e.append(f"{variant}:probe_recipe_drift")
    ai = obj.get("action_interaction_subspace") or {}
    if int(ai.get("dimension", -1)) <= 0 or "control_plus_scene_context" not in str(ai.get("definition", "")):
        e.append(f"{variant}:action_interaction_contract")
    for role in ROLES:
        c = (obj.get("cells") or {}).get(role) or {}
        ai_c = (obj.get("action_interaction_cells") or {}).get(role) or {}
        for name in ("state", "support_true", "reserve_true"):
            m = c.get(name) or {}
            if int(m.get("rows", 0)) <= 0 or m.get("auc") is None:
                e.append(f"{variant}:{role}:{name}:empty_or_null")
        for name in ("support_true", "reserve_true"):
            m = ai_c.get(name) or {}
            if int(m.get("rows", 0)) <= 0 or m.get("auc") is None:
                e.append(f"{variant}:{role}:ai:{name}:empty_or_null")
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


def _cross(rs: set[str], n: int) -> bool:
    return len(rs) >= n and any("near" in x for x in rs) and any("contact" in x for x in rs)


def _action_gate(docs: dict[str, dict[str, Any]], cell_key: str, metric: str) -> dict[str, Any]:
    positive: list[list[str]] = []
    top: list[list[str]] = []
    roles: set[str] = set(); top_roles: set[str] = set()
    for v in ("balanced", "precision"):
        for role in ROLES:
            c = docs[v][cell_key][role]
            m = c[f"{metric}_true"]
            if _ok(m.get("auc"), 0.65) and _ok(m.get("auc_vs_shuffled"), 0.05):
                positive.append([v, role]); roles.add(role)
            if _ok(m.get("top1_vs_shuffled"), 0.10):
                top.append([v, role]); top_roles.add(role)
    go = len(positive) >= 6 and _cross(roles, 3) and len(top) >= 4 and _cross(top_roles, 2)
    return {"go": bool(go), "positive_cells": positive, "top1_material_cells": top, "roles": sorted(roles), "top1_roles": sorted(top_roles)}


def _state_gate(docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    positive: list[list[str]] = []; roles: set[str] = set()
    for v in ("balanced", "precision"):
        for role in ROLES:
            m = docs[v]["cells"][role]["state"]
            if _ok(m.get("auc"), 0.70):
                positive.append([v, role]); roles.add(role)
    go = len(positive) >= 6 and _cross(roles, 3)
    return {"go": bool(go), "positive_cells": positive, "roles": sorted(roles)}


def _deltas_vs_v102(docs, refs) -> dict[str, list[dict[str, Any]]]:
    out = {"state": [], "support": [], "reserve": []}
    for v in ("balanced", "precision"):
        for role in ROLES:
            for name, key in (("state", "state"), ("support", "support_true"), ("reserve", "reserve_true")):
                a = docs[v]["cells"][role][key].get("auc")
                b = refs[v]["cells"][role][key].get("auc")
                if a is not None and b is not None:
                    out[name].append({
                        "variant": v,
                        "role": role,
                        "prelast_auc": float(a),
                        "v102_final_stage_i_auc": float(b),
                        "prelast_minus_v102_final": float(a) - float(b),
                    })
    return out


def _localization_summary(docs) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for axis in ("support", "reserve"):
        axis_out = {}
        for group in TOKEN_GROUP_ORDER:
            vals = []
            shares = []
            for v in ("balanced", "precision"):
                for role in ROLES:
                    m = (((docs[v].get("token_localization") or {}).get(role) or {}).get(axis) or {}).get(group) or {}
                    if m.get("cka_minus_shuffled") is not None:
                        vals.append(float(m["cka_minus_shuffled"]))
                    if m.get("mean_action_energy_share") is not None:
                        shares.append(float(m["mean_action_energy_share"]))
            axis_out[group] = {
                "mean_cka_minus_shuffled": None if not vals else sum(vals) / len(vals),
                "positive_cka_cells": sum(x > 0.0 for x in vals),
                "mean_action_energy_share": None if not shares else sum(shares) / len(shares),
                "cells": len(vals),
                "diagnostic_only": True,
            }
        out[axis] = axis_out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", type=Path, required=True)
    ap.add_argument("--precision", type=Path, required=True)
    ap.add_argument("--v102-balanced", type=Path, required=True)
    ap.add_argument("--v102-precision", type=Path, required=True)
    ap.add_argument("--v102-comparison", type=Path, required=True)
    ap.add_argument("--v104-comparison", type=Path, required=True)
    ap.add_argument("--v104-pipeline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    docs = {"balanced": json.loads(a.balanced.read_text()), "precision": json.loads(a.precision.read_text())}
    refs = {"balanced": json.loads(a.v102_balanced.read_text()), "precision": json.loads(a.v102_precision.read_text())}
    c102 = json.loads(a.v102_comparison.read_text())
    c104 = json.loads(a.v104_comparison.read_text())
    p104 = json.loads(a.v104_pipeline.read_text())
    errors: list[str] = []

    d102 = c102.get("preregistered_decision") or {}
    if not (c102.get("valid") and c102.get("attribution_ready") and c102.get("engineering_version") == V102_ENGINEERING_VERSION and d102.get("status") == "STAGE_I_ACTION_INFORMATION_SUFFICIENCY_STOP"):
        errors.append("v48_102_final_stage_i_audit_prerequisite_missing")
    d104 = c104.get("preregistered_decision") or {}
    if not (
        p104.get("valid") and p104.get("attribution_ready") and p104.get("engineering_version") == V104_ENGINEERING_VERSION
        and p104.get("preregistered_status") == "NOMINAL_INVARIANT_CONTROL_REFINEMENT_STOP"
        and c104.get("valid") and c104.get("attribution_ready")
        and d104.get("status") == "NOMINAL_INVARIANT_CONTROL_REFINEMENT_STOP"
        and d104.get("state_go") is True and d104.get("support_go") is False and d104.get("reserve_go") is False
        and d104.get("next_branch") == "close_last_stage_i_block_refinement_then_preregister_pre_last_token_action_equivariance_audit_no_broad_encoder_or_source_sweep"
    ):
        errors.append("v48_104_pre_last_audit_branch_prerequisite_missing")

    for v in ("balanced", "precision"):
        errors += _result_errors(docs[v], v)
        errors += _population_errors(docs[v], refs[v], v)

    state = _state_gate(docs) if not errors else {"go": False, "positive_cells": [], "roles": []}
    support = _action_gate(docs, "cells", "support") if not errors else {"go": False, "positive_cells": [], "top1_material_cells": [], "roles": [], "top1_roles": []}
    reserve = _action_gate(docs, "cells", "reserve") if not errors else {"go": False, "positive_cells": [], "top1_material_cells": [], "roles": [], "top1_roles": []}
    ai_support = _action_gate(docs, "action_interaction_cells", "support") if not errors else {"go": False, "positive_cells": [], "top1_material_cells": [], "roles": [], "top1_roles": []}
    ai_reserve = _action_gate(docs, "action_interaction_cells", "reserve") if not errors else {"go": False, "positive_cells": [], "top1_material_cells": [], "roles": [], "top1_roles": []}
    full = bool(state["go"] and support["go"] and reserve["go"])

    if errors:
        status = "V48_105_ENGINEERING_STOP"
        next_branch = "fix_v48_105_engineering_and_rerun_same_pre_last_audit"
    elif full:
        status = "PRELAST_CONTROL_SUFFICIENCY_GO"
        next_branch = "prelast_full_sufficiency_then_preregister_direct_nominal_invariant_prelast_response_transport_no_encoder_or_source_training"
    elif state["go"] and support["go"] and not reserve["go"]:
        status = "PRELAST_ACTION_EQUIVARIANCE_PARTIAL_SUPPORT"
        next_branch = "prelast_support_sufficient_then_preregister_one_signed_reserve_debt_localization_objective_no_source_or_broad_encoder"
    elif state["go"] and reserve["go"] and not support["go"]:
        status = "PRELAST_ACTION_EQUIVARIANCE_PARTIAL_RESERVE"
        next_branch = "prelast_reserve_sufficient_then_preregister_one_support_establishment_localization_objective_no_source_or_broad_encoder"
    elif ai_support["go"] and ai_reserve["go"]:
        status = "PRELAST_ACTION_INTERACTION_SUBSPACE_GO"
        next_branch = "action_interaction_subspace_sufficient_then_preregister_fixed_subspace_nominal_invariant_response_transport_no_encoder_or_source_sweep"
    else:
        status = "PRELAST_ACTION_EQUIVARIANCE_LOCALIZATION_STOP"
        next_branch = "prelast_action_equivariance_insufficient_then_preregister_one_block_earlier_action_interaction_audit_no_training_or_source_sweep"

    decision = {
        "status": status,
        "next_branch": next_branch,
        "prelast_state_go": state["go"],
        "prelast_support_go": support["go"],
        "prelast_reserve_go": reserve["go"],
        "prelast_control_sufficiency_go": full,
        "state_positive_cells": state["positive_cells"],
        "state_roles": state["roles"],
        "support_positive_cells": support["positive_cells"],
        "support_top1_material_cells": support["top1_material_cells"],
        "support_roles": support["roles"],
        "reserve_positive_cells": reserve["positive_cells"],
        "reserve_top1_material_cells": reserve["top1_material_cells"],
        "reserve_roles": reserve["roles"],
        "action_interaction_support_go": ai_support["go"],
        "action_interaction_reserve_go": ai_reserve["go"],
        "action_interaction_support_positive_cells": ai_support["positive_cells"],
        "action_interaction_reserve_positive_cells": ai_reserve["positive_cells"],
        "diagnostic_auc_deltas_vs_v102_final_stage_i": _deltas_vs_v102(docs, refs) if not errors else {},
        "token_localization": _localization_summary(docs) if not errors else {},
        "source_training_authorized": False,
        "broad_encoder_training_authorized": False,
        "boundary_transport_authorized": False,
        "dataset_reconstruction_authorized": False,
        "regime_conditioned_policy_authorized": False,
    }
    out = {
        "schema": "ocrap-v48.105-pael-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_pre_last_stage_i_action_equivariance_localization",
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "source_parameters_trained": 0,
        "dataset_reconstruction": False,
        "dataset_reselection": False,
        "teacher_metadata_input_to_model": False,
        "boundary_transport": False,
        "relative_ranker_modified": False,
        "regime_conditioning": False,
        "v48_102_comparison_sha256": _sha(a.v102_comparison),
        "v48_104_comparison_sha256": _sha(a.v104_comparison),
        "v48_104_pipeline_sha256": _sha(a.v104_pipeline),
        "preregistered_decision": decision,
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "state": state["go"], "support": support["go"], "reserve": reserve["go"], "action_interaction_support": ai_support["go"], "action_interaction_reserve": ai_reserve["go"]}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
