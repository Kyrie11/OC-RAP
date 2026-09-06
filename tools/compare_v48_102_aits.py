#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.v48_102_action_information_transport_sufficiency import ENGINEERING_VERSION

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
    if not obj.get("audit_only"):
        e.append(f"{variant}:not_audit_only")
    if obj.get("boundary_transport") or obj.get("regime_conditioning") or obj.get("teacher_metadata_input_to_model"):
        e.append(f"{variant}:forbidden_runtime_input_or_transport")
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
    ap.add_argument("--v101-balanced", type=Path, required=True)
    ap.add_argument("--v101-precision", type=Path, required=True)
    ap.add_argument("--v101-comparison", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    docs = {"balanced": json.loads(a.balanced.read_text()), "precision": json.loads(a.precision.read_text())}
    refs = {"balanced": json.loads(a.v101_balanced.read_text()), "precision": json.loads(a.v101_precision.read_text())}
    c101 = json.loads(a.v101_comparison.read_text())
    errors: list[str] = []
    d101 = c101.get("preregistered_decision") or {}
    if not (
        c101.get("valid") and c101.get("attribution_ready")
        and d101.get("status") == "ROOT_CROSS_ATTENTION_SEMANTIC_ALIGNMENT_STOP"
        and d101.get("state_representation_go") is True
        and d101.get("support_action_representation_go") is False
        and d101.get("reserve_debt_representation_go") is False
        and d101.get("next_branch") == "close_root_decoder_semantic_family_then_preregister_stage_i_action_information_transport_audit_no_capacity_or_source_sweep"
    ):
        errors.append("v48_101_stop_branch_prerequisite_missing")
    for v in ("balanced", "precision"):
        errors += _result_errors(docs[v], v)
        errors += _population_errors(docs[v], refs[v], v)

    state_cells: list[list[str]] = []
    support_cells: list[list[str]] = []
    reserve_cells: list[list[str]] = []
    support_top: list[list[str]] = []
    reserve_top: list[list[str]] = []
    state_roles: set[str] = set()
    support_roles: set[str] = set()
    reserve_roles: set[str] = set()
    support_top_roles: set[str] = set()
    reserve_top_roles: set[str] = set()
    transport = {"state": [], "support": [], "reserve": []}

    if not errors:
        for v in ("balanced", "precision"):
            obj, ref = docs[v], refs[v]
            for role in ROLES:
                c = obj["cells"][role]
                r = ref["cells"][role]
                st = c["state"]
                su = c["support_true"]
                re = c["reserve_true"]
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
                for name, metric in (("state", "state"), ("support", "support_true"), ("reserve", "reserve_true")):
                    av = (c.get(metric) or {}).get("auc")
                    bv = (r.get(metric) or {}).get("auc")
                    if av is not None and bv is not None:
                        transport[name].append({
                            "variant": v,
                            "role": role,
                            "stage_i_auc": float(av),
                            "v101_root_auc": float(bv),
                            "stage_i_minus_v101": float(av) - float(bv),
                        })

    def cross(rs: set[str], n: int) -> bool:
        return len(rs) >= n and any("near" in x for x in rs) and any("contact" in x for x in rs)

    state_go = bool(not errors and len(state_cells) >= 6 and cross(state_roles, 3))
    support_go = bool(not errors and len(support_cells) >= 6 and cross(support_roles, 3) and len(support_top) >= 4 and cross(support_top_roles, 2))
    reserve_go = bool(not errors and len(reserve_cells) >= 6 and cross(reserve_roles, 3) and len(reserve_top) >= 4 and cross(reserve_top_roles, 2))
    full_go = bool(state_go and support_go and reserve_go)

    if errors:
        status = "V48_102_ENGINEERING_STOP"
        next_branch = "fix_v48_102_engineering_and_rerun_same_stage_i_audit"
    elif full_go:
        status = "STAGE_I_ACTION_INFORMATION_SUFFICIENCY_GO"
        next_branch = "stage_i_semantics_sufficient_root_transport_bottleneck_then_preregister_one_direct_memory_to_recovery_semantic_transport_no_source_or_capacity_sweep"
    elif support_go or reserve_go:
        status = "STAGE_I_ACTION_INFORMATION_PARTIAL"
        next_branch = "stage_i_partial_action_sufficiency_then_preregister_minimal_stage_i_recovery_sufficient_representation_objective_no_source_or_broad_encoder_sweep"
    else:
        status = "STAGE_I_ACTION_INFORMATION_SUFFICIENCY_STOP"
        next_branch = "stage_i_action_information_insufficient_then_preregister_minimal_stage_i_recovery_representation_objective_no_source_or_broad_encoder_sweep"

    decision = {
        "stage_i_state_observability_go": state_go,
        "stage_i_support_action_observability_go": support_go,
        "stage_i_reserve_action_observability_go": reserve_go,
        "stage_i_action_information_sufficiency_go": full_go,
        "state_positive_cells": state_cells,
        "support_positive_cells": support_cells,
        "reserve_positive_cells": reserve_cells,
        "support_top1_material_cells": support_top,
        "reserve_top1_material_cells": reserve_top,
        "state_roles": sorted(state_roles),
        "support_roles": sorted(support_roles),
        "reserve_roles": sorted(reserve_roles),
        "transport_auc_stage_i_vs_v101": transport,
        "status": status,
        "next_branch": next_branch,
        "source_training_authorized": False,
        "boundary_transport_authorized": False,
        "dataset_reconstruction_authorized": False,
        "regime_conditioned_policy_authorized": False,
    }
    out = {
        "schema": "ocrap-v48.102-aits-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_stage_i_action_information_transport_sufficiency",
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
        "v48_101_comparison_sha256": _sha(a.v101_comparison),
        "preregistered_decision": decision,
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "state": state_go, "support": support_go, "reserve": reserve_go}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
