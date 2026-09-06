#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.v48_101_root_cross_attention_semantic_alignment import ENGINEERING_VERSION, expected_cross_attention_parameter_count

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _result_errors(o: dict, v: str) -> list[str]:
    e: list[str] = []
    if o.get("engineering_version") != ENGINEERING_VERSION or not o.get("valid"):
        e.append(f"{v}_contract")
    if int(o.get("root_cross_attention_parameters_trained", -1)) != expected_cross_attention_parameter_count(192):
        e.append(f"{v}_parameter_contract")
    for k in (
        "planner_parameters_trained", "source_parameters_trained", "stage_i_parameters_trained",
        "root_query_parameters_trained", "recovery_chart_parameters_trained", "root_self_attention_parameters_trained",
        "root_ffn_parameters_trained", "root_logit_head_parameters_trained",
    ):
        if int(o.get(k, -1)) != 0:
            e.append(f"{v}_{k}")
    if not (o.get("v100_baseline_identity") or {}).get("valid"):
        e.append(f"{v}_v100_baseline_identity")
    if not o.get("root_cross_attention_changed"):
        e.append(f"{v}_attention_noop")
    for role in ROLES:
        if not (((o.get("evaluation_contracts") or {}).get(role) or {}).get("valid")):
            e.append(f"{v}_{role}_eval_contract")
        for n in ("state", "support_true", "reserve_true"):
            m = (((o.get("cells") or {}).get(role) or {}).get(n) or {})
            if int(m.get("rows", 0)) <= 0 or m.get("auc") is None:
                e.append(f"{v}_{role}_{n}_empty")
    return e


def _state_errors(path: Path, v: str) -> list[str]:
    if not path.is_file():
        return [f"missing_{v}_state"]
    try:
        obj = __import__("torch").load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return [f"invalid_{v}_state:{exc}"]
    e: list[str] = []
    if obj.get("engineering_version") != ENGINEERING_VERSION:
        e.append(f"{v}_state_version")
    if int(obj.get("root_cross_attention_parameter_count", -1)) != expected_cross_attention_parameter_count(192):
        e.append(f"{v}_state_parameter_count")
    if not obj.get("v100_state_sha256") or not obj.get("v100_result_sha256") or not obj.get("l80_checkpoint_sha256"):
        e.append(f"{v}_state_prerequisite_provenance")
    if obj.get("initial_root_cross_attention_sha256") == obj.get("final_root_cross_attention_sha256"):
        e.append(f"{v}_state_attention_noop")
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    for x in (
        "runtime", "balanced", "precision", "balanced_state", "precision_state", "comparison",
        "v48_100_pipeline", "v48_100_comparison", "output",
    ):
        ap.add_argument("--" + x.replace("_", "-"), dest=x, type=Path, required=True)
    a = ap.parse_args(); errors: list[str] = []; objs = {}
    for name, p in (("runtime", a.runtime), ("balanced", a.balanced), ("precision", a.precision), ("comparison", a.comparison)):
        if not p.is_file():
            errors.append("missing_" + name); continue
        try:
            objs[name] = json.loads(p.read_text())
        except Exception as exc:
            errors.append(f"invalid_{name}:{exc}")

    rt = objs.get("runtime", {})
    if rt.get("engineering_version") != ENGINEERING_VERSION or not rt.get("valid") or not rt.get("attribution_ready"):
        errors.append("runtime_contract")
    errors += _result_errors(objs.get("balanced", {}), "balanced") + _result_errors(objs.get("precision", {}), "precision")
    errors += _state_errors(a.balanced_state, "balanced") + _state_errors(a.precision_state, "precision")
    comp = objs.get("comparison", {})
    if comp.get("engineering_version") != ENGINEERING_VERSION or not comp.get("valid") or not comp.get("attribution_ready"):
        errors.append("comparison_contract")

    if not a.v48_100_pipeline.is_file() or not a.v48_100_comparison.is_file():
        errors.append("missing_v48_100_prerequisite")
    else:
        v100p = json.loads(a.v48_100_pipeline.read_text())
        v100c = json.loads(a.v48_100_comparison.read_text())
        d = v100c.get("preregistered_decision") or {}
        if not (
            v100p.get("valid") and v100p.get("attribution_ready")
            and v100p.get("preregistered_status") == "JOINT_ROOT_SEMANTIC_DECODER_STOP"
            and v100c.get("valid") and v100c.get("attribution_ready")
            and d.get("next_branch") == "close_root_query_plus_chart_family_then_preregister_root_cross_attention_semantic_objective_no_source_sweep"
        ):
            errors.append("v48_100_stop_prerequisite")
        expected = ((v100p.get("artifacts") or {}).get("comparison") or {}).get("sha256")
        if not expected or comp.get("v48_100_comparison_sha256") != expected:
            errors.append("v48_100_comparison_provenance_mismatch")

    artifacts = {}
    for name, p in (
        ("runtime", a.runtime), ("balanced", a.balanced), ("precision", a.precision),
        ("balanced_state", a.balanced_state), ("precision_state", a.precision_state), ("comparison", a.comparison),
    ):
        if p.is_file():
            artifacts[name] = {"path": str(p.resolve()), "sha256": sha(p)}
    status = (comp.get("preregistered_decision") or {}).get("status")
    out = {
        "schema": "ocrap-v48.101-rcsa-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "root_cross_attention_only_semantic_alignment_from_frozen_v100_state",
        "planner_parameters_trained": 0,
        "source_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_query_parameters_trained": 0,
        "recovery_chart_parameters_trained": 0,
        "root_cross_attention_parameters_trained": int((objs.get("balanced") or {}).get("root_cross_attention_parameters_trained", 0)),
        "root_self_attention_parameters_trained": 0,
        "root_ffn_parameters_trained": 0,
        "root_logit_head_parameters_trained": 0,
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
