from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from ocrap.v48_103_factorized_control_sufficient_state import (
    ENGINEERING_VERSION,
    FactorizedControlSufficientState,
    build_nominal_index,
    expected_parameter_count,
    nominal_response_zero_check,
    state_response_parameter_disjoint_check,
    token_permutation_invariance_check,
)


def test_parameter_and_structure_contract():
    m = FactorizedControlSufficientState(192)
    assert m.trainable_parameter_count == 1540
    assert expected_parameter_count(192) == 1540
    assert state_response_parameter_disjoint_check(16)


def test_nominal_response_exact_zero():
    assert nominal_response_zero_check(16)


def test_token_pooling_permutation_invariant():
    assert token_permutation_invariance_check(16)


def test_group_anchor_builder():
    rows = [
        {"bucket": 1, "scene": "a", "time": 1, "nominal": True},
        {"bucket": 1, "scene": "a", "time": 1, "nominal": False},
        {"bucket": 2, "scene": "b", "time": 3, "nominal": True},
        {"bucket": 2, "scene": "b", "time": 3, "nominal": False},
        {"bucket": 2, "scene": "b", "time": 3, "nominal": False},
    ]
    x = build_nominal_index(rows)
    assert torch.equal(x, torch.tensor([0, 0, 2, 2, 2]))


def _cell(high: bool):
    if high:
        return {
            "state": {"rows": 10, "drs_state_rows": 5, "dep_state_rows": 5, "auc": 0.8},
            "support_true": {"rows": 20, "positive_rows": 10, "negative_rows": 10, "powered_groups": 5, "auc": 0.8, "auc_vs_shuffled": 0.2, "top1_vs_shuffled": 0.2},
            "support_shuffled": {"rows": 20, "positive_rows": 10, "negative_rows": 10, "powered_groups": 5, "auc": 0.6, "top1": 0.4},
            "reserve_true": {"rows": 20, "positive_rows": 10, "negative_rows": 10, "powered_groups": 5, "auc": 0.8, "auc_vs_shuffled": 0.2, "top1_vs_shuffled": 0.2},
            "reserve_shuffled": {"rows": 20, "positive_rows": 10, "negative_rows": 10, "powered_groups": 5, "auc": 0.6, "top1": 0.4},
        }
    return {
        "state": {"rows": 10, "drs_state_rows": 5, "dep_state_rows": 5, "auc": 0.5},
        "support_true": {"rows": 20, "positive_rows": 10, "negative_rows": 10, "powered_groups": 5, "auc": 0.5, "auc_vs_shuffled": 0.0, "top1_vs_shuffled": 0.0},
        "support_shuffled": {"rows": 20, "positive_rows": 10, "negative_rows": 10, "powered_groups": 5, "auc": 0.5, "top1": 0.5},
        "reserve_true": {"rows": 20, "positive_rows": 10, "negative_rows": 10, "powered_groups": 5, "auc": 0.5, "auc_vs_shuffled": 0.0, "top1_vs_shuffled": 0.0},
        "reserve_shuffled": {"rows": 20, "positive_rows": 10, "negative_rows": 10, "powered_groups": 5, "auc": 0.5, "top1": 0.5},
    }


def _result(high: bool, variant: str):
    return {
        "valid": True,
        "engineering_version": ENGINEERING_VERSION,
        "variant": variant,
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "representation_parameters_trained": 1540,
        "source_parameters_trained": 0,
        "boundary_transport": False,
        "regime_conditioning": False,
        "teacher_metadata_input_to_model": False,
        "state_response_learned_mixing": False,
        "nominal_response_exact_zero": True,
        "cells": {r: _cell(high) for r in ("dev_near", "dev_contact", "certificate_near", "certificate_contact")},
    }


def _v102_result():
    d = _result(False, "balanced")
    d.update({"engineering_version": "v48.102.0-OC-AITS", "audit_only": True})
    d.pop("representation_parameters_trained", None)
    return d


def test_compare_go_and_stop(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    tool = repo / "tools/compare_v48_103_fcss.py"
    c102 = {
        "valid": True,
        "attribution_ready": True,
        "preregistered_decision": {
            "status": "STAGE_I_ACTION_INFORMATION_SUFFICIENCY_STOP",
            "stage_i_state_observability_go": False,
            "stage_i_support_action_observability_go": False,
            "stage_i_reserve_action_observability_go": False,
            "next_branch": "stage_i_action_information_insufficient_then_preregister_minimal_stage_i_recovery_representation_objective_no_source_or_broad_encoder_sweep",
        },
    }
    cpath = tmp_path / "c102.json"; cpath.write_text(json.dumps(c102))
    refs = {}
    for v in ("balanced", "precision"):
        r = _v102_result(); r["variant"] = v
        p = tmp_path / f"v102_{v}.json"; p.write_text(json.dumps(r)); refs[v] = p
    # V101 only supplies diagnostic AUCs; reuse same population shape.
    refs101 = {}
    for v in ("balanced", "precision"):
        r = _result(False, v); r["engineering_version"] = "v48.101.0-OC-RCSA"
        p = tmp_path / f"v101_{v}.json"; p.write_text(json.dumps(r)); refs101[v] = p
    env = os.environ.copy(); env["PYTHONPATH"] = f"{repo/'src'}:{repo}" + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    for high, status in ((True, "FACTORIZED_CONTROL_SUFFICIENT_STATE_GO"), (False, "FACTORIZED_CONTROL_SUFFICIENT_STATE_STOP")):
        bp = tmp_path / f"b_{high}.json"; pp = tmp_path / f"p_{high}.json"; out = tmp_path / f"o_{high}.json"
        bp.write_text(json.dumps(_result(high, "balanced"))); pp.write_text(json.dumps(_result(high, "precision")))
        subprocess.run([
            sys.executable, str(tool), "--balanced", str(bp), "--precision", str(pp),
            "--v102-balanced", str(refs["balanced"]), "--v102-precision", str(refs["precision"]),
            "--v102-comparison", str(cpath), "--v101-balanced", str(refs101["balanced"]),
            "--v101-precision", str(refs101["precision"]), "--output", str(out),
        ], check=True, env=env)
        assert json.loads(out.read_text())["preregistered_decision"]["status"] == status


def test_runtime_contract(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "runtime.json"
    env = os.environ.copy(); env["PYTHONPATH"] = f"{repo/'src'}:{repo}" + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    subprocess.run([sys.executable, str(repo / "tools/check_v48_103_runtime_code_contract.py"), "--repo", str(repo), "--output", str(out)], check=True, env=env)
    d = json.loads(out.read_text())
    assert d["valid"] and d["scientific_contract"]["fixed_representation_parameters"] == 1540
