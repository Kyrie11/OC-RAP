from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ocrap.simulation.teacher.margins as margins_mod
from ocrap.algorithms.evidence_targets import ComponentVetoTolerances


def _fake_margin_call(monkeypatch: pytest.MonkeyPatch, *, mode: str) -> float:
    monkeypatch.setattr(
        margins_mod,
        "component_margins",
        lambda *a, **k: {
            "clearance": -0.4,
            "stop": 1.0,
            "control": 1.0,
            "route": 1.0,
            "harm": 1.0,
            "stability": 1.0,
            "secondary": 1.0,
        },
    )
    history = SimpleNamespace()
    prefix = SimpleNamespace()
    future = SimpleNamespace(metadata={})
    option = SimpleNamespace(mode="yield_rejoin", valid=True)
    val, _ = margins_mod.teacher_margin(
        history, prefix, future, option,
        np.zeros((1, 7), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        {
            "artifact": {"use_margin_override": False},
            "teacher_margin_semantics": {"mode": mode},
            "margin_scales": {"inactive": 10.0},
        },
    )
    return float(val)


def test_strict_min_slack_removes_legacy_positive_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _fake_margin_call(monkeypatch, mode="legacy") == pytest.approx(0.6)
    assert _fake_margin_call(monkeypatch, mode="strict_min_slack") == pytest.approx(-0.4)


def test_unknown_teacher_margin_semantics_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="teacher_margin_semantics.mode"):
        _fake_margin_call(monkeypatch, mode="not_a_mode")


def test_v48_56_audit_detects_pcd_veto_conflict(tmp_path: Path) -> None:
    # One group where a DRS jump makes legacy PCD beneficial although GAP degrades.
    nom = {
        "path": "/tmp/evidence_adapt_train_near_contact/nom.npz",
        "bucket": 1, "scene": "s", "time": 1, "candidate": 0, "macro": 0,
        "nominal": True, "teacher_pcd": 0.0, "teacher_drs": 0.0,
        "teacher_r_dep": -1.0, "teacher_gap": 0.0,
        "teacher_hard_violation": 0.0, "teacher_harm_proxy": 0.0,
        "component_veto_margin": 0.0, "component_harmful": False, "beneficial": False,
    }
    # Construct exact legacy fields using the repository semantics.
    from ocrap.evaluation.metrics import post_contact_deployability_score
    from ocrap.algorithms.evidence_targets import component_veto_margin_numpy
    cand = dict(nom)
    cand.update({
        "path": "/tmp/evidence_adapt_train_near_contact/cand.npz",
        "candidate": 1, "macro": 2, "nominal": False,
        "teacher_drs": 1.0, "teacher_r_dep": -0.5, "teacher_gap": 1.0,
    })
    nom["teacher_pcd"] = post_contact_deployability_score(
        nom["teacher_drs"], nom["teacher_r_dep"], nom["teacher_gap"]
    )
    cand["teacher_pcd"] = post_contact_deployability_score(
        cand["teacher_drs"], cand["teacher_r_dep"], cand["teacher_gap"]
    )
    margin = component_veto_margin_numpy(
        candidate_drs=cand["teacher_drs"], nominal_drs=nom["teacher_drs"],
        candidate_r_dep=cand["teacher_r_dep"], nominal_r_dep=nom["teacher_r_dep"],
        candidate_gap=cand["teacher_gap"], nominal_gap=nom["teacher_gap"],
        tolerances=ComponentVetoTolerances(),
    )
    cand["component_veto_margin"] = margin
    cand["component_harmful"] = bool(margin > 0)
    cand["beneficial"] = bool(cand["teacher_pcd"] - nom["teacher_pcd"] >= 0.015)
    assert cand["beneficial"] and cand["component_harmful"]

    idx = tmp_path / "idx.jsonl"
    idx.write_text("\n".join(json.dumps(x) for x in [nom, cand]) + "\n", encoding="utf-8")
    from importlib.util import module_from_spec, spec_from_file_location
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "audit_v48_56_teacher_component_semantics.py"
    spec = spec_from_file_location("v4856_audit", tool_path)
    assert spec and spec.loader
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = mod._audit_index(idx, {2, 3, 5, 6, 7})
    near = report["by_regime"]["near"]["legacy_pcd"]
    assert near["beneficial_and_harmful_candidates"] == 1
    assert near["overlap_max_culprit_counts"]["gap"] == 1


def test_strict_mode_keeps_explicit_intent_component_separate_from_postmin_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        margins_mod,
        "component_margins",
        lambda *a, **k: {
            "clearance": 0.4,
            "stop": 1.0,
            "control": 1.0,
            "route": 1.0,
            "harm": 1.0,
            "stability": 1.0,
            "secondary": 1.0,
            "intent": -0.7,
        },
    )
    future = SimpleNamespace(metadata={"hidden_intent": "yield"})
    option = SimpleNamespace(mode="yield_rejoin", valid=True)
    val, diag = margins_mod.teacher_margin(
        SimpleNamespace(), SimpleNamespace(), future, option,
        np.zeros((1, 7), dtype=np.float32), np.zeros((1, 4), dtype=np.float32),
        {
            "artifact": {"use_margin_override": False, "enable_branch_intent_margin": True},
            "teacher_margin_semantics": {"mode": "strict_min_slack"},
            "margin_scales": {"inactive": 10.0},
        },
    )
    assert val == pytest.approx(-0.7)
    assert diag.component_margins["intent"] == pytest.approx(-0.7)


def test_v48_56_audit_refuses_test_role_inputs() -> None:
    from importlib.util import module_from_spec, spec_from_file_location
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "audit_v48_56_teacher_component_semantics.py"
    spec = spec_from_file_location("v4856_audit_reject_test", tool_path)
    assert spec and spec.loader
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(ValueError, match="refuses test-role input"):
        mod._reject_test_inputs("/data/OCRAP/test_contact", "/tmp/dev_index.jsonl")
