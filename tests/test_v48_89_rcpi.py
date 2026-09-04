from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ocrap.v48_89_root_correspondence import (
    audit_candidate_nominal_pair,
    nested_tail_influence,
    root_option_physical_intervals,
    semantic_future_branch_keys,
)
from tools.build_v48_89_root_correspondence_audit import ROLE_FILES, _labels


def _sample(assignments, margins, *, mode="stop", metadata=None):
    if metadata is None:
        metadata = [{}, {"reactive_variant": 0}]
    s = {
        "m_star": np.asarray(margins, dtype=np.float32),
        "root_probs": np.asarray([0.5, 0.5], dtype=np.float32),
        "root_valid": np.asarray([1, 1], dtype=np.float32),
        "c_star": np.eye(2, dtype=np.float32),
        "option_valid": np.asarray([1], dtype=np.float32),
        "root_assignments": np.asarray(assignments, dtype=np.int64),
        "future_probs": np.asarray([0.5, 0.5], dtype=np.float32),
        "future_sources": np.asarray(["replay", "reactive"]),
        "future_metadata": json.dumps(metadata, sort_keys=True),
        "recovery_modes": np.asarray([mode]),
    }
    _, r, _, _ = nested_tail_influence(s)
    s["r_dep_star"] = np.float32(r)
    return s


def test_semantic_correspondence_recovers_root_slot_permutation():
    nominal = _sample([0, 1], [[-1.0], [1.0]])
    candidate = _sample([1, 0], [[1.2], [-0.5]])
    rec = audit_candidate_nominal_pair(candidate, nominal)
    assert rec.valid
    assert rec.candidate_to_nominal_root == [1, 0]
    assert rec.nested_tail_exact_correspondence_mass == 1.0
    assert rec.branch_vs_slot_mapping_disagreement_fraction == 1.0


def test_future_identity_is_candidate_independent_and_unique():
    a = _sample([0, 1], [[-1.0], [1.0]])
    b = _sample([1, 0], [[1.0], [-1.0]])
    ka, wa, ca = semantic_future_branch_keys(a)
    kb, wb, cb = semantic_future_branch_keys(b)
    assert ka == kb
    assert len(set(ka)) == 2
    assert not wa.any() and not wb.any()
    assert ca == 0 and cb == 0


def test_switch_inverse_distinguishes_active_from_inactive_floor():
    # Uniform yield_rejoin floor exists.  Stored 0.8 is above the 0.6 switch,
    # so the pre-floor physical value is point identified as 0.8.
    s = _sample([0, 1], [[0.8], [0.9]], mode="yield_rejoin")
    lo, hi, exact, informative = root_option_physical_intervals(s)
    assert np.allclose(lo, [[0.8], [0.9]])
    assert np.allclose(hi, [[0.8], [0.9]])
    assert exact.all() and informative.all()


def test_mixed_structural_profile_fails_closed():
    meta = [{"route_blocked": True}, {"route_blocked": False, "reactive_variant": 0}]
    # Both futures in the same root, but only one is route-blocked.  Root-level
    # inversion must be unidentifiable rather than fabricating a point target.
    s = _sample([0, 0], [[-0.8], [1.0]], mode="yield_rejoin", metadata=meta)
    lo, hi, exact, informative = root_option_physical_intervals(s)
    assert lo[0, 0] < -1e5 and hi[0, 0] > 1e5
    assert not exact[0, 0]
    assert not informative[0, 0]


def test_duplicate_semantic_branches_are_reported_as_weak_order_fallback():
    s = _sample(
        [0, 1],
        [[-1.0], [1.0]],
        metadata=[{"reactive_variant": 0}, {"reactive_variant": 0}],
    )
    s["future_sources"] = np.asarray(["reactive", "reactive"])
    keys, weak, collisions = semantic_future_branch_keys(s)
    assert len(set(keys)) == 2
    assert weak.tolist() == [True, True]
    assert collisions == 1


def test_invalid_future_does_not_contribute_to_correspondence_mass():
    nominal = _sample([0, 1], [[-1.0], [1.0]])
    candidate = _sample([1, 0], [[1.2], [-0.5]])
    nominal["future_valid"] = np.asarray([1, 0], dtype=np.float32)
    candidate["future_valid"] = np.asarray([1, 0], dtype=np.float32)
    rec = audit_candidate_nominal_pair(candidate, nominal)
    assert rec.valid
    assert rec.shared_future_mass_candidate == 1.0
    assert rec.shared_future_mass_nominal == 1.0


def _proposal_row(scene: str, time: int, candidate: int, *, adv: float = 0.1):
    return {
        "scene": scene,
        "time": time,
        "candidate": candidate,
        "teacher_adv": adv,
        "teacher_harmful": False,
        "teacher_candidate_r_dep": 0.2,
        "macro": 2,
    }


def _write_proposal_support(root: Path, variant: str, role: str, rows) -> None:
    p = root / "candidates" / variant / "calibration" / ROLE_FILES[role]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_label_identity_accepts_variant_specific_topk_membership_and_uses_union(tmp_path):
    for role in ROLE_FILES:
        _write_proposal_support(
            tmp_path,
            "balanced",
            role,
            [_proposal_row(f"{role}-scene", 1, 1), _proposal_row(f"{role}-scene", 1, 2)],
        )
        _write_proposal_support(
            tmp_path,
            "precision",
            role,
            [_proposal_row(f"{role}-scene", 1, 2), _proposal_row(f"{role}-scene", 1, 3)],
        )
    labels, identity = _labels(tmp_path)
    for role in ROLE_FILES:
        assert len(labels[role]) == 3
        assert identity[role]["shared_rows"] == 1
        assert identity[role]["union_rows"] == 3
        assert identity[role]["exact_key_identity"] is False
        assert identity[role]["teacher_value_identity_on_overlap"] is True
        assert labels[role][(f"{role}-scene", 1, 2)]["_v4889_label_variants"] == ["balanced", "precision"]


def test_label_identity_still_fails_on_teacher_value_conflict(tmp_path):
    for role in ROLE_FILES:
        balanced = [_proposal_row(f"{role}-scene", 1, 1)]
        precision = [_proposal_row(f"{role}-scene", 1, 1)]
        if role == "dev_near":
            precision[0] = _proposal_row(f"{role}-scene", 1, 1, adv=0.2)
        _write_proposal_support(tmp_path, "balanced", role, balanced)
        _write_proposal_support(tmp_path, "precision", role, precision)
    with pytest.raises(ValueError, match="teacher-value identity failed"):
        _labels(tmp_path)


def test_weak_occurrence_fallback_is_not_counted_as_exact_root_correspondence():
    nominal = _sample(
        [0, 1],
        [[-1.0], [1.0]],
        metadata=[{"reactive_variant": 0}, {"reactive_variant": 0}],
    )
    candidate = _sample(
        [0, 1],
        [[-0.5], [1.2]],
        metadata=[{"reactive_variant": 0}, {"reactive_variant": 0}],
    )
    nominal["future_sources"] = np.asarray(["reactive", "reactive"])
    candidate["future_sources"] = np.asarray(["reactive", "reactive"])
    rec = audit_candidate_nominal_pair(candidate, nominal)
    assert rec.valid
    assert rec.semantic_identity_fallback_fraction_candidate == 1.0
    assert rec.semantic_identity_fallback_fraction_nominal == 1.0
    assert rec.nested_tail_exact_correspondence_mass == 0.0


def test_recovery_option_identity_mismatch_fails_closed():
    nominal = _sample([0, 1], [[-1.0], [1.0]], mode="stop")
    candidate = _sample([0, 1], [[-0.5], [1.2]], mode="yield_rejoin")
    rec = audit_candidate_nominal_pair(candidate, nominal)
    assert not rec.valid
    assert "recovery option identity/order mismatch" in str(rec.error)
