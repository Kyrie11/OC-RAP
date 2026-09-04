from __future__ import annotations

import json

import numpy as np

from ocrap.v48_89_root_correspondence import (
    audit_candidate_nominal_pair,
    nested_tail_influence,
    root_option_physical_intervals,
    semantic_future_branch_keys,
)


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
