from __future__ import annotations

import numpy as np
import pytest

from ocrap.audits.executable_constraint_jacobian import (
    JACOBIAN_GEOMETRY_DIM,
    MATCHED_DIM,
    best_option_index,
    contract_checks,
    executable_constraint_field_from_sample,
    jacobian_geometry,
    validate_group_contract,
)


def _sample(*, candidate_index: int, lateral: float = 0.0, agent_x: float = 12.0):
    T = 8
    states = np.zeros((T, 9), dtype=np.float32)
    states[:, 0] = np.linspace(0.2, 3.0, T)
    states[:, 1] = np.linspace(0.0, lateral, T)
    states[:, 6] = 4.0
    states[:, 7] = 4.8
    states[:, 8] = 2.0
    controls = np.zeros((T, 4), dtype=np.float32)
    hist = np.zeros((3, 3, 16), dtype=np.float32)
    valid = np.ones((3, 3), dtype=np.float32)
    hist[:, 0, 10] = 4.8
    hist[:, 0, 11] = 2.0
    hist[:, 1, 0] = agent_x
    hist[:, 1, 3] = -0.5
    hist[:, 1, 10] = 4.5
    hist[:, 1, 11] = 2.0
    hist[:, 2, 0] = 25.0
    hist[:, 2, 3] = 0.5
    hist[:, 2, 10] = 4.0
    hist[:, 2, 11] = 1.8
    return {
        "scene_id": "s",
        "time_index": np.int64(5),
        "candidate_index": np.int64(candidate_index),
        "is_nominal": np.int64(candidate_index == 0),
        "agent_history": hist,
        "agent_valid": valid,
        "ego_state": np.asarray([0, 0, 4, 0, 0, 0, 4, 4.8, 2.0], dtype=np.float32),
        "prefix_states": states,
        "prefix_controls": controls,
        "prefix_macro_id": np.int64(0),
        "prefix_param": np.zeros((3,), dtype=np.float32),
        "recovery_modes": np.asarray(["brake_lane", "lateral_escape", "post_contact_stabilize"], dtype=object),
        "recovery_params": np.asarray([[-4.0, 1.2, 0.0], [2.0, 4.0, 1.0], [0.8, 1.2, -2.0]], dtype=np.float32),
        "option_valid": np.ones((3,), dtype=np.float32),
    }


def _cfg():
    return {
        "sample_rate_hz": 10.0,
        "recovery_horizon_s": 2.0,
        "d_safe0_m": 1.0,
        "safe_time_headway_s": 0.5,
        "route_dev_max_m": 2.5,
        "margin_scales": {"distance": 2.0, "stop": 5.0, "route": 1.0},
        "control_limits": {"a_min": -6.0, "a_max": 3.0, "delta_max": 0.55, "j_max": 6.0, "steer_rate_max": 0.5},
    }


def test_contract_checks_all_pass():
    checks = contract_checks()
    assert all(checks.values()), checks
    assert JACOBIAN_GEOMETRY_DIM == 64
    assert MATCHED_DIM == 220


def test_same_option_candidate_equal_nominal_is_exact_zero():
    d = _sample(candidate_index=0)
    f = executable_constraint_field_from_sample(d, _cfg())
    l = best_option_index(f)
    g = jacobian_geometry(f, f, l)
    assert g.shape == (64,)
    assert np.count_nonzero(g) == 0


def test_reentry_can_activate_from_physical_contact_without_regime_label():
    d = _sample(candidate_index=0, agent_x=1.0)
    f = executable_constraint_field_from_sample(d, _cfg())
    assert f.diagnostics["reentry_active_options"] > 0
    assert f.masks[:, :, 3].any()


def test_observation_mismatch_fails_closed():
    n = _sample(candidate_index=0)
    c = _sample(candidate_index=1, lateral=0.4)
    c["ego_state"] = c["ego_state"].copy()
    c["ego_state"][0] = 1.0
    with pytest.raises(ValueError, match="observation changed"):
        validate_group_contract(n, c)


def test_agent_permutation_is_invariant():
    d = _sample(candidate_index=1, lateral=0.4)
    f1 = executable_constraint_field_from_sample(d, _cfg())
    p = dict(d)
    p["agent_history"] = d["agent_history"][:, [0, 2, 1], :].copy()
    p["agent_valid"] = d["agent_valid"][:, [0, 2, 1]].copy()
    f2 = executable_constraint_field_from_sample(p, _cfg())
    assert np.allclose(f1.values, f2.values, rtol=0.0, atol=1e-10)
    assert np.array_equal(f1.masks, f2.masks)
    assert np.allclose(f1.option_scores, f2.option_scores, rtol=0.0, atol=1e-10)


def test_contact_bucket_name_does_not_activate_reentry_without_physical_contact():
    # Contact/post-contact is a dataset/evaluation role, not a deployable input.
    # The ECJ primitive may only activate re-entry from observed/prefix/recovery
    # physical contact, never from a regime label.
    d = _sample(candidate_index=0, agent_x=20.0)
    d["regime_contact"] = np.int64(1)
    d["bucket"] = np.asarray("contact")
    f = executable_constraint_field_from_sample(d, _cfg())
    assert f.diagnostics["observed_or_prefix_contact"] is False
    assert f.diagnostics["reentry_active_options"] == 0 or not f.masks[:, :, 3].any()
