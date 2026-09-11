from __future__ import annotations

import numpy as np

from ocrap.audits.common_option_constraint_work import (
    MATCHED_DIM,
    WORK_GEOMETRY_DIM,
    _bin_edges,
    constraint_work_geometry,
    contract_checks,
    integral_response_geometry,
    work_conservation_error,
)
from ocrap.audits.executable_constraint_jacobian import (
    best_option_index,
    executable_constraint_field_from_sample,
)


def _sample(candidate_index: int = 0, lateral: float = 0.0, agent_x: float = 1.0):
    T = 8
    states = np.zeros((T, 9), dtype=np.float32)
    states[:, 0] = np.linspace(0.1, 2.0, T)
    states[:, 1] = np.linspace(0.0, lateral, T)
    states[:, 6] = 3.0
    states[:, 7] = 4.8
    states[:, 8] = 2.0
    hist = np.zeros((3, 3, 16), dtype=np.float32)
    valid = np.ones((3, 3), dtype=np.float32)
    hist[:, 0, 10] = 4.8
    hist[:, 0, 11] = 2.0
    hist[:, 1, 0] = agent_x
    hist[:, 1, 10] = 4.5
    hist[:, 1, 11] = 2.0
    hist[:, 2, 0] = 15.0
    hist[:, 2, 3] = -1.0
    hist[:, 2, 10] = 4.0
    hist[:, 2, 11] = 1.8
    return {
        "scene_id": "s",
        "time_index": np.int64(2),
        "candidate_index": np.int64(candidate_index),
        "is_nominal": np.int64(candidate_index == 0),
        "agent_history": hist,
        "agent_valid": valid,
        "ego_state": np.asarray([0, 0, 3, 0, 0, 0, 3, 4.8, 2.0], dtype=np.float32),
        "prefix_states": states,
        "prefix_controls": np.zeros((T, 4), dtype=np.float32),
        "prefix_macro_id": np.int64(0),
        "prefix_param": np.zeros((3,), dtype=np.float32),
        "recovery_modes": np.asarray(["brake_lane", "post_contact_stabilize", "avoid_secondary"], dtype=object),
        "recovery_params": np.asarray([[-3.0, 1.0, 0.0], [0.8, 1.2, -2.0], [3.5, -3.0, 8.0]], dtype=np.float32),
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


def test_ccw_contract_checks_all_pass():
    checks = contract_checks()
    assert checks and all(checks.values()), checks
    assert WORK_GEOMETRY_DIM == 64
    assert MATCHED_DIM == 220


def test_nominal_work_and_integral_are_exact_zero():
    f = executable_constraint_field_from_sample(_sample(), _cfg())
    l = best_option_index(f)
    assert np.count_nonzero(integral_response_geometry(f, f, l)) == 0
    assert np.count_nonzero(constraint_work_geometry(f, f, l)) == 0


def test_reserve_plus_debt_work_exactly_conserves_signed_response():
    n = executable_constraint_field_from_sample(_sample(), _cfg())
    c = executable_constraint_field_from_sample(_sample(candidate_index=1, lateral=0.4), _cfg())
    l = best_option_index(c)
    assert work_conservation_error(c, n, l) <= 1.0e-12


def test_full_horizon_is_used_not_only_eight_sparse_knots():
    f = executable_constraint_field_from_sample(_sample(), _cfg())
    assert f.full_values.shape[1] == 20
    assert f.values.shape[1] == 8
    assert f.full_masks.shape == f.full_values.shape


def test_full_horizon_bins_cover_each_state_exactly_once():
    for length in (8, 9, 20, 31):
        edges = _bin_edges(length)
        seen = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            seen.extend(range(int(lo), int(hi)))
        assert seen == list(range(length))
