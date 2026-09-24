from __future__ import annotations

import numpy as np

from ocrap.audits.tail_boundary_crossing_flow import (
    BOUNDARY_GEOMETRY_DIM,
    MATCHED_DIM,
    _root_boundary_witness,
    boundary_measure_diagnostics,
    contract_checks,
    nominal_tail_boundary_measure,
)


def _measure():
    M = np.asarray([
        [-0.5, 0.10, 0.40, -0.05],
        [-0.2, 0.20, 0.30, -0.10],
        [0.2, -0.40, 0.05, 0.30],
        [0.5, -0.20, 0.10, -0.02],
    ], dtype=np.float64)
    logits = np.asarray([0.4, 0.1, -0.2, -0.5])
    C = np.asarray([
        [1.0, .8, .2, .1], [.8, 1.0, .4, .2],
        [.2, .4, 1.0, .7], [.1, .2, .7, 1.0],
    ])
    return nominal_tail_boundary_measure(
        M, logits, C, root_valid=np.ones(4, dtype=bool),
        option_valid=np.ones(4, dtype=bool), alpha=.5, beta=.5, top_m=3,
    )


def test_tbfc_contract_is_capacity_matched_and_nontrivial():
    checks = contract_checks()
    assert checks and all(checks.values()), checks
    assert BOUNDARY_GEOMETRY_DIM == 64
    assert MATCHED_DIM == 220


def test_boundary_witness_brackets_zero_without_threshold():
    w = _root_boundary_witness(np.asarray([-1.0, -0.1, 0.2, 2.0]), np.ones(4, dtype=bool))
    assert np.allclose(w, [0.0, 0.5, 0.5, 0.0])
    assert abs(float(w.sum()) - 1.0) <= 1e-12


def test_boundary_witness_ties_are_permutation_symmetric():
    m = np.asarray([-0.1, -0.1, 0.2, 0.2])
    w = _root_boundary_witness(m, np.ones(4, dtype=bool))
    assert np.allclose(w, [0.25, 0.25, 0.25, 0.25])


def test_boundary_measure_is_probability_and_more_than_one_option_on_synthetic():
    m = _measure()
    assert np.all(m.option_weights >= -1e-12)
    assert abs(float(m.option_weights.sum()) - 1.0) <= 1e-12
    assert abs(float(m.root_exposure.sum()) - 1.0) <= 1e-12
    d = boundary_measure_diagnostics(m)
    assert d["boundary_positive_option_count"] >= 2
    assert d["boundary_two_sided_root_mass"] > 0.0
