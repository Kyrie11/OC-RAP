from __future__ import annotations

import numpy as np

from ocrap.audits.weak_root_recovery_set_flow import (
    MATCHED_DIM,
    align_model_option_measure_to_physical_library,
    TAIL_GEOMETRY_DIM,
    contract_checks,
    nominal_ocmero_tail_measure,
    tail_measure_diagnostics,
)


def _measure():
    M = np.asarray([
        [-0.5, 0.2, 0.4, -0.1],
        [-0.2, 0.1, 0.3, 0.0],
        [0.2, -0.4, 0.1, 0.3],
        [0.5, 0.0, -0.2, 0.4],
    ], dtype=np.float64)
    logits = np.asarray([0.4, 0.1, -0.2, -0.5])
    C = np.asarray([
        [1.0, .8, .2, .1],
        [.8, 1.0, .4, .2],
        [.2, .4, 1.0, .7],
        [.1, .2, .7, 1.0],
    ])
    return nominal_ocmero_tail_measure(
        M, logits, C,
        root_valid=np.ones(4, dtype=bool),
        option_valid=np.ones(4, dtype=bool),
        alpha=.5, beta=.5, top_m=3,
    )


def test_weak_root_contract_is_exact_and_capacity_matched():
    checks = contract_checks()
    assert checks and all(checks.values()), checks
    assert TAIL_GEOMETRY_DIM == 64
    assert MATCHED_DIM == 220


def test_nominal_cotangent_is_probability_measure_after_pushforward():
    m = _measure()
    assert np.all(m.nested_cotangent >= -1e-12)
    assert abs(float(m.nested_cotangent.sum()) - 1.0) <= 1e-12
    assert np.all(m.option_weights >= -1e-12)
    assert abs(float(m.option_weights.sum()) - 1.0) <= 1e-12
    assert np.allclose(m.option_weights, m.nested_cotangent.sum(axis=0), atol=1e-12)


def test_tail_measure_is_deterministic_and_candidate_independent_by_interface():
    a = _measure()
    b = _measure()
    assert np.array_equal(a.best_option_per_anchor, b.best_option_per_anchor)
    assert np.array_equal(a.option_weights, b.option_weights)
    d = tail_measure_diagnostics(a)
    assert d["tail_outer_positive_root_count"] >= 1
    assert d["tail_positive_option_count"] >= 1


def test_model_physical_option_alignment_accepts_only_invalid_checkpoint_padding():
    physical, diag = align_model_option_measure_to_physical_library(
        np.asarray([True, False, True, False, False]),
        np.asarray([True, False, True]),
        np.asarray([0.4, 0.0, 0.6, 0.0, 0.0]),
    )
    assert np.array_equal(physical, np.asarray([0.4, 0.0, 0.6]))
    assert diag["model_option_count"] == 5
    assert diag["physical_option_count"] == 3
    assert diag["model_padding_count"] == 2
    assert diag["model_padding_all_invalid"] is True
    assert diag["model_physical_valid_prefix_match"] is True
    assert diag["padded_tail_mass"] == 0.0
    assert abs(diag["physical_tail_mass"] - 1.0) <= 1e-12


def test_model_physical_option_alignment_rejects_extra_valid_model_option():
    import pytest
    with pytest.raises(ValueError, match="extra valid recovery options"):
        align_model_option_measure_to_physical_library(
            np.asarray([True, True, True]),
            np.asarray([True, True]),
            np.asarray([0.5, 0.5, 0.0]),
        )


def test_model_physical_option_alignment_rejects_prefix_identity_mismatch():
    import pytest
    with pytest.raises(ValueError, match="option-valid prefix mismatch"):
        align_model_option_measure_to_physical_library(
            np.asarray([True, False, False]),
            np.asarray([True, True]),
            np.asarray([1.0, 0.0, 0.0]),
        )


def test_model_physical_option_alignment_rejects_padded_tail_mass():
    import pytest
    with pytest.raises(ValueError, match="mass to invalid padded options"):
        align_model_option_measure_to_physical_library(
            np.asarray([True, True, False]),
            np.asarray([True, True]),
            np.asarray([0.45, 0.45, 0.10]),
        )
