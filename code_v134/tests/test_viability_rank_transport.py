from __future__ import annotations

import numpy as np

from ocrap.audits.executable_constraint_jacobian import ExecutableConstraintField
from ocrap.audits.heterogeneous_constraint_normal_cone import NUM_CONSTRAINTS
from ocrap.audits.viability_rank_transport import (
    MATCHED_DIM,
    TRANSPORT_GEOMETRY_DIM,
    TRANSPORT_MODE_DEGREES,
    _basis_interval_integrals,
    _transport_modes_from_joint,
    contract_checks,
    option_permutation_invariance_error,
    full_transport_geometry,
)


def _field(levels: np.ndarray, *, time_steps: int = 8) -> ExecutableConstraintField:
    levels = np.asarray(levels, dtype=np.float64)
    L = len(levels)
    vals = np.repeat(levels[:, None, None], time_steps, axis=1)
    vals = np.repeat(vals, NUM_CONSTRAINTS, axis=2)
    masks = np.ones_like(vals, dtype=bool)
    return ExecutableConstraintField(
        values=vals.copy(), masks=masks.copy(), full_values=vals, full_masks=masks,
        option_valid=np.ones(L, dtype=bool), option_scores=np.zeros(L, dtype=np.float64),
        option_modes=tuple(f"m{i}" for i in range(L)), diagnostics={},
    )


def test_rank_transport_contract_checks_all_true():
    checks = contract_checks()
    assert checks and all(checks.values()), checks
    assert TRANSPORT_GEOMETRY_DIM == 64
    assert MATCHED_DIM == 220
    assert TRANSPORT_MODE_DEGREES == (0, 1, 2, 3)


def test_shifted_legendre_whole_interval_has_only_dc_mass():
    assert np.allclose(_basis_interval_integrals(0.0, 1.0), [1.0, 0.0, 0.0, 0.0], atol=1e-12, rtol=0.0)


def test_same_static_order_distribution_can_have_nonzero_same_option_transport():
    # Candidate is a pure identity swap: its sorted viability multiset equals nominal,
    # so an independently resorted static order profile cannot see the change.
    nominal = np.asarray([[-1.5], [-0.5], [0.5], [1.5]], dtype=np.float64)
    candidate = np.asarray([[1.5], [-0.5], [0.5], [-1.5]], dtype=np.float64)
    modes, diag = _transport_modes_from_joint(candidate, nominal)
    assert np.sort(candidate[:, 0]).tolist() == np.sort(nominal[:, 0]).tolist()
    assert np.linalg.norm(modes[0]) > 1e-6
    assert diag["mean_rank_inversion_fraction"] > 0.0


def test_nominal_ties_are_invariant_to_within_tie_identity_permutation():
    nominal = np.asarray([[0.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    candidate = np.asarray([[1.0], [-1.0], [1.25], [1.75]], dtype=np.float64)
    a, _ = _transport_modes_from_joint(candidate, nominal)
    perm = np.asarray([1, 0, 2, 3])
    b, _ = _transport_modes_from_joint(candidate[perm], nominal[perm])
    assert np.allclose(a, b, atol=1e-12, rtol=0.0)


def test_global_same_option_shift_lives_only_in_degree_zero_mode():
    nominal = np.asarray([[-2.0], [-1.0], [0.5], [3.0]], dtype=np.float64)
    candidate = nominal + 0.4
    modes, _ = _transport_modes_from_joint(candidate, nominal)
    assert np.isclose(modes[0, 0], 0.4, atol=1e-12)
    assert np.allclose(modes[0, 1:], 0.0, atol=1e-12, rtol=0.0)


def test_field_geometry_is_permutation_invariant_and_64d():
    nominal = _field(np.asarray([-1.0, -0.2, 0.4, 1.2]))
    candidate = _field(np.asarray([-0.7, -0.4, 0.9, 0.8]))
    g, diag = full_transport_geometry(candidate, nominal)
    assert g.shape == (64,)
    assert np.isfinite(g).all()
    assert diag.mean_prefix_rank_inversion_fraction > 0.0
    exposed = np.asarray([True, True, True, True])
    assert option_permutation_invariance_error(candidate, nominal, exposed) <= 1e-12
