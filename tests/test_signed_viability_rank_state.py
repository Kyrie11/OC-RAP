from __future__ import annotations

import numpy as np

from ocrap.audits.executable_constraint_jacobian import ExecutableConstraintField
from ocrap.audits.heterogeneous_constraint_normal_cone import NUM_CONSTRAINTS
from ocrap.audits.signed_viability_rank_state import (
    ENGINEERING_VERSION,
    SCIENTIFIC_VERSION,
    MATCHED_DIM,
    STATE_GEOMETRY_DIM,
    STATE_MODE_NAMES,
    _nominal_midranks,
    _signed_state_modes_from_joint,
    contract_checks,
    full_signed_state_geometry,
)


def test_signed_rank_state_versions_and_contract():
    assert ENGINEERING_VERSION == "v48.122.0-OC-SVRT"
    assert SCIENTIFIC_VERSION == "v48.122-OC-SVRT"
    assert MATCHED_DIM == 220
    assert STATE_GEOMETRY_DIM == 64
    assert STATE_MODE_NAMES == (
        "global_shift",
        "nominal_rank_tilt",
        "signed_nominal_state_coupling",
        "rank_signed_nominal_state_interaction",
    )
    assert all(contract_checks().values())


def test_same_rank_transport_different_absolute_state_changes_only_state_modes():
    T = 8
    base = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    delta = np.asarray([0.4, -0.1, 0.2, 0.6], dtype=np.float64)
    n0 = np.repeat(base[:, None], T, axis=1)
    c0 = n0 + delta[:, None]
    n1 = n0 + 3.0
    c1 = n1 + delta[:, None]
    a, _ = _signed_state_modes_from_joint(c0, n0)
    b, _ = _signed_state_modes_from_joint(c1, n1)
    assert np.allclose(a[:, :2], b[:, :2], atol=1e-12, rtol=0)
    assert np.max(np.abs(a[:, 2:] - b[:, 2:])) > 1e-6


def test_zero_boundary_annuls_state_specific_modes():
    T = 5
    n = np.zeros((3, T), dtype=np.float64)
    c = np.asarray([[0.4], [-0.2], [0.1]], dtype=np.float64) + n
    modes, d = _signed_state_modes_from_joint(c, n)
    assert np.allclose(modes[:, 2:], 0.0, atol=1e-12, rtol=0)
    assert d["mean_abs_nominal_state"] == 0.0


def test_exact_nominal_ties_share_midranks_and_are_identity_invariant():
    n = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    r = _nominal_midranks(n)
    assert np.allclose(r[0], r[1])
    c = np.asarray([[1.0, 0.5], [-1.0, -0.5], [1.5, 1.25]], dtype=np.float64)
    a, _ = _signed_state_modes_from_joint(c, n)
    b, _ = _signed_state_modes_from_joint(c[[1, 0, 2]], n[[1, 0, 2]])
    assert np.allclose(a, b, atol=1e-12, rtol=0)


def test_full_geometry_shape_and_activity():
    L, T, C = 4, 8, NUM_CONSTRAINTS
    masks = np.ones((L, T, C), dtype=bool)
    base = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    n = np.repeat(base[:, None], T, axis=1)
    c = n + np.asarray([[0.4], [-0.1], [0.2], [0.6]])
    nf = np.repeat(n[:, :, None], C, axis=2)
    cf = np.repeat(c[:, :, None], C, axis=2)
    ov = np.ones(L, dtype=bool)
    modes = tuple(f"m{i}" for i in range(L))
    nominal = ExecutableConstraintField(
        values=nf.copy(), masks=masks.copy(), full_values=nf, full_masks=masks,
        option_valid=ov.copy(), option_scores=np.zeros(L), option_modes=modes, diagnostics={},
    )
    cand = ExecutableConstraintField(
        values=cf.copy(), masks=masks.copy(), full_values=cf, full_masks=masks,
        option_valid=ov.copy(), option_scores=np.zeros(L), option_modes=modes, diagnostics={},
    )
    g, d = full_signed_state_geometry(cand, nominal)
    assert g.shape == (64,)
    assert np.isfinite(g).all()
    assert d.mean_prefix_signed_state_energy > 0.0
    assert d.mean_suffix_signed_state_energy > 0.0
    assert d.mean_prefix_abs_nominal_state > 0.0
