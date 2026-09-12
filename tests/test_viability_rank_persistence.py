from __future__ import annotations

import numpy as np

from ocrap.audits.executable_constraint_jacobian import ExecutableConstraintField
from ocrap.audits.heterogeneous_constraint_normal_cone import NUM_CONSTRAINTS
from ocrap.audits.viability_rank_persistence import (
    MATCHED_DIM,
    COUPLING_GEOMETRY_DIM,
    COUPLING_MODE_NAMES,
    _nominal_midranks,
    _directional_rank_persistence_defect,
    _persistence_modes_from_joint,
    contract_checks,
    option_permutation_invariance_error,
    full_persistence_geometry,
)


def _field(paths: np.ndarray) -> ExecutableConstraintField:
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths[:, None]
    L, T = paths.shape
    vals = np.repeat(paths[:, :, None], NUM_CONSTRAINTS, axis=2)
    masks = np.ones_like(vals, dtype=bool)
    return ExecutableConstraintField(
        values=vals.copy(), masks=masks.copy(), full_values=vals, full_masks=masks,
        option_valid=np.ones(L, dtype=bool), option_scores=np.zeros(L, dtype=np.float64),
        option_modes=tuple(f"m{i}" for i in range(L)), diagnostics={},
    )


def test_rank_persistence_contract_checks_all_true():
    checks = contract_checks()
    assert checks and all(checks.values()), checks
    assert COUPLING_GEOMETRY_DIM == 64
    assert MATCHED_DIM == 220
    assert COUPLING_MODE_NAMES == (
        "global_shift", "nominal_rank_tilt", "rank_persistence_defect", "rank_persistence_interaction"
    )


def test_nominal_midranks_share_exact_tie_block():
    no = np.asarray([[0.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    r = _nominal_midranks(no)[:, 0]
    assert np.isclose(r[0], r[1])
    assert r[0] < r[2] < r[3]


def test_persistent_rank_path_has_zero_defect():
    no = np.repeat(np.asarray([[-2.0], [-1.0], [0.5], [2.0]], dtype=np.float64), 6, axis=1)
    r = _nominal_midranks(no)
    pre = _directional_rank_persistence_defect(r, reverse=False)
    suf = _directional_rank_persistence_defect(r, reverse=True)
    assert np.allclose(pre, 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(suf, 0.0, atol=1e-12, rtol=0.0)


def test_same_instantaneous_rank_transport_but_different_identity_history_is_detected():
    L, T = 4, 6
    levels = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    delta = np.asarray([0.4, 0.1, -0.1, -0.4], dtype=np.float64)
    no_a = np.repeat(levels[:, None], T, axis=1)
    ca_a = no_a + delta[:, None]
    no_b = np.empty_like(no_a)
    ca_b = np.empty_like(ca_a)
    for t in range(T):
        perm = np.arange(L) if t % 2 == 0 else np.asarray([3, 1, 2, 0])
        no_b[:, t] = levels[perm]
        ca_b[:, t] = no_b[:, t]
        order = np.argsort(no_b[:, t], kind="mergesort")
        ca_b[order, t] += delta
    a, _ = _persistence_modes_from_joint(ca_a, no_a, reverse=False)
    b, bd = _persistence_modes_from_joint(ca_b, no_b, reverse=False)
    assert np.allclose(a[:, :2], b[:, :2], atol=1e-12, rtol=0.0)
    assert np.max(np.abs(a[:, 2:] - b[:, 2:])) > 1e-6
    assert bd["mean_rank_persistence_defect"] > 0.0


def test_global_shift_on_persistent_ranks_uses_only_global_mode():
    no = np.repeat(np.asarray([[-2.0], [-1.0], [0.5], [3.0]], dtype=np.float64), 5, axis=1)
    ca = no + 0.4
    modes, d = _persistence_modes_from_joint(ca, no, reverse=False)
    assert np.allclose(modes[:, 0], 0.4, atol=1e-12, rtol=0.0)
    assert np.allclose(modes[:, 1:], 0.0, atol=1e-12, rtol=0.0)
    assert d["mean_rank_persistence_defect"] == 0.0


def test_field_geometry_is_permutation_invariant_and_64d():
    nominal = _field(np.asarray([
        [-1.0, -1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
        [-0.2, 0.4, -0.2, 0.4, -0.2, 0.4, -0.2, 0.4],
        [0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2],
        [1.2, 1.2, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6],
    ]))
    candidate = _field(np.asarray([
        [-0.7, -0.7, 0.7, 0.7, 1.2, 1.2, 1.2, 1.2],
        [-0.4, 0.7, -0.5, 0.7, -0.4, 0.7, -0.4, 0.7],
        [0.9, 0.0, 0.8, 0.0, 0.8, 0.0, 0.8, 0.0],
        [0.8, 0.8, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4],
    ]))
    g, diag = full_persistence_geometry(candidate, nominal)
    assert g.shape == (64,)
    assert np.isfinite(g).all()
    assert diag.mean_prefix_rank_persistence_defect > 0.0 or diag.mean_suffix_rank_persistence_defect > 0.0
    exposed = np.asarray([True, True, True, True])
    assert option_permutation_invariance_error(candidate, nominal, exposed) <= 1e-12
