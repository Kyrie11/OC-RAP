from __future__ import annotations

import numpy as np

from ocrap.audits.constraint_native_orientation import PREFIX_COMPLETE_STEPS, PREFIX_STATE_START, PREFIX_STATE_WIDTH, RAW_CANDIDATE_DIM
from ocrap.audits.heterogeneous_constraint_normal_cone import (
    ConstraintConeConfig,
    active_constraint_indices,
    cone_geometry,
    heterogeneous_constraint_paths,
    selector_diagnostics,
)


def _raw_candidate(*, x_end: float = 5.0, y_end: float = 0.0, speed: float = 5.0) -> np.ndarray:
    r = np.zeros((1, RAW_CANDIDATE_DIM), dtype=np.float64)
    r[:, 7] = 4.8
    r[:, 8] = 2.0
    st = np.zeros((PREFIX_COMPLETE_STEPS, PREFIX_STATE_WIDTH), dtype=np.float64)
    st[:, 0] = np.linspace(0.5, x_end, PREFIX_COMPLETE_STEPS)
    st[:, 1] = np.linspace(0.0, y_end, PREFIX_COMPLETE_STEPS)
    st[:, 6] = speed
    st[:, 7] = 4.8
    st[:, 8] = 2.0
    r[0, PREFIX_STATE_START:PREFIX_STATE_START + PREFIX_COMPLETE_STEPS * PREFIX_STATE_WIDTH] = st.reshape(-1)
    return r


def _agents(x_m: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((1, 2, 10), dtype=np.float64)
    # Token positions are normalized by 80 m; velocity by 20 m/s.
    a[0, 0, 0] = x_m / 80.0
    a[0, 0, 7] = 0.48
    a[0, 0, 8] = 0.4
    a[0, 1, 0] = 60.0 / 80.0
    a[0, 1, 7] = 0.48
    a[0, 1, 8] = 0.4
    m = np.ones((1, 2), dtype=bool)
    return a, m


def _cfg() -> ConstraintConeConfig:
    return ConstraintConeConfig(1.0, 0.5, 2.0, 5.0, 1.0, 2.5, -6.0, 60.0)


def test_candidate_equal_nominal_has_exact_zero_cone_geometry():
    r = _raw_candidate()
    a, m = _agents(20.0)
    h, mask, _ = heterogeneous_constraint_paths(r, a, m, sample_rate_hz=10.0, config=_cfg())
    sel = active_constraint_indices(h, mask)
    g = cone_geometry(h, h, sel)
    assert np.count_nonzero(g) == 0


def test_route_change_can_switch_candidate_active_constraint_without_regime_id():
    nominal = _raw_candidate(y_end=0.0)
    candidate = _raw_candidate(y_end=30.0)
    a, m = _agents(40.0)
    cfg = ConstraintConeConfig(1.0, 0.5, 2.0, 5.0, 1.0, 20.0, -6.0, 60.0)
    h0, m0, _ = heterogeneous_constraint_paths(nominal, a, m, sample_rate_hz=10.0, config=cfg)
    hc, mc, _ = heterogeneous_constraint_paths(candidate, a, m, sample_rate_hz=10.0, config=cfg)
    ns = active_constraint_indices(h0, m0)
    cs = active_constraint_indices(hc, mc)
    d = selector_diagnostics(ns, cs)
    assert d["candidate_any_switch_fraction"] > 0.0
    assert d["candidate_active_type_counts"]["route"] > 0


def test_reentry_constraint_activates_from_contact_geometry_not_bucket_label():
    r = _raw_candidate(x_end=6.0, speed=2.0)
    # Put an observed agent close enough for current circle overlap.
    a, m = _agents(1.0)
    h, mask, _ = heterogeneous_constraint_paths(r, a, m, sample_rate_hz=10.0, config=_cfg())
    # Re-entry is constraint index 3 and must become active without any regime input.
    assert mask[0, :, 3].any()
    assert np.isfinite(h[0, :, 3]).all()


def test_agent_permutation_does_not_change_heterogeneous_constraint_paths():
    r = _raw_candidate(y_end=1.0)
    a, m = _agents(15.0)
    h1, m1, _ = heterogeneous_constraint_paths(r, a, m, sample_rate_hz=10.0, config=_cfg())
    h2, m2, _ = heterogeneous_constraint_paths(r, a[:, ::-1], m[:, ::-1], sample_rate_hz=10.0, config=_cfg())
    assert np.allclose(h1, h2, rtol=0.0, atol=1e-12)
    assert np.array_equal(m1, m2)
