from __future__ import annotations
import numpy as np
from ocrap.audits.zero_boundary_viability_transition import (
    ENGINEERING_VERSION, SCIENTIFIC_VERSION, MATCHED_DIM, TRANSITION_GEOMETRY_DIM,
    TRANSITION_MODE_NAMES, _transition_modes_from_joint, contract_checks,
)

def test_versions_and_contract():
    assert ENGINEERING_VERSION == "v48.123.0-OC-ZBST"
    assert SCIENTIFIC_VERSION == "v48.123-OC-ZBST"
    assert MATCHED_DIM == 220 and TRANSITION_GEOMETRY_DIM == 64
    assert TRANSITION_MODE_NAMES == (
        "reserve_transition", "nominal_rank_reserve_transition",
        "debt_repayment_transition", "nominal_rank_debt_repayment_transition",
    )
    assert all(contract_checks().values())

def test_zero_boundary_decomposition_is_exact_and_recovers_transport():
    q0=np.asarray([[-1.0,-.5],[.2,.4],[1.0,1.5]],dtype=float)
    qa=np.asarray([[.3,-.2],[.5,-.1],[.8,2.0]],dtype=float)
    m,d=_transition_modes_from_joint(qa,q0)
    assert d["max_displacement_decomposition_error"] <= 1e-12
    assert np.allclose(m[:,0]+m[:,2], np.mean(qa-q0,axis=0), atol=1e-12, rtol=0)

def test_safe_and_debt_sides_are_semantically_separated():
    n=np.repeat(np.asarray([.2,.5,1.0])[:,None],4,axis=1); c=n+.1
    m,_=_transition_modes_from_joint(c,n)
    assert np.any(np.abs(m[:,:2])>0) and np.allclose(m[:,2:],0,atol=1e-12,rtol=0)
    n=-np.repeat(np.asarray([.2,.5,1.0])[:,None],4,axis=1); c=n+.1
    m,_=_transition_modes_from_joint(c,n)
    assert np.any(np.abs(m[:,2:])>0) and np.allclose(m[:,:2],0,atol=1e-12,rtol=0)

def test_crossing_diagnostics_are_active():
    n=np.repeat(np.asarray([-1.0,.5])[:,None],4,axis=1)
    c=np.repeat(np.asarray([.5,-.5])[:,None],4,axis=1)
    _,d=_transition_modes_from_joint(c,n)
    assert d["mean_zero_crossing_fraction"] > 0
    assert d["mean_debt_to_reserve_fraction"] > 0
    assert d["mean_reserve_to_debt_fraction"] > 0
