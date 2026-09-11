from __future__ import annotations
import numpy as np
from ocrap.audits.executable_constraint_jacobian import ExecutableConstraintField
from ocrap.audits.viability_order_profile import (
    ORDER_MASSES, PROFILE_GEOMETRY_DIM, MATCHED_DIM, _upper_tail_mean,
    _profile_from_joint, full_profile_geometry, exposed_profile_geometry,
    option_permutation_invariance_error, contract_checks,
)

def _field(x: np.ndarray) -> ExecutableConstraintField:
    x=np.asarray(x,dtype=np.float64); L,T,C=x.shape; m=np.ones_like(x,dtype=bool)
    return ExecutableConstraintField(values=x.copy(),masks=m.copy(),full_values=x.copy(),full_masks=m.copy(),option_valid=np.ones(L,dtype=bool),option_scores=np.zeros(L),option_modes=tuple(f'm{i}' for i in range(L)),diagnostics={})

def test_vop_fixed_fractional_upper_order_means():
    vals=np.asarray([4.,3.,2.,1.])
    got=np.asarray([_upper_tail_mean(vals,float(g)) for g in ORDER_MASSES])
    assert np.allclose(got,[4.,3.5,3.,2.5],atol=1e-12,rtol=0)

def test_vop_fractional_mass_handles_three_options_exactly():
    vals=np.asarray([3.,2.,1.])
    # gamma=.5 has target mass 1.5 => (3 + .5*2)/1.5
    assert np.isclose(_upper_tail_mean(vals,.5), 8/3)

def test_vop_profile_is_monotone_and_noncollapsed():
    joint=np.asarray([[4.,4.],[3.,3.],[2.,2.],[1.,1.]])
    p,d=_profile_from_joint(joint)
    assert p.shape==(2,4)
    assert np.all(np.diff(p,axis=1)<=1e-12)
    assert d['noncollapsed_fraction']==1.0
    assert d['max_monotonicity_error']<=1e-12

def test_vop_rejects_cross_option_frankenstein_through_same_option_joint_min():
    x=np.ones((4,8,4),dtype=np.float64)
    # each option violates a different constraint, so every option joint margin is negative
    for l in range(4): x[l,:,l]=-0.2-0.1*l
    n=_field(x)
    # candidate worsens all violation depths slightly
    c=_field(x - 0.05)
    g,d=full_profile_geometry(c,n)
    assert g.shape==(PROFILE_GEOMETRY_DIM,)
    assert d.mean_eligible_option_count==4.0

def test_vop_exposed_and_full_have_same_matched_geometry_budget():
    x=np.ones((4,8,4),dtype=np.float64)
    n=_field(x); y=x.copy(); y[0,5:,0]-=.5; y[1,4:,1]-=.3; y[2,:3,2]-=.4; y[3,:,3]+=.2; c=_field(y)
    e,ed=exposed_profile_geometry(c,n,np.asarray([1,1,1,0],dtype=bool))
    f,fd=full_profile_geometry(c,n)
    assert e.shape==f.shape==(64,)
    assert MATCHED_DIM==220
    assert max(ed.max_monotonicity_error,fd.max_monotonicity_error)<=1e-12

def test_vop_joint_option_permutation_invariant():
    x=np.ones((4,8,4),dtype=np.float64); n=_field(x); y=x.copy(); y[0,5:,0]=-.2; y[1,4:,1]=-.3; c=_field(y)
    err=option_permutation_invariance_error(c,n,np.asarray([1,1,1,0],dtype=bool))
    assert err<=1e-12

def test_vop_contract_checks_all_true():
    checks=contract_checks(); assert checks and all(checks.values()),checks
