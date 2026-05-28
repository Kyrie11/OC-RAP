import numpy as np
from ocrap.proposals.action_lattice import generate_lattice_actions
from ocrap.proposals.action_projection import validate_prefix
from ocrap.utils.datatypes import EgoState, RouteInfo, MapFeatures, ActionPrefix


def test_dynamic_bounds_satisfied():
    acts=generate_lattice_actions(EgoState(v=8), RouteInfo.straight(), K=8)
    for a in acts:
        assert np.max(np.abs(a.controls[:,1])) <= 0.25 + 1e-5
        assert np.max(a.states[:,3]) <= 13.9 + 3.0 + 1e-5


def test_static_collision_rejected():
    states=np.zeros((11,6),np.float32); states[:,0]=np.linspace(0,10,11); controls=np.zeros((10,3),np.float32)
    pref=ActionPrefix(0,True,'test',states,controls,np.zeros(6,np.float32))
    obs=np.array([[4,-1],[6,-1],[6,1],[4,1]],np.float32)
    out=validate_prefix(pref, MapFeatures(static_obstacles=[obs]))
    assert not out.valid and out.mask_reason=='static_collision'


def test_dynamic_conflict_not_rejected_before_recovery():
    acts=generate_lattice_actions(EgoState(v=8), RouteInfo.straight(), K=8)
    assert any(a.valid for a in acts)
