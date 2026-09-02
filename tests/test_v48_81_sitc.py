import json
import numpy as np
import torch
from ocrap.v48_81_switch_inverse_truth_contract import nested_tail_switch_inverse_interval
from ocrap.cli.train import _absolute_feasibility_interval_huber

def sample(val, mode='post_contact_stabilize', meta=None):
    return {'m_star':np.array([[val]],np.float32),'root_probs':np.array([1.],np.float32),'root_valid':np.array([1],np.bool_),'option_valid':np.array([1],np.bool_),'c_star':np.eye(1,dtype=np.float32),'root_assignments':np.array([0]),'future_metadata':[meta or {}],'recovery_modes':np.array([mode]),'r_dep_star':np.array(val,np.float32)}

def test_inactive_floor_is_point_identified():
    r=nested_tail_switch_inverse_interval(sample(.7))
    assert r.exact_physical
    assert abs(r.physical_lower-.7)<1e-5 and abs(r.physical_upper-.7)<1e-5

def test_active_floor_is_upper_bound_only():
    r=nested_tail_switch_inverse_interval(sample(.6))
    assert not r.exact_physical
    assert r.upper_finite and not r.lower_finite
    assert abs(r.physical_upper-.6)<1e-5

def test_secondary_floor_inverse():
    r=nested_tail_switch_inverse_interval(sample(.9,mode='avoid_secondary',meta={'secondary_threat':True}))
    assert not r.exact_physical and r.upper_finite
    r2=nested_tail_switch_inverse_interval(sample(1.2,mode='avoid_secondary',meta={'secondary_threat':True}))
    assert r2.exact_physical and abs(r2.physical_lower-1.2)<1e-5

def test_floor_then_route_cap_is_unidentifiable():
    r=nested_tail_switch_inverse_interval(sample(-.8,mode='yield_rejoin',meta={'route_blocked':True}))
    assert not r.lower_finite and not r.upper_finite

def test_hidden_override_unidentifiable():
    r=nested_tail_switch_inverse_interval(sample(.2,mode='stop',meta={'artifact_branch':'yield'}))
    assert not r.lower_finite and not r.upper_finite

def test_interval_loss_accepts_switch_inverse_policy():
    b={'is_nominal':torch.tensor([0.]),'bucket_id':torch.tensor([1]),'time_index':torch.tensor([0]),'r_dep_star':torch.tensor([.5]),'absolute_truth_interval_informative':torch.tensor([1.]),'absolute_truth_physical_lower':torch.tensor([.2]),'absolute_truth_physical_upper':torch.tensor([.6])}
    z=_absolute_feasibility_interval_huber({'direct_recovery_absolute_feasibility_logit':torch.tensor([.4])},b,{'direct_value_absolute_feasibility_truth_contract':'switch_inverse_interval_bounds'})
    assert float(z)==0.0


def test_mixed_root_floor_exposure_is_unidentifiable_not_crash():
    # m_star is an intra-root aggregate. One future may be structurally floored
    # while another is not, so an aggregate value below 0.6 is legal and must
    # not be inverted as if the floor applied uniformly to the whole root.
    s={
        'm_star':np.array([[.5]],np.float32),
        'root_probs':np.array([1.],np.float32),
        'root_valid':np.array([1],np.bool_),
        'option_valid':np.array([1],np.bool_),
        'c_star':np.eye(1,dtype=np.float32),
        'root_assignments':np.array([0,0]),
        'future_metadata':[{}, {'artifact_branch':'yield'}],
        'recovery_modes':np.array(['post_contact_stabilize']),
        'r_dep_star':np.array(.5,np.float32),
    }
    r=nested_tail_switch_inverse_interval(s)
    assert r.valid
    assert not r.exact_physical
    assert not r.lower_finite and not r.upper_finite
    assert r.mixed_structural_cell_fraction > 0
