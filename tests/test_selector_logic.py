import numpy as np
import pytest
from ocrap.models.selector import select_action, SelectorParams


def prof():
    return {"R":np.array([0.9,0.8,0.2]),"B":np.array([0.1,0.2,0.1]),"U":np.array([0.9,0.1,0.1]),"dH":np.array([0.0,0.0,0.0]),"K_post":np.array([0.1,0.1,0.1]),"witness":np.zeros((3,2),dtype=int)}

def masks(): return {"action_mask":np.array([True,True,True])}

def test_nominal_accepted_when_feasible():
    r=select_action([0,1,2],prof(),np.array([3,2,1.]),0,0,masks())
    assert r['action_index']==0 and r['status']=='nominal_accepted'

def test_constrained_selection_when_nominal_infeasible():
    p=prof(); p['R'][0]=0.1
    r=select_action([0,1,2],p,np.array([3,2,1.]),0,0,masks())
    assert r['action_index']==1 and r['status']=='constrained'

def test_controlled_relaxation_when_feasible_set_empty():
    p=prof(); p['R'][:]=0.1
    r=select_action([0,1,2],p,np.array([3,2,1.]),0,0,masks())
    assert r['status']=='controlled_relaxation'

def test_emergency_stop_only_when_no_valid_action():
    r=select_action([0,1,2],prof(),np.array([3,2,1.]),0,0,{"action_mask":np.array([False,False,False])})
    assert r['status']=='no_valid_action' and r['used_emergency_fallback']

def test_selector_does_not_double_penalize_U_in_main_method():
    with pytest.raises(ValueError):
        select_action([0,1,2],prof(),np.array([3,2,1.]),0,0,masks(),SelectorParams(lambda_U_selector=1.0,method='ours'))
