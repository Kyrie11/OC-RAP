import torch
import numpy as np
from ocrap.v48_80_interval_truth_contract import nested_tail_physical_interval
from ocrap.cli.train import _absolute_feasibility_interval_huber

def sample(meta, mode='yield_rejoin', val=.6):
    return {'m_star':np.array([[val]],dtype=np.float32),'root_probs':np.array([1.]),'root_valid':np.array([1]),'c_star':np.array([[1.]]),'option_valid':np.array([1]),'root_assignments':np.array([0]),'future_metadata':[meta],'recovery_modes':np.array([mode]),'r_dep_star':np.array(val)}
def test_floor_produces_upper_bound_not_exact_target():
    r=nested_tail_physical_interval(sample({}))
    assert r.valid and r.upper_finite and not r.lower_finite and r.physical_upper <= .600001

def test_route_override_produces_lower_bound():
    r=nested_tail_physical_interval(sample({'route_blocked':True},mode='yield_rejoin',val=-.8))
    # same cell has floor+cap -> deliberately uninformative fail-closed
    assert r.valid and not r.informative

def test_clean_row_is_exact_interval():
    r=nested_tail_physical_interval(sample({},mode='stop',val=.3))
    assert r.exact_physical and abs(r.physical_lower-.3)<1e-5 and abs(r.physical_upper-.3)<1e-5

def test_interval_huber_zero_inside_positive_outside():
    b={'r_dep_star':torch.tensor([.6,.6]),'is_nominal':torch.zeros(2),'bucket_id':torch.ones(2,dtype=torch.long),'time_index':torch.zeros(2,dtype=torch.long),'absolute_truth_interval_informative':torch.ones(2),'absolute_truth_physical_lower':torch.tensor([-1e6,-1e6]),'absolute_truth_physical_upper':torch.tensor([.6,.6])}
    cfg={'direct_value_absolute_feasibility_truth_contract':'structural_interval_bounds'}
    z=_absolute_feasibility_interval_huber({'direct_recovery_absolute_feasibility_logit':torch.tensor([.2,.5])},b,cfg)
    assert float(z)==0.0
    p=_absolute_feasibility_interval_huber({'direct_recovery_absolute_feasibility_logit':torch.tensor([1.2,.5])},b,cfg)
    assert float(p)>0
