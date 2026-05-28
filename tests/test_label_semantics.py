import numpy as np
from ocrap.utils.datatypes import RolloutTrace
from ocrap.teacher.margins import compute_margins
from ocrap.teacher.evidence_labels import evidence_from_trace


def trace(contact=False):
    T=36; states=np.zeros((T,6),np.float32); states[:,3]=np.linspace(5,0,T); controls=np.zeros((T-1,2),np.float32)
    margins=np.ones(T,np.float32)*0.5
    fc=8 if contact else -1
    coll=margins.copy()
    if contact: coll[fc]=-1.0
    costs={"stop":np.linspace(1,0,T),"lane":np.ones(T),"route":np.linspace(0.5,0.1,T),"escape":np.ones(T),"stabilize":np.linspace(1,0,T)}
    return RolloutTrace(states,controls,stage_boundary_idx=10,first_contact_idx=fc,first_contact_stage=1 if contact else 0,contact_type='front',relative_speed_at_first_contact=2.0,collision_margin=coll,drivable_margin=margins,direction_margin=margins,route_margin=margins,speed_margin=margins,stability_margin=margins,ttc_margin=margins,affordance_costs=costs)

def test_stage_boundary_fixed_by_horizon_not_event():
    tr=trace(contact=True); assert tr.stage_boundary_idx==10 and tr.first_contact_idx!=tr.stage_boundary_idx

def test_normal_root_has_recovery_stage_without_contact():
    m=compute_margins(trace(contact=False)); assert m['M_return'] > -1 and m['M_post']==1.0

def test_first_contact_does_not_auto_kill_post_contact_recovery():
    m=compute_margins(trace(contact=True)); assert m['M_post'] >= 0 and m['M_secondary'] >= 0 and m['M_option'] >= 0

def test_H_is_prefix_level_and_option_independent():
    e=evidence_from_trace(trace(contact=True),{"H_p":10}); assert e['H_star'] > 0 and e['H_source']==1

def test_return_affordance_not_satisfied_by_initial_state():
    tr=trace(contact=False); tr.affordance_costs['stop'][:10]=0.0; tr.affordance_costs['stop'][10:]=1.0
    m=compute_margins(tr); assert m['M_return'] < 0.35
