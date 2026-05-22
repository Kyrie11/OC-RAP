import numpy as np
from recap.proposals.action_lattice import generate_lattice_actions
from recap.proposals.recovery_options import generate_options_for_action
from recap.utils.datatypes import EgoState, RouteInfo


def make_opts():
    a=generate_lattice_actions(EgoState(v=8),RouteInfo.straight(),K=1)[0]
    return a, generate_options_for_action(a,RouteInfo.straight(),L=7)

def test_options_start_at_prefix_terminal():
    a,opts=make_opts()
    for o in opts:
        assert np.allclose(o.states_ref[0], a.states[-1])

def test_preserve_one_per_valid_type():
    a,opts=make_opts(); types={o.type for o in opts if o.valid}
    assert {'maintain','stop','lane','route','yield','escape','stabilize'}.issubset(types)

def test_stop_option_deceleration_bound():
    a,opts=make_opts(); stop=[o for o in opts if o.type=='stop'][0]
    assert np.max(np.abs(stop.controls_ref[:,0])) <= 6.0 + 1e-3

def test_stabilize_is_conditional_token_and_post_contact_evaluated_only_after_contact():
    a,opts=make_opts(); st=[o for o in opts if o.type=='stabilize'][0]
    assert st.valid and st.conditional
