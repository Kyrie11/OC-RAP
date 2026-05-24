import numpy as np

from recap.proposals.action_lattice import generate_lattice_actions
from recap.utils.datatypes import EgoState, RouteInfo


def test_paper_lattice_has_many_distinct_prefixes():
    acts = generate_lattice_actions(EgoState(v=8), RouteInfo.straight(), K_raw=64, K=32)
    sigs = {
        tuple(np.round([a.states[-1, 0], a.states[-1, 1], a.states[-1, 3], a.params[0]], 3))
        for a in acts
        if a.valid
    }
    assert len(sigs) >= 24
