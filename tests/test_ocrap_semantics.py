import numpy as np
import torch
from recap.teacher.recovery_specs import option_margin_from_specs
from recap.teacher.observation_classes import build_obs_equivalence, beta_from_obs_equiv, class_consistent_witness
from recap.evaluation.metrics import weighted_lcvar_np
from recap.models.oc_mero import existential_mu_aggregate
from recap.models.recot import ReCoT


def test_option_margin_is_max_over_specs_not_min():
    margin, spec_id = option_margin_from_specs(np.array([-1.0, 0.5, -0.2]))
    assert margin == 0.5 and spec_id == 1


def test_observation_class_forces_same_witness():
    Y = np.array([[1, 0], [0, 1]], dtype=np.float32)
    margins = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    obs_class, _ = build_obs_equivalence([{"ego": np.zeros(2)}, {"ego": np.zeros(2)}])
    witness_oc, Y_oc, _ = class_consistent_witness(Y, margins, np.array([0.5, 0.5]), obs_class)
    assert witness_oc[0] == witness_oc[1]
    assert Y_oc.mean() < 1.0


def test_R_star_uses_Y_oc_not_oracle_max_over_options():
    Y = np.array([[1, 0], [0, 1]], dtype=np.float32)  # [L,M]
    margins = np.array([[0.4, -0.2], [-0.2, 0.4]], dtype=np.float32)
    obs_class = np.array([0, 0])
    w, Y_oc, _ = class_consistent_witness(Y, margins, np.array([0.5, 0.5]), obs_class)
    oracle = weighted_lcvar_np(Y.max(axis=0), np.array([0.5, 0.5]), 0.5)
    oc = weighted_lcvar_np(Y_oc, np.array([0.5, 0.5]), 0.5)
    assert oracle == 1.0 and oc < 1.0


def test_mu_weighted_equal_scores_duplicate_invariant():
    for L in (1, 3, 7):
        v = torch.ones(1, 1, L, 2) * 0.7
        logits = torch.zeros_like(v)
        mask = torch.ones(1, 1, L, dtype=torch.bool)
        pi, mu = existential_mu_aggregate(v, logits, mask, tau_R=0.2, c_R=0.0)
        assert torch.allclose(pi, torch.sigmoid(torch.ones_like(pi) * 0.7), atol=1e-5)


def test_forward_rejects_oc_oracle_keys():
    model = ReCoT(C_bev=24, H_h=5, M=2, hidden=32)
    b = dict(bev=torch.zeros(1,5,24,32,32), ego_info=torch.zeros(1,11), route_command=torch.zeros(1,20,6), actions_states=torch.zeros(1,2,11,6), token_states_ref=torch.zeros(1,2,3,26,6), action_mask=torch.ones(1,2,dtype=torch.bool), option_mask=torch.ones(1,2,3,dtype=torch.bool))
    try:
        model(**b, witness_oc=torch.zeros(1,2,2,dtype=torch.long))
        assert False
    except ValueError:
        pass
