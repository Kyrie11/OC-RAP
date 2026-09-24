import numpy as np

from ocrap.planning.selector import constrained_lcb_select
from ocrap.evaluation.baselines import select_baseline


def _base_arrays():
    # nominal + 4 non-nominal candidates.  Nominal fails the absolute recovery
    # gate; candidates 1--4 pass it. Current lcb scoring prefers candidate 1.
    return dict(
        utility=np.array([0.80, 1.00, 0.95, 0.90, 0.85]),
        r_dep=np.array([-0.20, 0.30, 0.25, 0.20, 0.15]),
        hard=np.zeros(5),
        harm=np.zeros(5),
        feasible=np.ones(5, dtype=bool),
        pred_gap=np.zeros(5),
        nominal_deviation=np.zeros(5),
    )


def test_historical_mode_execution_identical_to_default():
    x = _base_arrays()
    a = constrained_lcb_select(**x, gamma_rec=0.0, require_absolute_admission_for_intervention=True)
    b = constrained_lcb_select(**x, gamma_rec=0.0, require_absolute_admission_for_intervention=True, relative_policy_mode="off")
    assert a.selected_index == b.selected_index == 1
    assert a.reason == b.reason == "best_admitted_lcb_score"
    assert np.array_equal(a.admitted, b.admitted)


def test_delta_mode_rejects_negative_relative_and_reranks_positive_survivors():
    x = _base_arrays()
    # Candidate 1 has highest utility but negative frozen recovery advantage.
    # Candidate 3 has the strongest positive frozen recovery advantage.
    direct = np.array([0.50, 0.20, 0.55, 0.80, 0.60])
    out = constrained_lcb_select(
        **x,
        gamma_rec=0.0,
        require_absolute_admission_for_intervention=True,
        pred_direct_value=direct,
        relative_policy_mode="delta_positive",
    )
    assert out.selected_index == 3
    assert out.reason == "best_rifa_delta_positive"
    # `admitted` remains the absolute P_F set, not the downstream relative set.
    assert out.admitted.tolist() == [False, True, True, True, True]


def test_delta_mode_abstains_when_all_absolute_candidates_are_relative_negative():
    x = _base_arrays()
    direct = np.array([0.50, 0.20, 0.30, 0.40, 0.45])
    out = constrained_lcb_select(
        **x,
        gamma_rec=0.0,
        require_absolute_admission_for_intervention=True,
        pred_direct_value=direct,
        relative_policy_mode="delta_positive",
    )
    assert out.selected_index == 0
    assert out.reason == "nominal_rifa_no_relative_survivor"


def test_joint_rifa_is_nested_topk_absolute_and_relative_sign_filter():
    x = _base_arrays()
    # Top-2 by rank are candidate 1 and 2. Candidate 1 fails positive delta;
    # candidate 2 passes all sign-consistent relative evidence. Candidate 3 has
    # the largest delta but is outside P_K and must not be rescued.
    direct = np.array([0.50, 0.40, 0.65, 0.95, 0.70])
    rank = np.array([0.0, 0.9, 0.8, 0.7, 0.6])
    opp = np.array([0.5, 0.8, 0.7, 0.9, 0.9])
    rharm = np.array([0.5, 0.2, 0.2, 0.1, 0.1])
    out = constrained_lcb_select(
        **x,
        gamma_rec=0.0,
        require_absolute_admission_for_intervention=True,
        pred_direct_value=direct,
        pred_direct_rank=rank,
        pred_direct_opportunity=opp,
        pred_direct_harm=rharm,
        relative_policy_mode="joint_sign",
        relative_proposal_top_k=2,
        relative_min_advantage=0.0,
        relative_opportunity_threshold=0.5,
        relative_harm_threshold=0.5,
    )
    assert out.selected_index == 2
    assert out.reason == "best_rifa_joint_sign"
    assert out.admitted[3]  # absolutely admitted but not in frozen top-k


def test_joint_rifa_never_rescues_absolute_rejected_candidate():
    x = _base_arrays()
    x["r_dep"] = np.array([-0.2, -0.1, 0.2, 0.1, 0.05])
    direct = np.array([0.0, 10.0, 0.5, 0.4, 0.3])
    rank = np.array([0.0, 10.0, 0.9, 0.8, 0.7])
    opp = np.array([0.5, 0.99, 0.8, 0.8, 0.8])
    rharm = np.array([0.5, 0.01, 0.1, 0.1, 0.1])
    out = constrained_lcb_select(
        **x,
        gamma_rec=0.0,
        require_absolute_admission_for_intervention=True,
        pred_direct_value=direct,
        pred_direct_rank=rank,
        pred_direct_opportunity=opp,
        pred_direct_harm=rharm,
        relative_policy_mode="joint_sign",
        relative_proposal_top_k=5,
    )
    assert out.selected_index != 1
    assert out.admitted[1] is np.False_ or not bool(out.admitted[1])


def test_baseline_selector_names_activate_only_diagnostic_modes():
    x = _base_arrays()
    cfg = {"selection": {"ocrap_selector": "lcb_constrained_relative_delta", "require_absolute_admission_for_intervention": True}}
    out = select_baseline(
        "ocrap", x["utility"], x["r_dep"], np.zeros(5), np.zeros(5), x["hard"], x["harm"], x["feasible"],
        0.0, 0.0, 0.0, cfg,
        pred_gap=x["pred_gap"], nominal_deviation=x["nominal_deviation"],
        pred_direct_value=np.array([0.5, 0.2, 0.55, 0.8, 0.6]),
        pred_direct_rank=np.array([0.0, 0.9, 0.8, 0.7, 0.6]),
        pred_direct_opportunity=np.array([0.5, 0.9, 0.9, 0.9, 0.9]),
        pred_direct_harm=np.array([0.5, 0.1, 0.1, 0.1, 0.1]),
        nominal_index=0,
    )
    assert out.selected_index == 3


def test_nested_selector_name_activates_topk_and_frozen_evidence_filter():
    x = _base_arrays()
    cfg = {
        "selection": {
            "ocrap_selector": "lcb_constrained_nested_evidence",
            "require_absolute_admission_for_intervention": True,
            "rifa_relative_proposal_top_k": 2,
            "rifa_relative_min_advantage": 0.0,
            "rifa_relative_opportunity_threshold": 0.65,
            "rifa_relative_harm_threshold": 0.30,
        }
    }
    out = select_baseline(
        "ocrap", x["utility"], x["r_dep"], np.zeros(5), np.zeros(5), x["hard"], x["harm"], x["feasible"],
        0.0, 0.0, 0.0, cfg,
        pred_gap=x["pred_gap"], nominal_deviation=x["nominal_deviation"],
        pred_direct_value=np.array([0.50, 0.40, 0.75, 0.95, 0.70]),
        pred_direct_rank=np.array([0.0, 0.9, 0.8, 0.7, 0.6]),
        pred_direct_opportunity=np.array([0.5, 0.9, 0.8, 0.9, 0.9]),
        pred_direct_harm=np.array([0.5, 0.2, 0.2, 0.1, 0.1]),
        nominal_index=0,
    )
    # Candidate 2 is top-K, absolute-admitted, positive-delta, high-opportunity,
    # and low-harm. Candidate 3 has a larger delta but lies outside top-K.
    assert out.selected_index == 2
    assert out.reason == "best_rifa_joint_sign"
