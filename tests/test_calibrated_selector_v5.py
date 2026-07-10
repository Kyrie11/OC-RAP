import numpy as np

from ocrap.evaluation.baselines import select_baseline
from ocrap.planning.selector import calibrated_constrained_select


def test_budget_rate_by_bucket_is_applied_without_global_scalar():
    utility = np.array([1.0, 1.5])
    pred_r_dep = np.array([-0.01, 0.05])
    teacher = np.array([0.1, 0.1])
    hard = np.array([0.0, 0.0])
    harm = np.array([0.0, 0.0])
    feasible = np.array([True, True])
    cfg = {
        "selection": {
            "ocrap_selector": "calibrated_constrained",
            "active_bucket_name": "test_safe_v2",
            "intervention_budget_rate_by_bucket": {"safe": 0.05},
            "intervention_budget_used": 10,
            "intervention_budget_steps": 20,
            "intervention_budget_penalty": 100.0,
            "budget_nominal_slack": 0.20,
            "safe_nominal_slack": 0.20,
        }
    }
    sel = select_baseline(
        "ocrap",
        utility,
        pred_r_dep,
        teacher,
        teacher,
        hard,
        harm,
        feasible,
        gamma_rec=0.0,
        gamma_H=0.0,
        gamma_D=5.0,
        cfg=cfg,
        pred_gap=np.zeros(2),
        nominal_deviation=np.zeros(2),
    )
    assert sel.selected_index == 0
    assert sel.reason == "nominal_budget_preserved"


def test_safe_switch_guard_requires_material_recovery_gain():
    sel = calibrated_constrained_select(
        utility=np.array([1.00, 1.40]),
        r_dep=np.array([-0.01, 0.00]),
        hard=np.zeros(2),
        harm=np.zeros(2),
        feasible=np.array([True, True]),
        gamma_rec=0.0,
        pred_gap=np.array([0.05, 0.05]),
        nominal_deviation=np.array([0.0, 0.02]),
        regime_name="test_safe",
        safe_nominal_slack=0.15,
        safe_switch_score_margin=0.10,
        safe_min_rec_lcb_gain=0.10,
        safe_min_gap_reduction=0.20,
    )
    assert sel.selected_index == 0
    assert sel.reason == "nominal_safe_switch_guard"


def test_prefer_admitted_ranks_within_admitted_candidates():
    sel = calibrated_constrained_select(
        utility=np.array([1.0, 4.0, 2.0]),
        r_dep=np.array([-0.4, -0.3, 0.2]),
        hard=np.zeros(3),
        harm=np.zeros(3),
        feasible=np.array([True, True, True]),
        gamma_rec=0.0,
        pred_gap=np.zeros(3),
        nominal_deviation=np.zeros(3),
        regime_name="test_contact",
        prefer_admitted=True,
    )
    assert sel.selected_index == 2
    assert sel.reason == "best_calibrated_prefer_admitted"


def test_option_drs_certificate_can_admit_conservative_scalar_candidate():
    sel = calibrated_constrained_select(
        utility=np.array([1.0, 1.2]),
        r_dep=np.array([0.0, -0.15]),
        hard=np.zeros(2),
        harm=np.zeros(2),
        feasible=np.array([True, True]),
        gamma_rec=0.2,
        pred_gap=np.array([0.9, 0.4]),
        pred_drs=np.array([0.55, 0.88]),
        nominal_deviation=np.zeros(2),
        regime_name="test_contact",
        prefer_admitted=True,
        require_admitted_intervention=True,
        require_intervention_evidence=True,
        option_drs_certificate=True,
        option_drs_certificate_threshold=0.8,
        option_drs_certificate_max_gap=0.6,
        option_drs_certificate_rec_slack=0.5,
    )
    assert sel.selected_index == 1
    assert sel.admitted[1]
    assert sel.reason == "best_option_drs_certified_prefer_admitted"


def test_relative_recovery_certificate_admits_contact_opportunity_without_soft_fallback():
    sel = calibrated_constrained_select(
        utility=np.array([1.0, 0.8, 0.7]),
        r_dep=np.array([-0.95, -0.55, -0.90]),
        hard=np.zeros(3),
        harm=np.zeros(3),
        feasible=np.array([True, True, True]),
        gamma_rec=0.2,
        pred_gap=np.array([1.30, 1.25, 1.50]),
        pred_drs=np.array([0.76, 0.78, 0.91]),
        nominal_deviation=np.zeros(3),
        regime_name="test_contact",
        prefer_admitted=True,
        require_admitted_intervention=True,
        require_intervention_evidence=True,
        relative_recovery_certificate=True,
        relative_recovery_nominal_rec_lcb_max=-0.60,
        relative_recovery_nominal_gap_min=1.10,
        relative_recovery_min_rec_gain=0.25,
        relative_recovery_min_drs=0.70,
        relative_recovery_max_gap=1.40,
        relative_recovery_max_gap_increase=0.20,
        relative_recovery_bonus=1.0,
    )
    assert sel.selected_index == 1
    assert sel.admitted[1]
    assert sel.reason == "best_relative_recovery_certified_prefer_admitted"


def test_relative_recovery_certificate_uses_recovery_pool_in_contact():
    # Candidate 1 is a plausible post-contact recovery: it has a nominal hard-rule
    # violation, so v16's `safe & ...` gate would reject it.  The v17 recovery
    # pool allows it only under explicit contact-regime hard/harm bounds.
    sel = calibrated_constrained_select(
        utility=np.array([2.0, 0.6]),
        r_dep=np.array([-1.0, -0.82]),
        hard=np.array([0.0, 2.0]),
        harm=np.array([0.0, 0.2]),
        feasible=np.array([True, True]),
        gamma_rec=0.2,
        pred_gap=np.array([1.30, 1.10]),
        pred_drs=np.array([0.70, 0.78]),
        nominal_deviation=np.array([0.0, 0.4]),
        regime_name="test_contact",
        prefer_admitted=True,
        require_admitted_intervention=True,
        require_intervention_evidence=True,
        relative_recovery_certificate=True,
        relative_recovery_use_recovery_pool=True,
        recovery_cert_max_hard=2.0,
        recovery_cert_max_harm=1.0,
        relative_recovery_nominal_rec_lcb_max=-0.5,
        relative_recovery_nominal_gap_min=1.0,
        relative_recovery_min_rec_gain=0.02,
        relative_recovery_min_gap_reduction=0.05,
        relative_recovery_gate="any_gain",
        relative_recovery_min_drs=0.70,
        relative_recovery_max_gap=1.35,
        relative_recovery_max_gap_increase=0.20,
    )
    assert sel.selected_index == 1
    assert sel.admitted[1]
    assert sel.reason == "best_relative_recovery_certified_prefer_admitted"


def test_relative_recovery_gap_dominance_can_rescue_overconfident_nominal():
    # The model sometimes saturates DRS for a nominal action that has high gap and
    # low teacher recovery.  A candidate that materially reduces the predicted
    # oracle-deployability gap should be certifiable even when scalar R_dep gain is
    # small, but only when the explicit dominance gate is requested.
    sel = calibrated_constrained_select(
        utility=np.array([5.0, 0.5]),
        r_dep=np.array([-0.80, -0.79]),
        hard=np.zeros(2),
        harm=np.zeros(2),
        feasible=np.array([True, True]),
        gamma_rec=0.2,
        pred_gap=np.array([1.50, 1.05]),
        pred_drs=np.array([0.82, 0.83]),
        nominal_deviation=np.array([0.0, 0.2]),
        regime_name="test_contact",
        prefer_admitted=True,
        require_admitted_intervention=True,
        require_intervention_evidence=True,
        relative_recovery_certificate=True,
        relative_recovery_nominal_rec_lcb_max=-0.5,
        relative_recovery_nominal_gap_min=1.0,
        relative_recovery_min_rec_gain=0.10,
        relative_recovery_min_gap_reduction=0.20,
        relative_recovery_gate="any_gain",
        relative_recovery_min_drs=0.70,
        relative_recovery_max_gap=1.35,
        relative_recovery_max_gap_increase=0.20,
    )
    assert sel.selected_index == 1
    assert sel.admitted[1]
