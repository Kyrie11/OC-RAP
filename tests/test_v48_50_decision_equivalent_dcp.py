from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.losses import decision_equivalent_frontier_calibration_loss
from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def _model(*, exact: bool) -> OCRAPModel:
    return OCRAPModel(
        input_dim=FlatFeatureLayout().total_dim,
        num_roots=3,
        num_options=4,
        d_model=12,
        d_obs=6,
        encoder_type="structured_transformer",
        num_layers=1,
        num_heads=3,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_pooling="candidate_concat_raw",
        direct_recovery_delta_head=True,
        direct_recovery_delta_mode="ordinal_evidence",
        direct_recovery_delta_regime_experts=True,
        direct_recovery_delta_policy_features=True,
        direct_recovery_evidence_calibrator=True,
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_source="physical_interaction",
        direct_recovery_evidence_interaction_hidden=16,
        direct_recovery_evidence_interaction_dropout=0.0,
        direct_recovery_evidence_dual_interaction_bridge=True,
        direct_recovery_evidence_roct_benefit=True,
        direct_recovery_evidence_roct_deployability=True,
        direct_recovery_evidence_roct_scale=3.0,
        direct_recovery_evidence_roct_alpha=0.2,
        direct_recovery_evidence_roct_beta=0.2,
        direct_recovery_evidence_roct_top_m=8,
        direct_recovery_evidence_roct_option_temperature=0.35,
        direct_recovery_evidence_native_certificate_preservation=True,
        direct_recovery_evidence_native_margin_complete_preservation=False,
        direct_recovery_evidence_native_advantage_preservation=True,
        direct_recovery_evidence_native_exact_advantage_preservation=exact,
        direct_recovery_evidence_native_positive_gain=0.015,
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=5,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=False,
        direct_recovery_evidence_admission_prior_mode="joint_reserve",
        direct_recovery_evidence_reserve_factor_alignment=True,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_reliability="1,1,1,0,0",
        direct_recovery_evidence_benefit_margin_temperature=0.05,
    )


def _frontier_fixture(pred_q: torch.Tensor, pred_root_probs: torch.Tensor) -> torch.Tensor:
    # Two candidate groups, each [nominal, recovery].  Teacher and prediction use
    # the same hard feasibility events but the q depths can differ.  This lets us
    # verify that the *forward* deployed DRS is hard while gradients remain smooth.
    pred_r_dep = torch.tensor([0.20, 0.45, -0.10, 0.20], requires_grad=True)
    pred_gap = torch.tensor([0.25, 0.05, 0.30, 0.15], requires_grad=True)
    teacher_r_dep = torch.tensor([0.20, 0.50, -0.10, 0.25])
    teacher_r_orc = teacher_r_dep + torch.tensor([0.25, 0.05, 0.30, 0.10])
    teacher_q = torch.tensor(
        [
            [[0.40, -0.30], [-0.50, -0.20]],
            [[0.50, -0.20], [0.30, -0.10]],
            [[0.20, -0.40], [-0.30, -0.10]],
            [[0.35, -0.10], [0.25, -0.20]],
        ],
        dtype=torch.float32,
    )
    teacher_root_probs = torch.tensor(
        [[0.75, 0.25], [0.75, 0.25], [0.60, 0.40], [0.60, 0.40]], dtype=torch.float32
    )
    root_valid = torch.ones((4, 2), dtype=torch.bool)
    option_valid = torch.ones((4, 2), dtype=torch.bool)
    scene_hash = torch.tensor([101, 101, 202, 202])
    time_index = torch.tensor([7, 7, 9, 9])
    is_nominal = torch.tensor([1.0, 0.0, 1.0, 0.0])
    return decision_equivalent_frontier_calibration_loss(
        pred_r_dep,
        pred_gap,
        pred_q,
        teacher_r_dep,
        teacher_r_orc,
        teacher_q,
        pred_root_probs,
        teacher_root_probs,
        root_valid,
        option_valid,
        scene_hash,
        time_index,
        is_nominal,
        gamma=0.0,
        option_temperature=0.35,
        deployability_tolerance=0.05,
        drs_tolerance=0.05,
        gap_tolerance=0.05,
        positive_gain=0.015,
        sign_temperature=0.08,
        regression_weight=1.0,
        sign_weight=0.5,
        pcd_weight=1.0,
    )


def test_exact_native_advantage_uses_hard_drs_teacher_coordinate() -> None:
    smooth = _model(exact=False)
    exact = _model(exact=True)
    exact.load_state_dict(smooth.state_dict(), strict=True)
    # [hard DRS, sigmoid(R_dep), smooth boundary DRS, exp(-gap)]
    native = torch.tensor(
        [
            [1.0, 0.80, 0.55, 0.90],
            [1.0, 0.90, 0.57, 0.95],
            [0.0, 0.95, 0.90, 0.99],
        ]
    )
    groups = torch.tensor([[7], [7], [7]])
    nominal = torch.tensor([1.0, 0.0, 0.0])

    _, smooth_margin, smooth_value = smooth._native_certificate_benefit_logit(native, groups, nominal)
    _, exact_margin, exact_value = exact._native_certificate_benefit_logit(native, groups, nominal)
    assert smooth_value is not None and exact_value is not None
    assert smooth_margin is not None and exact_margin is not None
    assert torch.allclose(smooth_value, native[:, 2] * native[:, 1] * native[:, 3])
    assert torch.allclose(exact_value, native[:, 0] * native[:, 1] * native[:, 3])
    # The third candidate is a useful regression guard: smooth boundary mass can
    # look strongly positive although the deployed hard DRS is exactly zero.
    assert smooth_margin[2] > 0.0
    assert exact_margin[2] < 0.0


def test_exact_native_advantage_is_parameter_free_and_requires_nap() -> None:
    torch.manual_seed(4850)
    smooth = _model(exact=False)
    torch.manual_seed(4850)
    exact = _model(exact=True)
    assert set(smooth.state_dict()) == set(exact.state_dict())
    exact.load_state_dict(smooth.state_dict(), strict=True)

    kwargs = dict(
        input_dim=FlatFeatureLayout().total_dim,
        num_roots=2,
        num_options=2,
        d_model=8,
        d_obs=4,
        encoder_type="structured_transformer",
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        direct_recovery_evidence_native_certificate_preservation=True,
        direct_recovery_evidence_native_advantage_preservation=False,
        direct_recovery_evidence_native_exact_advantage_preservation=True,
    )
    with pytest.raises(ValueError, match="requires native advantage preservation"):
        OCRAPModel(**kwargs)


def test_decision_equivalent_frontier_forward_is_hard_but_backward_is_smooth() -> None:
    roots = torch.tensor(
        [[0.70, 0.30], [0.70, 0.30], [0.55, 0.45], [0.55, 0.45]], dtype=torch.float32
    )
    q_near = torch.tensor(
        [
            [[0.02, -0.03], [-0.04, -0.02]],
            [[0.03, -0.02], [0.01, -0.01]],
            [[0.02, -0.04], [-0.03, -0.01]],
            [[0.02, -0.01], [0.01, -0.02]],
        ],
        requires_grad=True,
    )
    q_far = torch.tensor(
        [
            [[2.0, -3.0], [-4.0, -2.0]],
            [[3.0, -2.0], [1.0, -1.0]],
            [[2.0, -4.0], [-3.0, -1.0]],
            [[2.0, -1.0], [1.0, -2.0]],
        ],
        requires_grad=True,
    )
    loss_near = _frontier_fixture(q_near, roots)
    loss_far = _frontier_fixture(q_far, roots)
    # Same hard feasible/infeasible pattern => exactly the same forward loss.
    assert torch.allclose(loss_near, loss_far, atol=1e-7, rtol=0.0)
    loss_near.backward()
    loss_far.backward()
    assert q_near.grad is not None and torch.isfinite(q_near.grad).all()
    assert q_far.grad is not None and torch.isfinite(q_far.grad).all()
    assert q_near.grad.abs().sum() > 0.0
    # The ST surrogate preserves boundary-sensitive gradients: near-boundary q
    # receives a substantially stronger signal than saturated far-away q.
    assert q_near.grad.abs().sum() > q_far.grad.abs().sum()


def test_decision_equivalent_frontier_uses_predicted_root_weights() -> None:
    q = torch.tensor(
        [
            [[0.40, -0.30], [-0.50, -0.20]],
            [[0.50, -0.20], [0.30, -0.10]],
            [[0.20, -0.40], [-0.30, -0.10]],
            [[0.35, -0.10], [0.25, -0.20]],
        ],
        requires_grad=True,
    )
    roots_a = torch.tensor(
        [[0.75, 0.25], [0.75, 0.25], [0.60, 0.40], [0.60, 0.40]], dtype=torch.float32
    )
    roots_b = torch.tensor(
        [[0.10, 0.90], [0.10, 0.90], [0.15, 0.85], [0.15, 0.85]], dtype=torch.float32
    )
    loss_a = _frontier_fixture(q, roots_a)
    loss_b = _frontier_fixture(q.detach().clone().requires_grad_(True), roots_b)
    assert not torch.allclose(loss_a, loss_b, atol=1e-5, rtol=0.0)


def test_v4850_arm_contract_is_clean_2x2_around_v4849_c() -> None:
    arm = (ROOT / "scripts/run_v48_50_dcp_de_ablation_arm.sh").read_text(encoding="utf-8")
    comp = (ROOT / "tools/compare_v48_50_dcp_de_2x2.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/run_v48_50_dcp_de_2x2_two_gpu.sh").read_text(encoding="utf-8")
    postgate = (ROOT / "scripts/run_v48_50_postgate_if_authorized.sh").read_text(encoding="utf-8")

    assert "EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=false" in arm
    assert "EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true" in arm
    assert "V4850_DECISION_EQUIVALENT_FRONTIER=true" in arm
    assert "EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=true" in arm
    assert "strategy_regime_conditioning':False" in arm
    assert '"A":(False,False)' in comp and '"D":(True,True)' in comp
    assert 'for pair in "A B" "C D"' in launcher
    assert "OC-RAP-v48.50-DCP-DRFC-DE-2x2-audit.json" in launcher
    # Regression guard for the output-directory version mix-up found during the
    # engineering audit of the first v48.50 implementation draft.
    assert "ocrap_v48_49_dcp" not in arm
    dedicated = (ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh").read_text(encoding="utf-8")
    assert "--expect-value-regime-conditioning false" in dedicated
    assert "V48_49_GPU_SCHEDULER_DECISION" not in launcher
    assert "V48_49_RUNTIME" not in launcher
    assert "native_margin_complete_preservation')" in postgate
    assert "decision_equivalent_frontier" in postgate


def test_v4850_exact_flag_is_stage_locally_disabled_in_nested_witness() -> None:
    stage = (ROOT / "scripts/adapt_ocrap_v48_47_dsofr_witness_stage.sh").read_text(encoding="utf-8")
    assert "EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false" in stage
    assert "'native_exact_advantage_preservation':False" in stage
    assert "RECOVERY_FRONTIER_DECISION_EQUIVALENT=\"${V4850_DECISION_EQUIVALENT_FRONTIER:-false}\"" in stage


def test_v4850_decision_transport_contains_no_regime_route() -> None:
    model = (ROOT / "src/ocrap/models/ocrap.py").read_text(encoding="utf-8")
    loss = (ROOT / "src/ocrap/models/losses.py").read_text(encoding="utf-8")
    m0 = model.index("def _native_certificate_benefit_logit")
    m1 = model.index("def _direct_outputs", m0)
    l0 = loss.index("def decision_equivalent_frontier_calibration_loss")
    l1 = loss.index("\ndef ", l0 + 5)
    executable = "\n".join(
        line for line in (model[m0:m1] + loss[l0:l1]).splitlines()
        if not line.lstrip().startswith("#")
    ).lower()
    assert "bucket_id" not in executable
    assert "safe_" not in executable and "near_" not in executable and "contact_" not in executable


def test_decision_equivalent_predicted_mask_does_not_depend_on_teacher_finiteness() -> None:
    # A teacher NaN is target-side missingness, not an inference-time validity
    # signal. The predicted hard DRS must still see a root/option that is valid
    # according to root_valid/option_valid.
    common = dict(
        pred_r_dep=torch.tensor([0.0, 0.0]),
        pred_gap=torch.tensor([0.0, 0.0]),
        teacher_r_dep=torch.tensor([0.0, 0.0]),
        teacher_r_orc=torch.tensor([0.0, 0.0]),
        teacher_q=torch.tensor([[[-0.2], [-0.2]], [[float("nan")], [-0.2]]]),
        pred_root_probs=torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
        teacher_root_probs=torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
        root_valid=torch.ones((2, 2), dtype=torch.bool),
        option_valid=torch.ones((2, 1), dtype=torch.bool),
        scene_hash=torch.tensor([77, 77]),
        time_index=torch.tensor([3, 3]),
        is_nominal=torch.tensor([1.0, 0.0]),
        sign_weight=0.0,
    )
    q_pos = torch.tensor([[[-0.2], [-0.2]], [[0.2], [-0.2]]], requires_grad=True)
    q_neg = torch.tensor([[[-0.2], [-0.2]], [[-0.2], [-0.2]]], requires_grad=True)
    loss_pos = decision_equivalent_frontier_calibration_loss(pred_q=q_pos, **common)
    loss_neg = decision_equivalent_frontier_calibration_loss(pred_q=q_neg, **common)
    assert not torch.allclose(loss_pos, loss_neg, atol=1e-7, rtol=0.0)

def test_v4850_calibration_native_diagnostics_use_named_tolerances() -> None:
    calibrator = (ROOT / "tools/calibrate_policy_risk_v48.py").read_text(encoding="utf-8")
    # Regression for the real v48.50 four-arm RC30: ComponentVetoTolerances is
    # a dataclass, not a tuple.  The diagnostic-only native pair margins must
    # use the same named coordinates as component_veto_terms_numpy().
    assert "component_tolerances[" not in calibrator
    assert "component_tolerances.drs" in calibrator
    assert "component_tolerances.deployability_gate" in calibrator
    assert "component_tolerances.gap_discount" in calibrator

