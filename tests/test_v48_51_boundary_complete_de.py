from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.losses import boundary_complete_frontier_calibration_loss
from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def _model(*, bc: bool) -> OCRAPModel:
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
        direct_recovery_evidence_dual_interaction_bridge=True,
        direct_recovery_evidence_roct_benefit=True,
        direct_recovery_evidence_roct_deployability=True,
        direct_recovery_evidence_roct_scale=3.0,
        direct_recovery_evidence_native_certificate_preservation=True,
        direct_recovery_evidence_native_margin_complete_preservation=False,
        direct_recovery_evidence_native_advantage_preservation=True,
        direct_recovery_evidence_native_exact_advantage_preservation=False,
        direct_recovery_evidence_native_boundary_complete_advantage_preservation=bc,
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


def test_bc_nap_preserves_material_hard_sign_and_smooth_deadband_order() -> None:
    m = _model(bc=True)
    # Rows are [nominal, material-hard-positive, material-hard-negative,
    # hard-tie/smooth-positive].  dep/gap are chosen so the exact recovery-value
    # difference crosses +/-positive_gain for rows 1/2, while row 3 stays in the
    # exact deadband and must be resolved by the smooth boundary mass.
    native = torch.tensor(
        [
            [0.50, 0.80, 0.50, 1.00],  # nominal: exact=smooth=.40
            [0.60, 0.80, 0.20, 1.00],  # exact=.48 (+.08), smooth=.16 (-.24)
            [0.40, 0.80, 0.90, 1.00],  # exact=.32 (-.08), smooth=.72 (+.32)
            [0.50, 0.80, 0.60, 1.00],  # exact tie, smooth=.48 (+.08)
        ],
        dtype=torch.float32,
    )
    groups = torch.tensor([[11], [11], [11], [11]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 0.0])
    _, margin, diagnostic_value = m._native_certificate_benefit_logit(native, groups, nominal)
    assert margin is not None and diagnostic_value is not None
    assert margin[1] > 0.0, "material hard-positive sign must survive contrary smooth ordering"
    assert margin[2] < 0.0, "material hard-negative sign must survive contrary smooth ordering"
    assert margin[3] > 0.0, "inside hard equivalence band, smooth coordinate must resolve ordering"
    # Diagnostic row value deliberately stays v48.49-smooth for comparability;
    # the authoritative BC decision quantity is the candidate-relative margin.
    assert torch.allclose(diagnostic_value, native[:, 2] * native[:, 1] * native[:, 3])


def test_bc_nap_is_parameter_free_and_mutually_exclusive_with_exact_only_nap() -> None:
    torch.manual_seed(4851)
    smooth = _model(bc=False)
    torch.manual_seed(4851)
    bc = _model(bc=True)
    assert set(smooth.state_dict()) == set(bc.state_dict())
    bc.load_state_dict(smooth.state_dict(), strict=True)

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
        direct_recovery_evidence_native_advantage_preservation=True,
        direct_recovery_evidence_native_exact_advantage_preservation=True,
        direct_recovery_evidence_native_boundary_complete_advantage_preservation=True,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        OCRAPModel(**kwargs)


def _bc_frontier_fixture(pred_q: torch.Tensor) -> torch.Tensor:
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
    roots = torch.tensor(
        [[0.75, 0.25], [0.75, 0.25], [0.60, 0.40], [0.60, 0.40]], dtype=torch.float32
    )
    root_valid = torch.ones((4, 2), dtype=torch.bool)
    option_valid = torch.ones((4, 2), dtype=torch.bool)
    return boundary_complete_frontier_calibration_loss(
        pred_r_dep,
        pred_gap,
        pred_q,
        teacher_r_dep,
        teacher_r_orc,
        teacher_q,
        roots,
        roots,
        root_valid,
        option_valid,
        torch.tensor([101, 101, 202, 202]),
        torch.tensor([7, 7, 9, 9]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        gamma=0.0,
        option_temperature=0.35,
        deployability_tolerance=0.05,
        drs_tolerance=0.05,
        gap_tolerance=0.05,
        positive_gain=0.015,
    )


def test_bc_frontier_retains_boundary_local_magnitude_information_and_gradients() -> None:
    # These tensors have the same hard q>=0 pattern but different distances to
    # the zero frontier. Unlike v48.50 DEFC, BC-FC intentionally keeps their
    # order-channel losses different while sign supervision remains hard-exact.
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
    near = _bc_frontier_fixture(q_near)
    far = _bc_frontier_fixture(q_far)
    assert torch.isfinite(near) and torch.isfinite(far)
    assert not torch.allclose(near, far, atol=1e-6, rtol=0.0)
    near.backward(); far.backward()
    assert q_near.grad is not None and torch.isfinite(q_near.grad).all()
    assert q_far.grad is not None and torch.isfinite(q_far.grad).all()
    assert q_near.grad.abs().sum() > 0.0
    assert q_far.grad.abs().sum() > 0.0


def test_v4851_arm_contract_is_clean_2x2_without_regime_router_or_old_exact_only_axes() -> None:
    arm = (ROOT / "scripts/run_v48_51_dcp_drfc_bcde_ablation_arm.sh").read_text(encoding="utf-8")
    comp = (ROOT / "tools/compare_v48_51_dcp_drfc_bcde_2x2.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/run_v48_51_dcp_drfc_bcde_2x2_two_gpu.sh").read_text(encoding="utf-8")
    stage = (ROOT / "scripts/adapt_ocrap_v48_47_dsofr_witness_stage.sh").read_text(encoding="utf-8")

    assert "EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=false" in arm
    assert "EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true" in arm
    assert "EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false" in arm
    assert "V4850_DECISION_EQUIVALENT_FRONTIER=false" in arm
    assert "V4851_BOUNDARY_COMPLETE_FRONTIER=true" in arm
    assert "EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=true" in arm
    assert "'strategy_regime_conditioning':False" in arm
    assert '"A":(False,False)' in comp and '"D":(True,True)' in comp
    assert 'for pair in "A B" "C D"' in launcher
    assert "OC-RAP-v48.51-DCP-DRFC-BCDE-2x2-audit.json" in launcher
    # Nested witness stages must not accidentally activate the downstream BC-NAP.
    assert "'native_boundary_complete_advantage_preservation':False" in stage
    assert "EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=false" in stage
