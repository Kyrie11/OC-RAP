from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def _model(*, base_native: bool = True, margin_complete: bool = False, native_advantage: bool = False) -> OCRAPModel:
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
        direct_recovery_evidence_native_certificate_preservation=base_native,
        direct_recovery_evidence_native_margin_complete_preservation=margin_complete,
        direct_recovery_evidence_native_advantage_preservation=native_advantage,
        direct_recovery_evidence_native_drs_tolerance=0.05,
        direct_recovery_evidence_native_deployability_tolerance=0.05,
        direct_recovery_evidence_native_gap_tolerance=0.05,
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


def test_dcp_requires_base_native_certificate_preservation() -> None:
    with pytest.raises(ValueError, match="requires native certificate preservation"):
        _model(base_native=False, margin_complete=True)
    with pytest.raises(ValueError, match="requires native certificate preservation"):
        _model(base_native=False, native_advantage=True)


def test_margin_complete_transport_preserves_boundary_dep_and_gap_signs() -> None:
    m = _model(margin_complete=True)
    # [paper-facing hard DRS, sigmoid(R_dep), boundary-resolved DRS, gap quality]
    native = torch.tensor([
        [1.0, 0.80, 0.90, 0.90],  # nominal
        [1.0, 0.70, 0.60, 0.70],  # worse on all margin-complete coordinates
        [0.0, 0.90, 0.95, 0.95],  # hard DRS flips, but boundary geometry is safer
    ])
    groups = torch.tensor([[0], [0], [0]])
    nominal = torch.tensor([1.0, 0.0, 0.0])
    logits, margins = m._native_certificate_component_logits(native, groups, nominal)
    assert logits is not None and margins is not None
    expected = torch.tensor([
        [-0.05, -0.05, -0.05],
        [ 0.25,  0.05,  0.15],
        [-0.10, -0.15, -0.10],
    ])
    assert torch.allclose(margins, expected, atol=1e-6)
    assert torch.allclose(logits, expected / 0.025, atol=1e-6)


def test_native_advantage_matches_fixed_native_recovery_value() -> None:
    m = _model(native_advantage=True)
    native = torch.tensor([
        [1.0, 0.80, 0.90, 0.90],  # V=0.648 nominal
        [1.0, 0.90, 0.95, 0.95],  # V=0.81225 beneficial
        [0.0, 0.60, 0.50, 0.50],  # V=0.15 harmful
    ])
    groups = torch.tensor([[7], [7], [7]])
    nominal = torch.tensor([1.0, 0.0, 0.0])
    logit, margin, value = m._native_certificate_benefit_logit(native, groups, nominal)
    assert logit is not None and margin is not None and value is not None
    expected_value = native[:, 2] * native[:, 1] * native[:, 3]
    expected_margin = expected_value - expected_value[0] - 0.015
    assert torch.allclose(value, expected_value, atol=1e-7)
    assert torch.allclose(margin, expected_margin, atol=1e-7)
    assert torch.allclose(logit, expected_margin / 0.05, atol=1e-6)


def test_dcp_is_parameter_free_and_keeps_paper_hard_drs_coordinate() -> None:
    torch.manual_seed(4849)
    base = _model().eval()
    torch.manual_seed(4849)
    dcp = _model(margin_complete=True, native_advantage=True).eval()
    assert set(base.state_dict()) == set(dcp.state_dict())
    missing, unexpected = dcp.load_state_dict(base.state_dict(), strict=True)
    assert not missing and not unexpected

    root_logits = torch.tensor([[1.2, 0.4, -0.7]], dtype=torch.float32)
    obs = torch.tensor([[[0.0, 0.0], [0.05, 0.0], [1.5, 1.5]]], dtype=torch.float32)
    margins = torch.tensor([[[0.8, -0.5], [0.7, -0.4], [-0.3, 0.9]]], dtype=torch.float32)
    rv = torch.ones((1, 3), dtype=torch.bool)
    ov = torch.ones((1, 2), dtype=torch.bool)
    _sig, native = OCRAPModel._recovery_option_compatibility_signature(
        root_logits, obs, margins, tau_obs=0.25, alpha=0.2, beta=0.2, top_m=8,
        option_temperature=0.35, root_valid=rv, option_valid=ov,
        return_native_certificate=True,
    )
    assert native.shape[-1] == 4
    # Coordinate 0 remains the exact hard paper-facing DRS. Coordinate 2 adds
    # boundary resolution for admission but does not redefine reported DRS.
    assert torch.all((native[:, 0] >= 0.0) & (native[:, 0] <= 1.0))
    assert torch.all((native[:, 2] > 0.0) & (native[:, 2] < 1.0))


def test_dcp_transport_is_monotone_and_regime_agnostic() -> None:
    m = _model(margin_complete=True, native_advantage=True)
    nominal = torch.tensor([[1.0, 0.60, 0.60, 0.60]])
    worse = torch.tensor([[1.0, 0.55, 0.55, 0.55]])
    better = torch.tensor([[1.0, 0.75, 0.75, 0.75]])
    native = torch.cat([nominal, worse, better], dim=0)
    groups = torch.tensor([[0], [0], [0]])
    is_nominal = torch.tensor([1.0, 0.0, 0.0])
    harm, _ = m._native_certificate_component_logits(native, groups, is_nominal)
    benefit, _, _ = m._native_certificate_benefit_logit(native, groups, is_nominal)
    assert harm is not None and benefit is not None
    assert torch.all(harm[2] < harm[1])
    assert benefit[2] > benefit[1]

    block = (ROOT / "src/ocrap/models/ocrap.py").read_text(encoding="utf-8")
    start = block.index("def _native_certificate_component_logits")
    end = block.index("def _direct_outputs", start)
    executable = "\n".join(
        line for line in block[start:end].splitlines()
        if not line.lstrip().startswith("#")
    ).lower()
    assert "bucket_id" not in executable
    assert "safe_" not in executable and "near_" not in executable and "contact_" not in executable


def test_v4849_arm_and_comparator_define_clean_2x2() -> None:
    arm = (ROOT / "scripts/run_v48_49_dcp_ablation_arm.sh").read_text(encoding="utf-8")
    comp = (ROOT / "tools/compare_v48_49_dcp_2x2.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/run_v48_49_dcp_2x2_two_gpu.sh").read_text(encoding="utf-8")
    assert "V4847_RECOVERY_FRONTIER=1" in arm
    assert "EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true" in arm
    assert "EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=true" in arm
    assert "EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true" in arm
    assert "strategy_regime_conditioning':False" in arm
    assert '"B":(True,False)' in comp and '"C":(False,True)' in comp and '"D":(True,True)' in comp
    assert 'for pair in "A B" "C D"' in launcher
    assert "V4849_VARIANT_MODE:-serial" in launcher
    assert "OC-RAP-v48.49-DCP-DRFC-2x2-audit.json" in launcher
    assert "OC-RAP-v48.49-NCP-DRFC-2x2-audit.json" not in launcher
