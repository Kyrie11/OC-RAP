from __future__ import annotations

from pathlib import Path

import torch

from ocrap.models.losses import (
    _physical_student_observation_consistent_success_st,
    boundary_complete_frontier_calibration_loss,
)
from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def test_v4853_student_physical_drs_forward_matches_q_selected_margin_success() -> None:
    q = torch.tensor([[[0.8, -0.2], [0.4, -0.1]]], dtype=torch.float32)
    margins = torch.tensor([[[-0.3, 0.5], [0.2, -0.4]]], dtype=torch.float32, requires_grad=True)
    p = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    rv = torch.ones((1, 2), dtype=torch.bool)
    ov = torch.ones((1, 2), dtype=torch.bool)
    physical = _physical_student_observation_consistent_success_st(q, margins, p, rv, ov)
    # q selects option 0 for both roots. Root 0 physical margin fails, root 1 succeeds.
    assert torch.allclose(physical.detach(), torch.tensor([0.3]))
    q_hard = (p * (q.amax(dim=-1) >= 0.0).float()).sum(dim=-1)
    assert torch.allclose(q_hard, torch.tensor([1.0]))


def test_v4853_student_physical_drs_gradient_only_uses_q_selected_margin() -> None:
    q = torch.tensor([[[0.8, -0.2], [0.4, -0.1]]], dtype=torch.float32)
    margins = torch.tensor([[[-0.3, 0.5], [0.2, -0.4]]], dtype=torch.float32, requires_grad=True)
    p = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    rv = torch.ones((1, 2), dtype=torch.bool); ov = torch.ones((1, 2), dtype=torch.bool)
    out = _physical_student_observation_consistent_success_st(q, margins, p, rv, ov)
    out.sum().backward()
    g = margins.grad
    assert g is not None and torch.isfinite(g).all()
    assert g[0, 0, 0].abs() > 0 and g[0, 1, 0].abs() > 0
    assert g[0, 0, 1] == 0 and g[0, 1, 1] == 0


def _loss(student_physical: bool, pred_margins: torch.Tensor | None) -> torch.Tensor:
    pred_r_dep=torch.tensor([0.2,0.2],requires_grad=True); pred_gap=torch.tensor([0.1,0.1],requires_grad=True)
    pred_q=torch.tensor([[[0.2,-0.1],[0.2,-0.1]],[[0.4,-0.1],[0.4,-0.1]]],requires_grad=True)
    teacher_q=pred_q.detach().clone(); root=torch.tensor([[0.6,0.4]]*2)
    rv=torch.ones((2,2),dtype=torch.bool); ov=torch.ones((2,2),dtype=torch.bool)
    return boundary_complete_frontier_calibration_loss(
        pred_r_dep,pred_gap,pred_q,torch.tensor([0.2,0.2]),torch.tensor([0.3,0.3]),teacher_q,
        root,root,rv,ov,torch.tensor([1,1]),torch.tensor([0,0]),torch.tensor([1.0,0.0]),
        pred_margins=pred_margins,physical_student_sign_alignment=student_physical,
        option_execution_semantics='observation_class')


def test_v4853_student_physical_frontier_fails_closed_without_predicted_margins() -> None:
    try:
        _loss(True, None)
    except ValueError as exc:
        assert 'requires pred_margins' in str(exc)
    else:
        raise AssertionError('physical student alignment must fail closed without predicted margins')


def test_v4853_native_deployment_coordinate_uses_physical_student_event_only_when_enabled() -> None:
    root_logits=torch.tensor([[0.0]],dtype=torch.float32)
    obs=torch.zeros((1,1,2),dtype=torch.float32)
    # q is computed from margins, so use two options and a root layout where q's
    # selected option is allowed to disagree with its selected per-root margin.
    # A two-root aliased set supplies the robust q coupling needed for disagreement.
    root_logits=torch.tensor([[1.0,0.5]],dtype=torch.float32)
    obs=torch.tensor([[[0.0,0.0],[0.0,0.0]]],dtype=torch.float32)
    margins=torch.tensor([[[-0.4,0.9],[0.8,-0.5]]],dtype=torch.float32)
    rv=torch.ones((1,2),dtype=torch.bool); ov=torch.ones((1,2),dtype=torch.bool)
    _, q_native=OCRAPModel._recovery_option_compatibility_signature(
        root_logits,obs,margins,0.25,0.2,0.2,8,0.35,root_valid=rv,option_valid=ov,
        return_native_certificate=True,physical_student_drs=False)
    _, p_native=OCRAPModel._recovery_option_compatibility_signature(
        root_logits,obs,margins,0.25,0.2,0.2,8,0.35,root_valid=rv,option_valid=ov,
        return_native_certificate=True,physical_student_drs=True)
    assert q_native.shape == p_native.shape == (1,4)
    # Only native DRS coordinate 0 is redefined; DEP, smooth boundary mass, gap are unchanged.
    assert torch.allclose(q_native[:,1:], p_native[:,1:], atol=0.0, rtol=0.0)
    assert not torch.allclose(q_native[:,0], p_native[:,0], atol=0.0, rtol=0.0)
    assert torch.isfinite(p_native).all()


def test_v4853_scripts_define_strict_teacher_student_2x2_without_regime_router() -> None:
    arm=(ROOT/'scripts/run_v48_53_dcp_drfc_bcde_cse_arm.sh').read_text(encoding='utf-8')
    launcher=(ROOT/'scripts/run_v48_53_dcp_drfc_bcde_cse_two_gpu.sh').read_text(encoding='utf-8')
    comp=(ROOT/'tools/compare_v48_53_dcp_drfc_bcde_cse_2x2.py').read_text(encoding='utf-8')
    stage=(ROOT/'scripts/adapt_ocrap_v48_47_dsofr_witness_stage.sh').read_text(encoding='utf-8')
    assert 'V4851_BOUNDARY_COMPLETE_FRONTIER=true' in arm
    assert 'EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=false' in arm
    assert 'EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false' in arm
    assert 'V4853_PHYSICAL_STUDENT_SIGN_ALIGNMENT=true' in arm
    assert 'EVIDENCE_PHYSICAL_STUDENT_DRS=true' in arm
    assert "'strategy_regime_conditioning':False" in arm
    assert 'check_v48_53_ab_reference_reuse.py' in launcher
    assert 'run_arm C "$GPU0"' in launcher and 'run_arm D "$GPU1"' in launcher
    assert 'D-B-C+A' in comp
    assert 'RECOVERY_FRONTIER_PHYSICAL_STUDENT_SIGN_ALIGNMENT' in stage
