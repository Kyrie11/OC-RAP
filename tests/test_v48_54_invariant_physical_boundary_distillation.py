from __future__ import annotations
from pathlib import Path
import torch
from ocrap.models.losses import selected_option_physical_boundary_distillation_loss
ROOT=Path(__file__).resolve().parents[1]

def _loss(pred):
    tq=torch.tensor([[[0.8,-0.2],[0.5,-0.1]]],dtype=torch.float32)
    tm=torch.tensor([[[0.4,-0.5],[-0.3,0.6]]],dtype=torch.float32)
    rp=torch.tensor([[0.6,0.4]],dtype=torch.float32)
    rv=torch.ones((1,2),dtype=torch.bool); ov=torch.ones((1,2),dtype=torch.bool)
    return selected_option_physical_boundary_distillation_loss(pred,tq,tm,rp,rv,ov,temperature=0.08)

def test_v4854_ipbd_prefers_correct_selected_physical_zero_signs():
    good=torch.tensor([[[0.5,-9.0],[-0.5,9.0]]],requires_grad=True)
    bad=torch.tensor([[[-0.5,-9.0],[0.5,9.0]]],requires_grad=True)
    assert _loss(good) < _loss(bad)

def test_v4854_ipbd_gradient_only_touches_teacher_q_selected_option():
    pred=torch.tensor([[[0.2,7.0],[-0.2,-8.0]]],requires_grad=True)
    loss=_loss(pred); loss.backward(); g=pred.grad
    assert g is not None and torch.isfinite(g).all()
    assert g[0,0,0].abs()>0 and g[0,1,0].abs()>0
    assert g[0,0,1]==0 and g[0,1,1]==0

def test_v4854_ipbd_nonselected_physical_geometry_is_invariant():
    a=torch.tensor([[[0.25,-1.0],[-0.25,1.0]]],dtype=torch.float32)
    b=torch.tensor([[[0.25,100.0],[-0.25,-100.0]]],dtype=torch.float32)
    assert torch.allclose(_loss(a),_loss(b),atol=0,rtol=0)

def test_v4854_scripts_keep_qhard_deployment_and_isolate_training_only_ipbd():
    arm=(ROOT/'scripts/run_v48_54_dcp_drfc_bcde_ipbd_arm.sh').read_text()
    stage=(ROOT/'scripts/adapt_ocrap_v48_47_dsofr_witness_stage.sh').read_text()
    launch=(ROOT/'scripts/run_v48_54_dcp_drfc_bcde_ipbd_two_gpu.sh').read_text()
    comp=(ROOT/'tools/compare_v48_54_dcp_drfc_bcde_ipbd_ab.py').read_text()
    assert 'V4852_PHYSICAL_TEACHER_SIGN_ALIGNMENT=false' in arm
    assert 'V4853_PHYSICAL_STUDENT_SIGN_ALIGNMENT=false' in arm
    assert 'EVIDENCE_PHYSICAL_STUDENT_DRS=false' in arm
    assert 'V4854_INVARIANT_PHYSICAL_BOUNDARY_DISTILLATION=true' in arm
    assert "'student_sign_coordinate':'hard_qbest_ge_zero_root_mass_exact_pcd'" in arm
    assert "'teacher_sign_coordinate':'q_hard_proxy_drs_exact_pcd'" in arm
    assert 'INVARIANT_PHYSICAL_BOUNDARY_DISTILLATION="$IPBD"' in stage
    assert 'check_v48_54_reference_reuse.py' in launch
    assert 'B-A isolates training-only Invariant-Preserving Boundary Distillation' in comp


def test_v4854_ipbd_excludes_nonfinite_teacher_selected_margin():
    tq=torch.tensor([[[0.8,-0.2],[0.5,-0.1]]],dtype=torch.float32)
    tm=torch.tensor([[[float("nan"),-0.5],[-0.3,0.6]]],dtype=torch.float32)
    pred=torch.tensor([[[100.0,-9.0],[-0.5,9.0]]],requires_grad=True)
    rp=torch.tensor([[0.6,0.4]],dtype=torch.float32); rv=torch.ones((1,2),dtype=torch.bool); ov=torch.ones((1,2),dtype=torch.bool)
    loss=selected_option_physical_boundary_distillation_loss(pred,tq,tm,rp,rv,ov,temperature=0.08)
    loss.backward(); assert pred.grad is not None
    assert pred.grad[0,0,0]==0  # invalid privileged target is excluded, not relabeled negative
    assert pred.grad[0,1,0].abs()>0
