from __future__ import annotations
from pathlib import Path
import torch
from ocrap.models.ocrap import (
    DualObservationConditionedActionFrontierBridge,
    OCRAPModel,
)
from ocrap.models.losses import frontier_normalize_signed_margin
from ocrap.models.encoders import FlatFeatureLayout

ROOT=Path(__file__).resolve().parents[1]

def test_dual_ocaf_zero_action_is_exactly_zero_for_both_tasks():
    b=DualObservationConditionedActionFrontierBridge(7,11,16,0.0).eval()
    benefit,harm=b(torch.zeros(5,7),torch.randn(5,11))
    assert torch.equal(benefit,torch.zeros_like(benefit))
    assert torch.equal(harm,torch.zeros_like(harm))

def test_dual_ocaf_task_gradients_are_parameter_decoupled():
    torch.manual_seed(4840)
    b=DualObservationConditionedActionFrontierBridge(7,11,16,0.0)
    a=torch.randn(8,7); o=torch.randn(8,11)
    benefit,_=b(a,o)
    benefit.square().mean().backward()
    benefit_grads=[p.grad for p in b.benefit.parameters() if p.requires_grad]
    harm_grads=[p.grad for p in b.harm.parameters() if p.requires_grad]
    assert benefit_grads and any(g is not None and g.abs().sum()>0 for g in benefit_grads)
    assert all(g is None or torch.equal(g,torch.zeros_like(g)) for g in harm_grads)

def test_frontier_normalization_preserves_sign_zero_and_compresses_far_tail():
    x=torch.tensor([-0.95,-0.05,0.0,0.05,0.95])
    y=frontier_normalize_signed_margin(x,0.10)
    assert y[2].item()==0.0
    assert torch.equal(torch.sign(x),torch.sign(y))
    assert torch.allclose(y, -torch.flip(y,[0]), atol=1e-7)
    assert abs(float(y[-1])) < 0.101
    assert abs(float(y[3])) > 0.04

def _model():
    return OCRAPModel(
        input_dim=FlatFeatureLayout().total_dim,num_roots=2,num_options=3,d_model=8,d_obs=4,
        encoder_type='structured_transformer',num_layers=1,num_heads=2,dropout=0.0,
        direct_recovery_value_head=True,direct_recovery_value_pooling='candidate_concat_raw',
        direct_recovery_delta_head=True,direct_recovery_delta_regime_experts=True,
        direct_recovery_evidence_calibrator=True,direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_source='physical_interaction',
        direct_recovery_evidence_interaction_hidden=16,direct_recovery_evidence_interaction_dropout=0.0,
        direct_recovery_evidence_dual_interaction_bridge=True,
        direct_recovery_evidence_unified_experts=True,direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=5,direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=False,direct_recovery_evidence_admission_prior_mode='joint_reserve',
        direct_recovery_evidence_frontier=True,
    ).eval()

def test_model_exposes_distinct_dual_contexts_without_regime_input():
    m=_model(); assert m.direct_recovery_evidence_dual_interaction_bridge
    x=torch.randn(6,FlatFeatureLayout().total_dim)
    groups=torch.tensor([[0],[0],[0],[1],[1],[1]])
    nominal=torch.tensor([1.,0.,0.,1.,0.,0.])
    with torch.no_grad():
        out=m(x,bucket_id=torch.ones(6,dtype=torch.long),group_index=groups,is_nominal=nominal,direct_only=True)
    b=out['direct_recovery_evidence_benefit_interaction_context']
    h=out['direct_recovery_evidence_harm_interaction_context']
    assert b.shape==h.shape
    assert torch.equal(b[nominal.bool()],torch.zeros_like(b[nominal.bool()]))
    assert torch.equal(h[nominal.bool()],torch.zeros_like(h[nominal.bool()]))
    assert not torch.allclose(b[~nominal.bool()],h[~nominal.bool()])

def test_v4840_main_reverts_unbounded_and_preregisters_clean_2x2():
    main=(ROOT/'scripts/run_v48_40_dcfr_dedicated.sh').read_text()
    arm=(ROOT/'scripts/run_v48_40_dcfr_ablation_arm.sh').read_text()
    assert 'EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false' in main
    assert 'EVIDENCE_UNBOUNDED_HARM_FACTORS=false' in main
    assert 'EVIDENCE_DUAL_INTERACTION_BRIDGE=true' in main
    assert 'FACTOR_COMPONENT_MARGIN_TARGET_MODE=frontier_tanh' in main
    assert 'EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve' in main
    assert 'PROPOSAL_TOP_K=5' in main
    for token in ('A)','B)','C)','D)'):
        assert token in arm
    assert 'regime' not in main.lower().replace('regime-free','') or 'regime-free' in main.lower()
