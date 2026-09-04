from __future__ import annotations
import torch
import pytest
from ocrap.models.ocrap import OCRAPModel
from ocrap.models.encoders import FlatFeatureLayout


def make_model():
    L=FlatFeatureLayout(feature_max_agents=2)
    return OCRAPModel(
        L.total_dim,num_roots=4,num_options=3,d_model=32,d_obs=8,
        encoder_type='structured_transformer',feature_layout=L.__dict__,
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_quotient_tail_response=True,
        direct_recovery_semantic_witness_boundary_transport=False,
    )


def test_v4888_qtrr_capacity_zero_init_and_family_isolation():
    m=make_model()
    w=m.direct_absolute_quotient_tail_response_weight
    assert w is not None
    assert tuple(w.shape)==(2,m.direct_candidate_physical_feature_dim)
    assert w.numel()==282
    assert torch.count_nonzero(w)==0
    assert m.direct_absolute_action_response_adapter is None
    assert m.direct_absolute_root_tail_source_scale is None
    assert m.direct_absolute_structured_tail_field_weight is None
    assert m.direct_absolute_semantic_witness_gain is None


def test_v4888_quotient_direction_removes_option_translation_and_is_unit_norm():
    torch.manual_seed(88)
    B,K,L=5,4,3
    p=torch.rand(B,K); p=p/p.sum(dim=1,keepdim=True)
    g=torch.randn(B,K,L)
    d=OCRAPModel._quotient_tail_direction_from_cotangent(p,g)
    # The closed option-translation degree is removed exactly up to fp error.
    translated=(p.unsqueeze(-1)*d).sum(dim=1)
    assert float(translated.abs().max()) < 2e-6
    norms=d.square().sum(dim=(1,2)).sqrt()
    assert torch.allclose(norms,torch.ones_like(norms),atol=2e-6,rtol=2e-6)
    # A pure row-space translation cotangent has no admissible quotient direction.
    c=torch.randn(B,1,L)
    pure=p.unsqueeze(-1)*c
    z=OCRAPModel._quotient_tail_direction_from_cotangent(p,pure)
    assert float(z.abs().max()) < 2e-6


def test_v4888_signed_action_coefficient_is_nominal_zero_and_reserve_debt_shared():
    torch.manual_seed(89)
    m=make_model(); A=m.direct_candidate_physical_feature_dim
    with torch.no_grad():
        m.direct_absolute_quotient_tail_response_weight[0].copy_(torch.linspace(-.4,.7,A))
        m.direct_absolute_quotient_tail_response_weight[1].copy_(torch.linspace(.6,-.3,A))
    zero=torch.zeros(3,A)
    rdep=torch.tensor([1.0,-1.0,0.2])
    eta0=m._quotient_tail_response_coefficient(zero,rdep)
    assert torch.equal(eta0,torch.zeros_like(eta0))
    a=torch.randn(2,A)
    eta=m._quotient_tail_response_coefficient(a,torch.tensor([1.0,-1.0]))
    assert torch.isfinite(eta).all()
    assert float(eta.detach().abs().max())>1e-6


def test_v4888_qtrr_mutually_exclusive_with_closed_families():
    L=FlatFeatureLayout(feature_max_agents=2)
    base=dict(
        input_dim=L.total_dim,num_roots=3,num_options=2,d_model=32,d_obs=8,
        encoder_type='structured_transformer',feature_layout=L.__dict__,
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_quotient_tail_response=True,
    )
    with pytest.raises(ValueError):
        OCRAPModel(**base,direct_recovery_semantic_witness_action_response_adapter=True)
    with pytest.raises(ValueError):
        OCRAPModel(**base,direct_recovery_semantic_witness_root_tail_source=True)
    with pytest.raises(ValueError):
        OCRAPModel(**base,direct_recovery_semantic_witness_boundary_transport=True)
