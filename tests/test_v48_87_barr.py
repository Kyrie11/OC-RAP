from __future__ import annotations
import torch
from ocrap.models.ocrap import ObservationConsistentBilinearActionRootResponseAdapter

def test_v4887_bilinear_adapter_capacity_zero_init_and_gradient():
    torch.manual_seed(3)
    m=ObservationConsistentBilinearActionRootResponseAdapter(141,192,rank=51)
    assert m.trainable_parameter_count==53550
    assert m.trainable_parameter_count <= 2*192*141  # Q85 = 54,144
    a=torch.randn(5,141);r=torch.randn(5,8,192);margin=torch.ones(5,8);margin[:,4:]=-1
    out=m(a,r,margin)
    assert torch.equal(out,torch.zeros_like(out))
    nominal=m(torch.zeros_like(a),r,margin)
    assert torch.equal(nominal,torch.zeros_like(nominal))
    loss=(out-torch.randn_like(out)).pow(2).mean();loss.backward()
    assert m.output_factor.grad is not None
    assert float(m.output_factor.grad.abs().sum())>0.0

def test_v4887_response_is_root_conditioned_not_posthoc_state_gate():
    torch.manual_seed(4)
    m=ObservationConsistentBilinearActionRootResponseAdapter(141,192,rank=51)
    with torch.no_grad():m.output_factor.normal_(0,0.04)
    a=torch.randn(2,141);r=torch.randn(2,8,192);margin=torch.ones(2,8)
    x=m(a,r,margin)
    r2=r.clone();r2[:,2]+=1.7
    y=m(a,r2,margin)
    assert float((x[:,2]-y[:,2]).detach().abs().sum())>1e-7
    # Other roots are independent of the changed root token.
    keep=[0,1,3,4,5,6,7]
    assert torch.allclose(x[:,keep],y[:,keep],atol=1e-6,rtol=1e-6)

def test_v4887_reserve_debt_channels_are_shared_not_regime_conditioned():
    torch.manual_seed(5)
    m=ObservationConsistentBilinearActionRootResponseAdapter(141,192,rank=51)
    with torch.no_grad():
        m.output_factor[0].fill_(0.02)
        m.output_factor[1].fill_(-0.02)
    a=torch.randn(1,141);r=torch.randn(1,2,192)
    reserve=m(a,r,torch.tensor([[1.0,1.0]]))
    debt=m(a,r,torch.tensor([[-1.0,-1.0]]))
    assert not torch.allclose(reserve,debt)
