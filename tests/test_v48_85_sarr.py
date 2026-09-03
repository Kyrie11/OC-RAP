import torch
from ocrap.models.ocrap import OCRAPModel
from ocrap.models.encoders import FlatFeatureLayout


def make_model(state=False):
    L=FlatFeatureLayout(feature_max_agents=2)
    return OCRAPModel(
        L.total_dim,num_roots=3,num_options=2,d_model=32,d_obs=8,
        encoder_type='structured_transformer',feature_layout=L.__dict__,
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_action_response_adapter=True,
        direct_recovery_semantic_witness_action_response_state_conditioning=state,
    )


def test_v4885_same_capacity_and_zero_init():
    q=make_model(False); r=make_model(True)
    qw=q.direct_absolute_action_response_adapter.action_projection
    rw=r.direct_absolute_action_response_adapter.action_projection
    assert qw.shape==rw.shape==(2,32,q.direct_candidate_physical_feature_dim)
    assert torch.count_nonzero(qw)==0 and torch.count_nonzero(rw)==0
    assert sum(p.numel() for p in q.direct_absolute_action_response_adapter.parameters()) == sum(p.numel() for p in r.direct_absolute_action_response_adapter.parameters())


def test_v4885_nominal_zero_and_state_gate_only_modulates_action():
    q=make_model(False); r=make_model(True)
    qa=q.direct_absolute_action_response_adapter; ra=r.direct_absolute_action_response_adapter
    D,H=qa.action_dim,qa.d_model
    feat=torch.linspace(-0.037,0.061,D)
    hidden=(1.0+0.017*torch.arange(H)).view(H,1)
    with torch.no_grad():
        for c,scale in enumerate((1.0,1.37)):
            qa.action_projection[c].copy_(scale*hidden*feat.view(1,D))
        ra.action_projection.copy_(qa.action_projection)
    x=torch.stack((torch.zeros(D),torch.linspace(-1.7,2.3,D),torch.sin(torch.linspace(-1.2,2.0,D))+0.19*torch.linspace(-1,1,D)))
    base=torch.linspace(-1.5,1.9,H)
    roots=torch.stack([
        torch.stack((base,0.7*base.flip(0),torch.cos(base))),
        torch.stack((1.2*base+0.1,-0.9*base+0.2,torch.sin(base))),
        torch.stack((0.5*base-0.3,1.1*base.flip(0),torch.tanh(base))),
    ])
    margins=torch.tensor([[1.,1.,1.],[-1.,-1.,-1.],[1.,-1.,1.]])
    yq=qa(x,roots,margins); yr=ra(x,roots,margins)
    assert torch.equal(yq[0],torch.zeros_like(yq[0]))
    assert torch.equal(yr[0],torch.zeros_like(yr[0]))
    assert torch.isfinite(yq).all() and torch.isfinite(yr).all()
    assert float(yq[1:].abs().amax().detach()) > 1e-5
    assert float((yr[1:]-yq[1:]).abs().amax().detach()) > 1e-5


def test_v4885_nominal_anchor_fail_closed():
    m=make_model(False)
    vals=torch.randn(3,2,4); groups=torch.tensor([[1,2,3],[1,2,3],[1,2,3]])
    bad=torch.tensor([0.,0.,0.])
    got=m._nominal_group_anchor(vals,groups,bad)
    assert torch.count_nonzero(got)==0


def test_v4885_mutual_exclusion_with_root_tail():
    L=FlatFeatureLayout(feature_max_agents=2)
    try:
        OCRAPModel(L.total_dim,num_roots=3,num_options=2,d_model=32,d_obs=8,encoder_type='structured_transformer',feature_layout=L.__dict__,direct_recovery_absolute_semantic_witness_correction=True,direct_recovery_semantic_witness_action_response_adapter=True,direct_recovery_semantic_witness_root_tail_source=True)
    except ValueError as e:
        assert 'mutually exclusive' in str(e)
    else:
        raise AssertionError('expected mutual-exclusion failure')
