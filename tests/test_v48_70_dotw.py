from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from ocrap.cli.train import _semantic_witness_checkpoint_feature_contract
from ocrap.models.data import (
    DIRECT_OCCUPANCY_TEMPERED_RECOVERY_WITNESS_FEATURE_DIM,
    DIRECT_OCCUPANCY_TEMPERED_RECOVERY_WITNESS_FEATURE_SCHEMA,
    OPTION_FEATURE_DIM,
    direct_semantic_recovery_witness_features_from_sample,
    option_features_from_sample,
)
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout():
    return FlatFeatureLayout(feature_max_agents=2)


def _sample(*, agent_ax: float = 0.0):
    ego = np.zeros(9, np.float32); ego[6] = 4.0; ego[7] = 4.8; ego[8] = 2.0
    states = np.zeros((10, 9), np.float32)
    states[:, 0] = np.arange(1, 11) * 0.4; states[:, 6] = 4.0; states[:, 7] = 4.8; states[:, 8] = 2.0
    controls = np.zeros((9, 4), np.float32)
    hist = np.zeros((1, 2, 16), np.float32)
    # Observable current agent state. Positive ax makes the bounded-CA
    # counterfactual close faster than CV for a same-lane lead agent.
    hist[0, 1, 0] = 18.0; hist[0, 1, 3] = 0.0; hist[0, 1, 5] = agent_ax
    hist[0, 1, 10] = 4.8; hist[0, 1, 11] = 2.0
    return {
        'ego_state': ego, 'prefix_states': states, 'prefix_controls': controls,
        'agent_history': hist, 'agent_valid': np.asarray([[1, 1]], bool),
        'recovery_modes': np.asarray(['stop', 'lateral_escape'], object),
        'recovery_params': np.asarray([[-5., 5., 0.], [3.5, 5., 1.5]], np.float32),
        'option_valid': np.asarray([1, 1], bool), 'prefix_macro_id': 0,
        'prefix_macro_name': 'candidate', 'prefix_param': np.zeros(0, np.float32),
        'utility': 0., 'feasible': 1., 'hard_violation': 0., 'harm_proxy': 0.,
    }


def _cfg(*, demand=False, soft_occ=False, robust=False):
    return {
        'sample_rate_hz': 10.0, 'recovery_horizon_s': 4.0, 'prefix_horizon_s': 1.0,
        'route_dev_max_m': 2.5,
        'control_limits': {'a_max': 3.0, 'a_min': -6.0, 'delta_max': 0.55, 'j_max': 6.0, 'steer_rate_max': 0.5},
        'model': {
            'feature_max_agents': 2,
            'direct_recovery_semantic_witness_route_alignment': True,
            'direct_recovery_semantic_witness_reentry_alignment': True,
            'direct_recovery_semantic_witness_control_projection': True,
            'direct_recovery_semantic_witness_boundary_transport': False,
            'direct_recovery_semantic_witness_projection_fidelity_weighting': True,
            'direct_recovery_semantic_witness_demand_normalized_fidelity': demand,
            'direct_recovery_semantic_witness_robust_occupancy': robust,
            'direct_recovery_semantic_witness_soft_occupancy_disagreement': soft_occ,
        },
        'default_available_distance_m': 60.0,
    }


def _model(*, demand=False, soft_occ=False):
    L = _layout()
    return OCRAPModel(
        input_dim=L.total_dim, num_roots=3, num_options=2, d_model=16, d_obs=8,
        encoder_type='structured_transformer', feature_layout=asdict(L), num_layers=1,
        num_heads=4, dropout=0.0, option_feature_dim=OPTION_FEATURE_DIM,
        direct_recovery_value_head=True,
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_active_set_alignment=True,
        direct_recovery_semantic_witness_path_stop_alignment=False,
        direct_recovery_semantic_witness_classlocal_transport=False,
        direct_recovery_semantic_witness_route_alignment=True,
        direct_recovery_semantic_witness_reentry_alignment=True,
        direct_recovery_semantic_witness_control_projection=True,
        direct_recovery_semantic_witness_boundary_transport=False,
        direct_recovery_semantic_witness_projection_fidelity_weighting=True,
        direct_recovery_semantic_witness_demand_normalized_fidelity=demand,
        direct_recovery_semantic_witness_robust_occupancy=False,
        direct_recovery_semantic_witness_soft_occupancy_disagreement=soft_occ,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _opt(batch=1):
    z = torch.from_numpy(option_features_from_sample(_sample())).float()
    return z.unsqueeze(0).repeat(batch, 1, 1)


def _force_support_and_margins(m):
    m.root_logit_head.forward = MethodType(
        lambda self, z: torch.zeros((*z.shape[:-1], 1), device=z.device, dtype=z.dtype),
        m.root_logit_head,
    )
    m.obs_embed_head.forward = MethodType(
        lambda self, z: torch.zeros((*z.shape[:-1], 8), device=z.device, dtype=z.dtype),
        m.obs_embed_head,
    )
    def mf(self, z):
        vals = torch.zeros((1, 3, 2), device=z.device, dtype=z.dtype)
        return vals.expand(z.shape[0], -1, -1).unsqueeze(-1)
    m.margin_head.forward = MethodType(mf, m.margin_head)


def _features15(*, demand_units=(0.0, 1.0), occ_units=(0.0, 1.0)):
    f = torch.full((1, 2, 15), 0.6, dtype=torch.float32)
    f[..., 4] = -float(np.tanh(1.0))  # raw projection violation u=1
    f[..., 8] = 0.3; f[..., 9] = 0.3; f[..., 11] = 0.0
    f[..., 12] = 0.6; f[..., 13] = 0.6
    for j,d in enumerate(demand_units):
        f[0,j,1] = float(np.tanh(0.2))
        f[0,j,2] = float(np.tanh(0.2 + d))
    for j,o in enumerate(occ_units):
        f[0,j,14] = float(np.tanh(o))
    return f


def _support(model, feat):
    _force_support_and_margins(model)
    x=torch.zeros((1,_layout().total_dim)); rv=torch.ones((1,3),dtype=torch.bool); ov=torch.ones((1,2),dtype=torch.bool)
    out=model._direct_semantic_witness_absolute_feasibility(model._scene_tokens(x),x,_opt(),feat,root_valid=rv,option_valid=ov)
    assert out is not None
    return out[4], out[5]


def test_v4870_schema6_checkpoint_contract():
    base={
        'direct_recovery_absolute_semantic_witness_correction':True,
        'direct_recovery_semantic_witness_route_alignment':True,
        'direct_recovery_semantic_witness_reentry_alignment':True,
        'direct_recovery_semantic_witness_control_projection':True,
        'direct_recovery_semantic_witness_projection_fidelity_weighting':True,
    }
    assert _semantic_witness_checkpoint_feature_contract({**base,'direct_recovery_semantic_witness_soft_occupancy_disagreement':True}) == (6,'demand_occupancy_tempered_projected_recovery_witness')
    assert _semantic_witness_checkpoint_feature_contract({**base,'direct_recovery_semantic_witness_demand_normalized_fidelity':True,'direct_recovery_semantic_witness_soft_occupancy_disagreement':True}) == (6,'demand_occupancy_tempered_projected_recovery_witness')
    assert DIRECT_OCCUPANCY_TEMPERED_RECOVERY_WITNESS_FEATURE_SCHEMA == 6
    assert DIRECT_OCCUPANCY_TEMPERED_RECOVERY_WITNESS_FEATURE_DIM == 15


def test_v4870_soft_occupancy_appends_only_one_coordinate_and_keeps_first14_cv_exact():
    base=direct_semantic_recovery_witness_features_from_sample(_sample(agent_ax=-2.0),_cfg(soft_occ=False),num_options=2)
    soft=direct_semantic_recovery_witness_features_from_sample(_sample(agent_ax=-2.0),_cfg(soft_occ=True),num_options=2)
    assert base.shape==(2,14) and soft.shape==(2,15)
    assert np.array_equal(base,soft[:,:14])
    # At least one option sees CV optimism under the observed acceleration hypothesis.
    assert np.any(soft[:,14] > 0.0)


def test_v4870_soft_occupancy_is_not_v4868_hard_min():
    soft=direct_semantic_recovery_witness_features_from_sample(_sample(agent_ax=-2.0),_cfg(soft_occ=True,robust=False),num_options=2)
    hard=direct_semantic_recovery_witness_features_from_sample(_sample(agent_ax=-2.0),_cfg(soft_occ=False,robust=True),num_options=2)
    cv=direct_semantic_recovery_witness_features_from_sample(_sample(agent_ax=-2.0),_cfg(soft_occ=False,robust=False),num_options=2)
    # Soft mode preserves the CV physical certificate exactly; hard U68 changes it.
    assert np.array_equal(soft[:,:14],cv)
    assert not np.array_equal(hard,cv)


def test_v4870_support_factorizes_projection_demand_and_occupancy_trust_without_sign_change():
    t=_model(demand=False,soft_occ=False).eval()
    d=_model(demand=True,soft_occ=False).eval()
    e=_model(demand=False,soft_occ=True).eval()
    g=_model(demand=True,soft_occ=True).eval()
    feat=_features15(demand_units=(0.0,1.0),occ_units=(0.0,1.0))
    vt,st=_support(t,feat[:,:,:14]); vd,sd=_support(d,feat[:,:,:14]); ve,se=_support(e,feat); vg,sg=_support(g,feat)
    assert torch.equal(vt>0,vd>0) and torch.equal(vt>0,ve>0) and torch.equal(vt>0,vg>0)
    # option0: u=1,d=0,occ=0 -> all factors 1/2
    assert torch.allclose(st[...,0],torch.tensor([[0.5]]),atol=2e-5)
    assert torch.allclose(sd[...,0],st[...,0],atol=2e-5)
    assert torch.allclose(se[...,0],st[...,0],atol=2e-5)
    assert torch.allclose(sg[...,0],st[...,0],atol=2e-5)
    # option1: T=1/2; D=(1+1)/(1+1+1)=2/3; occ trust=1/(1+1)=1/2.
    assert torch.allclose(st[...,1],torch.tensor([[0.5]]),atol=2e-5)
    assert torch.allclose(sd[...,1],torch.tensor([[2/3]]),atol=2e-5)
    assert torch.allclose(se[...,1],torch.tensor([[0.25]]),atol=2e-5)
    assert torch.allclose(sg[...,1],torch.tensor([[1/3]]),atol=2e-5)


def test_v4870_soft_occupancy_requires_projected_fidelity_path():
    L=_layout()
    with pytest.raises(ValueError,match='soft occupancy disagreement'):
        OCRAPModel(
            input_dim=L.total_dim,num_roots=3,num_options=2,d_model=16,d_obs=8,
            encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,dropout=0.0,
            option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,
            direct_recovery_absolute_semantic_witness_correction=True,
            direct_recovery_semantic_witness_control_projection=True,
            direct_recovery_semantic_witness_projection_fidelity_weighting=False,
            direct_recovery_semantic_witness_soft_occupancy_disagreement=True,
        )



def test_v4870_checkpoint_schema6_and_flags_roundtrip(tmp_path: Path):
    from ocrap.models.inference import load_model_bundle
    m=_model(demand=True,soft_occ=True).eval(); L=_layout()
    model_cfg={
        'transformer_layers':1,'transformer_heads':4,'dropout':0.0,
        'encoder_type':'structured_transformer','option_feature_dim':OPTION_FEATURE_DIM,
        'direct_recovery_value_head':True,
        'direct_recovery_absolute_semantic_witness_correction':True,
        'direct_recovery_semantic_witness_active_set_alignment':True,
        'direct_recovery_semantic_witness_path_stop_alignment':False,
        'direct_recovery_semantic_witness_classlocal_transport':False,
        'direct_recovery_semantic_witness_route_alignment':True,
        'direct_recovery_semantic_witness_reentry_alignment':True,
        'direct_recovery_semantic_witness_control_projection':True,
        'direct_recovery_semantic_witness_boundary_transport':False,
        'direct_recovery_semantic_witness_projection_fidelity_weighting':True,
        'direct_recovery_semantic_witness_demand_normalized_fidelity':True,
        'direct_recovery_semantic_witness_robust_occupancy':False,
        'direct_recovery_semantic_witness_soft_occupancy_disagreement':True,
        'direct_recovery_evidence_native_certificate_preservation':True,
    }
    ckpt={'model_state':m.state_dict(),'input_dim':L.total_dim,'num_roots':3,'num_options':2,'d_model':16,'d_obs':8,'tau_obs':1.0,'encoder_type':'structured_transformer','feature_layout':asdict(L),'d_signature':0,'d_future_signature':0,'option_feature_dim':OPTION_FEATURE_DIM,**model_cfg,'direct_recovery_absolute_semantic_witness_feature_schema':6,'direct_recovery_absolute_semantic_witness_feature_source':'demand_occupancy_tempered_projected_recovery_witness','cfg':{'sample_rate_hz':10.0,'recovery_horizon_s':4.0,'model':model_cfg,'runtime':{'device':'cpu'}}}
    out=tmp_path/'dotw.pt'; torch.save(ckpt,out); bundle=load_model_bundle(out)
    assert bundle.model.direct_recovery_semantic_witness_soft_occupancy_disagreement is True
    assert bundle.model.direct_recovery_semantic_witness_demand_normalized_fidelity is True
    assert bundle.model.direct_recovery_semantic_witness_robust_occupancy is False

def test_v4870_runner_contract_is_factorial_and_keeps_forbidden_branches_off():
    p=Path(__file__).resolve().parents[1]/'scripts'/'run_v48_70_dcp_drfc_bcde_rifa_dotw_two_gpu.sh'
    text=p.read_text()
    assert 'E70_OCCSOFT' in text and 'G70_Main_OCDOTW' in text
    assert 'SEMANTIC_WITNESS_SOFT_OCCUPANCY_DISAGREEMENT=true' in text
    assert 'SEMANTIC_WITNESS_ROBUST_OCCUPANCY=false' in text
    assert 'SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false' in text
    assert 'PROPOSAL_TOP_K=5' in text and 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' in text
