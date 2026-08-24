from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from ocrap.cli.train import _absolute_feasibility_bce
from ocrap.data.schema import CandidatePrefix, RecoveryOption
from ocrap.models.data import (
    DIRECT_ACTIVE_CONSTRAINT_RECOVERY_WITNESS_FEATURE_SCHEMA,
    DIRECT_PROJECTED_BOUNDARY_RECOVERY_WITNESS_FEATURE_SCHEMA,
    OPTION_FEATURE_DIM,
    direct_semantic_recovery_witness_features_from_sample,
    option_features_from_sample,
)
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel
from ocrap.simulation.teacher.controllers import rollout_recovery_controller


def _layout():
    return FlatFeatureLayout(feature_max_agents=2)


def _sample(privileged: bool = False):
    ego = np.zeros(9, np.float32); ego[6]=4.0; ego[7]=4.8; ego[8]=2.0
    states = np.zeros((10,9),np.float32); states[:,0]=np.arange(1,11)*0.4; states[:,6]=4.0; states[:,7]=4.8; states[:,8]=2.0
    controls = np.zeros((9,4),np.float32)
    hist=np.zeros((1,2,16),np.float32); hist[0,1,0]=30.0; hist[0,1,10]=4.8; hist[0,1,11]=2.0
    d={
        'ego_state':ego,'prefix_states':states,'prefix_controls':controls,
        'agent_history':hist,'agent_valid':np.asarray([[1,1]],bool),
        'recovery_modes':np.asarray(['stop','lateral_escape'],object),
        'recovery_params':np.asarray([[-5.,5.,0.],[3.5,5.,1.5]],np.float32),
        'option_valid':np.asarray([1,1],bool),'prefix_macro_id':0,'prefix_macro_name':'candidate',
        'prefix_param':np.zeros(0,np.float32),'utility':0.,'feasible':1.,'hard_violation':0.,'harm_proxy':0.,
    }
    if privileged:
        d.update({'m_star':np.ones((3,2),np.float32)*99.,'root_future_signature':np.ones((3,8),np.float32)*77.,'r_dep_star':np.float32(-999.),'bucket_id':np.int64(2)})
    return d


def _cfg(*, projection=False, boundary=False):
    return {
        'sample_rate_hz':10.0,'recovery_horizon_s':4.0,'route_dev_max_m':2.5,
        'control_limits':{'a_max':3.0,'a_min':-6.0,'delta_max':0.55,'j_max':6.0,'steer_rate_max':0.5},
        'model':{
            'feature_max_agents':2,
            'direct_recovery_semantic_witness_route_alignment':True,
            'direct_recovery_semantic_witness_reentry_alignment':True,
            'direct_recovery_semantic_witness_control_projection':projection,
            'direct_recovery_semantic_witness_boundary_transport':boundary,
        },'default_available_distance_m':60.0,
    }


def _model(*, projection=False, boundary=False):
    L=_layout()
    return OCRAPModel(
        input_dim=L.total_dim,num_roots=3,num_options=2,d_model=16,d_obs=8,
        encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,dropout=0.0,
        option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_active_set_alignment=True,
        direct_recovery_semantic_witness_path_stop_alignment=False,
        direct_recovery_semantic_witness_classlocal_transport=False,
        direct_recovery_semantic_witness_route_alignment=True,
        direct_recovery_semantic_witness_reentry_alignment=True,
        direct_recovery_semantic_witness_control_projection=projection,
        direct_recovery_semantic_witness_boundary_transport=boundary,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _opt(batch=1):
    z=torch.from_numpy(option_features_from_sample(_sample())).float()
    return z.unsqueeze(0).repeat(batch,1,1)


def _manual_features(batch=1):
    f=torch.full((batch,2,14),0.5,dtype=torch.float32)
    f[...,8]=0.2; f[...,9]=0.2; f[...,11]=0.0; f[...,12]=0.6; f[...,13]=0.6
    return f


def _force_support_and_margins(m, margins: torch.Tensor):
    m.root_logit_head.forward=MethodType(lambda self,z:torch.zeros((*z.shape[:-1],1),device=z.device,dtype=z.dtype),m.root_logit_head)
    m.obs_embed_head.forward=MethodType(lambda self,z:torch.zeros((*z.shape[:-1],8),device=z.device,dtype=z.dtype),m.obs_embed_head)
    def mf(self,z):
        vals=margins.to(device=z.device,dtype=z.dtype)
        if vals.ndim==2: vals=vals.unsqueeze(0)
        return vals.expand(z.shape[0],-1,-1).unsqueeze(-1)
    m.margin_head.forward=MethodType(mf,m.margin_head)


def test_pbrw_projected_controller_enforces_magnitude_jerk_and_rate_without_changing_default():
    d=_sample(); states=d['prefix_states']; ctrls=d['prefix_controls']
    prefix=CandidatePrefix(0,'candidate',np.zeros(0,np.float32),states,ctrls,0.0,True,0.0,0.0)
    opt=RecoveryOption(1,'lateral_escape',np.asarray([3.5,5.0,1.5],np.float32),True)
    cfg=_cfg(projection=True)
    a_s,a_c,_=rollout_recovery_controller(prefix,opt,40,cfg)
    p_s,p_c,diag=rollout_recovery_controller(prefix,opt,40,cfg,project_control_envelope=True)
    assert not np.array_equal(a_s,p_s)
    assert diag['projected_control_envelope'] is True
    lim=cfg['control_limits']
    assert np.max(p_c[:,0]) <= lim['a_max']+1e-6 and np.min(p_c[:,0]) >= lim['a_min']-1e-6
    assert np.max(np.abs(p_c[:,1])) <= lim['delta_max']+1e-6
    assert np.max(np.abs(p_c[:,2])) <= lim['j_max']+1e-5
    assert np.max(np.abs(p_c[:,3])) <= lim['steer_rate_max']+1e-5
    # Historical default path must remain unchanged when the kwarg is omitted/False.
    b_s,b_c,_=rollout_recovery_controller(prefix,opt,40,cfg,project_control_envelope=False)
    assert np.array_equal(a_s,b_s) and np.array_equal(a_c,b_c)


def test_pbrw_schema3_is_privilege_free_and_boundary_only_keeps_v4866_features_exact():
    v66=direct_semantic_recovery_witness_features_from_sample(_sample(),_cfg(),num_options=2)
    boundary=direct_semantic_recovery_witness_features_from_sample(_sample(),_cfg(boundary=True),num_options=2)
    projected=direct_semantic_recovery_witness_features_from_sample(_sample(),_cfg(projection=True),num_options=2)
    privileged=direct_semantic_recovery_witness_features_from_sample(_sample(True),_cfg(projection=True,boundary=True),num_options=2)
    projected2=direct_semantic_recovery_witness_features_from_sample(_sample(),_cfg(projection=True,boundary=True),num_options=2)
    assert DIRECT_ACTIVE_CONSTRAINT_RECOVERY_WITNESS_FEATURE_SCHEMA==2
    assert DIRECT_PROJECTED_BOUNDARY_RECOVERY_WITNESS_FEATURE_SCHEMA==3
    assert v66.shape==boundary.shape==projected.shape==(2,14)
    assert np.array_equal(v66,boundary), 'transport-only arm must not change side-channel features'
    assert np.array_equal(projected,privileged) and np.array_equal(projected,projected2)
    assert np.isfinite(projected).all()


def test_pbrw_projected_control_is_enforced_by_construction_not_rejected_as_a_barrier():
    m=_model(projection=True,boundary=False).eval()
    margins=torch.ones((3,2)); _force_support_and_margins(m,margins)
    feat=_manual_features(); feat[...,4]=-1.0  # raw historical control coordinate is deliberately bad
    x=torch.zeros((1,_layout().total_dim))
    out=m._direct_semantic_witness_absolute_feasibility(m._scene_tokens(x),x,_opt(),feat,
        root_valid=torch.ones((1,3),dtype=torch.bool),option_valid=torch.ones((1,2),dtype=torch.bool))
    assert out is not None
    assert torch.all(out[4] > 0), 'projected arm must not hard-veto on the pre-projection control certificate'
    assert torch.all(out[10][...,2] == 1.0), 'effective control barrier is satisfied by construction'


def test_pbrw_boundary_transport_is_monotone_bounded_and_gain_zero_exact():
    m=_model(projection=False,boundary=True).eval()
    base=torch.tensor([[-1.0,-1.0],[-0.5,-0.5],[-0.2,-0.2]],dtype=torch.float32)
    _force_support_and_margins(m,base)
    captured={}
    def compat(self,root_logits,obs_embeddings,margins,*args,**kwargs):
        captured['margins']=margins.detach().clone()
        # Minimal native certificate with p=sigmoid(LC-ish positive max) for this unit test.
        score=margins.amax(dim=(1,2))
        prob=torch.sigmoid(score)
        native=torch.stack([score,prob,score,score],dim=-1)
        return torch.zeros_like(score),native
    m._recovery_option_compatibility_signature=MethodType(compat,m)
    x=torch.zeros((1,_layout().total_dim)); feat=_manual_features()
    rv=torch.ones((1,3),dtype=torch.bool); ov=torch.ones((1,2),dtype=torch.bool)
    with torch.no_grad(): m.direct_absolute_semantic_witness_gain.copy_(torch.tensor([0.0,0.0]))
    m._direct_semantic_witness_absolute_feasibility(m._scene_tokens(x),x,_opt(),feat,root_valid=rv,option_valid=ov)
    assert torch.equal(captured['margins'],base.unsqueeze(0))
    with torch.no_grad(): m.direct_absolute_semantic_witness_gain.copy_(torch.tensor([2.0,0.0]))
    m._direct_semantic_witness_absolute_feasibility(m._scene_tokens(x),x,_opt(),feat,root_valid=rv,option_valid=ov)
    corrected=captured['margins'][0]
    cert_target=float(torch.atanh(torch.tensor(0.5)))
    assert torch.all(corrected >= base-1e-7)
    assert torch.allclose(corrected,torch.full_like(corrected,cert_target),atol=1e-5)
    # If native margins are already above the certified target, the residual is one-sided and preserves them.
    safer=torch.full((3,2),0.8,dtype=torch.float32); _force_support_and_margins(m,safer)
    m._direct_semantic_witness_absolute_feasibility(m._scene_tokens(x),x,_opt(),feat,root_valid=rv,option_valid=ov)
    assert torch.allclose(captured['margins'][0],safer,atol=0,rtol=0)


def test_pbrw_bce_gradient_stays_on_two_shared_gains():
    torch.manual_seed(4867); m=_model(projection=True,boundary=True).train()
    _force_support_and_margins(m,torch.full((3,2),-0.3))
    for n,p in m.named_parameters(): p.requires_grad_(n=='direct_absolute_semantic_witness_gain')
    x=torch.randn((4,_layout().total_dim)); out=m._direct_semantic_witness_absolute_feasibility(
        m._scene_tokens(x),x,_opt(4),_manual_features(4),
        root_valid=torch.ones((4,3),dtype=torch.bool),option_valid=torch.ones((4,2),dtype=torch.bool))
    assert out is not None
    loss=_absolute_feasibility_bce({'direct_recovery_absolute_feasibility_logit':out[0]},
        {'r_dep_star':torch.tensor([-.5,.5,-.5,.5]),'is_nominal':torch.zeros(4),'bucket_id':torch.tensor([1,1,2,2]),'time_index':torch.arange(4)})
    loss.backward(); g=m.direct_absolute_semantic_witness_gain.grad
    assert g is not None and torch.isfinite(g).all() and torch.any(g!=0)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad)==2


def test_pbrw_checkpoint_schema3_and_flags_roundtrip(tmp_path: Path):
    from ocrap.models.inference import load_model_bundle
    m=_model(projection=True,boundary=True).eval(); L=_layout()
    model_cfg={
      'transformer_layers':1,'transformer_heads':4,'dropout':0.0,'encoder_type':'structured_transformer','option_feature_dim':OPTION_FEATURE_DIM,
      'direct_recovery_value_head':True,'direct_recovery_absolute_semantic_witness_correction':True,
      'direct_recovery_semantic_witness_active_set_alignment':True,'direct_recovery_semantic_witness_path_stop_alignment':False,
      'direct_recovery_semantic_witness_classlocal_transport':False,'direct_recovery_semantic_witness_route_alignment':True,
      'direct_recovery_semantic_witness_reentry_alignment':True,'direct_recovery_semantic_witness_control_projection':True,
      'direct_recovery_semantic_witness_boundary_transport':True,'direct_recovery_evidence_native_certificate_preservation':True,
    }
    ckpt={'model_state':m.state_dict(),'input_dim':L.total_dim,'num_roots':3,'num_options':2,'d_model':16,'d_obs':8,'tau_obs':1.0,
      'encoder_type':'structured_transformer','feature_layout':asdict(L),'d_signature':0,'d_future_signature':0,'option_feature_dim':OPTION_FEATURE_DIM,
      **model_cfg,'direct_recovery_absolute_semantic_witness_feature_schema':3,
      'direct_recovery_absolute_semantic_witness_feature_source':'projected_boundary_common_executable_recovery_witness',
      'cfg':{'sample_rate_hz':10.0,'recovery_horizon_s':4.0,'model':model_cfg,'runtime':{'device':'cpu'}}}
    p=tmp_path/'pbrw.pt'; torch.save(ckpt,p); b=load_model_bundle(p)
    assert b.model.direct_recovery_semantic_witness_control_projection is True
    assert b.model.direct_recovery_semantic_witness_boundary_transport is True
    assert b.model.direct_recovery_semantic_witness_route_alignment is True
    assert b.model.direct_recovery_semantic_witness_reentry_alignment is True
