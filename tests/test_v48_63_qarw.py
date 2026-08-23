from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from types import MethodType
import numpy as np
import torch

from ocrap.cli.train import _absolute_feasibility_bce
from ocrap.models.data import DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA, OPTION_FEATURE_DIM, direct_common_recovery_witness_features_from_sample, option_features_from_sample
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout(): return FlatFeatureLayout(feature_max_agents=2)

def _model(num_options=2):
    L=_layout()
    return OCRAPModel(input_dim=L.total_dim,num_roots=3,num_options=num_options,d_model=16,d_obs=8,
        encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,dropout=0.0,
        option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,
        direct_recovery_absolute_quantifier_witness_correction=True,
        direct_recovery_evidence_native_certificate_preservation=True)

def _sample(priv=False):
    ego=np.zeros(9,np.float32);ego[6]=4.;ego[7]=4.8;ego[8]=2.
    states=np.zeros((10,9),np.float32);states[:,0]=np.arange(1,11)*.4;states[:,6]=4.;states[:,7]=4.8;states[:,8]=2.
    controls=np.zeros((9,4),np.float32)
    hist=np.zeros((1,2,16),np.float32);hist[0,1,0]=10.;hist[0,1,10]=4.8;hist[0,1,11]=2.
    d={'ego_state':ego,'prefix_states':states,'prefix_controls':controls,'agent_history':hist,'agent_valid':np.asarray([[1,1]],bool),
       'recovery_modes':np.asarray(['stop','lateral_escape'],object),'recovery_params':np.asarray([[-5.,5.,0.],[3.5,5.,1.5]],np.float32),
       'option_valid':np.asarray([1,1],bool),'prefix_macro_id':0,'prefix_macro_name':'candidate','prefix_param':np.zeros(0,np.float32),
       'utility':0.,'feasible':1.,'hard_violation':0.,'harm_proxy':0.}
    if priv: d.update({'m_star':np.ones((3,2),np.float32)*99,'root_future_signature':np.ones((3,8),np.float32)*77,'r_dep_star':np.float32(-999),'bucket_id':np.int64(2)})
    return d

def _cfg(): return {'sample_rate_hz':10.0,'recovery_horizon_s':4.0,'model':{'feature_max_agents':2}}
def _field(): return torch.from_numpy(direct_common_recovery_witness_features_from_sample(_sample(),_cfg(),num_options=2)).float()
def _opt(batch=1):
    z=torch.from_numpy(option_features_from_sample(_sample())).float()
    return z.unsqueeze(0).repeat(batch,1,1)

def _force_common_support(m):
    m.root_logit_head.forward=MethodType(lambda self,z: torch.zeros((*z.shape[:-1],1),device=z.device,dtype=z.dtype),m.root_logit_head)
    m.obs_embed_head.forward=MethodType(lambda self,z: torch.zeros((*z.shape[:-1],8),device=z.device,dtype=z.dtype),m.obs_embed_head)
    def margins(self,z):
        vals=torch.tensor([[[1.,1.],[1.,1.],[1.,1.]]],device=z.device,dtype=z.dtype)
        return vals.unsqueeze(-1).expand(z.shape[0],-1,-1,1)
    m.margin_head.forward=MethodType(margins,m.margin_head)

def test_qarw_reuses_privilege_free_common_witness_field():
    a=direct_common_recovery_witness_features_from_sample(_sample(),_cfg(),num_options=2)
    b=direct_common_recovery_witness_features_from_sample(_sample(True),_cfg(),num_options=2)
    assert a.shape==(2,10) and np.isfinite(a).all() and np.array_equal(a,b)

def test_qarw_exists_success_blocks_universal_failure():
    m=_model().eval();_force_common_support(m);L=_layout();x=torch.zeros((1,L.total_dim));mem=m._scene_tokens(x)
    feat=torch.zeros((1,2,10))
    # option0 valid finite-time witness, option1 physical failure.
    feat[0,0]=torch.tensor([.5,.5,.5,.5,.5,.5,.5,.5,.2,.2])
    feat[0,1]=torch.tensor([-.5,-.4,-.2,-.3,-.2,-.4,-.3,-.2,-.1,-.1])
    out=m._direct_quantifier_witness_absolute_feasibility(mem,x,_opt(),feat,root_valid=torch.ones((1,3),dtype=torch.bool),option_valid=torch.ones((1,2),dtype=torch.bool))
    assert out is not None
    viability,support,best,fail,pos=out[4],out[5],out[6],out[7],out[8]
    assert float(viability[0,0])>0 and float(viability[0,1])<0
    assert float(support.min())>.9 and float(best[0])>0 and float(fail[0])==0.0 and float(pos[0])>=1

def test_qarw_all_common_options_fail_enables_candidate_veto():
    m=_model().eval();_force_common_support(m);L=_layout();x=torch.zeros((1,L.total_dim));mem=m._scene_tokens(x)
    feat=torch.full((1,2,10),-.4)
    # negative terminal/gain/floors guarantee physical failure.
    out=m._direct_quantifier_witness_absolute_feasibility(mem,x,_opt(),feat,root_valid=torch.ones((1,3),dtype=torch.bool),option_valid=torch.ones((1,2),dtype=torch.bool))
    assert out is not None
    assert float(out[6][0])<0 and float(out[7][0])>0 and float(out[8][0])==0

def test_qarw_zero_gain_is_execution_exact_native_b():
    torch.manual_seed(4863);m=_model().eval();L=_layout();x=torch.randn((3,L.total_dim));mem=m._scene_tokens(x)
    _,native=m._direct_recovery_option_compatibility_evidence(mem,x,_opt(3),root_valid=torch.ones((3,3),dtype=torch.bool),option_valid=torch.ones((3,2),dtype=torch.bool))
    feat=_field().unsqueeze(0).repeat(3,1,1)
    out=m._direct_quantifier_witness_absolute_feasibility(mem,x,_opt(3),feat,root_valid=torch.ones((3,3),dtype=torch.bool),option_valid=torch.ones((3,2),dtype=torch.bool))
    assert out is not None and torch.allclose(out[1],native[:,1],atol=0,rtol=0) and torch.equal(out[3],torch.zeros(2))

def test_qarw_bce_gradient_isolated_to_two_shared_gains():
    torch.manual_seed(4863);m=_model().train()
    for n,p in m.named_parameters():p.requires_grad_(n=='direct_absolute_quantifier_witness_gain')
    L=_layout();x=torch.randn((4,L.total_dim));mem=m._scene_tokens(x);base=_field();feat=torch.stack([base,base.flip(0),base,base.flip(0)])
    out=m._direct_quantifier_witness_absolute_feasibility(mem,x,_opt(4),feat,root_valid=torch.ones((4,3),dtype=torch.bool),option_valid=torch.ones((4,2),dtype=torch.bool))
    assert out is not None
    loss=_absolute_feasibility_bce({'direct_recovery_absolute_feasibility_logit':out[0]}, {'r_dep_star':torch.tensor([-.5,.5,-.5,.5]),'is_nominal':torch.zeros(4),'bucket_id':torch.tensor([1,1,2,2]),'time_index':torch.arange(4)})
    loss.backward();g=m.direct_absolute_quantifier_witness_gain.grad
    assert g is not None and torch.isfinite(g).all() and torch.any(g!=0) and sum(p.numel() for p in m.parameters() if p.requires_grad)==2

def test_qarw_fails_closed_and_mutually_exclusive():
    m=_model().eval();L=_layout();x=torch.zeros((1,L.total_dim));mem=m._scene_tokens(x)
    try:m._direct_quantifier_witness_absolute_feasibility(mem,x,_opt(),None)
    except RuntimeError as e:assert 'OC-CWRF features missing' in str(e)
    else:raise AssertionError('missing side channel must fail closed')
    for extra in ({'direct_recovery_absolute_feasibility_head':True},{'direct_recovery_absolute_option_margin_correction':True},{'direct_recovery_absolute_physical_headroom_correction':True},{'direct_recovery_absolute_executable_witness_correction':True},{'direct_recovery_absolute_common_witness_correction':True}):
        try:OCRAPModel(input_dim=L.total_dim,num_roots=2,num_options=2,d_model=16,d_obs=8,encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,direct_recovery_absolute_quantifier_witness_correction=True,**extra)
        except ValueError as e:assert 'mutually exclusive' in str(e)
        else:raise AssertionError('OC-QARW must remain a single-axis source')

def _ckpt(model,schema):
    L=_layout();d={'model_state':model.state_dict(),'input_dim':L.total_dim,'num_roots':3,'num_options':2,'d_model':16,'d_obs':8,'tau_obs':1.0,'encoder_type':'structured_transformer','feature_layout':asdict(L),'d_signature':0,'d_future_signature':0,'option_feature_dim':OPTION_FEATURE_DIM,'direct_recovery_value_head':True,'direct_recovery_absolute_quantifier_witness_correction':True,'direct_recovery_evidence_native_certificate_preservation':True,'cfg':{'sample_rate_hz':10.0,'recovery_horizon_s':4.0,'model':{'transformer_layers':1,'transformer_heads':4,'dropout':0.0,'encoder_type':'structured_transformer','option_feature_dim':OPTION_FEATURE_DIM,'direct_recovery_value_head':True,'direct_recovery_absolute_quantifier_witness_correction':True,'direct_recovery_evidence_native_certificate_preservation':True},'runtime':{'device':'cpu'}}}
    if schema is not None:d.update({'direct_recovery_absolute_quantifier_witness_feature_schema':schema,'direct_recovery_absolute_quantifier_witness_feature_source':'quantifier_aligned_common_finite_time_recovery_witness'})
    return d

def test_qarw_checkpoint_roundtrip_schema_and_diagnostics(tmp_path):
    from ocrap.models.inference import load_model_bundle,predict_samples
    m=_model().eval();good=tmp_path/'good.pt';torch.save(_ckpt(m,DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA),good)
    b=load_model_bundle(good);preds=predict_samples([_sample(),_sample(True)],b)
    assert len(preds)==2 and all(p.direct_recovery_quantifier_best_common_viability is not None for p in preds)
    bad=tmp_path/'bad.pt';torch.save(_ckpt(m,None),bad)
    try:load_model_bundle(bad)
    except RuntimeError as e:assert 'legacy/unknown OC-QARW checkpoint feature semantics' in str(e)
    else:raise AssertionError('schema-less OC-QARW checkpoint must fail closed')

def test_qarw_shell_plumbing_and_launcher_contract():
    root=Path(__file__).resolve().parents[1];train=(root/'scripts/train_ocrap_v48_trac_sr.sh').read_text();adapt=(root/'scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh').read_text();launch=(root/'scripts/run_v48_63_dcp_drfc_bcde_rifa_ocqarw_two_gpu.sh').read_text()
    assert 'ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION' in train and 'direct_recovery_absolute_quantifier_witness_correction' in train
    assert 'ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION="$ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION"' in adapt
    for s in ('EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_quantifier_witness_gain','STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_quantifier_witness_gain','ABSOLUTE_COMMON_WITNESS_CORRECTION=false','ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION=true','MAX_EVIDENCE_CALIBRATOR_PARAMS=2','PROPOSAL_TOP_K=5','ABSOLUTE_FEASIBILITY_THRESHOLD=0.5'):assert s in launch
    assert 'EVIDENCE_CENTER' not in launch.upper() and 'PRED_ADV_CENTER' not in launch.upper()
