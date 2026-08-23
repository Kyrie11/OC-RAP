from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from types import MethodType
import numpy as np
import torch

from ocrap.cli.train import _absolute_feasibility_bce
from ocrap.models.data import (
    DIRECT_SEMANTIC_RECOVERY_WITNESS_FEATURE_SCHEMA,
    OPTION_FEATURE_DIM,
    direct_semantic_recovery_witness_features_from_sample,
    option_features_from_sample,
)
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout(): return FlatFeatureLayout(feature_max_agents=2)


def _model(num_options=2, *, active=True, path=True):
    L=_layout()
    return OCRAPModel(
        input_dim=L.total_dim,num_roots=3,num_options=num_options,d_model=16,d_obs=8,
        encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,dropout=0.0,
        option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_active_set_alignment=active,
        direct_recovery_semantic_witness_path_stop_alignment=path,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _sample(priv=False):
    ego=np.zeros(9,np.float32);ego[6]=4.;ego[7]=4.8;ego[8]=2.
    states=np.zeros((10,9),np.float32);states[:,0]=np.arange(1,11)*.4;states[:,6]=4.;states[:,7]=4.8;states[:,8]=2.
    controls=np.zeros((9,4),np.float32)
    hist=np.zeros((1,2,16),np.float32);hist[0,1,0]=30.;hist[0,1,10]=4.8;hist[0,1,11]=2.
    d={'ego_state':ego,'prefix_states':states,'prefix_controls':controls,'agent_history':hist,'agent_valid':np.asarray([[1,1]],bool),
       'recovery_modes':np.asarray(['stop','lateral_escape'],object),'recovery_params':np.asarray([[-5.,5.,0.],[3.5,5.,1.5]],np.float32),
       'option_valid':np.asarray([1,1],bool),'prefix_macro_id':0,'prefix_macro_name':'candidate','prefix_param':np.zeros(0,np.float32),
       'utility':0.,'feasible':1.,'hard_violation':0.,'harm_proxy':0.}
    if priv: d.update({'m_star':np.ones((3,2),np.float32)*99,'root_future_signature':np.ones((3,8),np.float32)*77,'r_dep_star':np.float32(-999),'bucket_id':np.int64(2)})
    return d


def _cfg(): return {'sample_rate_hz':10.0,'recovery_horizon_s':4.0,'model':{'feature_max_agents':2},'default_available_distance_m':60.0}

def _field(): return torch.from_numpy(direct_semantic_recovery_witness_features_from_sample(_sample(),_cfg(),num_options=2)).float()

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


def test_sarw_field_is_privilege_free_and_extends_v4863_exactly():
    a=direct_semantic_recovery_witness_features_from_sample(_sample(),_cfg(),num_options=2)
    b=direct_semantic_recovery_witness_features_from_sample(_sample(True),_cfg(),num_options=2)
    assert a.shape==(2,12) and np.isfinite(a).all() and np.array_equal(a,b)
    # New tail is path-stop reserve + observable stability-active bit.
    assert np.all(np.abs(a[:,10])<=1.0) and set(np.unique(a[:,11])).issubset({0.0,1.0})


def _manual_features():
    # [hmin,hterm,hgain,hstopLegacy,hctrl,hstabMin,hstabTerm,hstabGain,hclearFloor,hstabFloor,pathStop,stabActive]
    feat=torch.zeros((1,2,12),dtype=torch.float32)
    feat[0,0]=torch.tensor([.5,.5,.5,-.6,.5,-.7,-.5,-.2,.2,-.1,.6,0.0])  # stop
    feat[0,1]=torch.tensor([.5,.5,.5,.5,.5,-.7,-.5,-.2,.2,-.1,.5,0.0])   # lateral
    return feat


def test_sarw_active_set_alignment_removes_inactive_stability_false_veto():
    L=_layout();x=torch.zeros((1,L.total_dim));feat=_manual_features()
    m_on=_model(active=True,path=True).eval();_force_common_support(m_on)
    out_on=m_on._direct_semantic_witness_absolute_feasibility(m_on._scene_tokens(x),x,_opt(),feat,root_valid=torch.ones((1,3),dtype=torch.bool),option_valid=torch.ones((1,2),dtype=torch.bool))
    m_off=_model(active=False,path=True).eval();_force_common_support(m_off)
    out_off=m_off._direct_semantic_witness_absolute_feasibility(m_off._scene_tokens(x),x,_opt(),feat,root_valid=torch.ones((1,3),dtype=torch.bool),option_valid=torch.ones((1,2),dtype=torch.bool))
    assert out_on is not None and out_off is not None
    assert float(out_on[4][0,1])>0.0 and float(out_off[4][0,1])<0.0


def test_sarw_path_stop_alignment_replaces_terminal_clearance_stop_proxy():
    L=_layout();x=torch.zeros((1,L.total_dim));feat=_manual_features()
    m_on=_model(active=True,path=True).eval();_force_common_support(m_on)
    out_on=m_on._direct_semantic_witness_absolute_feasibility(m_on._scene_tokens(x),x,_opt(),feat,root_valid=torch.ones((1,3),dtype=torch.bool),option_valid=torch.ones((1,2),dtype=torch.bool))
    m_off=_model(active=True,path=False).eval();_force_common_support(m_off)
    out_off=m_off._direct_semantic_witness_absolute_feasibility(m_off._scene_tokens(x),x,_opt(),feat,root_valid=torch.ones((1,3),dtype=torch.bool),option_valid=torch.ones((1,2),dtype=torch.bool))
    assert out_on is not None and out_off is not None
    assert float(out_on[4][0,0])>0.0 and float(out_off[4][0,0])<0.0


def test_sarw_zero_gain_is_execution_exact_native_b():
    torch.manual_seed(4864);m=_model().eval();L=_layout();x=torch.randn((3,L.total_dim));mem=m._scene_tokens(x)
    _,native=m._direct_recovery_option_compatibility_evidence(mem,x,_opt(3),root_valid=torch.ones((3,3),dtype=torch.bool),option_valid=torch.ones((3,2),dtype=torch.bool))
    feat=_field().unsqueeze(0).repeat(3,1,1)
    out=m._direct_semantic_witness_absolute_feasibility(mem,x,_opt(3),feat,root_valid=torch.ones((3,3),dtype=torch.bool),option_valid=torch.ones((3,2),dtype=torch.bool))
    assert out is not None and torch.allclose(out[1],native[:,1],atol=0,rtol=0) and torch.equal(out[3],torch.zeros(2))


def test_sarw_bce_gradient_isolated_to_two_shared_gains():
    torch.manual_seed(4864);m=_model().train()
    for n,p in m.named_parameters(): p.requires_grad_(n=='direct_absolute_semantic_witness_gain')
    L=_layout();x=torch.randn((4,L.total_dim));mem=m._scene_tokens(x);base=_field();feat=torch.stack([base,base.flip(0),base,base.flip(0)])
    out=m._direct_semantic_witness_absolute_feasibility(mem,x,_opt(4),feat,root_valid=torch.ones((4,3),dtype=torch.bool),option_valid=torch.ones((4,2),dtype=torch.bool))
    assert out is not None
    loss=_absolute_feasibility_bce({'direct_recovery_absolute_feasibility_logit':out[0]}, {'r_dep_star':torch.tensor([-.5,.5,-.5,.5]),'is_nominal':torch.zeros(4),'bucket_id':torch.tensor([1,1,2,2]),'time_index':torch.arange(4)})
    loss.backward();g=m.direct_absolute_semantic_witness_gain.grad
    assert g is not None and torch.isfinite(g).all() and torch.any(g!=0) and sum(p.numel() for p in m.parameters() if p.requires_grad)==2


def test_sarw_fail_closed_and_mutually_exclusive():
    m=_model().eval();L=_layout();x=torch.zeros((1,L.total_dim));mem=m._scene_tokens(x)
    try: m._direct_semantic_witness_absolute_feasibility(mem,x,_opt(),None)
    except RuntimeError as e: assert 'OC-SARW features missing' in str(e)
    else: raise AssertionError('missing semantic side channel must fail closed')
    extras=[
      {'direct_recovery_absolute_feasibility_head':True},
      {'direct_recovery_absolute_option_margin_correction':True},
      {'direct_recovery_absolute_physical_headroom_correction':True},
      {'direct_recovery_absolute_executable_witness_correction':True},
      {'direct_recovery_absolute_common_witness_correction':True},
      {'direct_recovery_absolute_quantifier_witness_correction':True},
    ]
    for extra in extras:
        try: OCRAPModel(input_dim=L.total_dim,num_roots=2,num_options=2,d_model=16,d_obs=8,encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,direct_recovery_absolute_semantic_witness_correction=True,**extra)
        except ValueError as e: assert 'mutually exclusive' in str(e)
        else: raise AssertionError('OC-SARW must remain a single-axis source')


def _ckpt(model,schema):
    L=_layout();d={'model_state':model.state_dict(),'input_dim':L.total_dim,'num_roots':3,'num_options':2,'d_model':16,'d_obs':8,'tau_obs':1.0,'encoder_type':'structured_transformer','feature_layout':asdict(L),'d_signature':0,'d_future_signature':0,'option_feature_dim':OPTION_FEATURE_DIM,'direct_recovery_value_head':True,'direct_recovery_absolute_semantic_witness_correction':True,'direct_recovery_semantic_witness_active_set_alignment':True,'direct_recovery_semantic_witness_path_stop_alignment':True,'direct_recovery_evidence_native_certificate_preservation':True,'cfg':{'sample_rate_hz':10.0,'recovery_horizon_s':4.0,'model':{'transformer_layers':1,'transformer_heads':4,'dropout':0.0,'encoder_type':'structured_transformer','option_feature_dim':OPTION_FEATURE_DIM,'direct_recovery_value_head':True,'direct_recovery_absolute_semantic_witness_correction':True,'direct_recovery_semantic_witness_active_set_alignment':True,'direct_recovery_semantic_witness_path_stop_alignment':True,'direct_recovery_evidence_native_certificate_preservation':True},'runtime':{'device':'cpu'}}}
    if schema is not None:d.update({'direct_recovery_absolute_semantic_witness_feature_schema':schema,'direct_recovery_absolute_semantic_witness_feature_source':'semantics_aligned_common_executable_recovery_witness'})
    return d


def test_sarw_checkpoint_roundtrip_schema_and_diagnostics(tmp_path):
    from ocrap.models.inference import load_model_bundle,predict_samples
    m=_model().eval();good=tmp_path/'good.pt';torch.save(_ckpt(m,DIRECT_SEMANTIC_RECOVERY_WITNESS_FEATURE_SCHEMA),good)
    b=load_model_bundle(good);preds=predict_samples([_sample(),_sample(True)],b)
    assert len(preds)==2 and all(p.direct_recovery_semantic_best_common_viability is not None for p in preds)
    assert all(p.direct_recovery_semantic_best_barriers is not None for p in preds)
    bad=tmp_path/'bad.pt';torch.save(_ckpt(m,None),bad)
    try: load_model_bundle(bad)
    except RuntimeError as e: assert 'legacy/unknown OC-SARW checkpoint feature semantics' in str(e)
    else: raise AssertionError('schema-less OC-SARW checkpoint must fail closed')


def test_sarw_shell_plumbing_and_launcher_contract():
    root=Path(__file__).resolve().parents[1]
    train=(root/'scripts/train_ocrap_v48_trac_sr.sh').read_text();adapt=(root/'scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh').read_text()
    assert 'ABSOLUTE_SEMANTIC_WITNESS_CORRECTION' in train and 'direct_recovery_absolute_semantic_witness_correction' in train
    assert 'SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT' in train and 'SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT' in train
    assert 'ABSOLUTE_SEMANTIC_WITNESS_CORRECTION="$ABSOLUTE_SEMANTIC_WITNESS_CORRECTION"' in adapt
    launch=(root/'scripts/run_v48_64_dcp_drfc_bcde_rifa_sarw_two_gpu.sh').read_text()
    for z in ('EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_semantic_witness_gain','STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_semantic_witness_gain','ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true','SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT="$active"','SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT="$pathstop"','MAX_EVIDENCE_CALIBRATOR_PARAMS=2','PROPOSAL_TOP_K=5','ABSOLUTE_FEASIBILITY_THRESHOLD=0.5','I_ACTIVESET','J_PATHSTOP','K_Main_OCSARW'):
        assert z in launch
    assert 'EVIDENCE_CENTER' not in launch.upper() and 'PRED_ADV_CENTER' not in launch.upper()
    # Engineering contract: train.py writes the summary under model_v48_trac_sr/.
    # V48.64.0 accidentally checked candidates/<variant>/train_summary.json and
    # stopped after I_ACTIVESET despite successful training.  Keep both runtime
    # checkers pinned to the canonical producer path.
    vi=(root/'tools/check_v48_64_variant_isolation.py').read_text()
    pc=(root/'tools/check_v48_64_pipeline_complete.py').read_text()
    for checker in (vi,pc):
        assert "'model_v48_trac_sr'/'train_summary.json'" in checker
        assert "base/'train_summary.json'" not in checker
    assert 'TRAINING_COMPLETE.json' in vi and 'EVIDENCE_CORRECTION_COMPLETE.json' in vi
    assert 'TRAINING_COMPLETE.json' in pc and 'EVIDENCE_CORRECTION_COMPLETE.json' in pc
