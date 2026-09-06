from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import torch
from torch import nn
from ocrap.v48_104_nominal_invariant_control_refinement import (
    NominalInvariantLastBlockRefinement, initialization_identity_check, response_only_loss,
)


def test_nominal_and_initial_identity():
    assert initialization_identity_check(16,4)
    torch.manual_seed(104)
    layer=nn.TransformerEncoderLayer(d_model=16,nhead=4,dim_feedforward=64,dropout=.1,batch_first=True,activation='gelu',norm_first=True)
    m=NominalInvariantLastBlockRefinement(layer,nn.LayerNorm(16))
    pre=torch.randn(6,7,16); ni=torch.tensor([0,0,0,3,3,3]); raw=m.base_raw(pre); base=m.final_norm(raw)
    out=m.refined_memory(pre,ni,raw)
    assert torch.allclose(out,base,atol=1e-6,rtol=0.0)
    with torch.no_grad():
        next(m.adapted_last.parameters()).add_(0.5)
    out2=m.refined_memory(pre,ni,raw)
    # Nominal identity is a structural invariant, not a tolerance-based one.
    assert torch.equal(out2[[0,3]],base[[0,3]])
    assert m.nominal_identity_error(pre,ni,raw) == 0.0
    assert not torch.equal(out2[[1,2,4,5]],base[[1,2,4,5]])


def test_only_adapted_last_trainable():
    layer=nn.TransformerEncoderLayer(d_model=16,nhead=4,dim_feedforward=64,dropout=.1,batch_first=True,activation='gelu',norm_first=True)
    m=NominalInvariantLastBlockRefinement(layer,nn.LayerNorm(16))
    assert m.parameter_count==sum(p.numel() for p in m.adapted_last.parameters())
    assert all(not p.requires_grad for p in m.base_last.parameters())
    assert all(not p.requires_grad for p in m.final_norm.parameters())


def test_response_only_loss_ignores_common_shift():
    s=torch.tensor([.3,.5,.7,.2,.4]); r=torch.tensor([0.,.2,.4,-.1,.1]); td=s.clone(); tr=r.clone(); ci=torch.tensor([1,2,4]); ni=torch.tensor([0,0,3]); scales={'delta_support':1.,'delta_reserve':1.}
    a,_=response_only_loss(s,r,td,tr,ci,ni,scales)
    b,_=response_only_loss(s+.1,r+.4,td+.1,tr+.4,ci,ni,scales)
    assert float(a)==0.0 and float(b)==0.0


def _cell(high=True):
    a=.8 if high else .5
    return {'state':{'rows':10,'drs_state_rows':5,'dep_state_rows':5,'auc':.8},'support_true':{'rows':20,'positive_rows':10,'negative_rows':10,'powered_groups':5,'auc':a,'auc_vs_shuffled':.2 if high else 0.,'top1_vs_shuffled':.2 if high else 0.},'support_shuffled':{'rows':20,'positive_rows':10,'negative_rows':10,'powered_groups':5,'auc':.6,'top1':.4},'reserve_true':{'rows':20,'positive_rows':10,'negative_rows':10,'powered_groups':5,'auc':a,'auc_vs_shuffled':.2 if high else 0.,'top1_vs_shuffled':.2 if high else 0.},'reserve_shuffled':{'rows':20,'positive_rows':10,'negative_rows':10,'powered_groups':5,'auc':.6,'top1':.4}}

def _v103(v):
    return {'valid':True,'engineering_version':'v48.103.0-OC-FCSS','variant':v,'cells':{r:_cell(True) for r in ('dev_near','dev_contact','certificate_near','certificate_contact')}}

def _v104(v,high):
    return {'valid':True,'engineering_version':'v48.104.0-OC-NICR','variant':v,'stage_i_last_block_parameters_trained':444864,'planner_parameters_trained':0,'stage_i_other_parameters_trained':0,'root_decoder_parameters_trained':0,'source_parameters_trained':0,'frozen_v103_readout_parameters':1540,'response_only_objective':True,'nominal_memory_exact_identity':True,'initial_v103_function_identity':True,'state_metrics_exact_v103':{'valid':True},'boundary_transport':False,'regime_conditioning':False,'teacher_metadata_input_to_model':False,'cells':{r:_cell(high) for r in ('dev_near','dev_contact','certificate_near','certificate_contact')}}

def test_compare_go_stop(tmp_path:Path):
    repo=Path(__file__).resolve().parents[1]; tool=repo/'tools/compare_v48_104_nicr.py'; env=os.environ.copy(); env['PYTHONPATH']=f"{repo/'src'}:{repo}"+((':'+env['PYTHONPATH']) if env.get('PYTHONPATH') else '')
    c103={'valid':True,'attribution_ready':True,'preregistered_decision':{'status':'FACTORIZED_CONTROL_SUFFICIENT_STATE_STOP','factorized_state_representation_go':True,'factorized_support_action_go':False,'factorized_reserve_debt_go':False,'next_branch':'close_frozen_stage_i_readout_family_then_preregister_last_stage_i_block_control_sufficient_representation_objective_no_broad_encoder_or_source_sweep'}}; cp=tmp_path/'c103.json'; cp.write_text(json.dumps(c103))
    refs={}
    for v in ('balanced','precision'):
        p=tmp_path/f'r_{v}.json'; p.write_text(json.dumps(_v103(v))); refs[v]=p
    for high,status in ((True,'NOMINAL_INVARIANT_CONTROL_REFINEMENT_GO'),(False,'NOMINAL_INVARIANT_CONTROL_REFINEMENT_STOP')):
        bp=tmp_path/f'b{high}.json'; pp=tmp_path/f'p{high}.json'; out=tmp_path/f'o{high}.json'; bp.write_text(json.dumps(_v104('balanced',high))); pp.write_text(json.dumps(_v104('precision',high)))
        subprocess.run([sys.executable,str(tool),'--balanced',str(bp),'--precision',str(pp),'--v103-balanced',str(refs['balanced']),'--v103-precision',str(refs['precision']),'--v103-comparison',str(cp),'--output',str(out)],check=True,env=env)
        assert json.loads(out.read_text())['preregistered_decision']['status']==status


def test_runtime_contract(tmp_path:Path):
    repo=Path(__file__).resolve().parents[1]; out=tmp_path/'r.json'; env=os.environ.copy(); env['PYTHONPATH']=f"{repo/'src'}:{repo}"+((':'+env['PYTHONPATH']) if env.get('PYTHONPATH') else '')
    subprocess.run([sys.executable,str(repo/'tools/check_v48_104_runtime_code_contract.py'),'--repo',str(repo),'--output',str(out)],check=True,env=env)
    d=json.loads(out.read_text()); assert d['valid'] and d['scientific_contract']['expected_last_block_parameters_d192']==444864

def test_prelast_reconstruction_synthetic():
    from ocrap.models.encoders import FlatFeatureLayout, StructuredTokenEncoder
    from tools.run_v48_104_nominal_invariant_control_refinement import _prelast_and_base
    torch.manual_seed(481040)
    layout=FlatFeatureLayout(feature_max_agents=4)
    enc=StructuredTokenEncoder(layout=layout,d_model=16,num_layers=2,num_heads=4,dropout=.1).eval()
    x=torch.randn(3,layout.total_dim)
    with torch.no_grad():
        pre,raw,base=_prelast_and_base(enc,x)
        direct=enc.forward_tokens(x)
    assert pre.shape==base.shape==direct.shape
    assert torch.allclose(base,direct,atol=1e-6,rtol=0.0)
