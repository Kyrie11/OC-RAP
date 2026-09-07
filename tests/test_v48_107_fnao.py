from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import torch
from torch import nn
from ocrap.v48_107_first_block_nominal_invariant_action_orientation import (
    NominalInvariantFirstBlockOrientation, initialization_identity_check,
    ordinal_action_orientation_loss_sum, orientation_loss_sign_check,
)


def _layer(d=16):
    return nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=4*d,dropout=.1,batch_first=True,activation='gelu',norm_first=True)


def test_first_block_nominal_identity_exact():
    assert initialization_identity_check(16,4)
    torch.manual_seed(107)
    m=NominalInvariantFirstBlockOrientation(_layer(),[_layer()],nn.LayerNorm(16))
    x=torch.randn(6,7,16); ni=torch.tensor([0,0,0,3,3,3]); base1=m.base_after_first(x); base=m.tail_memory(base1)
    with torch.no_grad(): next(m.adapted_first.parameters()).add_(0.5)
    out=m.refined_memory(x,ni,base1)
    assert torch.equal(out[[0,3]],base[[0,3]])
    assert m.nominal_identity_error(x,ni,base1)==0.0
    assert not torch.equal(out[[1,2,4,5]],base[[1,2,4,5]])


def test_only_first_block_trainable():
    m=NominalInvariantFirstBlockOrientation(_layer(),[_layer()],nn.LayerNorm(16))
    assert m.parameter_count==sum(p.numel() for p in m.adapted_first.parameters())
    assert all(not p.requires_grad for p in m.base_first.parameters())
    assert all(not p.requires_grad for p in m.frozen_tail.parameters())
    assert all(not p.requires_grad for p in m.final_norm.parameters())


def test_ordinal_orientation_prefers_correct_order_and_ignores_common_shift():
    assert orientation_loss_sign_check()
    s=torch.tensor([.2,.8,.1,.9]);r=torch.tensor([-1.,1.,-.5,.5]);ts=torch.tensor([0.,1.,0.,1.]);tr=r.clone();g=torch.tensor([0,0,1,1]);sc={'delta_support':1.,'delta_reserve':1.}
    a,_=ordinal_action_orientation_loss_sum(s,r,ts,tr,g,sc)
    b,_=ordinal_action_orientation_loss_sum(s+3.,r-7.,ts*5.,tr*4.,g,sc)
    assert torch.allclose(a,b,atol=1e-7,rtol=0.0)


def _cell(high=True):
    a=.8 if high else .5
    return {'state':{'rows':10,'drs_state_rows':5,'dep_state_rows':5,'auc':.8},'support_true':{'rows':20,'positive_rows':10,'negative_rows':10,'powered_groups':5,'auc':a,'auc_vs_shuffled':.2 if high else 0.,'top1_vs_shuffled':.2 if high else 0.},'support_shuffled':{'rows':20,'positive_rows':10,'negative_rows':10,'powered_groups':5,'auc':.6,'top1':.4},'reserve_true':{'rows':20,'positive_rows':10,'negative_rows':10,'powered_groups':5,'auc':a,'auc_vs_shuffled':.2 if high else 0.,'top1_vs_shuffled':.2 if high else 0.},'reserve_shuffled':{'rows':20,'positive_rows':10,'negative_rows':10,'powered_groups':5,'auc':.6,'top1':.4}}


def _v103(v): return {'valid':True,'engineering_version':'v48.103.0-OC-FCSS','variant':v,'cells':{r:_cell(True) for r in ('dev_near','dev_contact','certificate_near','certificate_contact')}}
def _v107(v,high):
    return {'valid':True,'engineering_version':'v48.107.0-OC-FNAO','variant':v,'stage_i_first_block_parameters_trained':444864,'stage_i_other_parameters_trained':0,'frozen_stage_i_second_block':True,'frozen_v103_readout_parameters':1540,'root_decoder_parameters_trained':0,'source_parameters_trained':0,'planner_parameters_trained':0,'ordinal_action_orientation_objective':True,'ordinal_target_magnitude_discarded_after_sign':True,'nominal_first_block_exact_identity':True,'nominal_final_memory_exact_identity':True,'state_metrics_exact_v103':{'valid':True},'initial_v103_function_identity':{'valid':True},'boundary_transport':False,'regime_conditioning':False,'teacher_metadata_input_to_model':False,'cells':{r:_cell(high) for r in ('dev_near','dev_contact','certificate_near','certificate_contact')}}


def test_compare_go_stop(tmp_path:Path):
    repo=Path(__file__).resolve().parents[1];tool=repo/'tools/compare_v48_107_fnao.py';env=os.environ.copy();env['PYTHONPATH']=f"{repo/'src'}:{repo}"+((':'+env['PYTHONPATH']) if env.get('PYTHONPATH') else '')
    p106=tmp_path/'p106.json';c106=tmp_path/'c106.json'
    p106.write_text(json.dumps({'valid':True,'attribution_ready':True,'engineering_version':'v48.106.0-OC-PEAO','preregistered_status':'PREENCODER_ACTION_ORIENTATION_STOP'}))
    c106.write_text(json.dumps({'valid':True,'attribution_ready':True,'preregistered_decision':{'status':'PREENCODER_ACTION_ORIENTATION_STOP','next_branch':'preencoder_action_orientation_insufficient_then_preregister_first_stage_i_block_nominal_invariant_action_orientation_objective_no_source_or_broad_encoder_sweep'}}))
    refs={}
    for v in ('balanced','precision'):
        p=tmp_path/f'r{v}.json';p.write_text(json.dumps(_v103(v)));refs[v]=p
    for high,status in ((True,'FIRST_BLOCK_NOMINAL_INVARIANT_ACTION_ORIENTATION_GO'),(False,'FIRST_BLOCK_NOMINAL_INVARIANT_ACTION_ORIENTATION_STOP')):
        bp=tmp_path/f'b{high}.json';pp=tmp_path/f'p{high}.json';out=tmp_path/f'o{high}.json';bp.write_text(json.dumps(_v107('balanced',high)));pp.write_text(json.dumps(_v107('precision',high)))
        subprocess.run([sys.executable,str(tool),'--balanced',str(bp),'--precision',str(pp),'--v103-balanced',str(refs['balanced']),'--v103-precision',str(refs['precision']),'--v106-pipeline',str(p106),'--v106-comparison',str(c106),'--output',str(out)],check=True,env=env)
        assert json.loads(out.read_text())['preregistered_decision']['status']==status


def test_runtime_contract(tmp_path:Path):
    repo=Path(__file__).resolve().parents[1];out=tmp_path/'r.json';env=os.environ.copy();env['PYTHONPATH']=f"{repo/'src'}:{repo}"+((':'+env['PYTHONPATH']) if env.get('PYTHONPATH') else '')
    subprocess.run([sys.executable,str(repo/'tools/check_v48_107_runtime_code_contract.py'),'--repo',str(repo),'--output',str(out)],check=True,env=env)
    d=json.loads(out.read_text());assert d['valid'] and d['scientific_contract']['expected_first_block_parameters_d192']==444864


def test_historical_two_layer_reconstruction_synthetic():
    from ocrap.models.encoders import FlatFeatureLayout,StructuredTokenEncoder
    from tools.run_v48_107_first_block_nominal_invariant_action_orientation import _input_first_and_final
    torch.manual_seed(48107);layout=FlatFeatureLayout(feature_max_agents=4);enc=StructuredTokenEncoder(layout=layout,d_model=16,num_layers=2,num_heads=4,dropout=.1).eval();x=torch.randn(3,layout.total_dim)
    with torch.no_grad(): inp,first,final=_input_first_and_final(enc,x);direct=enc.forward_tokens(x)
    assert inp.shape==first.shape==final.shape==direct.shape
    assert torch.allclose(final,direct,atol=1e-6,rtol=0.0)
