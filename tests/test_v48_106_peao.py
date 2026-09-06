from __future__ import annotations
import numpy as np
import torch
from ocrap.models.encoders import FlatFeatureLayout, StructuredTokenEncoder
from ocrap.v48_105_prelast_action_equivariance_localization import action_interaction_slice, summary_group_slices
from ocrap.v48_106_preencoder_action_orientation_audit import (
    agent_permutation_invariance_check, candidate_zero_delta_check, mean_difference_direction,
    preencoder_contract_checks, preencoder_memory_summary, signed_orientation_cosine, signed_orientation_sign_flip_check,
)
from tools.run_v48_106_preencoder_action_orientation_audit import _preencoder_memory

def test_preencoder_contract_checks():
    assert all(preencoder_contract_checks(192).values())

def test_candidate_nominal_zero_delta_and_agent_invariance():
    assert candidate_zero_delta_check(16)
    assert agent_permutation_invariance_check(16)

def test_preencoder_summary_dimension_and_same_partition():
    x=torch.randn(3,43,16); z=preencoder_memory_summary(x); assert z.shape==(3,240)
    g=summary_group_slices(16); assert g['control']==slice(80,96); assert action_interaction_slice(16)==slice(80,240)

def test_preencoder_reconstructs_historical_final_tokens():
    torch.manual_seed(48106); layout=FlatFeatureLayout(feature_max_agents=4)
    enc=StructuredTokenEncoder(layout,d_model=16,num_layers=2,num_heads=4,dropout=0.1).eval(); x=torch.randn(5,layout.total_dim)
    with torch.no_grad(): inp,pre,final=_preencoder_memory(enc,x); direct=enc.forward_tokens(x)
    assert inp.shape[1]==15 and pre.shape[1]==15
    assert torch.equal(final,direct)

def test_signed_orientation_detects_sign_flip():
    assert signed_orientation_sign_flip_check()
    y=np.asarray([0,0,1,1]); X=np.asarray([[0.,0.],[0.,1.],[2.,0.],[2.,1.]])
    d=mean_difference_direction(X,y); assert d[0]>0
    assert signed_orientation_cosine(d,d)>0.999999
    assert signed_orientation_cosine(d,-d)<-0.999999

def test_signed_orientation_is_scale_invariant_but_sign_sensitive():
    a=np.asarray([1.,2.,-3.]); assert abs(signed_orientation_cosine(a,5*a)-1.0)<1e-12
    assert abs(signed_orientation_cosine(a,-5*a)+1.0)<1e-12
