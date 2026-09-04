from __future__ import annotations
import json
from types import SimpleNamespace
import numpy as np
from ocrap.algorithms.lcv import normalize_weights, weighted_lcvar
from ocrap.v48_91_common_exogenous_physical_margin import (
    physical_margin_from_teacher_diag, future_physical_matrix, future_nested_tail_influence,
    audit_future_physical_response,
)

def sample(assign, sources, metas, mf, probs=None):
    mf=np.asarray(mf,dtype=np.float64);assign=np.asarray(assign,dtype=np.int64);F,L=mf.shape;K=int(assign.max())+1
    probs=np.asarray(probs if probs is not None else np.ones(F)/F,dtype=np.float64);probs=normalize_weights(probs)
    rp=np.zeros(K); M=np.zeros((K,L))
    for k in range(K):
        idx=np.where(assign==k)[0];rp[k]=probs[idx].sum()
        w=normalize_weights(probs[idx])
        for l in range(L):M[k,l]=weighted_lcvar(mf[idx,l],w,.2)
    s={'m_star':M.astype(np.float32),'root_probs':rp.astype(np.float32),'root_valid':np.ones(K,np.float32),'c_star':np.eye(K,dtype=np.float32),'option_valid':np.ones(L,np.float32),'root_assignments':assign,'future_probs':probs.astype(np.float32),'future_valid':np.ones(F,np.float32),'future_sources':np.asarray(sources),'future_metadata':json.dumps(metas,sort_keys=True),'recovery_modes':np.asarray(['stop']*L)}
    # nested-tail helper needs a stored scalar; set exact recomputation.
    from ocrap.v48_89_root_correspondence import nested_tail_influence
    _,r,_,_=nested_tail_influence({**s,'r_dep_star':np.float32(0.)});s['r_dep_star']=np.float32(r)
    return s

def test_pre_structural_physical_min_ignores_inactive_and_structural_value():
    d=SimpleNamespace(active={'clearance':True,'route':False,'stability':True},component_margins={'clearance':.4,'route':-8.,'stability':-.2})
    assert physical_margin_from_teacher_diag(d)==-.2
    m=future_physical_matrix([[d]],np.array([True]))
    assert m.shape==(1,1) and m[0,0]==-.2

def test_future_tail_influence_reconstructs_root_tail_mass():
    mf=np.array([[-1.0],[1.0],[.5]],dtype=float)
    s=sample([0,0,1],['replay','reactive','targeted'],[{}, {'rollout_variant':'x'}, {'targeted_type':'z'}],mf,probs=[.2,.3,.5])
    fmass,err=future_nested_tail_influence(s,mf)
    assert fmass.shape==mf.shape
    assert err<1e-8
    assert fmass.sum()>0

def test_common_exogenous_future_response_is_root_slot_permutation_invariant_and_signed():
    metas=[{'rollout_variant':'shared-a'},{'targeted_type':'shared-b'}]
    # Candidate swaps root slots relative to nominal, but exogenous futures are the same.
    cs=sample([1,0],['reactive','targeted'],metas,[[-.5],[.2]],probs=[.5,.5])
    ns=sample([0,1],['reactive','targeted'],metas,[[-1.0],[.2]],probs=[.5,.5])
    cp=np.array([[.5],[.2]])   # physical response +1.0 on the weak shared-a future
    np_=np.array([[-.5],[.2]])
    rec=audit_future_physical_response(cs,ns,np.array([[-.5],[.2]]),np.array([[-1.0],[.2]]),cp,np_)
    assert rec.valid
    assert rec.common_exogenous_tail_coverage>0.99
    assert rec.response_sign_identifiable_mass>0.99
    assert rec.signed_response_score>0.99

def test_different_exogenous_realization_is_not_matched():
    ca={'targeted_type':'waymax_hidden_vehicle_yield','scenario_augmented':True,'hidden_spawn_xy':[1.,2.],'hidden_actor_object_index':3}
    na={'targeted_type':'waymax_hidden_vehicle_yield','scenario_augmented':True,'hidden_spawn_xy':[5.,2.],'hidden_actor_object_index':4}
    cs=sample([0],['targeted'],[ca],[[-1.]])
    ns=sample([0],['targeted'],[na],[[-1.]])
    rec=audit_future_physical_response(cs,ns,np.array([[-1.]]),np.array([[-1.]]),np.array([[1.]]),np.array([[-1.]]))
    assert rec.valid
    assert rec.common_exogenous_tail_coverage==0.0
    assert rec.response_sign_identifiable_mass==0.0
