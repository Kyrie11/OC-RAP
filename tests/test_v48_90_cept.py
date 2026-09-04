from __future__ import annotations

import json
import numpy as np

from ocrap.v48_89_root_correspondence import nested_tail_influence
from ocrap.v48_90_partition_transport import audit_partition_transport_pair, future_class_keys


def _sample(assignments, probs, metas, *, sources=None, margins=None, root_probs=None, mode="stop"):
    assignments=np.asarray(assignments,dtype=np.int64)
    probs=np.asarray(probs,dtype=np.float32)
    K=(max(assignments)+1) if assignments.size else 1
    if margins is None:
        margins=np.linspace(-1.0,1.0,K,dtype=np.float32)[:,None]
    margins=np.asarray(margins,dtype=np.float32)
    K=margins.shape[0]
    if root_probs is None:
        rp=np.zeros(K,dtype=np.float32)
        p=probs/probs.sum()
        for a,w in zip(assignments,p): rp[int(a)]+=float(w)
        root_probs=rp
    if sources is None:
        sources=["targeted"]*len(assignments)
    s={
        "m_star":margins,
        "root_probs":np.asarray(root_probs,dtype=np.float32),
        "root_valid":np.ones(K,dtype=np.float32),
        "c_star":np.eye(K,dtype=np.float32),
        "option_valid":np.ones(margins.shape[1],dtype=np.float32),
        "root_assignments":assignments,
        "future_probs":probs,
        "future_valid":np.ones(len(probs),dtype=np.float32),
        "future_sources":np.asarray(sources),
        "future_metadata":json.dumps(metas,sort_keys=True),
        "recovery_modes":np.asarray([mode]*margins.shape[1]),
    }
    _,r,_,_=nested_tail_influence(s)
    s["r_dep_star"]=np.float32(r)
    return s


def test_recipe_quotient_collapses_exchangeable_duplicates_without_weak_identity():
    metas=[
        {"targeted_type":"waymax_sdc_post_prefix_control_stress","ego_after_prefix_accel":-2.0},
        {"targeted_type":"waymax_sdc_post_prefix_control_stress","ego_after_prefix_accel":-2.0},
    ]
    s=_sample([0,0],[0.5,0.5],metas,margins=[[-0.5]])
    keys, unresolved, duplicate=future_class_keys(s,exogenous=False)
    assert keys[0]==keys[1]
    assert unresolved.tolist()==[False,False]
    assert duplicate.tolist()==[True,True]
    rec=audit_partition_transport_pair(s,s)
    assert rec.valid
    assert rec.recipe_unresolved_semantic_mass_candidate==0.0
    assert rec.exchangeable_duplicate_mass_candidate==1.0
    assert rec.duplicate_root_homogeneity_mass_candidate==1.0
    assert rec.recipe_tail_partition_stability==1.0


def test_candidate_dependent_augmented_realization_is_not_fabricated_as_exogenous_match():
    base={"artifact_branch":"yield","targeted_type":"waymax_hidden_vehicle_yield","scenario_augmented":True}
    a=dict(base,hidden_spawn_xy=[8.0,1.0],hidden_actor_object_index=3)
    b=dict(base,hidden_spawn_xy=[12.0,-2.0],hidden_actor_object_index=4)
    c=_sample([0],[1.0],[a],margins=[[-0.5]])
    n=_sample([0],[1.0],[b],margins=[[-1.0]])
    rec=audit_partition_transport_pair(c,n)
    assert rec.valid
    assert rec.recipe_shared_mass_candidate==1.0
    assert rec.exogenous_shared_mass_candidate==0.0
    assert rec.exogenous_tail_unmatched_mass==1.0


def test_transport_recovers_root_slot_permutation_without_root_id_assumption():
    metas=[{}, {"rollout_variant":"waymax_log_playback_sdc_coast","ego_after_prefix_accel":-1.0}]
    sources=["replay","reactive"]
    n=_sample([0,1],[0.5,0.5],metas,sources=sources,margins=[[-1.0],[1.0]])
    c=_sample([1,0],[0.5,0.5],metas,sources=sources,margins=[[1.2],[-0.5]])
    rec=audit_partition_transport_pair(c,n)
    assert rec.valid
    assert np.isclose(rec.exogenous_matched_transport_mass,1.0)
    assert np.isclose(rec.exogenous_tail_transport_coverage,1.0)
    assert np.isclose(rec.exogenous_tail_transport_purity,1.0)
    assert np.isclose(rec.exogenous_tail_partition_stability,1.0)


def test_split_merge_partition_reduces_transport_purity():
    # Same semantic class is split over two candidate roots but occupies one nominal root.
    metas=[
        {"targeted_type":"waymax_sdc_post_prefix_control_stress","ego_after_prefix_accel":-2.0},
        {"targeted_type":"waymax_sdc_post_prefix_control_stress","ego_after_prefix_accel":-2.0},
    ]
    n=_sample([0,0],[0.5,0.5],metas,margins=[[-1.0],[1.0]],root_probs=[1.0,0.0])
    c=_sample([0,1],[0.5,0.5],metas,margins=[[-0.6],[-0.4]],root_probs=[0.5,0.5])
    rec=audit_partition_transport_pair(c,n)
    assert rec.valid
    assert rec.duplicate_root_homogeneity_mass_candidate==0.0
    # Candidate rows each map cleanly to the same nominal root, so row-purity itself is 1;
    # duplicate homogeneity is the explicit fail-closed diagnostic for this quotient.
    assert rec.recipe_tail_transport_coverage==1.0


def test_missing_augmented_realization_fingerprint_is_unresolved():
    meta={"artifact_branch":"yield","targeted_type":"waymax_hidden_vehicle_yield","scenario_augmented":True}
    s=_sample([0],[1.0],[meta],margins=[[-0.5]])
    keys, unresolved, _=future_class_keys(s,exogenous=True)
    assert len(keys)==1 and unresolved.tolist()==[True]
    rec=audit_partition_transport_pair(s,s)
    assert rec.valid
    assert rec.exogenous_unresolved_mass_candidate==1.0
    assert rec.exogenous_matched_transport_mass==0.0


def test_recovery_option_mismatch_fails_closed():
    meta=[{}]
    n=_sample([0],[1.0],meta,sources=["replay"],margins=[[-1.0]],mode="stop")
    c=_sample([0],[1.0],meta,sources=["replay"],margins=[[-0.5]],mode="yield_rejoin")
    rec=audit_partition_transport_pair(c,n)
    assert not rec.valid
    assert "recovery option identity/order mismatch" in str(rec.error)


def test_deterministic_secondary_collision_recipe_does_not_require_impulse_fingerprint():
    meta={
        "scenario_augmented":False,
        "targeted_type":"waymax_secondary_collision_approach",
        "secondary_collision_approach":True,
        "contact_surrogate":True,
        "ego_after_prefix_accel":-1.2,
    }
    s=_sample([0],[1.0],[meta],margins=[[-0.5]])
    keys, unresolved, _=future_class_keys(s,exogenous=True)
    assert len(keys)==1 and unresolved.tolist()==[False]
    rec=audit_partition_transport_pair(s,s)
    assert rec.valid
    assert rec.exogenous_shared_mass_candidate==1.0
    assert rec.exogenous_tail_partition_stability==1.0
