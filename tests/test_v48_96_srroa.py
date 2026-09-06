from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np

from ocrap.v48_96_support_reserve_root_observability import (
    derive_candidate_semantics, feature_only_dataset_cfg, root_observability_features, ENGINEERING_VERSION
)


def test_drs_activation_semantics():
    n={'teacher_drs':0.0,'teacher_r_dep':0.0,'teacher_gap':0.0,'teacher_pcd':0.0,'component_harmful':False}
    c={'teacher_drs':1.0,'teacher_r_dep':0.0,'teacher_gap':0.0,'teacher_pcd':0.5,'component_harmful':False}
    x=derive_candidate_semantics(n,c)
    assert x['safe_positive'] and x['mediation_mode']=='drs_activation'


def test_deployability_gain_semantics():
    import math
    # nominal/candidate PCD values are constructed from DRS=1 and sigmoid(r_dep).
    n={'teacher_drs':1.0,'teacher_r_dep':-1.0,'teacher_gap':0.0,'teacher_pcd':1/(1+math.exp(1.0)),'component_harmful':False}
    c={'teacher_drs':1.0,'teacher_r_dep':1.0,'teacher_gap':0.0,'teacher_pcd':1/(1+math.exp(-1.0)),'component_harmful':False}
    x=derive_candidate_semantics(n,c)
    assert x['safe_positive'] and x['mediation_mode']=='deployability_gain'


def test_compare_go_and_stop(tmp_path:Path):
    def doc(high=True):
        cells={}
        for r in ('dev_near','dev_contact','certificate_near','certificate_contact'):
            cells[r]={'state':{'auc':0.8 if high else 0.5},
                      'support_true':{'auc':0.8 if high else 0.5,'auc_vs_shuffled':0.2 if high else 0.0,'top1_vs_shuffled':0.2 if high else 0.0},
                      'reserve_true':{'auc':0.8 if high else 0.5,'auc_vs_shuffled':0.2 if high else 0.0,'top1_vs_shuffled':0.2 if high else 0.0}}
        return {'valid':True,'engineering_version':ENGINEERING_VERSION,'cells':cells}
    repo=Path(__file__).resolve().parents[1]; tool=repo/'tools/compare_v48_96_srroa.py'
    for high,status in [(True,'FROZEN_ROOT_SUPPORT_RESERVE_OBSERVABILITY_GO'),(False,'FROZEN_ROOT_SUPPORT_RESERVE_OBSERVABILITY_STOP')]:
        b=tmp_path/f'b{high}.json';p=tmp_path/f'p{high}.json';o=tmp_path/f'o{high}.json';b.write_text(json.dumps(doc(high)));p.write_text(json.dumps(doc(high)))
        subprocess.run([sys.executable,str(tool),'--balanced',str(b),'--precision',str(p),'--output',str(o)],check=True)
        assert json.loads(o.read_text())['preregistered_decision']['status']==status


def test_runtime_contract(tmp_path:Path):
    repo=Path(__file__).resolve().parents[1]; out=tmp_path/'r.json'
    subprocess.run([sys.executable,str(repo/'tools/check_v48_96_runtime_code_contract.py'),'--repo',str(repo),'--output',str(out)],check=True)
    d=json.loads(out.read_text());assert d['valid'] and d['scientific_contract']['planner_parameters_trained']==0



def test_feature_only_cfg_strips_checkpoint_supervision_sidecars(tmp_path:Path):
    cfg={
        'training':{
            'direct_value_absolute_feasibility_truth_contract':'structural_interval_bounds',
            'direct_value_absolute_feasibility_truth_index':'/old/train-dev-only.jsonl',
            'direct_value_absolute_feasibility_supervision_objective':'signed_margin_interval_huber',
            'direct_value_action_response_truth_index':'/old/response.jsonl',
        },
        'model':{'d_model':192},
    }
    out,event=feature_only_dataset_cfg(cfg,cache_dir=str(tmp_path/'cache'),workers=3)
    t=out['training']
    assert t['direct_value_absolute_feasibility_truth_contract']=='legacy_full'
    assert t['direct_value_absolute_feasibility_truth_index']==''
    assert t['direct_value_absolute_feasibility_supervision_objective']=='binary_sign'
    assert t['direct_value_action_response_truth_index']==''
    assert t['persistent_tensor_cache_build_workers']==3
    assert event['truth_sidecars_attached'] is False
    # The serialized checkpoint config itself must remain untouched.
    assert cfg['training']['direct_value_absolute_feasibility_truth_contract']=='structural_interval_bounds'
    assert cfg['model']==out['model']


def test_root_features_are_nominal_state_pure_and_root_permutation_invariant():
    import torch
    # nominal + two candidates, K=3, D=2
    rt=torch.tensor([
        [[1.,0.],[2.,1.],[4.,2.]],
        [[3.,0.],[5.,1.],[7.,2.]],
        [[9.,0.],[8.,1.],[6.,2.]],
    ])
    p=torch.tensor([[0.2,0.3,0.5],[0.1,0.7,0.2],[0.4,0.2,0.4]])
    # Candidate validity differs on purpose: nominal state must still be identical.
    v=torch.tensor([[1,1,1],[1,1,0],[0,1,1]],dtype=torch.bool)
    state,delta,context=root_observability_features(rt,p,v)
    assert torch.equal(state[0],state[1])  # pure nominal state; candidate cannot change it
    # Independently permute candidate-1 root slots with probabilities/validity.
    perm=torch.tensor([2,0,1])
    rt2=rt.clone(); p2=p.clone(); v2=v.clone()
    rt2[1]=rt2[1,perm]; p2[1]=p2[1,perm]; v2[1]=v2[1,perm]
    state2,delta2,context2=root_observability_features(rt2,p2,v2)
    assert torch.allclose(state,state2,atol=0,rtol=0)
    assert torch.allclose(delta,delta2,atol=1e-7,rtol=0)
    assert torch.allclose(context,context2,atol=1e-7,rtol=0)
