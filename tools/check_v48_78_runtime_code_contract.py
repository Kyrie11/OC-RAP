#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,importlib,json,sys
from dataclasses import asdict
from pathlib import Path
import torch

def sha(p:Path):
    h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();src=repo/'src';sys.path.insert(0,str(src));sys.path.insert(0,str(repo));errors=[];mods={}
    expected={'ocrap':src/'ocrap/__init__.py','ocrap.cli.train':src/'ocrap/cli/train.py','ocrap.models.data':src/'ocrap/models/data.py','ocrap.models.ocrap':src/'ocrap/models/ocrap.py','ocrap.models.inference':src/'ocrap/models/inference.py','ocrap.algorithms.lcv':src/'ocrap/algorithms/lcv.py','ocrap.algorithms.ocmero':src/'ocrap/algorithms/ocmero.py'}
    for name,ep in expected.items():
        try:
            m=importlib.import_module(name);p=Path(m.__file__).resolve();ok=p==ep.resolve() and repo in p.parents;mods[name]={'path':str(p),'expected_path':str(ep.resolve()),'exact_path':p==ep.resolve(),'inside_repo':repo in p.parents,'sha256':sha(p)}
            if not ok:errors.append(f'runtime module mismatch: {name}')
        except Exception as e:mods[name]={'error':repr(e),'inside_repo':False};errors.append(f'runtime import failed: {name}')
    from ocrap.algorithms.lcv import torch_weighted_lcvar,torch_weighted_lcvar_influence
    from ocrap.cli.train import _absolute_feasibility_supervision_loss,_semantic_witness_checkpoint_feature_contract
    from ocrap.models.data import OPTION_FEATURE_DIM
    from ocrap.models.encoders import FlatFeatureLayout
    from ocrap.models.ocrap import OCRAPModel
    # Exact tail-attribution algebra check.
    scores=torch.tensor([[[-2.,0.,3.],[-1.,2.,4.]]]);weights=torch.tensor([[[.2,.5,.3],[.2,.5,.3]]]);alpha=.6
    inf=torch_weighted_lcvar_influence(scores,weights,alpha);val=torch_weighted_lcvar(scores,weights,alpha)
    influence_ok=bool(torch.allclose(inf.sum(-1),torch.ones_like(val),atol=1e-7) and torch.allclose((inf*scores).sum(-1),val,atol=1e-7))
    if not influence_ok:errors.append('LCVAR influence contract failed')
    batch={'r_dep_star':torch.tensor([.5,.2,-.7,-2.0]),'is_nominal':torch.zeros(4),'bucket_id':torch.tensor([1,1,2,2]),'time_index':torch.zeros(4,dtype=torch.long)};out={'direct_recovery_absolute_feasibility_logit':torch.tensor([99.0,.1,-.2,-1.0])};cfg={'direct_value_absolute_feasibility_truth_contract':'censor_exact_0p5','direct_value_absolute_feasibility_supervision_objective':'signed_margin_huber'}
    got=float(_absolute_feasibility_supervision_loss(out,batch,cfg));exp=float(torch.nn.functional.smooth_l1_loss(torch.tensor([.1,-.2,-1.0]),torch.tensor([.2,-.7,-2.0]),beta=1.0));supervision_ok=abs(got-exp)<=1e-8
    if not supervision_ok:errors.append('signed-margin supervision contract failed')
    base={'direct_recovery_absolute_semantic_witness_correction':True,'direct_recovery_semantic_witness_active_set_alignment':True,'direct_recovery_semantic_witness_path_stop_alignment':False,'direct_recovery_semantic_witness_classlocal_transport':False,'direct_recovery_semantic_witness_route_alignment':True,'direct_recovery_semantic_witness_reentry_alignment':True,'direct_recovery_semantic_witness_control_projection':True,'direct_recovery_semantic_witness_boundary_transport':False,'direct_recovery_semantic_witness_projection_fidelity_weighting':False,'direct_recovery_semantic_witness_active_constraint_typed_source':False,'direct_recovery_semantic_witness_root_tail_source':True,'direct_recovery_evidence_native_certificate_preservation':True}
    L=FlatFeatureLayout(feature_max_agents=2);models={};serializers={}
    for label,tail in [('I78_ROOT_SHAPE',False),('J78_MAIN_RTSI',True)]:
        mc=dict(base);mc['direct_recovery_semantic_witness_tail_localization']=tail
        schema,source=_semantic_witness_checkpoint_feature_contract(mc);sok=(schema==3 and source=='projected_boundary_common_executable_recovery_witness');serializers[label]={'schema':schema,'source':source,'expected_schema':3,'expected_source':'projected_boundary_common_executable_recovery_witness','valid':sok}
        if not sok:errors.append(f'{label} serializer mismatch')
        m=OCRAPModel(input_dim=L.total_dim,num_roots=3,num_options=2,d_model=16,d_obs=8,encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,dropout=0.0,option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,**mc)
        w=m.direct_absolute_root_tail_source_scale;mk=bool(w is not None and tuple(w.shape)==(1,) and w.numel()==1 and torch.count_nonzero(w).item()==0 and m.direct_absolute_semantic_witness_gain is None)
        models[label]={'source_scale_shape':list(w.shape) if w is not None else None,'source_scale_numel':w.numel() if w is not None else None,'zero_init':bool(w is not None and torch.count_nonzero(w).item()==0),'legacy_gain_absent':m.direct_absolute_semantic_witness_gain is None,'tail_localization':tail,'valid':mk}
        if not mk:errors.append(f'{label} root-tail source contract failed')
    valid=not errors
    doc={'schema':'ocrap-v48.78-rtsi-runtime-code-contract-v1','engineering_version':'v48.78.0-OC-RTSI','valid':valid,'attribution_ready':valid,'errors':errors,'runtime_modules':mods,'serializer_contracts':serializers,'root_tail_source_model_contracts':models,'tail_influence_contract':{'valid':influence_ok,'influence_sum':inf.sum(-1).tolist(),'lcvar':val.tolist()},'supervision_contract':{'truth_contract':'censor_exact_0p5','objective':'signed_margin_huber','huber_beta':1.0,'regime_conditioned':False,'teacher_future_input':False,'floor_relabelled':False,'valid':supervision_ok},'source_contract':{'trainable_parameter':'direct_absolute_root_tail_source_scale[1]','option_translation_zero_mean':True,'option_id_input':False,'regime_id_input':False,'classlocal_transport':False,'boundary_transport':False,'projection_fidelity':False,'active_constraint_typed_source':False},'dataset_reconstruction':False,'uses_test_roots':False,'test_roots_read':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_78_runtime_contract','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
