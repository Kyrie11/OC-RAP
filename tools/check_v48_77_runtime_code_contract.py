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
 expected={'ocrap':src/'ocrap/__init__.py','ocrap.cli.train':src/'ocrap/cli/train.py','ocrap.models.data':src/'ocrap/models/data.py','ocrap.models.ocrap':src/'ocrap/models/ocrap.py','ocrap.models.inference':src/'ocrap/models/inference.py','ocrap.simulation.teacher.margins':src/'ocrap/simulation/teacher/margins.py'}
 for name,ep in expected.items():
  try:
   m=importlib.import_module(name);p=Path(m.__file__).resolve();ok=p==ep.resolve() and repo in p.parents;mods[name]={'path':str(p),'expected_path':str(ep.resolve()),'exact_path':p==ep.resolve(),'inside_repo':repo in p.parents,'sha256':sha(p)}
   if not ok:errors.append(f'runtime module mismatch: {name}')
  except Exception as e:mods[name]={'error':repr(e),'inside_repo':False};errors.append(f'runtime import failed: {name}')
 from ocrap.cli.train import _absolute_feasibility_supervision_loss,_semantic_witness_checkpoint_feature_contract
 from ocrap.models.data import OPTION_FEATURE_DIM
 from ocrap.models.encoders import FlatFeatureLayout
 from ocrap.models.ocrap import OCRAPModel
 batch={'r_dep_star':torch.tensor([.5,.2,-.7,-2.0]),'is_nominal':torch.zeros(4),'bucket_id':torch.tensor([1,1,2,2]),'time_index':torch.zeros(4,dtype=torch.long)};out={'direct_recovery_absolute_feasibility_logit':torch.tensor([99.0,.1,-.2,-1.0])}
 cfg={'direct_value_absolute_feasibility_truth_contract':'censor_exact_0p5','direct_value_absolute_feasibility_supervision_objective':'signed_margin_huber'}
 got=float(_absolute_feasibility_supervision_loss(out,batch,cfg));exp=float(torch.nn.functional.smooth_l1_loss(torch.tensor([.1,-.2,-1.0]),torch.tensor([.2,-.7,-2.0]),beta=1.0));synth={'loss':got,'expected_loss':exp,'floor_prediction_ignored':True,'valid':abs(got-exp)<=1e-8}
 if not synth['valid']:errors.append('signed-margin supervision synthetic check failed')
 base_flags={'direct_recovery_absolute_semantic_witness_correction':True,'direct_recovery_semantic_witness_active_set_alignment':True,'direct_recovery_semantic_witness_path_stop_alignment':False,'direct_recovery_semantic_witness_classlocal_transport':False,'direct_recovery_semantic_witness_route_alignment':True,'direct_recovery_semantic_witness_reentry_alignment':True,'direct_recovery_semantic_witness_control_projection':True,'direct_recovery_semantic_witness_boundary_transport':False,'direct_recovery_semantic_witness_demand_normalized_fidelity':False,'direct_recovery_semantic_witness_robust_occupancy':False,'direct_recovery_semantic_witness_soft_occupancy_disagreement':False,'direct_recovery_semantic_witness_boundary_localized_occupancy_trust':False,'direct_recovery_semantic_witness_history_occupancy_reachability':False,'direct_recovery_semantic_witness_interaction_box_support':False,'direct_recovery_semantic_witness_interaction_hull_support':False,'direct_recovery_semantic_witness_interaction_anchor_support':False,'direct_recovery_semantic_witness_interaction_response_support':False}
 serializers={}
 models={}
 for label,fid in [('G77_TYPED_PROJ',False),('H77_MAIN_ACTSI',True)]:
  mc=dict(base_flags);mc['direct_recovery_semantic_witness_projection_fidelity_weighting']=fid;mc['direct_recovery_semantic_witness_active_constraint_typed_source']=True
  schema,source=_semantic_witness_checkpoint_feature_contract(mc);expected_schema=4 if fid else 3;expected_source='robust_trust_projected_recovery_witness' if fid else 'projected_boundary_common_executable_recovery_witness';ok=(schema,source)==(expected_schema,expected_source);serializers[label]={'schema':schema,'source':source,'expected_schema':expected_schema,'expected_source':expected_source,'valid':ok}
  if not ok:errors.append(f'{label} serializer mismatch')
  L=FlatFeatureLayout(feature_max_agents=2)
  m=OCRAPModel(input_dim=L.total_dim,num_roots=3,num_options=2,d_model=16,d_obs=8,encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,dropout=0.0,option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,direct_recovery_evidence_native_certificate_preservation=True,**mc)
  g=m.direct_absolute_semantic_witness_gain;mk=bool(g is not None and tuple(g.shape)==(6,2) and g.numel()==12 and torch.count_nonzero(g).item()==0 and m.direct_recovery_semantic_witness_active_constraint_typed_source)
  models[label]={'gain_shape':list(g.shape) if g is not None else None,'gain_numel':g.numel() if g is not None else None,'zero_init':bool(g is not None and torch.count_nonzero(g).item()==0),'valid':mk}
  if not mk:errors.append(f'{label} typed-source model contract failed')
 # Historical global interface stays execution-compatible and 2-parameter.
 L=FlatFeatureLayout(feature_max_agents=2);old=OCRAPModel(input_dim=L.total_dim,num_roots=3,num_options=2,d_model=16,d_obs=8,encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,dropout=0.0,option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,direct_recovery_absolute_semantic_witness_correction=True,direct_recovery_semantic_witness_active_set_alignment=True,direct_recovery_semantic_witness_path_stop_alignment=False,direct_recovery_semantic_witness_route_alignment=True,direct_recovery_semantic_witness_reentry_alignment=True,direct_recovery_semantic_witness_control_projection=True,direct_recovery_semantic_witness_projection_fidelity_weighting=False,direct_recovery_semantic_witness_active_constraint_typed_source=False,direct_recovery_evidence_native_certificate_preservation=True)
 old_ok=old.direct_absolute_semantic_witness_gain is not None and tuple(old.direct_absolute_semantic_witness_gain.shape)==(2,)
 if not old_ok:errors.append('historical global two-gain interface changed')
 t=(src/'ocrap/simulation/teacher/margins.py').read_text();teacher={'positive_structural_floor_0p6':'max(val, 0.6)' in t,'route_override_neg_0p8':'min(val, -0.8)' in t,'secondary_floor_0p9':'max(val, 0.9)' in t};teacher['valid']=all(teacher.values())
 valid=not errors and synth['valid'] and teacher['valid'] and old_ok and all(z['valid'] for z in serializers.values()) and all(z['valid'] for z in models.values())
 doc={'schema':'ocrap-v48.77-actsi-runtime-code-contract-v1','engineering_version':'v48.77.0-OC-ACTSI','valid':valid,'attribution_ready':valid,'errors':errors,'runtime_modules':mods,'serializer_contracts':serializers,'typed_source_model_contracts':models,'historical_global_two_gain_shape':[2],'historical_interface_valid':old_ok,'supervision_contract':{'truth_contract':'censor_exact_0p5','objective':'signed_margin_huber','huber_beta':1.0,'regime_conditioned':False,'teacher_future_input':False,'floor_relabelled':False,'synthetic_check':synth},'typed_source_contract':{'active_constraints':['clearance','stopping','control','stability','route','persistent_reentry'],'gain_columns':['positive_rescue','universal_failure'],'trainable_parameters':12,'option_id_input':False,'regime_id_input':False,'boundary_transport':False},'teacher_structural_contract_present':teacher,'dataset_reconstruction':False,'uses_test_roots':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_77_runtime_contract','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
