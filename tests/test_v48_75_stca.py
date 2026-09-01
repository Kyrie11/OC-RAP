from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import torch
from ocrap.cli.train import _absolute_feasibility_supervision_mask

REPO=Path(__file__).resolve().parents[1]

def test_v4875_legacy_truth_contract_execution_exact():
 batch={'r_dep_star':torch.tensor([0.5,0.500001,-0.2,0.7,0.5,0.5]),'is_nominal':torch.tensor([0.,0.,0.,0.,1.,0.]),'bucket_id':torch.tensor([1,1,2,2,1,0]),'time_index':torch.arange(6)}
 m,t,f=_absolute_feasibility_supervision_mask(batch,{'direct_value_absolute_feasibility_truth_contract':'legacy_full'})
 assert m.tolist()==[True,True,True,True,False,False]
 assert f.tolist()==[False]*6
 assert t.tolist()==[1.,1.,0.,1.,1.,1.]

def test_v4875_censor_only_exact_0p5_without_relabel():
 batch={'r_dep_star':torch.tensor([0.5,0.500001,-0.2,0.7,0.5,0.5]),'is_nominal':torch.tensor([0.,0.,0.,0.,1.,0.]),'bucket_id':torch.tensor([1,1,2,2,1,0]),'time_index':torch.arange(6)}
 lm,lt,_=_absolute_feasibility_supervision_mask(batch,{'direct_value_absolute_feasibility_truth_contract':'legacy_full'})
 cm,ct,cf=_absolute_feasibility_supervision_mask(batch,{'direct_value_absolute_feasibility_truth_contract':'censor_exact_0p5'})
 assert cm.tolist()==[False,True,True,True,False,False]
 assert cf.tolist()==[True,False,False,False,False,False]
 assert torch.equal(lt,ct)
 assert lm.sum().item()==4 and cm.sum().item()==3

def test_v4875_unknown_truth_contract_fails_closed():
 batch={'r_dep_star':torch.tensor([0.5]),'is_nominal':torch.tensor([0.]),'bucket_id':torch.tensor([1]),'time_index':torch.tensor([0])}
 try:_absolute_feasibility_supervision_mask(batch,{'direct_value_absolute_feasibility_truth_contract':'sweep_me'})
 except ValueError:return
 raise AssertionError('unknown truth contract must fail closed')

def test_v4875_shell_wires_truth_contract_to_training_config():
 low=(REPO/'scripts/train_ocrap_v48_trac_sr.sh').read_text()
 adapt=(REPO/'scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh').read_text()
 run=(REPO/'scripts/run_v48_75_dcp_drfc_bcde_rifa_stca_two_gpu.sh').read_text()
 assert 'training.direct_value_absolute_feasibility_truth_contract' in low
 assert 'ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=${ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT:-legacy_full}' in adapt or 'ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT="${ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT:-legacy_full}"' in adapt
 assert 'ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=censor_exact_0p5' in run
 assert 'OCRAP_V48_74_SIGNED_VIABILITY=0' in run

def test_v4875_runner_keeps_forbidden_mechanisms_off():
 s=(REPO/'scripts/run_v48_75_dcp_drfc_bcde_rifa_stca_two_gpu.sh').read_text()
 for text in ['SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false','SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY=false','SEMANTIC_WITNESS_ROBUST_OCCUPANCY=false','SEMANTIC_WITNESS_SOFT_OCCUPANCY_DISAGREEMENT=false','SEMANTIC_WITNESS_INTERACTION_BOX_SUPPORT=false','SEMANTIC_WITNESS_INTERACTION_HULL_SUPPORT=false','SEMANTIC_WITNESS_INTERACTION_ANCHOR_SUPPORT=false','SEMANTIC_WITNESS_INTERACTION_RESPONSE_SUPPORT=false']:
  assert text in s
 assert 'PROPOSAL_TOP_K=5' in s

def test_v4875_runtime_preflight_passes(tmp_path):
 out=tmp_path/'runtime.json'
 env=dict(__import__('os').environ); env['PYTHONPATH']=f"{REPO/'src'}:{REPO}";env['OCRAP_V48_74_SIGNED_VIABILITY']='0'
 r=subprocess.run([sys.executable,str(REPO/'tools/check_v48_75_runtime_code_contract.py'),'--repo',str(REPO),'--output',str(out)],env=env,capture_output=True,text=True)
 assert r.returncode==0,(r.stdout,r.stderr)
 d=json.loads(out.read_text());assert d['valid'] and d['attribution_ready'];assert d['truth_contract']['synthetic_check']['valid'];assert d['teacher_structural_contract_present']['valid']

def _ckpt(path:Path, *, adapted:bool, fidelity:bool=False):
 state={'w':torch.tensor([1.0])}
 if adapted:state['direct_absolute_semantic_witness_gain']=torch.tensor([0.2,0.0])
 d={'model_state':state}
 if adapted:
  d.update({'direct_recovery_absolute_semantic_witness_feature_schema':4 if fidelity else 3,'direct_recovery_absolute_semantic_witness_feature_source':'robust_trust_projected_recovery_witness' if fidelity else 'projected_boundary_common_executable_recovery_witness','direct_recovery_absolute_semantic_witness_correction':True,'direct_recovery_semantic_witness_active_set_alignment':True,'direct_recovery_semantic_witness_path_stop_alignment':False,'direct_recovery_semantic_witness_classlocal_transport':False,'direct_recovery_semantic_witness_route_alignment':True,'direct_recovery_semantic_witness_reentry_alignment':True,'direct_recovery_semantic_witness_control_projection':True,'direct_recovery_semantic_witness_boundary_transport':False,'direct_recovery_semantic_witness_projection_fidelity_weighting':fidelity,'direct_recovery_semantic_witness_demand_normalized_fidelity':False,'direct_recovery_semantic_witness_robust_occupancy':False,'direct_recovery_semantic_witness_soft_occupancy_disagreement':False,'direct_recovery_semantic_witness_boundary_localized_occupancy_trust':False,'direct_recovery_semantic_witness_history_occupancy_reachability':False,'direct_recovery_semantic_witness_interaction_box_support':False,'direct_recovery_semantic_witness_interaction_hull_support':False,'direct_recovery_semantic_witness_interaction_anchor_support':False,'direct_recovery_semantic_witness_interaction_response_support':False,'cfg':{'training':{'direct_value_absolute_feasibility_truth_contract':'censor_exact_0p5'}}})
 torch.save(d,path)

def test_v4875_state_isolation_projection_and_fidelity(tmp_path):
 ref=tmp_path/'ref.pt';_ckpt(ref,adapted=False)
 for fidelity in (False,True):
  dst=tmp_path/f"{'fid' if fidelity else 'proj'}.pt";out=tmp_path/f"{'fid' if fidelity else 'proj'}.json";_ckpt(dst,adapted=True,fidelity=fidelity)
  r=subprocess.run([sys.executable,str(REPO/'tools/check_v48_75_state_isolation.py'),'--reference',str(ref),'--adapted',str(dst),'--fidelity',str(fidelity).lower(),'--output',str(out)],capture_output=True,text=True)
  assert r.returncode==0,(r.stdout,r.stderr);d=json.loads(out.read_text());assert d['valid'] and d['truth_contract_valid'] and d['stage_i_bitwise_identity']
