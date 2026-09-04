from __future__ import annotations

import json
from pathlib import Path
import subprocess,sys

ROLES=("dev_near","certificate_near","dev_contact","certificate_contact")


def role(*, exo=True, response=False, direction=True):
    q=lambda x:{"q10":x,"median":x,"q90":x,"mean":x}
    return {
      "labeled_rows":200,"safe_positive_rows":20,
      "recipe_unresolved_semantic_mass_candidate":q(0.0),"recipe_unresolved_semantic_mass_nominal":q(0.0),
      "duplicate_root_homogeneity_mass_candidate":q(1.0),"duplicate_root_homogeneity_mass_nominal":q(1.0),
      "recipe_shared_mass_candidate":q(1.0),"recipe_shared_mass_nominal":q(1.0),
      "recipe_tail_transport_coverage":q(1.0),"recipe_tail_partition_stability":q(1.0),
      "exogenous_unresolved_mass_candidate":q(0.0),"exogenous_unresolved_mass_nominal":q(0.0),
      "exogenous_shared_mass_candidate":q(1.0 if exo else .5),"exogenous_shared_mass_nominal":q(1.0 if exo else .5),
      "exogenous_tail_transport_coverage":q(1.0 if exo else .5),"exogenous_tail_transport_purity":q(1.0),
      "exogenous_tail_partition_stability":q(1.0 if exo else .5),
      "partition_stability_safe_vs_harmful_auc":.75 if direction else .5,
      "partition_stability_macro_stratified_auc":.72 if direction else .5,
      "partition_stability_safe_positive_mean":1.0,"partition_stability_harmful_mean":.7,
      "exogenous_transport_sign_identifiable_mass":q(.8 if response else .1),
      "exogenous_transport_informative_response_mass":q(.8 if response else .1),
      "response_safe_vs_harmful_auc":.7 if response else .5,
      "response_top1_lift":.2 if response else 0.0,
    }


def run_compare(tmp_path:Path, roles, prev_ok=True):
    summary=tmp_path/'s.json'; prev=tmp_path/'p.json'; out=tmp_path/'o.json'
    summary.write_text(json.dumps({"valid":True,"attribution_ready":True,"roles":roles}))
    prev_dec={"status":"COUNTERFACTUAL_ROOT_CORRESPONDENCE_STOP","root_correspondence_go":False,"next_branch":"do_not_train_new_source_then_audit_counterfactual_future_identity_and_root_partition_stability_no_encoder_or_adapter_sweep"}
    if not prev_ok: prev_dec["status"]="OTHER"
    prev.write_text(json.dumps({"valid":True,"preregistered_decision":prev_dec}))
    tool=Path(__file__).resolve().parents[1]/'tools/compare_v48_90_cept.py'
    cp=subprocess.run([sys.executable,str(tool),'--audit-summary',str(summary),'--v48-89-comparison',str(prev),'--output',str(out)],capture_output=True,text=True)
    return cp,json.loads(out.read_text())


def test_recipe_quotient_go_exogenous_stop_branch(tmp_path):
    roles={r:role(exo=False,response=False) for r in ROLES}
    cp,d=run_compare(tmp_path,roles)
    assert cp.returncode==0
    q=d['preregistered_decision']
    assert q['recipe_equivalence_quotient_go'] is True
    assert q['exogenous_partition_transport_go'] is False
    assert q['status']=='RECIPE_QUOTIENT_GO_EXOGENOUS_TRANSPORT_STOP'


def test_exogenous_transport_go_response_stop_keeps_only_rejector_scaffold(tmp_path):
    roles={r:role(exo=True,response=False,direction=True) for r in ROLES}
    cp,d=run_compare(tmp_path,roles)
    assert cp.returncode==0
    q=d['preregistered_decision']
    assert q['exogenous_partition_transport_go'] is True
    assert q['partition_stability_directional_relevance_go'] is True
    assert q['transport_physical_response_identifiability_go'] is False
    assert q['status']=='PARTITION_TRANSPORT_GO_PHYSICAL_RESPONSE_UNDERIDENTIFIED'
    assert 'structural_rejector_scaffold_only' in q['next_branch']


def test_full_transport_response_go_authorizes_only_fixed_capacity_next_stage(tmp_path):
    roles={r:role(exo=True,response=True,direction=True) for r in ROLES}
    cp,d=run_compare(tmp_path,roles)
    assert cp.returncode==0
    q=d['preregistered_decision']
    assert q['matched_transport_response_training_authorized'] is True
    assert q['status']=='COUNTERFACTUAL_PARTITION_TRANSPORT_RESPONSE_GO'
    assert 'fixed_capacity' in q['next_branch'] and 'no_boundary_transport' in q['next_branch']


def test_bad_v4889_prerequisite_fails_engineering_contract(tmp_path):
    roles={r:role(exo=True,response=True,direction=True) for r in ROLES}
    cp,d=run_compare(tmp_path,roles,prev_ok=False)
    assert cp.returncode==30
    assert d['valid'] is False


def test_runtime_contract_serializes_and_passes(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    tool=repo/'tools/check_v48_90_runtime_code_contract.py'
    out=tmp_path/'runtime.json'
    env=dict(__import__('os').environ)
    env['PYTHONPATH']=str(repo/'src')+__import__('os').pathsep+str(repo)+(__import__('os').pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONNOUSERSITE']='1'
    cp=subprocess.run([sys.executable,str(tool),'--repo',str(repo),'--output',str(out)],capture_output=True,text=True,env=env)
    assert cp.returncode==0, cp.stderr+cp.stdout
    d=json.loads(out.read_text())
    assert d['valid'] is True and d['attribution_ready'] is True
    assert d['engineering_version']=='v48.90.0-OC-CEPT'
    assert all(d['synthetic_checks'].values())
