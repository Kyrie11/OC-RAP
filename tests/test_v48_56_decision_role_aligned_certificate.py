from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from ocrap.algorithms.evidence_targets import ComponentVetoTolerances, component_veto_terms_numpy
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel

ROOT=Path(__file__).resolve().parents[1]

def _model(dep_abs: bool, reliability: str="1,1,1,0,0")->OCRAPModel:
    return OCRAPModel(
        input_dim=FlatFeatureLayout().total_dim,num_roots=3,num_options=4,d_model=12,d_obs=6,
        encoder_type='structured_transformer',num_layers=1,num_heads=3,dropout=0.0,
        direct_recovery_value_head=True,direct_recovery_value_pooling='candidate_concat_raw',
        direct_recovery_delta_head=True,direct_recovery_delta_mode='ordinal_evidence',
        direct_recovery_delta_regime_experts=True,direct_recovery_delta_policy_features=True,
        direct_recovery_evidence_calibrator=True,direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_source='physical_interaction',
        direct_recovery_evidence_interaction_hidden=16,direct_recovery_evidence_interaction_dropout=0.0,
        direct_recovery_evidence_dual_interaction_bridge=True,direct_recovery_evidence_roct_benefit=True,
        direct_recovery_evidence_roct_deployability=True,direct_recovery_evidence_roct_scale=3.0,
        direct_recovery_evidence_roct_alpha=.2,direct_recovery_evidence_roct_beta=.2,
        direct_recovery_evidence_roct_top_m=8,direct_recovery_evidence_roct_option_temperature=.35,
        direct_recovery_evidence_native_certificate_preservation=True,
        direct_recovery_evidence_native_dep_boundary_aligned=dep_abs,
        direct_recovery_evidence_native_drs_tolerance=.05,direct_recovery_evidence_native_deployability_tolerance=.05,
        direct_recovery_evidence_unified_experts=True,direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=5,direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=False,direct_recovery_evidence_admission_prior_mode='joint_reserve',
        direct_recovery_evidence_reserve_factor_alignment=True,direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_reliability=reliability)

def test_dep_boundary_is_absolute_and_nominal_invariant():
    t=ComponentVetoTolerances(deployability_boundary_aligned=True)
    a=component_veto_terms_numpy(candidate_drs=1,nominal_drs=1,candidate_r_dep=-.2,nominal_r_dep=-3,candidate_gap=0,nominal_gap=0,tolerances=t)
    b=component_veto_terms_numpy(candidate_drs=1,nominal_drs=1,candidate_r_dep=-.2,nominal_r_dep=3,candidate_gap=0,nominal_gap=0,tolerances=t)
    assert a[1] > 0 and np.isclose(a[1],b[1])
    c=component_veto_terms_numpy(candidate_drs=1,nominal_drs=1,candidate_r_dep=.2,nominal_r_dep=-3,candidate_gap=0,nominal_gap=0,tolerances=t)
    assert c[1] < 0

def test_gap_ordinal_only_never_owns_hard_veto_but_dep_still_can():
    t=ComponentVetoTolerances(deployability_boundary_aligned=True,gap_ordinal_only=True)
    x=component_veto_terms_numpy(candidate_drs=1,nominal_drs=1,candidate_r_dep=.5,nominal_r_dep=.5,candidate_gap=20,nominal_gap=0,tolerances=t)
    assert x[2] < 0 and max(x) <= 0
    y=component_veto_terms_numpy(candidate_drs=1,nominal_drs=1,candidate_r_dep=-.5,nominal_r_dep=.5,candidate_gap=0,nominal_gap=0,tolerances=t)
    assert y[1] > 0 and y[2] < 0 and max(y) > 0

def test_native_dep_component_uses_exact_sigmoid_half_boundary():
    m=_model(True)
    native=torch.tensor([[.8,.9],[.8,.4],[.8,.6]],dtype=torch.float32)
    groups=torch.tensor([[0],[0],[0]]); nominal=torch.tensor([1.,0.,0.])
    _, margins=m._native_certificate_component_logits(native,groups,nominal)
    assert margins is not None
    assert torch.allclose(margins[:,1], torch.tensor([-.4,.1,-.1]), atol=1e-6)
    # v48.56 adds no learned parameters.
    assert set(_model(False).state_dict())==set(m.state_dict())

def test_gap_ordinal_reliability_neutralizes_deployed_gap_hard_veto():
    torch.manual_seed(4856)
    m=_model(True, reliability='1,1,0,0,0').eval()
    x=torch.randn(4,FlatFeatureLayout().total_dim)
    groups=torch.tensor([[0],[0],[1],[1]]); nominal=torch.tensor([1.,0.,1.,0.])
    rv=torch.ones((4,3),dtype=torch.bool); ov=torch.ones((4,4),dtype=torch.bool)
    with torch.no_grad():
        out=m(x,bucket_id=torch.ones(4,dtype=torch.long),group_index=groups,is_nominal=nominal,direct_only=True,root_valid=rv,option_valid=ov)
    probs=out['direct_recovery_evidence_component_harm_probabilities']
    # Reliability 0 maps GAP to the semantic non-harm prior sigmoid(-2), so it
    # cannot independently cross the 0.5 hard-veto threshold.
    rec=nominal < .5
    assert torch.allclose(probs[rec,2], torch.full_like(probs[rec,2], torch.sigmoid(torch.tensor(-2.0))), atol=1e-6)
    assert torch.allclose(probs[~rec,2], torch.full_like(probs[~rec,2], .5), atol=1e-6)


def test_v4856_factor_map_and_fresh_control_are_exact():
    arm=(ROOT/'scripts/run_v48_56_dcp_drfc_bcde_drac_arm.sh').read_text()
    launch=(ROOT/'scripts/run_v48_56_dcp_drfc_bcde_drac_two_gpu.sh').read_text()
    assert 'EVIDENCE_DEP_BOUNDARY_ALIGNED=true' in arm
    assert 'EVIDENCE_GAP_ORDINAL_ONLY=true' in arm
    assert 'FACTOR_COMPONENT_MARGIN_REGRESSION_RELIABILITY="1,1,0,0,0"' in arm
    assert "'gap_in_hard_component_veto':not y" in arm
    assert 'run_arm A "$A_RUN" "$GPU0" "$GPU1" 0' in launch
    assert 'check_v48_56_reference_reuse.py' not in launch
    assert 'compare_v48_56_dcp_drfc_bcde_drac_2x2.py' in launch

def test_teacher_index_contract_seals_decision_role_flags(tmp_path:Path):
    root=tmp_path/'d'; root.mkdir(); (root/'manifest.csv').write_text('x\n')
    summary=tmp_path/'s.json'; out=tmp_path/'o.json'
    summary.write_text(json.dumps({'index_contract':{'dataset_roots':[str(root.resolve())],
      'dataset_manifests':[{'root':str(root.resolve()),'manifest':str(root/'manifest.csv'),'manifest_sha256':__import__('hashlib').sha256((root/'manifest.csv').read_bytes()).hexdigest()}],
      'alpha':.2,'beta':.2,'top_m':8,'positive_gain':.015,'deployable_macro_ids':[2,3,5,6,7],
      'component_harm_tolerances':{'drs':.05,'deployability_gate':.05,'gap_discount':.05,'hard_violation':.05,'harm_proxy':.05,'deployability_boundary_aligned':True,'gap_ordinal_only':True}}}))
    cmd=[sys.executable,str(ROOT/'tools/check_v48_19_target_support.py'),'--summary',str(summary),'--expected-dataset',str(root),'--mode','contract','--dep-boundary-aligned','--gap-ordinal-only','--output',str(out)]
    assert subprocess.run(cmd,cwd=ROOT).returncode==0
    stale=cmd.copy(); stale.remove('--gap-ordinal-only')
    assert subprocess.run(stale,cwd=ROOT).returncode==5


def _load_v4856_comparator():
    import importlib.util
    path=ROOT/'tools/compare_v48_56_dcp_drfc_bcde_drac_2x2.py'
    spec=importlib.util.spec_from_file_location('v4856cmp', path)
    assert spec is not None and spec.loader is not None
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_fixed_external_labels_do_not_move_with_arm_semantics():
    cmp=_load_v4856_comparator()
    # Positive total PCD benefit plus a large GAP degradation: legacy promotes
    # GAP to an independent hard veto, whereas DRAC keeps it ordinal-only.  The
    # fixed comparator must expose that semantic disagreement explicitly.
    gap_conflict={
      'teacher_adv':.03,'pred_adv':.2,'scene':'s','time':1,'deployed_rule_chosen':True,
      'teacher_candidate_drs':1.,'teacher_nominal_drs':1.,
      'teacher_candidate_r_dep':.4,'teacher_nominal_r_dep':.4,
      'teacher_candidate_gap':3.,'teacher_nominal_gap':0.,
      'teacher_candidate_hard':0.,'teacher_nominal_hard':0.,
      'teacher_candidate_harm_proxy':0.,'teacher_nominal_harm_proxy':0.,
    }
    legacy=cmp._fixed_semantic_readout([gap_conflict],.015,dep_boundary_aligned=False,gap_ordinal_only=False)
    drac=cmp._fixed_semantic_readout([gap_conflict],.015,dep_boundary_aligned=True,gap_ordinal_only=True)
    assert legacy['raw_benefit_and_harm_conflicts']==1 and legacy['safe_positive_candidates']==0
    assert drac['raw_benefit_and_harm_conflicts']==0 and drac['safe_positive_candidates']==1
    # Absolute non-deployability remains harmful under DRAC even if the total
    # teacher benefit is positive; removing GAP hard veto is not label relaxing.
    rdep_conflict=dict(gap_conflict, teacher_candidate_r_dep=-.4, teacher_candidate_gap=0.)
    drac2=cmp._fixed_semantic_readout([rdep_conflict],.015,dep_boundary_aligned=True,gap_ordinal_only=True)
    assert drac2['raw_benefit_and_harm_conflicts']==1 and drac2['safe_positive_candidates']==0


def test_postgate_requires_full_drac_and_forbids_tcbc_artifact():
    text=(ROOT/'scripts/run_v48_56_postgate_if_authorized.sh').read_text()
    assert 'factor_x_deployability_zero_boundary' in text
    assert 'factor_y_gap_ordinal_only' in text
    assert 'component_margin_regression_reliability' in text
    assert '1,1,0,0,0' in text
    assert 'V48_55_COMPONENT_BOUNDARY_SCALES.json' in text
    assert 'unexpected TCBC scale artifact' in text
