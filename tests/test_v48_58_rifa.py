from __future__ import annotations
import torch
from ocrap.models.ocrap import OCRAPModel
from ocrap.cli.train import _absolute_feasibility_bce
from tools.calibrate_policy_risk_v48 import _top1


def _model() -> OCRAPModel:
    return OCRAPModel(
        input_dim=5, num_roots=3, num_options=2, d_model=16, d_obs=8,
        num_heads=4, dropout=0.0, direct_recovery_value_head=True,
        direct_recovery_absolute_feasibility_head=True,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def test_afe_initialization_is_exact_native_dep_zero_boundary() -> None:
    torch.manual_seed(58)
    model=_model().eval()
    x=torch.randn(3,5); groups=torch.zeros((3,1),dtype=torch.long); nominal=torch.tensor([1.,0.,0.])
    rv=torch.ones((3,3),dtype=torch.bool); ov=torch.ones((3,2),dtype=torch.bool)
    out=model(x,group_index=groups,is_nominal=nominal,root_valid=rv,option_valid=ov)
    native=out['direct_recovery_evidence_native_certificate']
    logit=out['direct_recovery_absolute_feasibility_logit']
    assert torch.allclose(logit, 4.0*native[:,1]-2.0, atol=1e-6, rtol=0)


def test_afe_features_are_detached_from_stage_i() -> None:
    torch.manual_seed(5801)
    model=_model().train()
    x=torch.randn(3,5); groups=torch.zeros((3,1),dtype=torch.long); nominal=torch.tensor([1.,0.,0.])
    rv=torch.ones((3,3),dtype=torch.bool); ov=torch.ones((3,2),dtype=torch.bool)
    out=model(x,group_index=groups,is_nominal=nominal,root_valid=rv,option_valid=ov)
    out['direct_recovery_absolute_feasibility_logit'].sum().backward()
    assert model.direct_absolute_feasibility_head.weight.grad is not None
    assert model.direct_absolute_feasibility_head.bias.grad is not None
    leaked=[n for n,p in model.named_parameters() if not n.startswith('direct_absolute_feasibility_head.') and p.grad is not None and torch.any(p.grad != 0)]
    assert leaked == []


def test_afe_bce_is_candidate_only_and_critical_regime_only() -> None:
    logits=torch.tensor([0.0,0.0,0.0,0.0],requires_grad=True)
    out={'direct_recovery_absolute_feasibility_logit':logits}
    batch={
      'r_dep_star':torch.tensor([-1.0,1.0,1.0,-1.0]),
      'is_nominal':torch.tensor([1.0,0.0,0.0,0.0]),
      'bucket_id':torch.tensor([1,1,0,2]),
      'time_index':torch.arange(4),
    }
    # Only rows 1 (target 1) and 3 (target 0) are included -> BCE(logit=0)=ln2.
    loss=_absolute_feasibility_bce(out,batch)
    assert torch.allclose(loss, torch.tensor(0.69314718), atol=1e-6)


def test_rifa_filters_after_frozen_rank_topk_before_relative_evidence() -> None:
    pairs=[
      {'candidate':1,'macro':2,'rank_adv':3.0,'pred_adv':2.0,'opportunity':.9,'harm':.1,'absolute_feasibility_pass':False},
      {'candidate':2,'macro':2,'rank_adv':2.0,'pred_adv':1.0,'opportunity':.9,'harm':.1,'absolute_feasibility_pass':True},
      {'candidate':3,'macro':2,'rank_adv':1.0,'pred_adv':.5,'opportunity':.9,'harm':.1,'absolute_feasibility_pass':True},
    ]
    selected=_top1([{'pairs':pairs,'scene':'s','time':1,'fold':0,'oracle_best_teacher_adv':1.0}],.5,.5,{2},proposal_top_k=2,evidence_rerank_top_k=True)
    assert len(selected)==1
    assert selected[0]['candidate']==2
    assert selected[0]['proposal_rank']==2


def test_rifa_does_not_expand_the_frozen_proposal() -> None:
    pairs=[
      {'candidate':1,'macro':2,'rank_adv':3.0,'pred_adv':3.0,'opportunity':.9,'harm':.1,'absolute_feasibility_pass':False},
      {'candidate':2,'macro':2,'rank_adv':2.0,'pred_adv':2.0,'opportunity':.9,'harm':.1,'absolute_feasibility_pass':False},
      {'candidate':3,'macro':2,'rank_adv':1.0,'pred_adv':1.0,'opportunity':.9,'harm':.1,'absolute_feasibility_pass':True},
    ]
    # Candidate 3 is feasible but outside top-2; lexicographic Stage II abstains
    # rather than silently replacing Stage-I's proposal with a lower-ranked action.
    assert _top1([{'pairs':pairs,'scene':'s','time':1,'fold':0,'oracle_best_teacher_adv':1.0}],.5,.5,{2},proposal_top_k=2,evidence_rerank_top_k=True)==[]


def test_enabling_afe_does_not_change_stage_i_outputs_before_training() -> None:
    torch.manual_seed(5858)
    base=OCRAPModel(
        input_dim=5, num_roots=3, num_options=2, d_model=16, d_obs=8,
        num_heads=4, dropout=0.0, direct_recovery_value_head=True,
        direct_recovery_evidence_native_certificate_preservation=True,
        direct_recovery_absolute_feasibility_head=False,
    ).eval()
    rifa=_model().eval()
    shared={k:v for k,v in base.state_dict().items() if k in rifa.state_dict()}
    missing,unexpected=rifa.load_state_dict(shared,strict=False)
    assert set(missing)=={
        'direct_absolute_feasibility_head.weight',
        'direct_absolute_feasibility_head.bias',
    }
    assert unexpected==[]
    x=torch.randn(4,5); groups=torch.zeros((4,1),dtype=torch.long); nominal=torch.tensor([1.,0.,0.,0.])
    rv=torch.ones((4,3),dtype=torch.bool); ov=torch.ones((4,2),dtype=torch.bool)
    with torch.inference_mode():
        a=base(x,group_index=groups,is_nominal=nominal,root_valid=rv,option_valid=ov)
        b=rifa(x,group_index=groups,is_nominal=nominal,root_valid=rv,option_valid=ov)
    for key in ('root_logits','margins','c_star','direct_recovery_value_logit',
                'direct_recovery_evidence_native_certificate'):
        assert torch.equal(a[key],b[key]), key


def test_v58_launcher_keeps_parallel_variants_path_isolated() -> None:
    from pathlib import Path
    script = (Path(__file__).resolve().parents[1] / 'scripts' / 'run_v48_58_dcp_drfc_bcde_rifa_two_gpu.sh').read_text()
    assert 'local v src dst' in script
    assert 'local v="$1"\n  local gpu="$2"\n  local src="$REFERENCE_A/candidates/$v"\n  local dst="$C_RUN/candidates/$v"' in script
    assert 'local v="$1" gpu="$2" src="$REFERENCE_A/candidates/$v"' not in script
    assert 'check_v48_58_variant_isolation.py' in script


def test_v58_launcher_forbids_legacy_admission_head_and_allows_only_afe_init_missing() -> None:
    from pathlib import Path
    script = (Path(__file__).resolve().parents[1] / 'scripts' / 'run_v48_58_dcp_drfc_bcde_rifa_two_gpu.sh').read_text()
    assert 'export EVIDENCE_ADMISSION_HEAD=false' in script
    assert 'STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_feasibility_head' in script


def test_v58_metric_contract_accepts_registered_rifa_selection_semantics() -> None:
    from tools.check_v48_36_metric_calibration_contract import (
        _selection_contract, LEGACY_SELECTION_SEMANTICS, RIFA_SELECTION_SEMANTICS,
    )
    legacy=_selection_contract(
        {"SELECTION_SEMANTICS":LEGACY_SELECTION_SEMANTICS},
        {"selection_semantics":LEGACY_SELECTION_SEMANTICS},
    )
    assert legacy["mode"] == "off"
    assert legacy["mode_valid"] and legacy["threshold_valid"] and legacy["selection_semantics_valid"]
    for mode in ("native","learned"):
        rifa=_selection_contract(
            {"SELECTION_SEMANTICS":RIFA_SELECTION_SEMANTICS,
             "ABSOLUTE_FEASIBILITY_MODE":mode,
             "ABSOLUTE_FEASIBILITY_THRESHOLD":"0.5"},
            {"selection_semantics":RIFA_SELECTION_SEMANTICS,
             "absolute_feasibility_mode":mode,
             "absolute_feasibility_threshold":0.5},
        )
        assert rifa["mode_valid"] and rifa["threshold_valid"] and rifa["selection_semantics_valid"]
        assert rifa["expected_selection_semantics"] == RIFA_SELECTION_SEMANTICS


def test_v58_metric_contract_rejects_legacy_order_when_rifa_is_enabled() -> None:
    from tools.check_v48_36_metric_calibration_contract import (
        _selection_contract, LEGACY_SELECTION_SEMANTICS, RIFA_SELECTION_SEMANTICS,
    )
    doc=_selection_contract(
        {"SELECTION_SEMANTICS":LEGACY_SELECTION_SEMANTICS,
         "ABSOLUTE_FEASIBILITY_MODE":"native",
         "ABSOLUTE_FEASIBILITY_THRESHOLD":"0.5"},
        {"selection_semantics":RIFA_SELECTION_SEMANTICS,
         "absolute_feasibility_mode":"native",
         "absolute_feasibility_threshold":0.5},
    )
    assert doc["mode_valid"] and doc["threshold_valid"]
    assert not doc["selection_semantics_valid"]


def test_v58_native_policy_rewrite_removes_conflicting_selector_keys(tmp_path) -> None:
    import subprocess, sys
    from pathlib import Path
    contract=tmp_path/'POLICY_CONTRACT.env'
    contract.write_text(
        'PROPOSAL_TOP_K=5\n'
        'SELECTION_SEMANTICS=rank_topk_then_filter_then_evidence_rerank\n'
        'ABSOLUTE_FEASIBILITY_MODE=off\n'
        'ABSOLUTE_FEASIBILITY_THRESHOLD=0.25\n', encoding='utf-8'
    )
    tool=Path(__file__).resolve().parents[1]/'tools'/'rewrite_v48_58_policy_contract.py'
    subprocess.run([sys.executable,str(tool),'--contract',str(contract),'--mode','native','--threshold','0.5'],check=True)
    rows=[x for x in contract.read_text(encoding='utf-8').splitlines() if '=' in x]
    vals={}
    counts={}
    for row in rows:
        k,v=row.split('=',1); vals[k]=v; counts[k]=counts.get(k,0)+1
    for k in ('SELECTION_SEMANTICS','ABSOLUTE_FEASIBILITY_MODE','ABSOLUTE_FEASIBILITY_THRESHOLD'):
        assert counts[k] == 1
    assert vals['ABSOLUTE_FEASIBILITY_MODE']=='native'
    assert vals['ABSOLUTE_FEASIBILITY_THRESHOLD']=='0.5'
    assert vals['SELECTION_SEMANTICS']=='rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank'


def test_v58_launcher_pins_reference_snapshot_and_hotfix_version() -> None:
    from pathlib import Path
    script=(Path(__file__).resolve().parents[1]/'scripts'/'run_v48_58_dcp_drfc_bcde_rifa_two_gpu.sh').read_text()
    assert 'rewrite_v48_58_policy_contract.py' in script
    assert '--reference-contract "$REF_AUDIT"' in script
    assert 'v48.58.2-RIFA-SELECTION-CONTRACT-HOTFIX' in script


def test_v58_comparison_reads_real_nested_deployment_and_rifa_fields(tmp_path) -> None:
    import json
    from tools.compare_v48_58_rifa import metric
    run=tmp_path/'run'; cal=run/'candidates'/'balanced'/'calibration'; cal.mkdir(parents=True)
    (cal/'dev_diagnostic_near_v48.json').write_text(json.dumps({
        'development_fit_only':True,'certificate_mode':'development_fit_only','valid_for_deployment':False,
        'selection_rule':'rifa','absolute_feasibility_mode':'learned','absolute_feasibility_threshold':0.5,
        'fit':{'num_groups':10,'num_selected':4,'positive_recall':0.75,'precision':0.5,'precision_wilson_lcb90':0.2,
               'harmful_group_exposure_ucb90':0.1,'harmful_selected_ucb90':0.3},
        'candidate_safe_positive_auc':0.7,'proposal_evidence_top1_safe_positive_auc':0.8,
        'proposal_deployed_rule_selected_count':4,'proposal_deployed_rule_abstention_rate':0.6,
    }))
    d=metric(run,'balanced','dev_near')
    assert d['development_or_certificate_phase']=='fit'
    assert d['deployment']['positive_recall']==0.75
    assert d['ranking_and_selector_diagnostics']['proposal_deployed_rule_selected_count']==4
    assert d['absolute_feasibility_mode']=='learned'


def test_v58_launcher_requires_final_pipeline_complete_sentinel() -> None:
    from pathlib import Path
    script=(Path(__file__).resolve().parents[1]/'scripts'/'run_v48_58_dcp_drfc_bcde_rifa_two_gpu.sh').read_text()
    assert 'check_v48_58_pipeline_complete.py' in script
    assert 'OC-RAP-v48.58-PIPELINE_COMPLETE.json' in script
    assert '"$PIPELINE_COMPLETE"' in script
