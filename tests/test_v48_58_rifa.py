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
