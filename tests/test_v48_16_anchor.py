from __future__ import annotations
from pathlib import Path
import torch
from ocrap.models.data import expand_split_roles, split_id_matches
from ocrap.models.losses import direct_uncertainty_recovery_value_loss

ROOT=Path(__file__).parents[1]

def _loss(*, benefit_logit: float, harm_logit: float, class_balanced: float=0.0, benefit_margin: float=0.0, harm_margin: float=0.0, residual: torch.Tensor|None=None, anchor: float=0.0):
    return direct_uncertainty_recovery_value_loss(
      pred_logit=torch.zeros(3), pred_logvar=torch.zeros(3), pred_rank_logit=torch.tensor([0.,2.,1.]),
      pred_opportunity_logit=torch.tensor([0., harm_logit, benefit_logit]), pred_harm_logit=torch.tensor([0., harm_logit, -2.]),
      teacher_r_dep=torch.zeros(3),teacher_r_orc=torch.zeros(3),teacher_q=torch.ones((3,1,1)),
      teacher_m_star=torch.tensor([[[1.]], [[-1.]], [[1.]]]),root_probs=torch.ones((3,1)),root_valid=torch.ones((3,1),dtype=torch.bool),option_valid=torch.ones((3,1),dtype=torch.bool),
      scene_hash=torch.tensor([1,1,1]),time_index=torch.zeros(3,dtype=torch.long),macro_type_id=torch.tensor([0,5,3]),is_nominal=torch.tensor([1.,0.,0.]),bucket_id=torch.ones(3,dtype=torch.long),
      macro_ids=(3,5),bucket_ids=(1,),output_mode='score',exact_teacher_pcd=True,positive_gain=.01,negative_gain=.01,
      point_weight=0,centered_weight=0,listwise_weight=0,advantage_weight=0,pairwise_weight=0,top_rank_weight=0,opportunity_weight=0,harm_weight=0,setwise_admission_weight=0,policy_distill_weight=0,policy_regret_weight=0,preference_weight=0,preference_regret_weight=0,preference_listwise_weight=0,preference_gap_weight=0,preference_set_weight=0,preference_all_group_set_weight=0,delta_nll_weight=0,
      ordinal_evidence_ordered_nll_all_weight=.1,ordinal_evidence_proposal_topk_weight=1.,ordinal_evidence_proposal_topk=2,
      ordinal_evidence_class_balanced_weight=class_balanced,ordinal_evidence_benefit_margin_weight=benefit_margin,ordinal_evidence_harm_margin_weight=harm_margin,ordinal_evidence_target_probability=.6,
      evidence_calibrator_residual=residual,evidence_calibrator_anchor_weight=anchor,
    )

def test_dedicated_protocol_roles_are_expanded_without_val_leakage():
    assert expand_split_roles({'calibration'}) == {'calibration','certificate_pool'}
    assert split_id_matches('certificate_pool','calibration')
    assert split_id_matches('evidence_adapt_train','train')
    assert split_id_matches('evidence_adapt_dev','val')
    assert not split_id_matches('evidence_adapt_dev','certificate_pool')

def test_balanced_margins_penalize_all_abstain_evidence():
    base=_loss(benefit_logit=-4.,harm_logit=-4.)
    anchored=_loss(benefit_logit=-4.,harm_logit=-4.,class_balanced=1.,benefit_margin=1.,harm_margin=1.)
    assert anchored.item() > base.item()

def test_calibrator_anchor_penalizes_large_target_domain_residual():
    zero=_loss(benefit_logit=0.,harm_logit=0.,residual=torch.zeros(3,2),anchor=1.)
    large=_loss(benefit_logit=0.,harm_logit=0.,residual=torch.ones(3,2),anchor=1.)
    assert large.item() > zero.item()

def test_certificate_script_requires_nonempty_protocol_data():
    text=(ROOT/'scripts'/'calibrate_v48_14_certificate_pool.sh').read_text()
    assert '--allowed-splits=certificate_pool' in text
    assert 'empty policy certificate' in text
    assert 'certificate_data_valid' in text
