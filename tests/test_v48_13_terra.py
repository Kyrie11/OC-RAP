from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ocrap.models.losses import direct_uncertainty_recovery_value_loss

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("calibrate_policy_risk_v48_terra", ROOT / "tools" / "calibrate_policy_risk_v48.py")
assert _spec is not None and _spec.loader is not None
_cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cal)


def _loss(rank: torch.Tensor, opp: torch.Tensor | None = None, harm: torch.Tensor | None = None, **extra) -> torch.Tensor:
    kwargs = dict(
        pred_logit=torch.zeros(4), pred_logvar=torch.zeros(4), pred_rank_logit=rank,
        pred_opportunity_logit=opp, pred_harm_logit=harm,
        teacher_r_dep=torch.tensor([0.0, 2.0, 0.0, 0.0]),
        teacher_r_orc=torch.tensor([0.0, 2.0, 0.0, 0.0]),
        teacher_q=torch.ones((4, 1, 1)),
        teacher_m_star=torch.tensor([[[1.0]], [[1.0]], [[1.0]], [[-1.0]]]),
        root_probs=torch.ones((4, 1)), root_valid=torch.ones((4, 1), dtype=torch.bool),
        option_valid=torch.ones((4, 1), dtype=torch.bool),
        scene_hash=torch.tensor([101, 101, 101, 101]), time_index=torch.ones(4, dtype=torch.long),
        macro_type_id=torch.tensor([0, 5, 3, 2]), is_nominal=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        bucket_id=torch.ones(4, dtype=torch.long), macro_ids=(2, 3, 5), bucket_ids=(1,),
        output_mode="score", exact_teacher_pcd=True, positive_gain=0.01, negative_gain=0.01,
        point_weight=0.0, centered_weight=0.0, listwise_weight=0.0, advantage_weight=0.0,
        pairwise_weight=0.0, top_rank_weight=0.0, opportunity_weight=0.0, harm_weight=0.0,
        setwise_admission_weight=0.0, policy_distill_weight=0.0, policy_regret_weight=0.0,
        preference_weight=0.0, preference_regret_weight=0.0, preference_listwise_weight=0.0,
        preference_gap_weight=0.0, preference_set_weight=0.0, preference_all_group_set_weight=0.0,
        delta_nll_weight=0.0,
    )
    kwargs.update(extra)
    return direct_uncertainty_recovery_value_loss(**kwargs)


def test_topk_proposal_loss_rewards_acceptable_candidate_in_proposal() -> None:
    good = _loss(
        torch.tensor([0.0, 1.5, 1.0, -1.0]),
        preference_conditional_set_weight=1.0,
        preference_conditional_regret_weight=0.0,
        preference_proposal_topk_weight=1.0,
        preference_proposal_topk=2,
    )
    bad = _loss(
        torch.tensor([0.0, -1.0, 1.5, 1.0]),
        preference_conditional_set_weight=1.0,
        preference_conditional_regret_weight=0.0,
        preference_proposal_topk_weight=1.0,
        preference_proposal_topk=2,
    )
    assert good.item() < bad.item()


def test_intragroup_ordinal_evidence_rewards_harm_separation() -> None:
    rank = torch.tensor([0.0, 2.0, 1.0, 0.5])
    # Candidate 1 beneficial, candidate 2 dead, candidate 3 harmful.
    good_opp = torch.tensor([0.0, 3.0, 0.0, -2.0])
    good_harm = torch.tensor([0.0, -3.0, -1.0, 3.0])
    bad_opp = -good_opp
    bad_harm = -good_harm
    good = _loss(
        rank, good_opp, good_harm,
        ordinal_evidence_ordered_nll_all_weight=0.1,
        ordinal_evidence_proposal_topk_weight=1.0,
        ordinal_evidence_proposal_topk=3,
        ordinal_evidence_intragroup_benefit_weight=1.0,
        ordinal_evidence_intragroup_harm_weight=1.0,
    )
    bad = _loss(
        rank, bad_opp, bad_harm,
        ordinal_evidence_ordered_nll_all_weight=0.1,
        ordinal_evidence_proposal_topk_weight=1.0,
        ordinal_evidence_proposal_topk=3,
        ordinal_evidence_intragroup_benefit_weight=1.0,
        ordinal_evidence_intragroup_harm_weight=1.0,
    )
    assert good.item() < bad.item()


def test_calibration_reranks_only_inside_frozen_topk_proposal() -> None:
    groups = [{
        "scene": "s", "time": 0, "fold": 0, "oracle_best_teacher_adv": 0.3,
        "pairs": [
            {"macro": 5, "candidate": 1, "rank_adv": 3.0, "pred_adv": 0.2, "opportunity": 0.8, "harm": 0.1, "teacher_adv": 0.1},
            {"macro": 3, "candidate": 2, "rank_adv": 2.0, "pred_adv": 0.6, "opportunity": 0.9, "harm": 0.1, "teacher_adv": 0.3},
            # Highest evidence but outside top-2; must never be selected.
            {"macro": 2, "candidate": 3, "rank_adv": 1.0, "pred_adv": 0.9, "opportunity": 0.95, "harm": 0.05, "teacher_adv": -0.2},
        ],
    }]
    selected = _cal._top1(
        groups, 0.5, 0.3, {2, 3, 5}, conditional_rank_margin=True,
        proposal_top_k=2, evidence_rerank_top_k=True,
    )
    assert len(selected) == 1
    assert selected[0]["candidate"] == 2
    assert selected[0]["proposal_rank"] == 2


def test_controller_sources_staged_policy_contract() -> None:
    controller = (ROOT / "run_v48_two_gpu_fast_commands.txt").read_text()
    stage = (ROOT / "scripts" / "train_ocrap_v48_13_terra.sh").read_text()
    assert 'source "$run/POLICY_CONTRACT.env"' in controller
    assert 'EVIDENCE_RERANK_TOP_K="${EVIDENCE_RERANK_TOP_K:-true}"' in stage
    assert "PREFERENCE_PROPOSAL_TOPK_WEIGHT" in stage


def test_stage_e_checkpoint_metric_matches_proposal_rerank_contract() -> None:
    stage = (ROOT / "scripts" / "train_ocrap_v48_13_terra.sh").read_text()
    trainer = (ROOT / "scripts" / "train_ocrap_v48_trac_sr.sh").read_text()
    train_py = (ROOT / "src" / "ocrap" / "cli" / "train.py").read_text()
    assert 'POLICY_METRIC_PROPOSAL_TOP_K="$PROPOSAL_TOP_K"' in stage
    assert 'POLICY_METRIC_EVIDENCE_RERANK_TOP_K="$EVIDENCE_RERANK_TOP_K"' in stage
    assert "direct_policy_metric_proposal_top_k" in trainer
    assert "metric_evidence_rerank" in train_py
    assert "certificate_positive_regret_sum" in train_py
