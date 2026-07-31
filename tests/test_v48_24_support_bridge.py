from __future__ import annotations

from pathlib import Path

import torch

from test_v48_23_frontier_bridge import _loss_common


def test_safe_utility_directly_pushes_harmful_gain_below_safe_gain() -> None:
    admission = torch.zeros(3, requires_grad=True)
    loss = _loss_common(
        pred_opportunity_logit=torch.tensor([0.0, 4.0, 4.0]),
        pred_harm_logit=torch.tensor([0.0, 4.0, -4.0]),
        pred_component_harm_logits=torch.tensor(
            [[0.0, 0.0, 0.0], [4.0, -4.0, -4.0], [-4.0, -4.0, -4.0]]
        ),
        pred_admission_logit=admission,
        ordinal_evidence_safe_utility_regression_weight=1.0,
        ordinal_evidence_safe_utility_listwise_weight=0.75,
        ordinal_evidence_safe_utility_temperature=0.10,
    )
    loss.backward()
    assert admission.grad is not None
    # Candidate 1 has benefit but violates a component envelope; candidate 2 is
    # safe-beneficial. Gradient descent must lower the former and raise the latter.
    assert admission.grad[1].item() > 0.0
    assert admission.grad[2].item() < 0.0


def test_support_bridge_persists_oracle_width_curve_and_fit_only_shadow_rule() -> None:
    root = Path(__file__).resolve().parents[1]
    calibration = (root / "tools" / "calibrate_policy_risk_v48.py").read_text()
    assert 'support_k_values = sorted({' in calibration
    assert '"proposal_support_curve": proposal_support_curve' in calibration
    assert '"diagnostic_selector_overrides": diagnostic_selector_overrides' in calibration
    assert 'selected_from": "fit_nearest_frontier"' in calibration
    assert "proposal-constrained safe-positive oracle cannot satisfy" in calibration


def test_closed_loop_loads_the_complete_certified_selector_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = (root / "scripts" / "run_ocrap_v48_trac_sr.sh").read_text()
    shadow = (root / "scripts" / "run_v48_24_dev_shadow_closed_loop.sh").read_text()
    for key in (
        "direct_value_proposal_top_k_by_bucket",
        "direct_value_evidence_rerank_top_k_by_bucket",
        "direct_value_min_rank_margin_by_bucket",
        "direct_value_conditional_rank_margin_by_bucket",
    ):
        assert key in runtime
    assert "DEV_SHADOW_DIAGNOSTIC=1" in shadow
    assert "fit_nearest_frontier_diagnostic_only" in runtime


def test_v48_24_uses_safe_labels_one_action_training_and_two_gpu_waves() -> None:
    root = Path(__file__).resolve().parents[1]
    adapt = (root / "scripts" / "adapt_ocrap_v48_24_support_variant.sh").read_text()
    parallel = (root / "scripts" / "run_v48_24_parallel_ablations.sh").read_text()
    assert "OPPORTUNITY_LABEL_MODE=safe_benefit" in adapt
    assert 'GROUP_OPPORTUNITY_WEIGHT="${ORDINAL_EVIDENCE_GROUP_OPPORTUNITY_WEIGHT:-0.0}"' in adapt
    assert "ORDINAL_EVIDENCE_SAFE_UTILITY_REGRESSION_WEIGHT" in adapt
    assert "ORDINAL_EVIDENCE_SAFE_UTILITY_LISTWISE_WEIGHT" in adapt
    assert "max_concurrent_tasks':2" in parallel
    assert "one task per A30" in parallel
    assert "A_top3_safe_label_baseline" in parallel
    assert "D_full_support_bridge" in parallel
