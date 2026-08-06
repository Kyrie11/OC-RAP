from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from ocrap.models.losses import direct_uncertainty_recovery_value_loss

ROOT = Path(__file__).resolve().parents[1]


def _benefit_margin_loss(positive_gain: float) -> tuple[float, torch.Tensor]:
    opportunity = torch.zeros(3, requires_grad=True)
    loss = direct_uncertainty_recovery_value_loss(
        pred_logit=torch.zeros(3),
        pred_logvar=torch.zeros(3),
        pred_rank_logit=torch.tensor([0.0, 3.0, 2.0]),
        pred_opportunity_logit=opportunity,
        teacher_r_dep=torch.tensor([0.0, 1.4, 1.4]),
        teacher_r_orc=torch.tensor([0.0, 1.4, 1.4]),
        teacher_q=torch.ones((3, 5, 1)),
        teacher_m_star=torch.tensor(
            [
                [[1.0], [1.0], [1.0], [1.0], [1.0]],
                [[-1.0], [1.0], [1.0], [1.0], [1.0]],
                [[1.0], [1.0], [1.0], [1.0], [1.0]],
            ]
        ),
        teacher_hard_violation=torch.zeros(3),
        teacher_harm_proxy=torch.zeros(3),
        root_probs=torch.full((3, 5), 0.2),
        root_valid=torch.ones((3, 5), dtype=torch.bool),
        option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.tensor([4837, 4837, 4837]),
        time_index=torch.zeros(3, dtype=torch.long),
        macro_type_id=torch.tensor([0, 2, 3]),
        is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.ones(3, dtype=torch.long),
        macro_ids=(2, 3),
        bucket_ids=(1,),
        output_mode="score",
        exact_teacher_pcd=True,
        positive_gain=positive_gain,
        negative_gain=0.01,
        point_weight=0.0,
        centered_weight=0.0,
        listwise_weight=0.0,
        advantage_weight=0.0,
        pairwise_weight=0.0,
        top_rank_weight=0.0,
        opportunity_weight=0.0,
        harm_weight=0.0,
        setwise_admission_weight=0.0,
        selective_risk_weight=0.0,
        selective_coverage_weight=0.0,
        policy_distill_weight=0.0,
        policy_regret_weight=0.0,
        preference_weight=0.0,
        preference_regret_weight=0.0,
        preference_listwise_weight=0.0,
        preference_gap_weight=0.0,
        preference_set_weight=0.0,
        preference_all_group_set_weight=0.0,
        delta_nll_weight=0.0,
        ordinal_evidence_benefit_margin_regression_weight=1.0,
        ordinal_evidence_benefit_margin_temperature=0.05,
    )
    loss.backward()
    assert opportunity.grad is not None
    return float(loss.item()), opportunity.grad.detach().clone()


def test_benefit_headroom_is_anchored_to_positive_gain_boundary() -> None:
    _, low_boundary_grad = _benefit_margin_loss(0.01)
    _, high_boundary_grad = _benefit_margin_loss(0.90)
    # With a reachable gain threshold, both recovery candidates are pushed to
    # positive opportunity logits. Moving the physical boundary beyond their
    # realized gain reverses the direction. No regime label participates.
    assert bool((low_boundary_grad[1:] < 0.0).all())
    assert bool((high_boundary_grad[1:] > 0.0).all())


def _save(path: Path, state: dict[str, torch.Tensor]) -> None:
    torch.save({"model_state": state}, path)


def _arch(trainable: str, *, bridge_trainable: bool) -> dict:
    return {
        "version": "v48.36-OCAF",
        "algorithm_variant": "v48.37-HAF",
        "trainable": [trainable],
        "context_source": "physical_interaction",
        "observation_conditioned_action_frontier": True,
        "interaction_bridge_trainable_this_stage": bridge_trainable,
        "regime_id_exposed_to_evidence_model": False,
        "shared_deployment_rule_required": True,
        "test_roots_read": False,
    }


def test_stage_transfer_accepts_explicitly_frozen_ocaf_factor_bridge(tmp_path: Path) -> None:
    factor = {
        "direct_evidence_concord_benefit_calibrator.0.weight": torch.zeros(2),
        "direct_evidence_concord_harm_calibrator.0.weight": torch.zeros(2),
        "direct_evidence_interaction_bridge.action_raw.weight": torch.ones(2),
        "encoder.weight": torch.ones(2),
    }
    identity = {k: v.clone() for k, v in factor.items()}
    identity["direct_evidence_concord_admission_calibrator.0.weight"] = torch.ones(2)
    factor_path, identity_path, final_path = [tmp_path / x for x in ("factor.pt", "identity.pt", "final.pt")]
    _save(factor_path, factor)
    _save(identity_path, identity)
    _save(final_path, {k: v.clone() for k, v in identity.items()})
    arch = _arch("direct_evidence_concord_admission_calibrator", bridge_trainable=False)
    identity_arch = tmp_path / "identity.json"
    final_arch = tmp_path / "final.json"
    identity_arch.write_text(json.dumps(arch), encoding="utf-8")
    final_arch.write_text(json.dumps(arch), encoding="utf-8")
    output = tmp_path / "transfer.json"
    cp = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_v48_36_stage_transfer.py"),
            "--factor", str(factor_path),
            "--identity", str(identity_path),
            "--final", str(final_path),
            "--identity-architecture", str(identity_arch),
            "--final-architecture", str(final_arch),
            "--identity-allowed-prefixes", "direct_evidence_concord_admission_calibrator",
            "--final-stage-disabled",
            "--implementation-version", "v48.37-HAF-test",
            "--output", str(output),
        ],
        cwd=ROOT,
        check=False,
    )
    doc = json.loads(output.read_text(encoding="utf-8"))
    assert cp.returncode == 0
    assert doc["valid"] is True
    assert doc["identity_disallowed_changed_parameter_count"] == 0


def test_haf_wrapper_keeps_one_continuous_policy_and_preserves_factors() -> None:
    wrapper = (ROOT / "scripts" / "run_v48_37_haf_dedicated.sh").read_text(encoding="utf-8")
    variant = (ROOT / "scripts" / "adapt_ocrap_v48_36_ocaf_variant.sh").read_text(encoding="utf-8")
    losses = (ROOT / "src" / "ocrap" / "models" / "losses.py").read_text(encoding="utf-8")
    assert "FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT" in wrapper
    assert "V4837_FACTOR_PRESERVING_IDENTITY" in wrapper
    controller = (ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh").read_text(encoding="utf-8")
    assert "--expect-algorithm-variant" in controller
    learning = (ROOT / "tools" / "check_v48_16_learning_gates.py").read_text(encoding="utf-8")
    assert "resolve_v48_36_authoritative_result" in learning
    assert "learning_gates_post_terminal.log" in controller
    assert 'V4836_IDENTITY_TRAIN_ALL="0"' in wrapper
    assert 'V4836_COUPLE_ADMISSION_PRIOR="0"' in wrapper
    assert "identity_prefixes=direct_evidence_concord_admission_calibrator" in variant
    assert "identity_benefit_listwise=0" in variant
    block = losses[losses.index("# v48.37 HAF:") : losses.index("# v48.10 COPE ordinal evidence")]
    assert "bucket_id" not in block
    assert "regime_id" not in block
    assert "t_delta" in block and "positive_gain" in block
