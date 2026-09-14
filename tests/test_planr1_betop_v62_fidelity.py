from __future__ import annotations

from pathlib import Path

import torch
import yaml

from ocrap.external_baselines.generative_ports import PlanR1Port


ROOT = Path(__file__).resolve().parents[1]


def test_planr1_default_corner_geometry_matches_uploaded_source() -> None:
    cb = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    corners = PlanR1Port._token_corners(cb)[0]
    expected = torch.tensor(
        [[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]],
        dtype=torch.float32,
    )
    assert torch.allclose(corners, expected)


def test_planr1_stage_transition_copies_and_freezes_reference() -> None:
    model = PlanR1Port(
        input_dim=8,
        max_candidates=2,
        d_model=16,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        future_len=10,
        scene_layers=1,
        token_interval=5,
        num_tokens=32,
        token_codebook=None,
        pretrain_epochs=1,
    )
    model.set_training_epoch(2)
    assert model.training_stage == "plan"
    assert model._planner_initialized
    for a, b in zip(model.reference_transformer.parameters(), model.plan_transformer.parameters()):
        assert torch.equal(a.detach(), b.detach())
    assert not any(p.requires_grad for p in model.reference_transformer.parameters())
    model.train()
    assert model.plan_transformer.training
    assert not model.reference_transformer.training


def test_v62_configs_preserve_paper_core_hyperparameters() -> None:
    plan = yaml.safe_load((ROOT / "configs/external_baselines/plan_r1.yaml").read_text())["external_baselines"]
    assert plan["model"]["num_layers"] == 6
    assert plan["model"]["d_model"] == 128
    assert plan["model"]["num_heads"] == 8
    assert plan["model"]["pretrain_epochs"] == 32
    assert plan["training"]["epochs"] == 37
    assert plan["training"]["finetune_lr"] == 4e-6
    assert plan["plan_r1"]["beta"] == 0.1
    assert plan["plan_r1"]["scaling_factor"] == 0.1

    betop = yaml.safe_load((ROOT / "configs/external_baselines/betopnet.yaml").read_text())["external_baselines"]
    assert betop["model"]["d_model"] == 128
    assert betop["model"]["num_layers"] == 4
    assert betop["model"]["num_topo"] == 32
    assert betop["model"]["num_topology_agents"] == 32
    assert betop["model"]["history_len"] == 21
    assert betop["training"]["epochs"] == 25
    assert betop["loss_weights"]["topology"] == 50.0
    assert betop["policy"]["betop_branch_steps"] == 3
    assert betop["policy"]["betop_short_term_cost_weight"] == 0.5
