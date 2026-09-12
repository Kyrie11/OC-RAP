from __future__ import annotations

from pathlib import Path

import torch
import yaml

from ocrap.external_baselines.models import build_model_from_cfg
from ocrap.external_baselines.provenance import SUPPLEMENTARY_BY_REGIME, find_provenance
from ocrap.external_baselines.train import _loss_dict

ROOT = Path(__file__).resolve().parents[1]


def _scene_batch(*, B: int = 1, N: int = 4, D: int = 24, T: int = 20) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(11)
    A, H, M, P, R = 4, 11, 6, 5, 10
    inputs = {
        "x": torch.randn(B, N, D),
        "mask": torch.ones(B, N, dtype=torch.bool),
        "prefix_traj": torch.cumsum(torch.randn(B, N, T, 2) * 0.1, dim=2),
        "prefix_valid": torch.ones(B, N, T, dtype=torch.bool),
        "source_agent_history": torch.randn(B, A, H, 9),
        "source_agent_valid": torch.ones(B, A, H, dtype=torch.bool),
        "source_current_state": torch.randn(B, 9),
        "source_map_points": torch.randn(B, M, P, 6),
        "source_map_point_valid": torch.ones(B, M, P, dtype=torch.bool),
        "source_map_meta": torch.randn(B, M, 4),
        "source_map_center": torch.randn(B, M, 3),
        "source_map_valid": torch.ones(B, M, dtype=torch.bool),
        "source_centerline": torch.randn(B, R, 3),
        "actor_topology_features": torch.randn(B, N, 4, 16),
        "actor_topology_mask": torch.ones(B, N, 4, dtype=torch.bool),
        "map_topology_features": torch.randn(B, N, 6, 14),
        "map_topology_mask": torch.ones(B, N, 6, dtype=torch.bool),
    }
    batch = {
        **inputs,
        "target_index": torch.zeros(B, dtype=torch.long),
        "utility": torch.randn(B, N),
        "hard": torch.rand(B, N),
        "harm": torch.rand(B, N),
        "r_orc": torch.randn(B, N),
        "r_dep": torch.randn(B, N),
        "feasible": torch.ones(B, N, dtype=torch.bool),
        "actor_topology_target": torch.randint(0, 2, (B, N, 4)).float(),
        "map_topology_target": torch.randint(0, 2, (B, N, 6)).float(),
    }
    return inputs, batch


def test_supplementary_regime_assignment() -> None:
    assert SUPPLEMENTARY_BY_REGIME["safe"] == ("diffusion_planner",)
    assert SUPPLEMENTARY_BY_REGIME["near"] == ("flow_planner", "plan_r1", "betopnet")
    assert find_provenance("plan_r1").regimes == ("near",)
    assert find_provenance("betopnet").canonical_name == "betopnet"


def test_new_learned_ports_forward_and_native_losses_are_finite() -> None:
    inputs, batch = _scene_batch()
    D = inputs["x"].shape[-1]
    N = inputs["x"].shape[1]
    for name in ("diffusion_planner", "flow_planner", "plan_r1", "betopnet"):
        cfg = yaml.safe_load((ROOT / f"configs/external_baselines/{name}.yaml").read_text())
        cfg["external_baselines"]["model"]["max_candidates"] = N
        if name == "betopnet":
            cfg["external_baselines"]["model"]["num_topology_agents"] = 4
            cfg["external_baselines"]["model"]["num_topology_map"] = 6
            cfg["external_baselines"]["model"]["num_topo"] = 4
        model = build_model_from_cfg(D, cfg)
        model.train()
        out = model(inputs["x"], inputs["mask"], **{k: v for k, v in inputs.items() if k not in {"x", "mask"}})
        assert out["logits"].shape == inputs["mask"].shape
        losses = _loss_dict(out, batch, cfg)
        assert torch.isfinite(losses["loss"])
