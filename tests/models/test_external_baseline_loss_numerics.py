from __future__ import annotations

import torch

from ocrap.external_baselines.train import _gameformer_traj_loss, _loss_dict


def _base_batch(batch: int = 2, candidates: int = 4, future: int = 5) -> dict[str, torch.Tensor]:
    mask = torch.ones(batch, candidates, dtype=torch.bool)
    prefix = torch.randn(batch, candidates, future, 2)
    return {
        "mask": mask,
        "target_index": torch.zeros(batch, dtype=torch.long),
        "utility": torch.zeros(batch, candidates),
        "hard": torch.zeros(batch, candidates),
        "harm": torch.zeros(batch, candidates),
        "r_orc": torch.zeros(batch, candidates),
        "r_dep": torch.zeros(batch, candidates),
        "prefix_traj": prefix,
        "prefix_valid": torch.ones(batch, candidates, future, dtype=torch.bool),
    }


def test_zero_weight_nan_auxiliary_cannot_poison_policy_loss() -> None:
    batch = _base_batch()
    B, N = batch["mask"].shape
    logits = torch.randn(B, N, requires_grad=True)
    nan_scalar = torch.full((B, N), float("nan"), requires_grad=True)
    out = {
        "logits": logits,
        # These heads are deliberately NaN.  All of their weights are zero.
        "utility": nan_scalar,
        "hard": nan_scalar,
        "harm": nan_scalar,
        "r_orc": nan_scalar,
        "r_dep": nan_scalar,
        "gameformer_level_trajs": [torch.full((B, N, 2, 5, 4), float("nan"))],
        "gameformer_level_scores": [torch.full((B, N, 2), float("nan"))],
    }
    cfg = {"external_baselines": {"loss_weights": {
        "policy": 1.0, "levelk": 0.0, "level_response": 0.0,
        "gameformer_traj": 0.0, "pluto_contrastive": 0.0, "topology": 0.0,
        "utility": 0.0, "hard": 0.0, "harm": 0.0,
        "oracle_rec": 0.0, "deploy_rec": 0.0,
    }}}
    losses = _loss_dict(out, batch, cfg)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_gameformer_traj_loss_ignores_nonfinite_padded_candidate_without_nan_gradients() -> None:
    B, N, M, T = 1, 3, 2, 6
    batch = _base_batch(B, N, T)
    batch["mask"][0, -1] = False
    batch["prefix_valid"][0, -1] = False

    traj = torch.randn(B, N, M, T, 4, requires_grad=True)
    scores = torch.randn(B, N, M, requires_grad=True)
    # Padded candidate may contain arbitrary storage; it must never affect the loss.
    with torch.no_grad():
        traj[0, -1].fill_(float("inf"))
        scores[0, -1].fill_(float("nan"))
    out = {
        "logits": torch.zeros(B, N, requires_grad=True),
        "gameformer_level_trajs": [traj],
        "gameformer_level_scores": [scores],
    }
    loss = _gameformer_traj_loss(out, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert traj.grad is not None
    assert scores.grad is not None
    assert torch.isfinite(traj.grad[0, :-1]).all()
    assert torch.isfinite(scores.grad[0, :-1]).all()
    assert torch.equal(traj.grad[0, -1], torch.zeros_like(traj.grad[0, -1]))
    assert torch.equal(scores.grad[0, -1], torch.zeros_like(scores.grad[0, -1]))


def test_gameformer_active_loss_is_float32_and_finite_for_large_prefix_coordinates() -> None:
    B, N, M, T = 1, 4, 3, 8
    batch = _base_batch(B, N, T)
    batch["prefix_traj"] *= 1.0e3
    traj = torch.randn(B, N, M, T, 4, dtype=torch.bfloat16, requires_grad=True)
    scores = torch.randn(B, N, M, dtype=torch.bfloat16, requires_grad=True)
    out = {
        "logits": torch.zeros(B, N, dtype=torch.bfloat16, requires_grad=True),
        "gameformer_level_trajs": [traj],
        "gameformer_level_scores": [scores],
    }
    loss = _gameformer_traj_loss(out, batch)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert traj.grad is not None and torch.isfinite(traj.grad).all()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_inactive_zero_loss_is_finite_even_if_reference_logits_are_nan() -> None:
    batch = _base_batch(batch=1, candidates=2, future=3)
    nan_logits = torch.full((1, 2), float("nan"), requires_grad=True)
    # Policy is inactive; a finite auxiliary loss should not be contaminated by
    # the differentiable graph-zero used for inactive heads.
    utility = torch.zeros(1, 2, requires_grad=True)
    out = {
        "logits": nan_logits,
        "utility": utility,
        "hard": utility,
        "harm": utility,
        "r_orc": utility,
        "r_dep": utility,
    }
    cfg = {"external_baselines": {"loss_weights": {
        "policy": 0.0, "levelk": 0.0, "level_response": 0.0,
        "gameformer_traj": 0.0, "pluto_contrastive": 0.0, "topology": 0.0,
        "utility": 1.0, "hard": 0.0, "harm": 0.0,
        "oracle_rec": 0.0, "deploy_rec": 0.0,
    }}}
    losses = _loss_dict(out, batch, cfg)
    assert torch.isfinite(losses["loss"])
