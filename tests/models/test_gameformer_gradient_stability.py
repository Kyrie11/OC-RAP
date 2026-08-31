from __future__ import annotations

import math

import torch

from ocrap.external_baselines.models import GameFormerFutureEncoder
from ocrap.external_baselines.train import _stable_clip_grad_norm_


def test_gameformer_future_encoder_zero_motion_has_finite_gradients() -> None:
    encoder = GameFormerFutureEncoder(d_model=32, future_len=5, dropout=0.0)
    # Zero trajectories exercise the formerly singular atan2(0, 0) path.
    traj = torch.zeros(2, 3, 4, 5, 2, requires_grad=True)
    scores = torch.zeros(2, 3, 4, requires_grad=True)
    loss = encoder(traj, scores).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert traj.grad is not None and torch.isfinite(traj.grad).all()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    for parameter in encoder.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_stable_clip_handles_finite_float32_norm_overflow() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1024, dtype=torch.float32))
    parameter.grad = torch.full_like(parameter, 1.0e20)
    norm = _stable_clip_grad_norm_([parameter], 5.0)
    assert math.isfinite(norm)
    assert norm > 5.0
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    clipped_norm = float(torch.linalg.vector_norm(parameter.grad.double()))
    assert abs(clipped_norm - 5.0) < 1.0e-4
    assert float(parameter.grad.abs().max()) > 0.0


def test_gameformer_future_encoder_never_calls_atan2_at_origin(monkeypatch) -> None:
    original = torch.atan2
    seen = {"calls": 0}

    def checked(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        seen["calls"] += 1
        assert not bool(((x == 0) & (y == 0)).any()), "atan2 received the singular origin"
        return original(y, x)

    monkeypatch.setattr(torch, "atan2", checked)
    traj = torch.zeros(1, 1, 2, 4, 2, requires_grad=True)
    vel, heading = GameFormerFutureEncoder._stable_motion_state(traj)
    (vel.square().mean() + heading.square().mean()).backward()
    assert seen["calls"] == 1
    assert traj.grad is not None and torch.isfinite(traj.grad).all()


def test_gameformer_future_encoder_first_step_is_measured_from_current_origin() -> None:
    traj = torch.tensor([[[[[0.2, 0.0], [0.5, 0.0], [0.5, 0.0]]]]], requires_grad=True)
    vel, heading = GameFormerFutureEncoder._stable_motion_state(traj)
    assert torch.allclose(vel[0, 0, 0, 0], torch.tensor([2.0, 0.0]), atol=1e-6)
    assert torch.allclose(vel[0, 0, 0, 1], torch.tensor([3.0, 0.0]), atol=1e-6)
    # The stationary third step takes the neutral heading branch rather than the
    # undefined atan2(0, 0) branch.
    assert torch.allclose(heading[0, 0, 0, 2], torch.tensor([0.0]), atol=1e-6)
