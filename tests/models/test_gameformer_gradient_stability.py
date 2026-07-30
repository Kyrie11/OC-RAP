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
