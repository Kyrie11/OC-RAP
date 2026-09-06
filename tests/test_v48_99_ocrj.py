from __future__ import annotations

import torch

from ocrap.v48_97_executable_recovery_state import ExecutableRecoverySufficientState
from ocrap.v48_99_recovery_jacobian import (
    ENGINEERING_VERSION,
    ObservationConditionedRecoveryJacobian,
    action_magnitude_linearity_synthetic_check,
    coordinate_scale_invariance_synthetic_check,
    nominal_identity_synthetic_check,
    observation_conditioning_synthetic_check,
    root_permutation_equivariance_synthetic_check,
    semantic_attention_weights,
    zero_init_nonzero_gradient_synthetic_check,
)


def test_v48_99_engineering_version():
    assert ENGINEERING_VERSION == "v48.99.0-OC-RJCA"


def test_v48_99_fixed_parameter_count():
    m = ObservationConditionedRecoveryJacobian(d_model=192, action_dim=125)
    assert m.trainable_parameter_count == 1404


def test_v48_99_nominal_identity():
    assert nominal_identity_synthetic_check(32, 17)


def test_v48_99_root_permutation_equivariance():
    assert root_permutation_equivariance_synthetic_check(32, 17)


def test_v48_99_observation_conditioning():
    assert observation_conditioning_synthetic_check(32, 17)


def test_v48_99_zero_init_nonzero_gradient():
    assert zero_init_nonzero_gradient_synthetic_check(32, 17)


def test_v48_99_action_magnitude_linearity():
    assert action_magnitude_linearity_synthetic_check(32, 17)


def test_v48_99_coordinate_scale_invariance():
    assert coordinate_scale_invariance_synthetic_check()


def test_v48_99_semantic_weights_permutation_equivariant():
    torch.manual_seed(99)
    d = 16
    e = ExecutableRecoverySufficientState(d).eval()
    r = torch.randn(3, 7, d)
    p = torch.softmax(torch.randn(3, 7), dim=-1)
    v = torch.ones(3, 7, dtype=torch.bool)
    perm = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    a1, b1 = semantic_attention_weights(e, r, p, v)
    a2, b2 = semantic_attention_weights(e, r[:, perm], p[:, perm], v[:, perm])
    assert torch.allclose(a1[:, perm], a2, atol=1e-6, rtol=0.0)
    assert torch.allclose(b1[:, perm], b2, atol=1e-6, rtol=0.0)
