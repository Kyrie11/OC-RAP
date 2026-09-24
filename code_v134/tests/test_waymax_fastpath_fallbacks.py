from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import ocrap.simulation.waymax_rollout as wr


class _Meta(NamedTuple):
    is_sdc: object


class _State(NamedTuple):
    x: object
    num_objects: object
    object_metadata: _Meta


class _Action:
    def __init__(self, data, valid):
        self.data = data
        self.valid = valid


class _Datatypes:
    Action = _Action


class _Env:
    def step(self, state, action, rng=None):
        dx = action.data[:, 0] + 0.25 * action.data[:, 1]
        return _State(state.x + dx, state.num_objects, state.object_metadata)

    def metrics(self, state):
        x = state.x
        return {
            "overlap": SimpleNamespace(value=jnp.abs(x) * 0.01),
            "offroad": SimpleNamespace(value=jnp.maximum(x, 0) * 0.02),
            "sdc_wrongway": SimpleNamespace(value=jnp.zeros_like(x)),
            "sdc_off_route": SimpleNamespace(value=jnp.abs(x) * 0.03),
            "kinematic_infeasibility": SimpleNamespace(value=jnp.zeros_like(x)),
            "log_divergence": SimpleNamespace(value=jnp.abs(x) * 0.04),
        }


def _fake_require_waymax():
    return jax, jnp, None, _Datatypes, None, None


def test_batched_teacher_metrics_validate_against_scalar_and_prefix_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(wr, "_require_waymax", _fake_require_waymax)
    wr._JIT_CONTROL_ROLLOUT_CACHE.clear()
    wr._JIT_TEACHER_BATCH_CACHE.clear()
    wr._JIT_TEACHER_BATCH_VALIDATION.clear()
    wr._JIT_PREFIX_ROLLOUT_VALIDATION.clear()

    state = _State(jnp.zeros(3, dtype=jnp.float32), 3, _Meta(jnp.array([True, False, False])))
    env = _Env()
    cfg = {
        "wheelbase_m": 2.8,
        "waymax": {
            "use_jit_scan_rollouts": True,
            "batch_teacher_option_rollouts": True,
            "validate_batched_teacher_metrics": True,
            "validate_jit_prefix_rollout": True,
        },
    }
    controls = np.zeros((4, 5, 2), dtype=np.float32)
    controls[:, :, 0] = np.arange(4, dtype=np.float32)[:, None] * 0.1
    controls[:, :, 1] = 0.02

    reference_validation_call = wr._rollout_teacher_options_final_metrics(state, env, controls, cfg)
    fast_call = wr._rollout_teacher_options_final_metrics(state, env, controls, cfg)
    assert reference_validation_call is not None and fast_call is not None
    for reference, fast in zip(reference_validation_call, fast_call):
        assert set(reference) == set(fast)
        for key in reference:
            np.testing.assert_allclose(reference[key], fast[key], rtol=0.0, atol=1.0e-6)
    assert any(wr._JIT_TEACHER_BATCH_VALIDATION.values())

    rng = jax.random.PRNGKey(1)
    scan_or_fallback = wr._rollout_bicycle_controls_scan(state, env, controls[2], cfg, rng=rng)
    scalar = wr._rollout_bicycle_controls_loop(state, env, controls[2], cfg, rng=rng)
    np.testing.assert_array_equal(np.asarray(scan_or_fallback.x), np.asarray(scalar.x))
