from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ocrap.simulation import waymax_rollout
from ocrap.simulation.teacher.margins import TeacherDiagnostics


def test_invalid_waymax_recovery_option_skips_exact_rollout(monkeypatch) -> None:
    state = SimpleNamespace()
    env = SimpleNamespace()
    history = SimpleNamespace(metadata={"_waymax_state": state})
    prefix = SimpleNamespace()
    future = SimpleNamespace(metadata={}, _waymax_state_after_prefix=state, _waymax_env=env)
    option = SimpleNamespace(mode="pull_over", valid=False)

    monkeypatch.setattr(
        waymax_rollout,
        "rollout_recovery_controller",
        lambda *_args, **_kwargs: (
            np.zeros((40, 7), dtype=np.float32),
            np.zeros((39, 4), dtype=np.float32),
            {"mode": "pull_over"},
        ),
    )
    monkeypatch.setattr(waymax_rollout, "_sdc_index", lambda _state: 0)
    monkeypatch.setattr(
        waymax_rollout,
        "teacher_margin",
        lambda *_args, **_kwargs: (
            0.25,
            TeacherDiagnostics(active={"clearance": True}, component_margins={"clearance": 0.25}, controller_diagnostics={}),
        ),
    )

    def should_not_roll(*_args, **_kwargs):  # pragma: no cover - executed only on regression
        raise AssertionError("invalid option must not execute exact Waymax rollout")

    monkeypatch.setattr(waymax_rollout, "_rollout_bicycle_controls_scan", should_not_roll)
    margins, diag = waymax_rollout.compute_waymax_future_option_margins(
        history,
        prefix,
        [future],
        [option],
        {
            "recovery_horizon_s": 4.0,
            "sample_rate_hz": 10.0,
            "waymax": {"teacher_backend": "hybrid", "teacher_rollout_top_k_options": 0},
            "artifact": {"use_margin_override": False},
        },
    )
    assert margins.shape == (1, 1)
    assert float(margins[0, 0]) == -1e9
    assert diag[0][0].controller_diagnostics["waymax_invalid_option_skipped"] is True
    assert diag[0][0].controller_diagnostics["waymax_recovery_rollout"] is False
