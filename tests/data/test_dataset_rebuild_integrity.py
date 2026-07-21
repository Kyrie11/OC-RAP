from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ocrap.config.defaults import DEFAULT_CONFIG
from ocrap.data.build import builder, regimes


def _history() -> SimpleNamespace:
    return SimpleNamespace(
        agent_history=np.zeros((1, 2, 16), dtype=np.float32),
        agent_valid=np.ones((1, 2), dtype=bool),
        ego_state=np.zeros((16,), dtype=np.float32),
        occ_mask=np.zeros((4, 4), dtype=np.uint8),
        metadata={"time_sampling_reasons": ["uniform"]},
    )


def _sample(*, contact_surrogate: bool) -> SimpleNamespace:
    future = SimpleNamespace(metadata={"contact_surrogate": contact_surrogate})
    prefix = SimpleNamespace(
        diagnostics={"prefix_collision": False, "prefix_contact": False},
        feasible=True,
        hard_violation=0.0,
        harm_proxy=0.0,
    )
    return SimpleNamespace(
        r_dep_star=1.0,
        r_orc_star=1.0,
        is_nominal=True,
        i_art_star=False,
        prefix=prefix,
        futures=[future],
        regime_label={},
    )


def test_assign_regimes_separates_counterfactual_contact(monkeypatch):
    monkeypatch.setattr(regimes, "agent_state_to_box", lambda _x: np.zeros((9,), dtype=np.float32))
    monkeypatch.setattr(regimes, "min_box_clearance", lambda *_args, **_kwargs: 99.0)
    monkeypatch.setattr(regimes, "compute_ttc", lambda *_args, **_kwargs: 99.0)
    monkeypatch.setattr(regimes, "unknown_ratio_in_corridor", lambda _x: 0.0)

    sample = _sample(contact_surrogate=True)
    regimes.assign_regimes([sample], _history(), {"regime_thresholds": {}})

    assert sample.regime_label["post_contact"] is True
    assert sample.regime_label["post_contact_observed"] is False
    assert sample.regime_label["post_contact_counterfactual"] is True


def test_assign_regimes_separates_observed_contact(monkeypatch):
    monkeypatch.setattr(regimes, "agent_state_to_box", lambda _x: np.zeros((9,), dtype=np.float32))
    monkeypatch.setattr(regimes, "min_box_clearance", lambda *_args, **_kwargs: 0.1)
    monkeypatch.setattr(regimes, "compute_ttc", lambda *_args, **_kwargs: 99.0)
    monkeypatch.setattr(regimes, "unknown_ratio_in_corridor", lambda _x: 0.0)

    sample = _sample(contact_surrogate=False)
    regimes.assign_regimes([sample], _history(), {"regime_thresholds": {"tau_contact": 0.8}})

    assert sample.regime_label["post_contact"] is True
    assert sample.regime_label["post_contact_observed"] is True
    assert sample.regime_label["post_contact_counterfactual"] is False


def test_builder_applies_scenario_start_index_once(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_iterator(cfg):
        captured.update(cfg)
        return iter(())

    monkeypatch.setattr(builder, "scenario_iterator", fake_iterator)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        {
            "progress": False,
            "scenario_start_index": 3,
            "max_scenarios": 2,
            "io": {"compress_npz": False, "fsync_npz": False},
        }
    )

    result = builder.build_dataset(tmp_path / "empty", cfg)

    assert result["num_samples"] == 0
    assert captured["scenario_start_index"] == 0
    assert captured["max_scenarios"] == 5
