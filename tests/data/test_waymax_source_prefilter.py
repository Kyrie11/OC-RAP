from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocrap.data import waymax_loader


class _DummyLoader:
    @staticmethod
    def preprocess_serialized_womd_data(_serialized, config=None):
        return {"config": config}

    @staticmethod
    def get_data_generator(_dataset_cfg, _parse, _postprocess):
        # Global source indices 0..5. The fallback selection tests below use
        # start=1, stride=2, worker=1 -> indices 2 and 4.
        for i in range(6):
            yield {"state": f"legacy-state-{i}", "scenario_id": None}


class _DummyFactories:
    @staticmethod
    def simulator_state_from_womd_dict(example, include_sdc_paths=True):
        return example


def _patch_minimal_waymax(monkeypatch):
    monkeypatch.setattr(
        waymax_loader,
        "_require_waymax",
        lambda: (None, None, None, _DummyLoader, _DummyFactories),
    )
    monkeypatch.setattr(
        waymax_loader,
        "_make_dataset_config",
        lambda _patterns, _cfg: SimpleNamespace(max_num_objects=64),
    )
    monkeypatch.setattr(
        waymax_loader,
        "_scenario_identity_from_payload",
        lambda _payload, idx, _state, _cfg=None: (
            f"scene__wx{idx:08d}", "scene", "waymax_legacy"
        ),
    )

    def fake_raw(_state, scenario_id, scenario_index, _cfg):
        return SimpleNamespace(
            scenario_id=scenario_id,
            metadata={"_waymax_scenario_index": int(scenario_index)},
        )

    monkeypatch.setattr(waymax_loader, "raw_scenario_from_waymax_state", fake_raw)


def test_prefiltered_waymax_iterator_preserves_global_source_indices(monkeypatch):
    _patch_minimal_waymax(monkeypatch)

    def fake_prefilter(**_kwargs):
        yield 11004, {"state": "fast-a", "scenario_id": None}
        yield 11010, {"state": "fast-b", "scenario_id": None}

    monkeypatch.setattr(waymax_loader, "_iter_prefiltered_waymax_payloads", fake_prefilter)
    cfg = {
        "max_agents": 64,
        "scenario_start_index": 11000,
        "scenario_stride": 6,
        "scenario_worker_index": 4,
        "waymax": {
            "retain_official_scenario_id": False,
            "prefilter_source_scan_controls": True,
        },
    }
    rows = list(waymax_loader.iter_waymax_womd_scenarios("dummy@150", 2, cfg))
    assert [r.metadata["_waymax_scenario_index"] for r in rows] == [11004, 11010]
    assert [r.scenario_id for r in rows] == ["scene__wx00011004", "scene__wx00011010"]
    assert all(r.metadata["waymax_source_scan_prefiltered"] for r in rows)


def test_prefilter_failure_before_first_row_falls_back_to_legacy_partition(monkeypatch):
    _patch_minimal_waymax(monkeypatch)

    def failing_prefilter(**_kwargs):
        raise RuntimeError("unsupported raw API")
        yield  # pragma: no cover

    monkeypatch.setattr(waymax_loader, "_iter_prefiltered_waymax_payloads", failing_prefilter)
    cfg = {
        "max_agents": 64,
        "scenario_start_index": 1,
        "scenario_stride": 2,
        "scenario_worker_index": 1,
        "waymax": {
            "retain_official_scenario_id": False,
            "prefilter_source_scan_controls": True,
        },
    }
    rows = list(waymax_loader.iter_waymax_womd_scenarios("dummy@2", 2, cfg))
    assert [r.metadata["_waymax_scenario_index"] for r in rows] == [2, 4]
    assert all(not r.metadata["waymax_source_scan_prefiltered"] for r in rows)


def test_prefilter_midstream_failure_does_not_restart_and_duplicate(monkeypatch):
    _patch_minimal_waymax(monkeypatch)

    def failing_midstream(**_kwargs):
        yield 10, {"state": "fast-a", "scenario_id": None}
        raise RuntimeError("midstream failure")

    monkeypatch.setattr(waymax_loader, "_iter_prefiltered_waymax_payloads", failing_midstream)
    cfg = {
        "max_agents": 64,
        "scenario_start_index": 10,
        "scenario_stride": 1,
        "scenario_worker_index": 0,
        "waymax": {
            "retain_official_scenario_id": False,
            "prefilter_source_scan_controls": True,
        },
    }
    it = waymax_loader.iter_waymax_womd_scenarios("dummy@2", 2, cfg)
    first = next(it)
    assert first.metadata["_waymax_scenario_index"] == 10
    with pytest.raises(RuntimeError, match="midstream failure"):
        next(it)


def test_required_prefilter_fails_fast_instead_of_slow_fallback(monkeypatch):
    _patch_minimal_waymax(monkeypatch)

    def failing_prefilter(**_kwargs):
        raise RuntimeError("unsupported raw API")
        yield  # pragma: no cover

    monkeypatch.setattr(waymax_loader, "_iter_prefiltered_waymax_payloads", failing_prefilter)
    cfg = {
        "max_agents": 64,
        "scenario_start_index": 11000,
        "scenario_stride": 6,
        "scenario_worker_index": 4,
        "waymax": {
            "retain_official_scenario_id": False,
            "prefilter_source_scan_controls": True,
            "require_source_scan_prefilter": True,
        },
    }
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        list(waymax_loader.iter_waymax_womd_scenarios("dummy@150", 2, cfg))
