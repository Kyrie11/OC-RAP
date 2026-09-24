from __future__ import annotations
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

from ocrap.data.waymax_loader import ObservationLegalRouteUnavailable, _route_from_sdc_paths_with_source
from ocrap.data.build.history import _sanitize_route


def _dummy_state_with_paths():
    pts = 6
    paths = SimpleNamespace(
        x=np.asarray([[0,1,2,3,4,5]],dtype=np.float32),
        y=np.zeros((1,pts),dtype=np.float32),
        valid=np.ones((1,pts),dtype=bool),
        on_route=np.asarray([True]),
    )
    traj = SimpleNamespace(
        x=np.asarray([[0.0]],dtype=np.float32),
        y=np.asarray([[0.0]],dtype=np.float32),
        yaw=np.asarray([[0.0]],dtype=np.float32),
    )
    meta=SimpleNamespace(is_sdc=np.asarray([True]))
    return SimpleNamespace(sdc_paths=paths,sim_trajectory=traj,object_metadata=meta,timestep=np.asarray(0))


def test_v131_sdc_paths_are_observation_legal_route_source():
    route, source = _route_from_sdc_paths_with_source(_dummy_state_with_paths(), 8, allow_logged_fallback=False)
    assert route.shape == (8,6)
    assert source == "womd_v1_3_1_sdc_paths_connectivity_only"
    assert np.isfinite(route).all()


def test_publication_route_refuses_logged_future_fallback():
    state = SimpleNamespace(sdc_paths=None)
    with pytest.raises(ValueError, match="refusing to fall back"):
        _route_from_sdc_paths_with_source(state, 8, allow_logged_fallback=False)


def test_publication_route_types_malformed_connectivity_as_method_independent_ineligibility():
    pts = 6
    paths = SimpleNamespace(
        x=np.zeros((2, pts), dtype=np.float32),
        y=np.zeros((2, pts), dtype=np.float32),
        valid=np.zeros((2, pts), dtype=bool),
        on_route=np.asarray([False, False]),
    )
    state = SimpleNamespace(sdc_paths=paths)
    with pytest.raises(ObservationLegalRouteUnavailable, match="no valid WOMD v1.3.1 connectivity path"):
        _route_from_sdc_paths_with_source(state, 8, allow_logged_fallback=False)


def test_observation_legal_route_unavailable_remains_a_value_error_for_legacy_fail_closed_callers():
    assert issubclass(ObservationLegalRouteUnavailable, ValueError)


def test_history_refuses_future_ego_route_proxy_in_publication_mode():
    cfg={"route_points":8,"closed_loop":{"require_observation_legal_route":True,"allow_future_route_proxy":False}}
    future=np.zeros((5,1,16),dtype=np.float32); valid=np.ones((5,1),dtype=bool)
    with pytest.raises(ValueError, match="not observation-legal"):
        _sanitize_route(np.zeros((8,6),dtype=np.float32),future,valid,cfg,route_source="logged_sdc_future_proxy")


def test_stable_launcher_defaults_to_small_candidate_quality_audit_after_near_stop():
    text=(Path(__file__).resolve().parents[1]/"scripts/run_constraint_native_orientation_audit.sh").read_text()
    assert 'OCRAP_CONSTRAINT_AUDIT_MODE:-terminal_internal_closure' in text
    assert 'run_near_all_state_support_localization_two_gpu.sh' in text
    assert 'run_near_pcd_oracle_ceiling_two_gpu.sh' in text
    assert 'run_near_candidate_quality_audit_two_gpu.sh' in text
    assert 'run_observation_legal_near_axis_two_gpu.sh' in text


def test_planner_route_is_invariant_to_on_route_label():
    pts=6
    x=np.asarray([[0,1,2,3,4,5],[0,1,2,3,4,5]],dtype=np.float32)
    y=np.asarray([[0,0,0,0,0,0],[0,1,2,3,4,5]],dtype=np.float32)
    valid=np.ones_like(x,dtype=bool)
    traj=SimpleNamespace(x=np.asarray([[0.0]],dtype=np.float32),y=np.asarray([[0.0]],dtype=np.float32),yaw=np.asarray([[0.0]],dtype=np.float32))
    meta=SimpleNamespace(is_sdc=np.asarray([True]))
    def state(labels):
        return SimpleNamespace(sdc_paths=SimpleNamespace(x=x,y=y,valid=valid,on_route=np.asarray(labels)),sim_trajectory=traj,object_metadata=meta,timestep=np.asarray(0))
    a,sa=_route_from_sdc_paths_with_source(state([True,False]),8,allow_logged_fallback=False)
    b,sb=_route_from_sdc_paths_with_source(state([False,True]),8,allow_logged_fallback=False)
    assert sa==sb=="womd_v1_3_1_sdc_paths_connectivity_only"
    assert np.array_equal(a,b)


def test_closed_loop_result_serializes_relative_contract_keys():
    text=(Path(__file__).resolve().parents[1]/"src/ocrap/simulation/closed_loop_runner.py").read_text()
    for key in ("rifa_relative_proposal_top_k","rifa_relative_min_advantage","rifa_relative_opportunity_threshold","rifa_relative_harm_threshold"):
        assert f'"{key}"' in text


def test_route_legal_launcher_fail_closes_frozen_checkpoint_hashes():
    root=Path(__file__).resolve().parents[1]
    text=(root/"scripts/run_observation_legal_near_axis_two_gpu.sh").read_text()
    assert "check_frozen_checkpoint_contract.py" in text
    checker=(root/"tools/check_frozen_checkpoint_contract.py").read_text()
    assert "checkpoint_sha_mismatch" in checker
    assert "balanced_checkpoint" in checker and "precision_checkpoint" in checker


def test_route_legal_launcher_exports_exact_runtime_source_snapshot():
    root=Path(__file__).resolve().parents[1]
    text=(root/"scripts/run_observation_legal_near_axis_two_gpu.sh").read_text()
    assert "package_runtime_source_snapshot.py" in text
    assert "runtime_source_snapshot.zip" in text
