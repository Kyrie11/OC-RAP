from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from ocrap.cli.train import _semantic_witness_checkpoint_feature_contract
from ocrap.models.data import (
    DIRECT_INTERACTION_ORIENTED_RECOVERY_WITNESS_FEATURE_DIM,
    DIRECT_INTERACTION_ORIENTED_RECOVERY_WITNESS_FEATURE_SCHEMA,
    OPTION_FEATURE_DIM,
    direct_semantic_recovery_witness_features_from_sample,
    option_features_from_sample,
    _build_items_ordered,
    _persistent_tensor_cache_key,
)
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout():
    return FlatFeatureLayout(feature_max_agents=2)


def _sample(*, xy=(80.0, 0.0), acc_hist=((-2.0, 0.0), (0.0, -2.0))):
    ego = np.zeros(9, np.float32)
    ego[6] = 4.0
    ego[7] = 4.8
    ego[8] = 2.0
    states = np.zeros((10, 9), np.float32)
    states[:, 0] = np.arange(1, 11) * 0.4
    states[:, 6] = 4.0
    states[:, 7] = 4.8
    states[:, 8] = 2.0
    controls = np.zeros((9, 4), np.float32)
    ah = np.asarray(acc_hist, np.float32).reshape(-1, 2)
    hist = np.zeros((len(ah), 2, 16), np.float32)
    valid = np.ones((len(ah), 2), bool)
    hist[:, 1, 0] = float(xy[0])
    hist[:, 1, 1] = float(xy[1])
    hist[:, 1, 5:7] = ah
    hist[:, 1, 10] = 4.8
    hist[:, 1, 11] = 2.0
    return {
        "ego_state": ego,
        "prefix_states": states,
        "prefix_controls": controls,
        "agent_history": hist,
        "agent_valid": valid,
        "recovery_modes": np.asarray(["stop", "lateral_escape"], object),
        "recovery_params": np.asarray([[-5.0, 5.0, 0.0], [3.5, 5.0, 1.5]], np.float32),
        "option_valid": np.asarray([1, 1], bool),
        "prefix_macro_id": 0,
        "prefix_macro_name": "candidate",
        "prefix_param": np.zeros(0, np.float32),
        "utility": 0.0,
        "feasible": 1.0,
        "hard_violation": 0.0,
        "harm_proxy": 0.0,
    }


def _cfg(*, box=False, hull=False, history=False):
    return {
        "sample_rate_hz": 10.0,
        "recovery_horizon_s": 4.0,
        "prefix_horizon_s": 1.0,
        "route_dev_max_m": 2.5,
        "control_limits": {
            "a_max": 3.0,
            "a_min": -6.0,
            "delta_max": 0.55,
            "j_max": 6.0,
            "steer_rate_max": 0.5,
        },
        "model": {
            "feature_max_agents": 2,
            "direct_recovery_semantic_witness_route_alignment": True,
            "direct_recovery_semantic_witness_reentry_alignment": True,
            "direct_recovery_semantic_witness_control_projection": True,
            "direct_recovery_semantic_witness_boundary_transport": False,
            "direct_recovery_semantic_witness_projection_fidelity_weighting": True,
            "direct_recovery_semantic_witness_demand_normalized_fidelity": False,
            "direct_recovery_semantic_witness_robust_occupancy": False,
            "direct_recovery_semantic_witness_soft_occupancy_disagreement": False,
            "direct_recovery_semantic_witness_boundary_localized_occupancy_trust": False,
            "direct_recovery_semantic_witness_history_occupancy_reachability": history,
            "direct_recovery_semantic_witness_interaction_box_support": box,
            "direct_recovery_semantic_witness_interaction_hull_support": hull,
        },
        "default_available_distance_m": 60.0,
    }


def _model(*, box=True, hull=False):
    L = _layout()
    return OCRAPModel(
        input_dim=L.total_dim,
        num_roots=3,
        num_options=2,
        d_model=16,
        d_obs=8,
        encoder_type="structured_transformer",
        feature_layout=asdict(L),
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        option_feature_dim=OPTION_FEATURE_DIM,
        direct_recovery_value_head=True,
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_active_set_alignment=True,
        direct_recovery_semantic_witness_path_stop_alignment=False,
        direct_recovery_semantic_witness_classlocal_transport=False,
        direct_recovery_semantic_witness_route_alignment=True,
        direct_recovery_semantic_witness_reentry_alignment=True,
        direct_recovery_semantic_witness_control_projection=True,
        direct_recovery_semantic_witness_boundary_transport=False,
        direct_recovery_semantic_witness_projection_fidelity_weighting=True,
        direct_recovery_semantic_witness_demand_normalized_fidelity=False,
        direct_recovery_semantic_witness_robust_occupancy=False,
        direct_recovery_semantic_witness_soft_occupancy_disagreement=False,
        direct_recovery_semantic_witness_boundary_localized_occupancy_trust=False,
        direct_recovery_semantic_witness_history_occupancy_reachability=False,
        direct_recovery_semantic_witness_interaction_box_support=box,
        direct_recovery_semantic_witness_interaction_hull_support=hull,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _force(m):
    m.root_logit_head.forward = MethodType(
        lambda self, z: torch.zeros((*z.shape[:-1], 1), device=z.device, dtype=z.dtype),
        m.root_logit_head,
    )
    m.obs_embed_head.forward = MethodType(
        lambda self, z: torch.zeros((*z.shape[:-1], 8), device=z.device, dtype=z.dtype),
        m.obs_embed_head,
    )

    def mf(self, z):
        return torch.zeros((z.shape[0], 3, 2, 1), device=z.device, dtype=z.dtype)

    m.margin_head.forward = MethodType(mf, m.margin_head)


def _support(m, feat):
    _force(m)
    x = torch.zeros((1, _layout().total_dim))
    opt = torch.from_numpy(option_features_from_sample(_sample())).float().unsqueeze(0)
    rv = torch.ones((1, 3), dtype=torch.bool)
    ov = torch.ones((1, 2), dtype=torch.bool)
    out = m._direct_semantic_witness_absolute_feasibility(
        m._scene_tokens(x), x, opt, feat, root_valid=rv, option_valid=ov
    )
    return out[4], out[5]


def _feat20():
    f = torch.full((1, 2, 20), 0.6)
    f[..., 4] = -float(np.tanh(1.0))  # projection-fidelity -> 1/2
    f[..., 8] = 0.3
    f[..., 9] = 0.3
    f[..., 11] = 0.0
    f[..., 12] = 0.6
    f[..., 13] = 0.6
    f[0, 0, 18] = float(np.tanh(0.0))
    f[0, 1, 18] = float(np.tanh(1.0))
    f[0, 0, 19] = float(np.tanh(0.0))
    f[0, 1, 19] = float(np.tanh(2.0))
    return f


def test_v4872_schema8_contract_and_dim():
    base = {
        "direct_recovery_absolute_semantic_witness_correction": True,
        "direct_recovery_semantic_witness_route_alignment": True,
        "direct_recovery_semantic_witness_reentry_alignment": True,
        "direct_recovery_semantic_witness_control_projection": True,
        "direct_recovery_semantic_witness_projection_fidelity_weighting": True,
    }
    for hull in (False, True):
        cfg = {
            **base,
            "direct_recovery_semantic_witness_interaction_box_support": True,
            "direct_recovery_semantic_witness_interaction_hull_support": hull,
        }
        assert _semantic_witness_checkpoint_feature_contract(cfg) == (
            8,
            "interaction_oriented_history_reachability_projected_recovery_witness",
        )
    assert DIRECT_INTERACTION_ORIENTED_RECOVERY_WITNESS_FEATURE_SCHEMA == 8
    assert DIRECT_INTERACTION_ORIENTED_RECOVERY_WITNESS_FEATURE_DIM == 20


def test_v4872_first18_coordinates_are_execution_exact_v4871_history_diagnostics():
    d = _sample(xy=(30.0, 5.0), acc_hist=((-2.0, 0.0), (0.0, -2.0), (-1.0, -0.5)))
    j71 = direct_semantic_recovery_witness_features_from_sample(
        d, _cfg(history=True), num_options=2
    )
    for hull in (False, True):
        f = direct_semantic_recovery_witness_features_from_sample(
            d, _cfg(box=True, hull=hull), num_options=2
        )
        assert f.shape == (2, 20)
        assert np.array_equal(j71, f[:, :18])


def test_v4872_directional_box_removes_isotropic_tangential_overpenalty():
    # For an agent almost straight ahead, ay history is primarily tangential.
    # The v48.71 circumball charges it through an L2 radius; directional box
    # support should never be more pessimistic than that circumball.
    d = _sample(xy=(80.0, 0.0), acc_hist=((-2.0, 0.0), (0.0, -2.0)))
    f = direct_semantic_recovery_witness_features_from_sample(
        d, _cfg(box=True), num_options=2
    )
    assert np.all(f[:, 18] <= f[:, 16] + 1e-6)
    assert np.any(f[:, 18] < f[:, 16] - 1e-4)


def test_v4872_empirical_hull_excludes_unobserved_box_corners():
    # Diagonal interaction direction: the component box contains (-2,-2), but
    # history observed only (-2,0) and (0,-2).  Convex-hull support is tighter.
    d = _sample(xy=(55.0, 55.0), acc_hist=((-2.0, 0.0), (0.0, -2.0)))
    f = direct_semantic_recovery_witness_features_from_sample(
        d, _cfg(box=True, hull=True), num_options=2
    )
    assert np.all(f[:, 19] <= f[:, 18] + 1e-6)
    assert np.any(f[:, 19] < f[:, 18] - 1e-4)


def test_v4872_model_selects_box_vs_hull_coordinate_and_keeps_sign():
    feat = _feat20()
    box = _model(box=True, hull=False).eval()
    hull = _model(box=True, hull=True).eval()
    vb, sb = _support(box, feat)
    vh, sh = _support(hull, feat)
    assert torch.equal(vb > 0, vh > 0)
    # projection fidelity contributes 1/2.  option1 box risk=1 -> 1/4 total;
    # hull risk=2 -> 1/6 total.
    assert torch.allclose(sb[..., 1], torch.tensor([[0.25]]), atol=2e-5)
    assert torch.allclose(sh[..., 1], torch.tensor([[1.0 / 6.0]]), atol=2e-5)


def test_v4872_rejects_stacking_and_hull_without_box():
    L = _layout()
    kw = dict(
        input_dim=L.total_dim,
        num_roots=3,
        num_options=2,
        d_model=16,
        d_obs=8,
        encoder_type="structured_transformer",
        feature_layout=asdict(L),
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        option_feature_dim=OPTION_FEATURE_DIM,
        direct_recovery_value_head=True,
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_control_projection=True,
        direct_recovery_semantic_witness_projection_fidelity_weighting=True,
    )
    with pytest.raises(ValueError, match="hull support requires"):
        OCRAPModel(**kw, direct_recovery_semantic_witness_interaction_hull_support=True)
    with pytest.raises(ValueError, match="replaces prior occupancy"):
        OCRAPModel(
            **kw,
            direct_recovery_semantic_witness_interaction_box_support=True,
            direct_recovery_semantic_witness_history_occupancy_reachability=True,
        )


def test_v4872_runner_freezes_boundary_transport_and_old_occupancy_trust():
    text = (Path(__file__).resolve().parents[1] / "scripts/run_v48_72_dcp_drfc_bcde_rifa_iorw_two_gpu.sh").read_text()
    assert "L72_BOX_SUPPORT" in text and "M72_Main_OCIORW" in text
    assert "SEMANTIC_WITNESS_INTERACTION_BOX_SUPPORT=true" in text
    assert "SEMANTIC_WITNESS_INTERACTION_HULL_SUPPORT" in text
    assert "SEMANTIC_WITNESS_HISTORY_OCCUPANCY_REACHABILITY=false" in text
    assert "SEMANTIC_WITNESS_BOUNDARY_LOCALIZED_OCCUPANCY_TRUST=false" in text
    assert "SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false" in text
    assert "SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY=false" in text
    assert "PROPOSAL_TOP_K=5" in text


def test_v4872_box_and_hull_arms_materialize_identical_schema8_tensors():
    d = _sample(xy=(42.0, 11.0), acc_hist=((-2.0, 0.2), (0.1, -1.7), (-0.8, -0.4)))
    box = direct_semantic_recovery_witness_features_from_sample(
        d, _cfg(box=True, hull=False), num_options=2
    )
    hull = direct_semantic_recovery_witness_features_from_sample(
        d, _cfg(box=True, hull=True), num_options=2
    )
    assert np.array_equal(box, hull)


def test_v4872_box_and_hull_share_persistent_tensor_cache_key(tmp_path):
    sample = tmp_path / "samples" / "sample_000.npz"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(b"cache-key-fixture")
    base = _cfg(box=True, hull=False)
    base["model"]["direct_recovery_absolute_semantic_witness_correction"] = True
    base["training"] = {"persistent_tensor_cache": True}
    alt = _cfg(box=True, hull=True)
    alt["model"]["direct_recovery_absolute_semantic_witness_correction"] = True
    alt["training"] = {"persistent_tensor_cache": True}
    kwargs = dict(
        num_roots=3, num_options=2, d_signature=4,
        d_future_signature=4, feature_dim=123,
    )
    assert _persistent_tensor_cache_key([sample], base, **kwargs) == _persistent_tensor_cache_key(
        [sample], alt, **kwargs
    )


def test_v4872_ordered_parallel_cache_builder_preserves_values_and_order():
    def build(i):
        return {"x": torch.tensor([float(i)]), "y": torch.tensor(i, dtype=torch.int64)}

    serial = _build_items_ordered(build, 37, 1)
    parallel = _build_items_ordered(build, 37, 8)
    assert len(serial) == len(parallel) == 37
    for expected, actual in zip(serial, parallel):
        assert expected.keys() == actual.keys()
        assert all(torch.equal(expected[k], actual[k]) for k in expected)


def test_v4872_training_wrappers_wire_interaction_flags_and_cache_workers():
    root = Path(__file__).resolve().parents[1]
    train = (root / "scripts/train_ocrap_v48_trac_sr.sh").read_text()
    adapt = (root / "scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh").read_text()
    assert "persistent_tensor_cache_build_workers" in train
    assert "SEMANTIC_WITNESS_INTERACTION_BOX_SUPPORT" in train
    assert "SEMANTIC_WITNESS_INTERACTION_HULL_SUPPORT" in train
    assert "SEMANTIC_WITNESS_INTERACTION_BOX_SUPPORT" in adapt
    assert "SEMANTIC_WITNESS_INTERACTION_HULL_SUPPORT" in adapt
