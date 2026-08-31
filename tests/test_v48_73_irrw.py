from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from ocrap.cli.train import _semantic_witness_checkpoint_feature_contract
from ocrap.models.data import (
    DIRECT_INTERACTION_RESPONSE_RECOVERY_WITNESS_FEATURE_DIM,
    DIRECT_INTERACTION_RESPONSE_RECOVERY_WITNESS_FEATURE_SCHEMA,
    OPTION_FEATURE_DIM,
    direct_semantic_recovery_witness_features_from_sample,
    option_features_from_sample,
    _persistent_tensor_cache_key,
)
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout():
    return FlatFeatureLayout(feature_max_agents=2)


def _sample(*, xy=(70.0, 0.0), acc_hist=((-4.0, 0.0), (-4.0, 0.0), (0.0, 0.0))):
    ego = np.zeros(9, np.float32)
    ego[6] = 4.0; ego[7] = 4.8; ego[8] = 2.0
    states = np.zeros((10, 9), np.float32)
    states[:, 0] = np.arange(1, 11) * 0.4
    states[:, 6] = 4.0; states[:, 7] = 4.8; states[:, 8] = 2.0
    controls = np.zeros((9, 4), np.float32)
    ah = np.asarray(acc_hist, np.float32).reshape(-1, 2)
    hist = np.zeros((len(ah), 2, 16), np.float32)
    valid = np.ones((len(ah), 2), bool)
    hist[:, 1, 0] = float(xy[0]); hist[:, 1, 1] = float(xy[1])
    hist[:, 1, 5:7] = ah
    hist[:, 1, 10] = 4.8; hist[:, 1, 11] = 2.0
    return {
        "ego_state": ego, "prefix_states": states, "prefix_controls": controls,
        "agent_history": hist, "agent_valid": valid,
        "recovery_modes": np.asarray(["stop", "lateral_escape"], object),
        "recovery_params": np.asarray([[-5.0, 5.0, 0.0], [3.5, 5.0, 1.5]], np.float32),
        "option_valid": np.asarray([1, 1], bool),
        "prefix_macro_id": 0, "prefix_macro_name": "candidate",
        "prefix_param": np.zeros(0, np.float32), "utility": 0.0,
        "feasible": 1.0, "hard_violation": 0.0, "harm_proxy": 0.0,
    }


def _cfg(*, anchor=False, response=False):
    return {
        "sample_rate_hz": 10.0, "recovery_horizon_s": 4.0, "prefix_horizon_s": 1.0,
        "route_dev_max_m": 2.5,
        "control_limits": {"a_max": 3.0, "a_min": -6.0, "delta_max": 0.55, "j_max": 6.0, "steer_rate_max": 0.5},
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
            "direct_recovery_semantic_witness_history_occupancy_reachability": False,
            "direct_recovery_semantic_witness_interaction_box_support": True,
            "direct_recovery_semantic_witness_interaction_hull_support": True,
            "direct_recovery_semantic_witness_interaction_anchor_support": anchor,
            "direct_recovery_semantic_witness_interaction_response_support": response,
        },
        "default_available_distance_m": 60.0,
    }


def _model(*, anchor=True, response=False):
    L = _layout()
    return OCRAPModel(
        input_dim=L.total_dim, num_roots=3, num_options=2, d_model=16, d_obs=8,
        encoder_type="structured_transformer", feature_layout=asdict(L), num_layers=1,
        num_heads=4, dropout=0.0, option_feature_dim=OPTION_FEATURE_DIM,
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
        direct_recovery_semantic_witness_interaction_box_support=True,
        direct_recovery_semantic_witness_interaction_hull_support=True,
        direct_recovery_semantic_witness_interaction_anchor_support=anchor,
        direct_recovery_semantic_witness_interaction_response_support=response,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _force(m):
    m.root_logit_head.forward = MethodType(lambda self,z: torch.zeros((*z.shape[:-1],1),device=z.device,dtype=z.dtype),m.root_logit_head)
    m.obs_embed_head.forward = MethodType(lambda self,z: torch.zeros((*z.shape[:-1],8),device=z.device,dtype=z.dtype),m.obs_embed_head)
    def mf(self,z): return torch.zeros((z.shape[0],3,2,1),device=z.device,dtype=z.dtype)
    m.margin_head.forward = MethodType(mf,m.margin_head)


def _support(m, feat):
    _force(m)
    x=torch.zeros((1,_layout().total_dim)); opt=torch.from_numpy(option_features_from_sample(_sample())).float().unsqueeze(0)
    rv=torch.ones((1,3),dtype=torch.bool); ov=torch.ones((1,2),dtype=torch.bool)
    out=m._direct_semantic_witness_absolute_feasibility(m._scene_tokens(x),x,opt,feat,root_valid=rv,option_valid=ov)
    return out[4],out[5]


def _feat22():
    f=torch.full((1,2,22),0.6)
    f[...,4]=-float(np.tanh(1.0))  # projection fidelity = 1/2
    f[...,8]=0.3; f[...,9]=0.3; f[...,11]=0.0; f[...,12]=0.6; f[...,13]=0.6
    f[...,19]=float(np.tanh(5.0))  # must be ignored by v48.73 trust selector
    f[0,0,20]=float(np.tanh(0.0)); f[0,1,20]=float(np.tanh(1.0))
    f[0,0,21]=float(np.tanh(0.0)); f[0,1,21]=float(np.tanh(2.0))
    return f


def test_v4873_schema9_contract_and_dim():
    base={
        "direct_recovery_absolute_semantic_witness_correction":True,
        "direct_recovery_semantic_witness_route_alignment":True,
        "direct_recovery_semantic_witness_reentry_alignment":True,
        "direct_recovery_semantic_witness_control_projection":True,
        "direct_recovery_semantic_witness_projection_fidelity_weighting":True,
        "direct_recovery_semantic_witness_interaction_box_support":True,
        "direct_recovery_semantic_witness_interaction_hull_support":True,
    }
    for response in (False,True):
        cfg={**base,"direct_recovery_semantic_witness_interaction_anchor_support":True,"direct_recovery_semantic_witness_interaction_response_support":response}
        assert _semantic_witness_checkpoint_feature_contract(cfg)==(9,"interaction_response_history_reachability_projected_recovery_witness")
    assert DIRECT_INTERACTION_RESPONSE_RECOVERY_WITNESS_FEATURE_SCHEMA==9
    assert DIRECT_INTERACTION_RESPONSE_RECOVERY_WITNESS_FEATURE_DIM==22


def test_v4873_first20_coordinates_are_execution_exact_v4872_main():
    d=_sample()
    m72=direct_semantic_recovery_witness_features_from_sample(d,_cfg(anchor=False,response=False),num_options=2)
    for response in (False,True):
        f=direct_semantic_recovery_witness_features_from_sample(d,_cfg(anchor=True,response=response),num_options=2)
        assert f.shape==(2,22)
        assert np.array_equal(m72,f[:,:20])


def test_v4873_temporal_anchor_does_not_instantly_reuse_stale_acceleration_extreme():
    # Historical inward -x acceleration is stale and the current acceleration is
    # zero. Static M72 can use -4 m/s^2 immediately; N73 must ramp to it.
    d=_sample(acc_hist=((-4.0,0.0),(-4.0,0.0),(0.0,0.0)))
    f=direct_semantic_recovery_witness_features_from_sample(d,_cfg(anchor=True),num_options=2)
    assert np.all(f[:,20] <= f[:,19] + 1e-6)
    assert np.any(f[:,20] < f[:,19] - 1e-4)


def test_v4873_observed_response_rate_can_reject_stale_inward_extreme():
    # The latest observed transition moves acceleration from inward (-4 x) to 0;
    # along the inward interaction normal there is no positive observed jerk.
    # O73 therefore cannot immediately ramp back toward the stale inward extreme.
    d=_sample(acc_hist=((-4.0,0.0),(-4.0,0.0),(0.0,0.0)))
    f=direct_semantic_recovery_witness_features_from_sample(d,_cfg(anchor=True,response=True),num_options=2)
    assert np.all(f[:,21] <= f[:,20] + 1e-6)
    assert np.any(f[:,21] < f[:,20] - 1e-4)


def test_v4873_model_replaces_static_hull_trust_instead_of_stacking():
    feat=_feat22()
    anchor=_model(anchor=True,response=False).eval(); response=_model(anchor=True,response=True).eval()
    va,sa=_support(anchor,feat); vr,sr=_support(response,feat)
    assert torch.equal(va>0,vr>0)
    # option1: projection 1/2. Anchor risk 1 -> total 1/4. Response risk 2 -> 1/6.
    # Static hull risk=5 must not also multiply either branch.
    assert torch.allclose(sa[...,1],torch.tensor([[0.25]]),atol=2e-5)
    assert torch.allclose(sr[...,1],torch.tensor([[1.0/6.0]]),atol=2e-5)


def test_v4873_requires_nested_empirical_hull_chain():
    L=_layout(); kw=dict(input_dim=L.total_dim,num_roots=3,num_options=2,d_model=16,d_obs=8,encoder_type="structured_transformer",feature_layout=asdict(L),num_layers=1,num_heads=4,dropout=0.0,option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,direct_recovery_absolute_semantic_witness_correction=True,direct_recovery_semantic_witness_control_projection=True,direct_recovery_semantic_witness_projection_fidelity_weighting=True)
    with pytest.raises(ValueError,match="empirical-hull"):
        OCRAPModel(**kw,direct_recovery_semantic_witness_interaction_anchor_support=True)
    with pytest.raises(ValueError,match="response support requires"):
        OCRAPModel(**kw,direct_recovery_semantic_witness_interaction_box_support=True,direct_recovery_semantic_witness_interaction_hull_support=True,direct_recovery_semantic_witness_interaction_response_support=True)


def test_v4873_runner_contract_freezes_forbidden_directions():
    p=Path(__file__).resolve().parents[1]/"scripts/run_v48_73_dcp_drfc_bcde_rifa_irrw_two_gpu.sh"
    assert p.is_file()
    text=p.read_text()
    assert "N73_ANCHORED_HULL" in text and "O73_Main_OCIRRW" in text
    assert "SEMANTIC_WITNESS_INTERACTION_BOX_SUPPORT=true" in text
    assert "SEMANTIC_WITNESS_INTERACTION_HULL_SUPPORT=true" in text
    assert "SEMANTIC_WITNESS_INTERACTION_ANCHOR_SUPPORT=true" in text
    assert "SEMANTIC_WITNESS_INTERACTION_RESPONSE_SUPPORT" in text
    assert "SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false" in text
    assert "SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY=false" in text
    assert "SEMANTIC_WITNESS_HISTORY_OCCUPANCY_REACHABILITY=false" in text
    assert "PROPOSAL_TOP_K=5" in text



def test_v4873_shell_wiring_reaches_train_config():
    root = Path(__file__).resolve().parents[1]
    adapt = (root / "scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh").read_text()
    train = (root / "scripts/train_ocrap_v48_trac_sr.sh").read_text()
    for name in (
        "SEMANTIC_WITNESS_INTERACTION_ANCHOR_SUPPORT",
        "SEMANTIC_WITNESS_INTERACTION_RESPONSE_SUPPORT",
    ):
        assert f'{name}="${{{name}:-false}}"' in adapt
        assert f'{name}="${name}"' in adapt
        assert name in train
    assert "model.direct_recovery_semantic_witness_interaction_anchor_support" in train
    assert "model.direct_recovery_semantic_witness_interaction_response_support" in train


def test_v4873_schema9_anchor_and_response_share_persistent_cache(tmp_path):
    root = tmp_path / "dataset"
    samples = root / "samples"
    samples.mkdir(parents=True)
    sample = samples / "row.npz"
    sample.write_bytes(b"not-opened-by-cache-key")
    (root / "manifest.csv").write_text(
        "path,split_id\nsamples/row.npz,evidence_adapt_train_near_contact\n"
    )

    def cfg(response: bool):
        c = _cfg(anchor=True, response=response)
        c["model"]["direct_recovery_absolute_semantic_witness_correction"] = True
        c["training"] = {"direct_policy_metric_exact_eligibility": False}
        return c

    kwargs = dict(
        paths=[sample], num_roots=3, num_options=2,
        d_signature=8, d_future_signature=8, feature_dim=_layout().total_dim,
    )
    n_key = _persistent_tensor_cache_key(cfg=cfg(False), **kwargs)
    o_key = _persistent_tensor_cache_key(cfg=cfg(True), **kwargs)
    assert n_key == o_key


def test_v4873_runner_preregisters_required_attribution_order():
    text = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_v48_73_dcp_drfc_bcde_rifa_irrw_two_gpu.sh"
    ).read_text()
    assert 'train_irrw_arm "$N_RUN" N73_ANCHORED_HULL false' in text
    assert 'train_irrw_arm "$O_RUN" O73_Main_OCIRRW true' in text
    assert "audit_v48_73_interaction_response.py" in text
    assert "compare_v48_73_irrw.py" in text
    assert "check_v48_73_pipeline_complete.py" in text
    assert "retain_supported_directional_set_then_interaction_response_dynamics" in text



def test_v4873_response_audit_preserves_exact_zero_teacher_feasibility():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "audit_v48_73_interaction_response",
        root / "tools/audit_v48_73_interaction_response.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.feas({"teacher_candidate_r_dep": 0.0}) is True
    assert module.feas({"teacher_candidate_r_dep": -1.0e-6}) is False
