from __future__ import annotations
import os

from dataclasses import asdict
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from ocrap.cli.train import _semantic_witness_checkpoint_feature_contract
from ocrap.models.data import (
    DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_DIM,
    DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_SCHEMA,
    OPTION_FEATURE_DIM,
    direct_semantic_recovery_witness_features_from_sample,
    option_features_from_sample,
    _persistent_tensor_cache_key,
)
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel
from ocrap.v48_74_signed_viability import signed_viability_diagnostics


@pytest.fixture(autouse=True)
def _enable_v4874(monkeypatch):
    monkeypatch.setenv("OCRAP_V48_74_SIGNED_VIABILITY", "1")


def _layout():
    return FlatFeatureLayout(feature_max_agents=2)


def _sample(*, xy=(16.0, 0.0), acc_hist=((-4.0, 0.0), (-4.0, 0.0), (0.0, 0.0))):
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


def _cfg(*, anchor=True, response=False):
    return {
        "sample_rate_hz": 10.0, "recovery_horizon_s": 4.0, "prefix_horizon_s": 1.0,
        "route_dev_max_m": 2.5,
        "control_limits": {"a_max": 3.0, "a_min": -6.0, "delta_max": 0.55, "j_max": 6.0, "steer_rate_max": 0.5},
        "model": {
            "feature_max_agents": 2,
            "direct_recovery_absolute_semantic_witness_correction": True,
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


def _model(*, response=False):
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
        direct_recovery_semantic_witness_interaction_anchor_support=True,
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
    f[...,19]=float(np.tanh(5.0))  # historical hull coordinate must not be stacked
    f[0,0,20]=0.0; f[0,1,20]=1.0
    f[0,0,21]=0.0; f[0,1,21]=2.0
    return f


def _contract_cfg(response=False):
    return {
        "direct_recovery_absolute_semantic_witness_correction":True,
        "direct_recovery_semantic_witness_route_alignment":True,
        "direct_recovery_semantic_witness_reentry_alignment":True,
        "direct_recovery_semantic_witness_control_projection":True,
        "direct_recovery_semantic_witness_projection_fidelity_weighting":True,
        "direct_recovery_semantic_witness_interaction_box_support":True,
        "direct_recovery_semantic_witness_interaction_hull_support":True,
        "direct_recovery_semantic_witness_interaction_anchor_support":True,
        "direct_recovery_semantic_witness_interaction_response_support":response,
    }


def test_v4874_schema10_contract_and_dim(monkeypatch):
    monkeypatch.setenv("OCRAP_V48_74_SIGNED_VIABILITY","1")
    for response in (False,True):
        assert _semantic_witness_checkpoint_feature_contract(_contract_cfg(response))==(10,"signed_finite_time_viability_projected_recovery_witness")
    assert DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_SCHEMA==10
    assert DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_DIM==22
    monkeypatch.delenv("OCRAP_V48_74_SIGNED_VIABILITY",raising=False)
    assert _semantic_witness_checkpoint_feature_contract(_contract_cfg(False))==(9,"interaction_response_history_reachability_projected_recovery_witness")


def test_v4874_first20_coordinates_are_execution_exact_v4872(monkeypatch):
    d=_sample()
    monkeypatch.setenv("OCRAP_V48_74_SIGNED_VIABILITY","1")
    base=direct_semantic_recovery_witness_features_from_sample(d,_cfg(anchor=False,response=False),num_options=2)
    for response in (False,True):
        f=direct_semantic_recovery_witness_features_from_sample(d,_cfg(anchor=True,response=response),num_options=2)
        assert f.shape==(2,22)
        assert np.array_equal(base,f[:,:20])
        assert np.all(np.isfinite(f[:,20:]))
        assert np.all(f[:,20:]>=0.0)
        assert np.all(f[:,21]>=f[:,20])


def test_v4874_finite_time_signed_barrier_contact_recovery_vs_stall():
    t=np.array([0.0,1.0,2.0])
    recover=np.array([-1.0,-0.5,0.0])
    stall=np.array([-1.0,-1.0,-1.0])
    a=signed_viability_diagnostics(recover,t,clearance_scale=1.0)
    b=signed_viability_diagnostics(stall,t,clearance_scale=1.0)
    assert float(a.first_order_debt) < 1e-10
    assert float(b.first_order_debt) > 0.9
    assert float(a.second_order_debt) >= float(a.first_order_debt)
    assert float(b.second_order_debt) >= float(b.first_order_debt)


def test_v4874_model_consumes_raw_debt_without_tanh_decoder(monkeypatch):
    monkeypatch.setenv("OCRAP_V48_74_SIGNED_VIABILITY","1")
    feat=_feat22()
    p=_model(response=False).eval(); q=_model(response=True).eval()
    vp,sp=_support(p,feat); vq,sq=_support(q,feat)
    assert torch.equal(vp>0,vq>0)
    # option1 projection fidelity=1/2; P debt=1 -> 1/4 total support;
    # Q debt=2 -> 1/6. Static hull risk=5 must not also multiply.
    assert torch.allclose(sp[...,1],torch.tensor([[0.25]]),atol=2e-5)
    assert torch.allclose(sq[...,1],torch.tensor([[1.0/6.0]]),atol=2e-5)


def test_v4874_p_q_share_schema10_persistent_cache(tmp_path,monkeypatch):
    monkeypatch.setenv("OCRAP_V48_74_SIGNED_VIABILITY","1")
    root=tmp_path/"dataset";samples=root/"samples";samples.mkdir(parents=True);sample=samples/"row.npz";sample.write_bytes(b"not-opened-by-cache-key")
    (root/"manifest.csv").write_text("path,split_id\nsamples/row.npz,evidence_adapt_train_near_contact\n")
    def cfg(response):
        c=_cfg(anchor=True,response=response);c["training"]={"direct_policy_metric_exact_eligibility":False};return c
    kwargs=dict(paths=[sample],num_roots=3,num_options=2,d_signature=8,d_future_signature=8,feature_dim=_layout().total_dim)
    assert _persistent_tensor_cache_key(cfg=cfg(False),**kwargs)==_persistent_tensor_cache_key(cfg=cfg(True),**kwargs)



def test_v4874_checkpoint_schema10_roundtrip_through_inference(tmp_path: Path, monkeypatch):
    """Calibration/inference must load both registered V48.74 selector arms.

    This is the exact path that previously raised RC30 after training because the
    loader inferred schema 9 solely from the inherited V48.73 selector flags.
    """
    from ocrap.models.inference import load_model_bundle
    monkeypatch.setenv("OCRAP_V48_74_SIGNED_VIABILITY", "1")
    L = _layout()
    for response in (False, True):
        m = _model(response=response).eval()
        model_cfg = {
            "transformer_layers": 1, "transformer_heads": 4, "dropout": 0.0,
            "encoder_type": "structured_transformer",
            "option_feature_dim": OPTION_FEATURE_DIM,
            "direct_recovery_value_head": True,
            "direct_recovery_absolute_semantic_witness_correction": True,
            "direct_recovery_semantic_witness_active_set_alignment": True,
            "direct_recovery_semantic_witness_path_stop_alignment": False,
            "direct_recovery_semantic_witness_classlocal_transport": False,
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
            "direct_recovery_semantic_witness_interaction_anchor_support": True,
            "direct_recovery_semantic_witness_interaction_response_support": response,
            "direct_recovery_evidence_native_certificate_preservation": True,
        }
        ckpt = {
            "model_state": m.state_dict(), "input_dim": L.total_dim,
            "num_roots": 3, "num_options": 2, "d_model": 16, "d_obs": 8,
            "tau_obs": 1.0, "encoder_type": "structured_transformer",
            "feature_layout": asdict(L), "d_signature": 0, "d_future_signature": 0,
            "option_feature_dim": OPTION_FEATURE_DIM, **model_cfg,
            "direct_recovery_absolute_semantic_witness_feature_schema": 10,
            "direct_recovery_absolute_semantic_witness_feature_source":
                "signed_finite_time_viability_projected_recovery_witness",
            "cfg": {"sample_rate_hz": 10.0, "recovery_horizon_s": 4.0,
                    "training": {"device": "cpu"}, "model": model_cfg},
        }
        path = tmp_path / ("q74.pt" if response else "p74.pt")
        torch.save(ckpt, path)
        bundle = load_model_bundle(path)
        assert bundle is not None
        assert bundle.model.direct_recovery_semantic_witness_interaction_response_support is response


def test_v4874_checkpoint_schema10_inference_fails_closed_without_overlay(tmp_path: Path, monkeypatch):
    from ocrap.models.inference import load_model_bundle
    L = _layout(); m = _model(response=False).eval()
    model_cfg = {
        "transformer_layers": 1, "transformer_heads": 4, "dropout": 0.0,
        "encoder_type": "structured_transformer", "option_feature_dim": OPTION_FEATURE_DIM,
        "direct_recovery_value_head": True,
        "direct_recovery_absolute_semantic_witness_correction": True,
        "direct_recovery_semantic_witness_active_set_alignment": True,
        "direct_recovery_semantic_witness_path_stop_alignment": False,
        "direct_recovery_semantic_witness_classlocal_transport": False,
        "direct_recovery_semantic_witness_route_alignment": True,
        "direct_recovery_semantic_witness_reentry_alignment": True,
        "direct_recovery_semantic_witness_control_projection": True,
        "direct_recovery_semantic_witness_boundary_transport": False,
        "direct_recovery_semantic_witness_projection_fidelity_weighting": True,
        "direct_recovery_semantic_witness_interaction_box_support": True,
        "direct_recovery_semantic_witness_interaction_hull_support": True,
        "direct_recovery_semantic_witness_interaction_anchor_support": True,
        "direct_recovery_semantic_witness_interaction_response_support": False,
        "direct_recovery_evidence_native_certificate_preservation": True,
    }
    ckpt = {
        "model_state": m.state_dict(), "input_dim": L.total_dim, "num_roots": 3,
        "num_options": 2, "d_model": 16, "d_obs": 8, "tau_obs": 1.0,
        "encoder_type": "structured_transformer", "feature_layout": asdict(L),
        "d_signature": 0, "d_future_signature": 0, "option_feature_dim": OPTION_FEATURE_DIM,
        **model_cfg, "direct_recovery_absolute_semantic_witness_feature_schema": 10,
        "direct_recovery_absolute_semantic_witness_feature_source":
            "signed_finite_time_viability_projected_recovery_witness",
        "cfg": {"sample_rate_hz": 10.0, "recovery_horizon_s": 4.0,
                "training": {"device": "cpu"}, "model": model_cfg},
    }
    path = tmp_path / "p74.pt"; torch.save(ckpt, path)
    monkeypatch.delenv("OCRAP_V48_74_SIGNED_VIABILITY", raising=False)
    with pytest.raises(RuntimeError, match="OCRAP_V48_74_SIGNED_VIABILITY=1"):
        load_model_bundle(path)

def test_v4874_runner_preregisters_signed_viability_and_frozen_directions():
    text=(Path(__file__).resolve().parents[1]/"scripts/run_v48_74_dcp_drfc_bcde_rifa_svbw_two_gpu.sh").read_text()
    assert 'train_svbw_arm "$N_RUN" P74_FIRST_ORDER_SVBW false' in text
    assert 'train_svbw_arm "$O_RUN" Q74_MAIN_OC_SVBW true' in text
    assert "semantic_witness_feature_schema':10" in text
    assert "signed_finite_time_viability_projected_recovery_witness" in text
    assert "SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false" in text
    assert "SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY=false" in text
    assert "PROPOSAL_TOP_K=5" in text
    assert "V73_COMPLETE" in text and "interaction_response_reachability_stop_no_parameter_sweep" in text
    assert 'audit_v48_74_interaction_response.py --reference "$T68_RUN" --p74 "$N_RUN" --q74 "$O_RUN"' in text


def test_v4874_response_audit_preserves_exact_zero_teacher_feasibility():
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("audit_v48_74_interaction_response",root/"tools/audit_v48_74_interaction_response.py")
    module=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(module)
    assert module.feas({"teacher_candidate_r_dep":0.0}) is True
    assert module.feas({"teacher_candidate_r_dep":-1.0e-6}) is False
