from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from ocrap.cli.train import _absolute_feasibility_bce
from ocrap.models.data import (
    DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA,
    OPTION_FEATURE_DIM,
    RECOVERY_MODE_TO_ID,
    direct_common_recovery_witness_features_from_sample,
    option_features_from_sample,
)
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout() -> FlatFeatureLayout:
    return FlatFeatureLayout(feature_max_agents=2)


def _model(num_options: int = 2) -> OCRAPModel:
    L = _layout()
    return OCRAPModel(
        input_dim=L.total_dim,
        num_roots=3,
        num_options=num_options,
        d_model=16,
        d_obs=8,
        encoder_type="structured_transformer",
        feature_layout=asdict(L),
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        option_feature_dim=OPTION_FEATURE_DIM,
        direct_recovery_value_head=True,
        direct_recovery_absolute_common_witness_correction=True,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _sample(*, include_privileged_noise: bool = False) -> dict:
    ego = np.zeros(9, dtype=np.float32)
    ego[6] = 4.0; ego[7] = 4.8; ego[8] = 2.0
    states = np.zeros((10, 9), dtype=np.float32)
    states[:, 0] = np.arange(1, 11, dtype=np.float32) * 0.4
    states[:, 6] = 4.0; states[:, 7] = 4.8; states[:, 8] = 2.0
    controls = np.zeros((9, 4), dtype=np.float32)
    history = np.zeros((1, 2, 16), dtype=np.float32)
    history[0, 1, 0] = 10.0
    history[0, 1, 10] = 4.8; history[0, 1, 11] = 2.0
    d = {
        "ego_state": ego,
        "prefix_states": states,
        "prefix_controls": controls,
        "agent_history": history,
        "agent_valid": np.asarray([[True, True]], dtype=bool),
        "recovery_modes": np.asarray(["stop", "lateral_escape"], dtype=object),
        "recovery_params": np.asarray([[-5.0, 5.0, 0.0], [3.5, 5.0, 1.5]], dtype=np.float32),
        "option_valid": np.asarray([True, True], dtype=bool),
        "prefix_macro_id": 0,
        "prefix_macro_name": "candidate",
        "prefix_param": np.zeros((0,), dtype=np.float32),
        "utility": 0.0,
        "feasible": 1.0,
        "hard_violation": 0.0,
        "harm_proxy": 0.0,
    }
    if include_privileged_noise:
        d.update({
            "m_star": np.random.default_rng(4862).normal(size=(3, 2)).astype(np.float32),
            "root_future_signature": np.random.default_rng(4863).normal(size=(3, 8)).astype(np.float32),
            "r_dep_star": np.float32(-999.0),
            "bucket_id": np.int64(2),
        })
    return d


def _cfg() -> dict:
    return {"sample_rate_hz": 10.0, "recovery_horizon_s": 4.0, "model": {"feature_max_agents": 2}}


def _field(n: int = 2, *, privileged_noise: bool = False) -> torch.Tensor:
    return torch.from_numpy(direct_common_recovery_witness_features_from_sample(
        _sample(include_privileged_noise=privileged_noise), _cfg(), num_options=n
    )).float()


def _option_features(batch: int = 1) -> torch.Tensor:
    f = torch.from_numpy(option_features_from_sample(_sample())).float()
    return f.unsqueeze(0).repeat(batch, 1, 1)


def test_occwrf_field_is_option_resolved_finite_and_privilege_free() -> None:
    f = _field(4)
    assert f.shape == (4, 10)
    assert torch.isfinite(f).all()
    assert torch.equal(f[2:], torch.zeros((2, 10)))
    assert not torch.allclose(f[0], f[1])
    assert torch.equal(_field(2), _field(2, privileged_noise=True))


def test_occwrf_contact_recovery_barrier_accepts_finite_time_recovery_not_initial_violation() -> None:
    torch.manual_seed(4862)
    m = _model().eval(); L = _layout()
    x = torch.zeros((1, L.total_dim))
    feat = torch.zeros((1, 2, 10))
    # option 0: initial/min clearance and stability violated, but terminal is
    # positive and both directions improve -> recovery barrier must be positive.
    feat[0, 0] = torch.tensor([-0.8, 0.6, 0.7, 0.5, 0.4, -0.7, 0.5, 0.6, 0.0, 0.0])
    # option 1: terminal looks recovered but the continuation first becomes
    # worse than the initial violated state -> secondary excursion must veto.
    feat[0, 1] = torch.tensor([-0.8, 0.6, 0.7, 0.5, 0.4, -0.7, 0.5, 0.6, -0.2, -0.2])
    memory = m._scene_tokens(x)
    out = m._direct_common_witness_absolute_feasibility(
        memory, x, _option_features(), feat,
        root_valid=torch.ones((1,3),dtype=torch.bool), option_valid=torch.ones((1,2),dtype=torch.bool),
    )
    assert out is not None
    viability = out[4]
    assert float(viability[0,0]) > 0.0
    assert float(viability[0,1]) < 0.0


def test_occwrf_common_option_support_requires_same_option_across_aliased_roots() -> None:
    m = _model().eval(); L = _layout(); x = torch.zeros((1, L.total_dim))
    memory = m._scene_tokens(x)
    # Make all roots equally probable and observation-identical.
    m.root_logit_head.forward = MethodType(lambda self, z: torch.zeros((*z.shape[:-1],1), device=z.device), m.root_logit_head)
    m.obs_embed_head.forward = MethodType(lambda self, z: torch.zeros((*z.shape[:-1],8), device=z.device), m.obs_embed_head)
    # Two roots prefer different options; third weakly agrees with option0.
    def margin_forward(self, z):
        vals = torch.tensor([[[3.0, -3.0],[-3.0,3.0],[2.5,-2.5]]], device=z.device, dtype=z.dtype)
        return vals.unsqueeze(-1).expand(z.shape[0], -1, -1, 1)
    m.margin_head.forward = MethodType(margin_forward, m.margin_head)
    feat = torch.ones((1,2,10))*0.5
    out = m._direct_common_witness_absolute_feasibility(
        memory, x, _option_features(), feat,
        root_valid=torch.ones((1,3),dtype=torch.bool), option_valid=torch.ones((1,2),dtype=torch.bool),
    )
    assert out is not None
    support = out[5]
    # No option can receive near-one common support because aliased roots disagree.
    assert float(support.max()) < 0.75


def test_occwrf_zero_gain_is_execution_exact_native_b() -> None:
    torch.manual_seed(4862)
    m = _model().eval(); L = _layout(); x = torch.randn((3,L.total_dim))
    memory = m._scene_tokens(x)
    _, native = m._direct_recovery_option_compatibility_evidence(
        memory, x, _option_features(3), root_valid=torch.ones((3,3),dtype=torch.bool), option_valid=torch.ones((3,2),dtype=torch.bool)
    )
    feat = _field(2).unsqueeze(0).repeat(3,1,1)
    out = m._direct_common_witness_absolute_feasibility(
        memory, x, _option_features(3), feat,
        root_valid=torch.ones((3,3),dtype=torch.bool), option_valid=torch.ones((3,2),dtype=torch.bool)
    )
    assert out is not None
    assert torch.allclose(out[1], native[:,1], atol=0.0, rtol=0.0)
    assert torch.equal(out[3], torch.zeros(2))


def test_occwrf_bce_gradient_isolated_to_two_shared_gains() -> None:
    torch.manual_seed(4862)
    m = _model().train()
    for name,p in m.named_parameters(): p.requires_grad_(name == "direct_absolute_common_witness_gain")
    L=_layout(); x=torch.randn((4,L.total_dim)); memory=m._scene_tokens(x)
    base=_field(2); feat=torch.stack([base,base.flip(0),base,base.flip(0)])
    out=m._direct_common_witness_absolute_feasibility(
        memory,x,_option_features(4),feat,
        root_valid=torch.ones((4,3),dtype=torch.bool),option_valid=torch.ones((4,2),dtype=torch.bool)
    )
    assert out is not None
    loss=_absolute_feasibility_bce(
        {"direct_recovery_absolute_feasibility_logit":out[0]},
        {"r_dep_star":torch.tensor([-0.5,0.5,-0.5,0.5]),"is_nominal":torch.zeros(4),"bucket_id":torch.tensor([1,1,2,2]),"time_index":torch.arange(4)},
    )
    loss.backward(); g=m.direct_absolute_common_witness_gain.grad
    assert g is not None and torch.isfinite(g).all() and bool(torch.any(g != 0))
    assert sum(p.numel() for p in m.parameters() if p.requires_grad)==2


def test_occwrf_fails_closed_and_is_mutually_exclusive() -> None:
    m=_model().eval(); L=_layout(); x=torch.zeros((1,L.total_dim)); memory=m._scene_tokens(x)
    try:
        m._direct_common_witness_absolute_feasibility(memory,x,_option_features(),None)
    except RuntimeError as exc:
        assert "OC-CWRF features missing" in str(exc)
    else: raise AssertionError("missing side channel must fail closed")
    for extra in (
        {"direct_recovery_absolute_feasibility_head":True},
        {"direct_recovery_absolute_option_margin_correction":True},
        {"direct_recovery_absolute_physical_headroom_correction":True},
        {"direct_recovery_absolute_executable_witness_correction":True},
    ):
        try:
            OCRAPModel(input_dim=L.total_dim,num_roots=2,num_options=2,d_model=16,d_obs=8,
                encoder_type="structured_transformer",feature_layout=asdict(L),num_layers=1,num_heads=4,
                option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_value_head=True,
                direct_recovery_absolute_common_witness_correction=True,**extra)
        except ValueError as exc: assert "mutually exclusive" in str(exc)
        else: raise AssertionError("OC-CWRF must be single-axis")


def _checkpoint(model: OCRAPModel, *, schema: int | None) -> dict:
    L=_layout()
    ckpt={
        "model_state":model.state_dict(),"input_dim":L.total_dim,"num_roots":3,"num_options":2,
        "d_model":16,"d_obs":8,"tau_obs":1.0,"encoder_type":"structured_transformer","feature_layout":asdict(L),
        "d_signature":0,"d_future_signature":0,"option_feature_dim":OPTION_FEATURE_DIM,
        "direct_recovery_value_head":True,"direct_recovery_absolute_common_witness_correction":True,
        "direct_recovery_evidence_native_certificate_preservation":True,
        "cfg":{"sample_rate_hz":10.0,"recovery_horizon_s":4.0,"model":{"transformer_layers":1,"transformer_heads":4,"dropout":0.0,
            "encoder_type":"structured_transformer","option_feature_dim":OPTION_FEATURE_DIM,"direct_recovery_value_head":True,
            "direct_recovery_absolute_common_witness_correction":True,"direct_recovery_evidence_native_certificate_preservation":True},"runtime":{"device":"cpu"}},
    }
    if schema is not None:
        ckpt["direct_recovery_absolute_common_witness_feature_schema"]=schema
        ckpt["direct_recovery_absolute_common_witness_feature_source"]="observation_consistent_option_resolved_finite_time_recovery_witness"
    return ckpt


def test_occwrf_checkpoint_roundtrip_legacy_rejection_and_vectorized_inference(tmp_path) -> None:
    from ocrap.models.inference import load_model_bundle,predict_samples
    m=_model().eval(); good=tmp_path/'good.pt'; torch.save(_checkpoint(m,schema=DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA),good)
    b=load_model_bundle(good); assert b is not None and b.model.direct_absolute_common_witness_gain is not None
    preds=predict_samples([_sample(),_sample(include_privileged_noise=True)],b)
    assert len(preds)==2 and all(np.isfinite(p.r_dep) and np.isfinite(p.r_orc) for p in preds)
    bad=tmp_path/'bad.pt'; torch.save(_checkpoint(m,schema=None),bad)
    try: load_model_bundle(bad)
    except RuntimeError as exc: assert "legacy/unknown OC-CWRF checkpoint feature semantics" in str(exc)
    else: raise AssertionError("schema-less checkpoint must fail closed")


def test_occwrf_shell_plumbing_present() -> None:
    root=Path(__file__).resolve().parents[1]
    train=(root/'scripts/train_ocrap_v48_trac_sr.sh').read_text(); adapt=(root/'scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh').read_text()
    assert 'ABSOLUTE_COMMON_WITNESS_CORRECTION' in train and 'direct_recovery_absolute_common_witness_correction' in train
    assert 'ABSOLUTE_COMMON_WITNESS_CORRECTION' in adapt
    launcher=root/'scripts/run_v48_62_dcp_drfc_bcde_rifa_occwrf_two_gpu.sh'
    assert launcher.is_file()
    text=launcher.read_text()
    assert 'EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_common_witness_gain' in text
    assert 'STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_common_witness_gain' in text
    assert 'ABSOLUTE_COMMON_WITNESS_CORRECTION=true' in text
    assert 'ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION=false' in text
    assert 'MAX_EVIDENCE_CALIBRATOR_PARAMS=2' in text
    assert 'PROPOSAL_TOP_K=5' in text
    assert 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' in text
    assert 'EVIDENCE_CENTER' not in text.upper() and 'PRED_ADV_CENTER' not in text.upper()
