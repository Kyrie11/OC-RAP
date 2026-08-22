from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ocrap.cli.train import _absolute_feasibility_bce
from ocrap.models.data import (
    DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_SCHEMA,
    direct_executable_recovery_witness_features_from_sample,
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
        direct_recovery_value_head=True,
        direct_recovery_absolute_executable_witness_correction=True,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _sample(*, include_privileged_noise: bool = False) -> dict:
    ego = np.zeros(9, dtype=np.float32)
    ego[6] = 4.0
    ego[7] = 4.8
    ego[8] = 2.0
    states = np.zeros((10, 9), dtype=np.float32)
    states[:, 0] = np.arange(1, 11, dtype=np.float32) * 0.4
    states[:, 6] = 4.0
    states[:, 7] = 4.8
    states[:, 8] = 2.0
    controls = np.zeros((9, 4), dtype=np.float32)
    history = np.zeros((1, 2, 16), dtype=np.float32)
    history[0, 1, 0] = 10.0
    history[0, 1, 10] = 4.8
    history[0, 1, 11] = 2.0
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
        # These are deliberately teacher/future-like fields.  ERWF must not read them.
        d.update({
            "m_star": np.random.default_rng(4861).normal(size=(3, 2)).astype(np.float32),
            "future_signature": np.random.default_rng(4862).normal(size=(3, 8)).astype(np.float32),
            "r_dep_star": np.float32(-123.0),
            "bucket_id": np.int64(2),
        })
    return d


def _field(num_options: int = 2, *, privileged_noise: bool = False) -> torch.Tensor:
    f = direct_executable_recovery_witness_features_from_sample(
        _sample(include_privileged_noise=privileged_noise),
        {
            "sample_rate_hz": 10.0,
            "recovery_horizon_s": 4.0,
            "model": {"feature_max_agents": 2},
        },
        num_options=num_options,
    )
    return torch.from_numpy(f).float()


def test_erwf_option_resolved_field_is_finite_and_padded() -> None:
    f = _field(4)
    assert f.shape == (4, 6)
    assert torch.isfinite(f).all()
    assert not torch.allclose(f[0], f[1])
    assert torch.equal(f[2:], torch.zeros((2, 6)))
    # The lateral escape should improve terminal clearance relative to stop in this scene.
    assert float(f[1, 1]) > float(f[0, 1])
    assert float(f[1, 2]) > float(f[0, 2])


def test_erwf_ignores_teacher_future_and_regime_fields() -> None:
    clean = _field(2, privileged_noise=False)
    noisy = _field(2, privileged_noise=True)
    assert torch.equal(clean, noisy)


def test_erwf_zero_weight_is_execution_exact_native_source() -> None:
    torch.manual_seed(4861)
    model = _model().eval()
    L = _layout()
    x = torch.randn((3, L.total_dim), dtype=torch.float32)
    memory = model._scene_tokens(x)
    _, native = model._direct_recovery_option_compatibility_evidence(memory, x, None)
    f = _field(2).unsqueeze(0).repeat(3, 1, 1)
    out = model._direct_executable_witness_absolute_feasibility(memory, x, None, f)
    assert out is not None
    _, p, features, weights = out
    assert torch.allclose(p, native[:, 1], atol=0.0, rtol=0.0)
    assert features.shape == (3, 2, 6)
    assert torch.equal(weights, torch.zeros(6))


def test_erwf_fails_closed_without_option_resolved_side_channel() -> None:
    model = _model().eval()
    L = _layout()
    x = torch.zeros((1, L.total_dim), dtype=torch.float32)
    memory = model._scene_tokens(x)
    try:
        model._direct_executable_witness_absolute_feasibility(memory, x, None, None)
    except RuntimeError as exc:
        assert "ERWF features missing" in str(exc)
    else:
        raise AssertionError("ERWF must fail closed without executable witness features")


def test_erwf_bce_gradient_isolated_to_six_shared_weights() -> None:
    torch.manual_seed(4861)
    model = _model().train()
    for name, p in model.named_parameters():
        p.requires_grad_(name == "direct_absolute_executable_witness_weight")
    L = _layout()
    x = torch.randn((4, L.total_dim), dtype=torch.float32)
    memory = model._scene_tokens(x)
    base = _field(2)
    f = torch.stack([base, base.flip(0), base, base.flip(0)], dim=0)
    out = model._direct_executable_witness_absolute_feasibility(memory, x, None, f)
    assert out is not None
    logit, _, _, _ = out
    loss = _absolute_feasibility_bce(
        {"direct_recovery_absolute_feasibility_logit": logit},
        {
            "r_dep_star": torch.tensor([-0.5, 0.5, -0.5, 0.5]),
            "is_nominal": torch.zeros(4),
            "bucket_id": torch.tensor([1, 1, 2, 2]),
            "time_index": torch.arange(4),
        },
    )
    loss.backward()
    grad = model.direct_absolute_executable_witness_weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert bool(torch.any(grad != 0))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 6


def test_erwf_mutual_exclusion_with_all_previous_absolute_source_variants() -> None:
    L = _layout()
    variants = (
        {"direct_recovery_absolute_feasibility_head": True},
        {"direct_recovery_absolute_option_margin_correction": True},
        {"direct_recovery_absolute_physical_headroom_correction": True},
    )
    for extra in variants:
        try:
            OCRAPModel(
                input_dim=L.total_dim,
                num_roots=2,
                num_options=2,
                d_model=16,
                d_obs=8,
                encoder_type="structured_transformer",
                feature_layout=asdict(L),
                num_layers=1,
                num_heads=4,
                direct_recovery_value_head=True,
                direct_recovery_absolute_executable_witness_correction=True,
                **extra,
            )
        except ValueError as exc:
            assert "mutually exclusive" in str(exc)
        else:
            raise AssertionError("ERWF must be a single-axis absolute-source intervention")


def _checkpoint(model: OCRAPModel, *, schema: int | None) -> dict:
    L = _layout()
    ckpt = {
        "model_state": model.state_dict(),
        "input_dim": L.total_dim,
        "num_roots": 3,
        "num_options": 2,
        "d_model": 16,
        "d_obs": 8,
        "tau_obs": 1.0,
        "encoder_type": "structured_transformer",
        "feature_layout": asdict(L),
        "d_signature": 0,
        "d_future_signature": 0,
        "option_feature_dim": 0,
        "direct_recovery_value_head": True,
        "direct_recovery_absolute_executable_witness_correction": True,
        "direct_recovery_evidence_native_certificate_preservation": True,
        "cfg": {
            "sample_rate_hz": 10.0,
            "recovery_horizon_s": 4.0,
            "model": {
                "transformer_layers": 1,
                "transformer_heads": 4,
                "dropout": 0.0,
                "encoder_type": "structured_transformer",
                "direct_recovery_value_head": True,
                "direct_recovery_absolute_executable_witness_correction": True,
                "direct_recovery_evidence_native_certificate_preservation": True,
            },
            "runtime": {"device": "cpu"},
        },
    }
    if schema is not None:
        ckpt["direct_recovery_absolute_executable_witness_feature_schema"] = schema
        ckpt["direct_recovery_absolute_executable_witness_feature_source"] = (
            "option_resolved_executable_recovery_continuation_side_channel"
        )
    return ckpt


def test_erwf_checkpoint_inference_roundtrip_and_legacy_rejection(tmp_path) -> None:
    from ocrap.models.inference import load_model_bundle

    model = _model().eval()
    good = tmp_path / "erwf.pt"
    torch.save(_checkpoint(model, schema=DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_SCHEMA), good)
    bundle = load_model_bundle(good)
    assert bundle is not None
    assert bundle.model.direct_recovery_absolute_executable_witness_correction
    assert bundle.model.direct_absolute_executable_witness_weight is not None
    assert torch.equal(bundle.model.direct_absolute_executable_witness_weight, torch.zeros(6))

    legacy = tmp_path / "legacy.pt"
    torch.save(_checkpoint(model, schema=None), legacy)
    try:
        load_model_bundle(legacy)
    except RuntimeError as exc:
        assert "legacy/unknown ERWF checkpoint feature semantics" in str(exc)
    else:
        raise AssertionError("schema-less ERWF checkpoint must be rejected")


def test_v4861_plumbing_and_launcher_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    train = (root / "scripts" / "train_ocrap_v48_trac_sr.sh").read_text()
    adapt = (root / "scripts" / "adapt_ocrap_v48_36_ocaf_single_stage.sh").read_text()
    assert "ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION" in train
    assert "direct_recovery_absolute_executable_witness_correction" in train
    assert "ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION" in adapt
    launcher = root / "scripts" / "run_v48_61_dcp_drfc_bcde_rifa_erwf_two_gpu.sh"
    assert launcher.is_file()
    text = launcher.read_text()
    assert "EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_executable_witness_weight" in text
    assert "STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_executable_witness_weight" in text
    assert "ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION=true" in text
    assert "ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=false" in text
    assert "MAX_EVIDENCE_CALIBRATOR_PARAMS=6" in text
    assert "PROPOSAL_TOP_K=5" in text
    assert "ABSOLUTE_FEASIBILITY_THRESHOLD=0.5" in text
    assert "check_v48_61_state_isolation.py" in text
    assert "check_v48_61_pipeline_complete.py" in text
    assert "EVIDENCE_CENTER" not in text.upper()
    assert "PRED_ADV_CENTER" not in text.upper()


def test_erwf_checkpoint_to_vectorized_inference_executes_side_channel(tmp_path) -> None:
    from ocrap.models.inference import load_model_bundle, predict_samples

    model = _model().eval()
    ckpt = tmp_path / "erwf_predict.pt"
    torch.save(_checkpoint(model, schema=DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_SCHEMA), ckpt)
    bundle = load_model_bundle(ckpt)
    assert bundle is not None
    preds = predict_samples([_sample(), _sample(include_privileged_noise=True)], bundle)
    assert len(preds) == 2
    assert all(np.isfinite(p.r_dep) and np.isfinite(p.r_orc) for p in preds)
    assert all(p.margins.shape == (3, 2) for p in preds)
