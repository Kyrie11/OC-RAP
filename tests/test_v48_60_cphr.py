from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ocrap.cli.train import _absolute_feasibility_bce
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout() -> FlatFeatureLayout:
    return FlatFeatureLayout(feature_max_agents=2)


def _model() -> OCRAPModel:
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
        direct_recovery_value_head=True,
        direct_recovery_absolute_physical_headroom_correction=True,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _physical_x(escape: bool = False) -> torch.Tensor:
    L = _layout()
    x = torch.zeros((1, L.total_dim), dtype=torch.float32)
    # ego_state: [x,y,vx,vy,heading,yaw_rate,speed,length,width]
    x[0, 6] = 2.0
    x[0, 7] = 4.8
    x[0, 8] = 2.0
    prefix_start = L.ego_dim + L.prefix_param_dim + L.num_macros + L.scalar_dim
    n_state = L.prefix_flat_dim // 9
    states = torch.zeros((n_state, 9), dtype=torch.float32)
    states[:, 6] = 2.0
    states[:, 7] = 4.8
    states[:, 8] = 2.0
    if escape:
        states[:, 0] = torch.linspace(0.0, 8.0, n_state)
    x[0, prefix_start:prefix_start + n_state * 9] = states.reshape(-1)
    controls_start = prefix_start + L.prefix_flat_dim
    agents_start = controls_start + L.control_flat_dim + L.agent_summary_dim
    # One stationary observed vehicle close enough to overlap at t=0.
    a = torch.zeros(10)
    a[0] = 0.5 / 80.0
    a[7] = 4.8 / 10.0
    a[8] = 2.0 / 5.0
    x[0, agents_start:agents_start + 10] = a
    return x


def test_cphr_zero_weight_is_exact_native_source() -> None:
    model = _model().eval()
    x = _physical_x(escape=True).repeat(3, 1)
    native = torch.tensor([
        [0.2, 0.10, 0.3, 0.4],
        [0.2, 0.50, 0.3, 0.4],
        [0.2, 0.90, 0.3, 0.4],
    ])
    out = model._direct_physical_headroom_absolute_feasibility(x, native)
    assert out is not None
    _, p, feat, weight = out
    assert torch.allclose(p, native[:, 1], atol=2e-6, rtol=0)
    assert feat.shape == (3, 6)
    assert torch.equal(weight, torch.zeros(6))


def test_cphr_signed_clearance_retains_contact_and_escape_headroom() -> None:
    model = _model().eval()
    stay = model._direct_absolute_physical_headroom_features(_physical_x(False))[0]
    escape = model._direct_absolute_physical_headroom_features(_physical_x(True))[0]
    assert torch.isfinite(stay).all() and torch.isfinite(escape).all()
    # Contact remains negative rather than clipped to zero.
    assert float(stay[0]) < 0.0
    # Terminal clearance and clearance recovery gain increase for an escaping prefix.
    assert float(escape[1]) > float(stay[1])
    assert float(escape[2]) > float(stay[2])


def test_cphr_bce_gradient_isolated_to_six_weights() -> None:
    torch.manual_seed(4860)
    model = _model().train()
    for name, p in model.named_parameters():
        p.requires_grad_(name == "direct_absolute_physical_headroom_weight")
    x = torch.cat([_physical_x(False), _physical_x(True), _physical_x(False), _physical_x(True)], dim=0)
    native = torch.tensor([
        [0.0, 0.45, 0.0, 0.0],
        [0.0, 0.45, 0.0, 0.0],
        [0.0, 0.55, 0.0, 0.0],
        [0.0, 0.55, 0.0, 0.0],
    ])
    direct = model._direct_physical_headroom_absolute_feasibility(x, native)
    assert direct is not None
    logit, prob, feat, weight = direct
    out = {"direct_recovery_absolute_feasibility_logit": logit}
    batch = {
        "r_dep_star": torch.tensor([-0.5, 0.5, -0.5, 0.5]),
        "is_nominal": torch.zeros(4),
        "bucket_id": torch.tensor([1, 1, 2, 2]),
        "time_index": torch.arange(4),
    }
    loss = _absolute_feasibility_bce(out, batch)
    loss.backward()
    grad = model.direct_absolute_physical_headroom_weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert bool(torch.any(grad != 0))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 6


def test_cphr_is_mutually_exclusive_with_afe_and_orfc() -> None:
    L = _layout()
    for extra in (
        {"direct_recovery_absolute_feasibility_head": True},
        {"direct_recovery_absolute_option_margin_correction": True},
    ):
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
                direct_recovery_absolute_physical_headroom_correction=True,
                **extra,
            )
        except ValueError as exc:
            assert "mutually exclusive" in str(exc)
        else:
            raise AssertionError("multiple absolute-source corrections must fail closed")


def test_v4860_plumbing_and_v4859_launcher_compatibility() -> None:
    root = Path(__file__).resolve().parents[1]
    train = (root / "scripts" / "train_ocrap_v48_trac_sr.sh").read_text()
    adapt = (root / "scripts" / "adapt_ocrap_v48_36_ocaf_single_stage.sh").read_text()
    old = (root / "scripts" / "run_v48_59_dcp_drfc_bcde_rifa_orfc_two_gpu.sh").read_text()
    assert "ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION" in train
    assert "direct_recovery_absolute_physical_headroom_correction" in train
    assert "ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION" in adapt
    # Existing V48.59 command and single-axis ORFC settings are untouched.
    assert "EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_option_margin_bias" in old
    assert "ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=true" in old
    assert "ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION" not in old


def test_v4860_formal_launcher_is_single_axis() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_v48_60_dcp_drfc_bcde_rifa_cphr_two_gpu.sh").read_text()
    assert "OC-RAP-v48.59-PIPELINE_COMPLETE.json" in script
    assert "V4860_ORFC_D" in script
    assert "EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_physical_headroom_weight" in script
    assert "STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_physical_headroom_weight" in script
    assert "ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=false ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=true" in script
    assert "MAX_EVIDENCE_CALIBRATOR_PARAMS=6" in script
    assert "PROPOSAL_TOP_K=5" in script
    assert "ABSOLUTE_FEASIBILITY_THRESHOLD=0.5" in script
    assert "check_v48_60_state_isolation.py" in script
    assert "check_v48_60_pipeline_complete.py" in script
    assert "EVIDENCE_CENTER" not in script.upper()
    assert "PRED_ADV_CENTER" not in script.upper()


def test_cphr_checkpoint_inference_roundtrip(tmp_path) -> None:
    from ocrap.models.inference import load_model_bundle
    L = _layout()
    model = _model().eval()
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
        "direct_recovery_absolute_physical_headroom_correction": True,
        "direct_recovery_evidence_native_certificate_preservation": True,
        "cfg": {
            "model": {
                "transformer_layers": 1,
                "transformer_heads": 4,
                "dropout": 0.0,
                "encoder_type": "structured_transformer",
                "direct_recovery_value_head": True,
                "direct_recovery_absolute_physical_headroom_correction": True,
                "direct_recovery_evidence_native_certificate_preservation": True,
            },
            "runtime": {"device": "cpu"},
        },
    }
    p = tmp_path / "cphr.pt"
    torch.save(ckpt, p)
    bundle = load_model_bundle(p)
    assert bundle is not None
    assert bundle.model.direct_recovery_absolute_physical_headroom_correction
    assert bundle.model.direct_absolute_physical_headroom_weight is not None
    assert torch.equal(bundle.model.direct_absolute_physical_headroom_weight, torch.zeros(6))
