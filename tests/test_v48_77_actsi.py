from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import MethodType

import torch

from ocrap.models.data import OPTION_FEATURE_DIM
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout():
    return FlatFeatureLayout(feature_max_agents=2)


def _model(*, typed: bool, fidelity: bool = False, boundary: bool = False):
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
        direct_recovery_semantic_witness_boundary_transport=boundary,
        direct_recovery_semantic_witness_projection_fidelity_weighting=fidelity,
        direct_recovery_semantic_witness_active_constraint_typed_source=typed,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _force_native(m: OCRAPModel, margins: torch.Tensor):
    m.root_logit_head.forward = MethodType(
        lambda self, z: torch.zeros((*z.shape[:-1], 1), device=z.device, dtype=z.dtype),
        m.root_logit_head,
    )
    m.obs_embed_head.forward = MethodType(
        lambda self, z: torch.zeros((*z.shape[:-1], 8), device=z.device, dtype=z.dtype),
        m.obs_embed_head,
    )

    def margin_forward(self, z):
        vals = margins.to(device=z.device, dtype=z.dtype)
        if vals.ndim == 2:
            vals = vals.unsqueeze(0)
        return vals.expand(z.shape[0], -1, -1).unsqueeze(-1)

    m.margin_head.forward = MethodType(margin_forward, m.margin_head)


def _options(batch: int = 1):
    # First semantic flag is enough to make stopping active.  Remaining option
    # identity does not enter OC-ACTSI's learned table.
    z = torch.zeros((batch, 2, OPTION_FEATURE_DIM), dtype=torch.float32)
    z[..., 0] = 1.0
    return z


def _features(*, route_first: bool = True, batch: int = 1):
    # 14-D historical projected-recovery semantic side channel.  Keep all
    # barriers safely positive and make route/re-entry the only distinct active
    # constraints.  Control is ignored because projection is ON.
    f = torch.full((batch, 2, 14), 0.8, dtype=torch.float32)
    # stability-active observation flag
    f[..., 11] = 1.0
    # route (12), persistent re-entry (13)
    if route_first:
        f[:, 0, 12] = 0.2
        f[:, 0, 13] = 0.8
        f[:, 1, 12] = 0.8
        f[:, 1, 13] = 0.2
    else:
        f[:, 0, 12] = 0.8
        f[:, 0, 13] = 0.2
        f[:, 1, 12] = 0.2
        f[:, 1, 13] = 0.8
    return f


def _run(m: OCRAPModel, features: torch.Tensor):
    x = torch.zeros((features.shape[0], _layout().total_dim), dtype=torch.float32)
    return m._direct_semantic_witness_absolute_feasibility(
        m._scene_tokens(x),
        x,
        _options(features.shape[0]),
        features,
        root_valid=torch.ones((features.shape[0], 3), dtype=torch.bool),
        option_valid=torch.ones((features.shape[0], 2), dtype=torch.bool),
    )


def test_v4877_typed_table_is_exactly_six_by_two_and_zero_init():
    m = _model(typed=True)
    g = m.direct_absolute_semantic_witness_gain
    assert g is not None and tuple(g.shape) == (6, 2)
    assert g.numel() == 12 and torch.count_nonzero(g).item() == 0
    assert m.direct_recovery_semantic_witness_active_constraint_typed_source is True


def test_v4877_zero_typed_source_is_execution_exact_global_zero_source():
    margins = torch.full((3, 2), -0.25)
    old = _model(typed=False).eval()
    new = _model(typed=True).eval()
    _force_native(old, margins)
    _force_native(new, margins)
    a = _run(old, _features())
    b = _run(new, _features())
    assert a is not None and b is not None
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
    assert torch.equal(a[4], b[4])
    assert torch.equal(a[5], b[5])


def test_v4877_active_constraint_type_not_option_id_controls_rescue():
    margins = torch.full((3, 2), -0.25)
    m = _model(typed=True).eval()
    _force_native(m, margins)
    with torch.no_grad():
        m.direct_absolute_semantic_witness_gain.zero_()
        # barrier order: clearance, stopping, control, stability, route, re-entry
        m.direct_absolute_semantic_witness_gain[4, 0] = 1.0
        m.direct_absolute_semantic_witness_gain[5, 0] = 0.0
    a = _run(m, _features(route_first=True))
    b = _run(m, _features(route_first=False))
    assert a is not None and b is not None
    # Swapping which option is route-limited must not change the result: there
    # is no learned option-ID bias, only active-constraint type.
    assert torch.allclose(a[0], b[0], atol=1e-6, rtol=1e-6)
    # The route gain must produce a real source intervention relative to zero.
    z = _model(typed=True).eval(); _force_native(z, margins)
    base = _run(z, _features(route_first=True))
    assert base is not None and torch.all(a[0] > base[0])


def test_v4877_typed_source_rejects_boundary_transport_reopening():
    m = _model(typed=True, boundary=True).eval()
    _force_native(m, torch.full((3, 2), -0.25))
    try:
        _run(m, _features())
    except RuntimeError as e:
        assert "does not reopen boundary transport" in str(e)
        return
    raise AssertionError("typed source silently combined with frozen boundary transport")


def test_v4877_typed_checkpoint_roundtrip(tmp_path: Path):
    from ocrap.models.inference import load_model_bundle

    m = _model(typed=True, fidelity=True).eval(); L = _layout()
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
        "direct_recovery_semantic_witness_active_constraint_typed_source": True,
        "direct_recovery_evidence_native_certificate_preservation": True,
    }
    ckpt = {
        "model_state": m.state_dict(), "input_dim": L.total_dim, "num_roots": 3, "num_options": 2,
        "d_model": 16, "d_obs": 8, "tau_obs": 1.0, "encoder_type": "structured_transformer",
        "feature_layout": asdict(L), "d_signature": 0, "d_future_signature": 0,
        "option_feature_dim": OPTION_FEATURE_DIM, **model_cfg,
        "direct_recovery_absolute_semantic_witness_feature_schema": 4,
        "direct_recovery_absolute_semantic_witness_feature_source": "robust_trust_projected_recovery_witness",
        "cfg": {"sample_rate_hz": 10.0, "recovery_horizon_s": 4.0, "model": model_cfg, "runtime": {"device": "cpu"}},
    }
    p = tmp_path / "actsi.pt"; torch.save(ckpt, p)
    bundle = load_model_bundle(p)
    assert bundle.model.direct_recovery_semantic_witness_active_constraint_typed_source is True
    assert tuple(bundle.model.direct_absolute_semantic_witness_gain.shape) == (6, 2)
