from __future__ import annotations

from dataclasses import asdict
from types import MethodType

import torch

from ocrap.algorithms.lcv import torch_weighted_lcvar, torch_weighted_lcvar_influence
from ocrap.models.data import OPTION_FEATURE_DIM
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout():
    return FlatFeatureLayout(feature_max_agents=2)


def _model(*, tail: bool):
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
        direct_recovery_semantic_witness_projection_fidelity_weighting=False,
        direct_recovery_semantic_witness_active_constraint_typed_source=False,
        direct_recovery_semantic_witness_root_tail_source=True,
        direct_recovery_semantic_witness_tail_localization=tail,
        direct_recovery_evidence_native_certificate_preservation=True,
        direct_recovery_evidence_roct_alpha=0.5,
        direct_recovery_evidence_roct_beta=0.5,
        direct_recovery_evidence_roct_top_m=3,
    )


def _force_native(m: OCRAPModel, margins: torch.Tensor):
    m.root_logit_head.forward = MethodType(
        lambda self, z: torch.tensor([[[0.0], [-0.3], [-0.8]]], dtype=z.dtype, device=z.device).expand(z.shape[0], -1, -1),
        m.root_logit_head,
    )
    # Frozen root observations remain part of the historical OC-MERO
    # compatibility kernel, but V48.78 learns no observation-class/root vector.
    obs = torch.tensor(
        [[[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [ 0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [ 1.0,-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    m.obs_embed_head.forward = MethodType(
        lambda self, z: obs.to(device=z.device, dtype=z.dtype).expand(z.shape[0], -1, -1),
        m.obs_embed_head,
    )
    def margin_forward(self, z):
        vals = margins.to(device=z.device, dtype=z.dtype)
        if vals.ndim == 2:
            vals = vals.unsqueeze(0)
        return vals.expand(z.shape[0], -1, -1).unsqueeze(-1)
    m.margin_head.forward = MethodType(margin_forward, m.margin_head)


def _options(batch: int = 1):
    z = torch.zeros((batch, 2, OPTION_FEATURE_DIM), dtype=torch.float32)
    z[..., 0] = 1.0
    return z


def _features(batch: int = 1):
    f = torch.full((batch, 2, 14), 0.75, dtype=torch.float32)
    f[..., 11] = 1.0
    f[:, 0, 12] = 0.35
    f[:, 1, 12] = 0.55
    f[:, :, 13] = 0.65
    return f


def _run(m: OCRAPModel):
    x = torch.zeros((1, _layout().total_dim), dtype=torch.float32)
    return m._direct_semantic_witness_absolute_feasibility(
        m._scene_tokens(x), x, _options(), _features(),
        root_valid=torch.ones((1, 3), dtype=torch.bool),
        option_valid=torch.ones((1, 2), dtype=torch.bool),
    )


def test_v4878_lcvar_influence_matches_exact_tail_value():
    scores = torch.tensor([[[-2.0, 0.0, 3.0], [-1.0, 2.0, 4.0]]])
    weights = torch.tensor([[[0.2, 0.5, 0.3], [0.2, 0.5, 0.3]]])
    alpha = 0.6
    inf = torch_weighted_lcvar_influence(scores, weights, alpha)
    val = torch_weighted_lcvar(scores, weights, alpha)
    assert torch.allclose(inf.sum(dim=-1), torch.ones_like(val), atol=1e-7)
    assert torch.allclose((inf * scores).sum(dim=-1), val, atol=1e-7)


def test_v4878_zero_init_is_execution_exact_native_semantic_source():
    margins = torch.tensor([[-0.5, -0.3], [-0.1, 0.0], [0.4, 0.2]])
    root = _model(tail=False).eval(); _force_native(root, margins)
    base = _model(tail=False).eval(); _force_native(base, margins)
    # Disable only the new branch while keeping the exact same historical source.
    base.direct_recovery_semantic_witness_root_tail_source = False
    a = _run(root); b = _run(base)
    assert a is not None and b is not None
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_v4878_root_shape_changes_source_without_option_translation():
    margins = torch.tensor([[-0.5, -0.3], [-0.1, 0.0], [0.4, 0.2]])
    m = _model(tail=False).eval(); _force_native(m, margins)
    with torch.no_grad():
        m.direct_absolute_root_tail_source_scale.zero_()
        m.direct_absolute_root_tail_source_scale[0] = 1.0
    out = _run(m)
    z = _model(tail=False).eval(); _force_native(z, margins)
    base = _run(z)
    assert out is not None and base is not None
    assert not torch.equal(out[0], base[0])
    assert m.direct_absolute_root_tail_source_scale.numel() == 1


def test_v4878_tail_localization_is_nested_and_zero_init_exact():
    margins = torch.tensor([[-0.7, 0.2], [-0.2, -0.1], [0.3, 0.4]])
    a = _model(tail=False).eval(); b = _model(tail=True).eval()
    _force_native(a, margins); _force_native(b, margins)
    # zero-init must make tail localization inert
    oa, ob = _run(a), _run(b)
    assert oa is not None and ob is not None and torch.equal(oa[0], ob[0])
    with torch.no_grad():
        a.direct_absolute_root_tail_source_scale[0] = 1.0
        b.direct_absolute_root_tail_source_scale[0] = 1.0
    oa, ob = _run(a), _run(b)
    assert oa is not None and ob is not None
    # Nested tail attribution is a distinct registered intervention.
    assert not torch.equal(oa[0], ob[0])


def test_v4878_checkpoint_roundtrip(tmp_path):
    from ocrap.models.inference import load_model_bundle
    m = _model(tail=True).eval(); L = _layout()
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
        "direct_recovery_semantic_witness_projection_fidelity_weighting": False,
        "direct_recovery_semantic_witness_active_constraint_typed_source": False,
        "direct_recovery_semantic_witness_root_tail_source": True,
        "direct_recovery_semantic_witness_tail_localization": True,
        "direct_recovery_evidence_native_certificate_preservation": True,
        "direct_recovery_evidence_roct_alpha": 0.5,
        "direct_recovery_evidence_roct_beta": 0.5,
        "direct_recovery_evidence_roct_top_m": 3,
    }
    ckpt = {
        "model_state": m.state_dict(), "input_dim": L.total_dim, "num_roots": 3, "num_options": 2,
        "d_model": 16, "d_obs": 8, "tau_obs": 1.0, "encoder_type": "structured_transformer",
        "feature_layout": asdict(L), "d_signature": 0, "d_future_signature": 0,
        "option_feature_dim": OPTION_FEATURE_DIM, **model_cfg,
        "direct_recovery_absolute_semantic_witness_feature_schema": 3,
        "direct_recovery_absolute_semantic_witness_feature_source": "projected_boundary_common_executable_recovery_witness",
        "cfg": {"sample_rate_hz": 10.0, "recovery_horizon_s": 4.0, "model": model_cfg, "runtime": {"device": "cpu"}},
    }
    p = tmp_path / "rtsi.pt"; torch.save(ckpt, p)
    bundle = load_model_bundle(p)
    assert bundle.model.direct_recovery_semantic_witness_root_tail_source is True
    assert bundle.model.direct_recovery_semantic_witness_tail_localization is True
    assert bundle.model.direct_absolute_root_tail_source_scale.numel() == 1


def test_v4878_optionwise_translation_cannot_change_lcvar_tail_shape():
    """Formalize the V48.77 STOP branch's algebraic limitation.

    Every V48.64--77 candidate-global gain adds one scalar per option to all
    roots.  Weighted LCVAR is translation equivariant, so such a transport can
    move an option's absolute level but cannot alter its within-option root-tail
    geometry.  V48.78 must therefore intervene below that translation family.
    """
    scores = torch.tensor([[[-0.7, -0.1, 0.4], [-0.4, 0.2, 0.9]]], dtype=torch.float32)
    weights = torch.tensor([[[0.2, 0.5, 0.3], [0.2, 0.5, 0.3]]], dtype=torch.float32)
    shift = torch.tensor([[[0.37], [-0.21]]], dtype=torch.float32)
    alpha = 0.4
    base = torch_weighted_lcvar(scores, weights, alpha)
    moved = torch_weighted_lcvar(scores + shift, weights, alpha)
    assert torch.allclose(moved, base + shift.squeeze(-1), atol=1e-7)


def test_v4878_root_tail_residual_has_exact_zero_p_translation():
    margins = torch.tensor([[-0.8, -0.4], [-0.2, 0.1], [0.5, 0.6]])
    for tail in (False, True):
        m = _model(tail=tail).eval(); _force_native(m, margins)
        with torch.no_grad():
            m.direct_absolute_root_tail_source_scale[0] = 1.0
        captured = {}
        def fake_signature(self, root_logits, obs_embeddings, corrected_margins, *args, **kwargs):
            captured['root_logits'] = root_logits.detach().clone()
            captured['corrected'] = corrected_margins.detach().clone()
            native = torch.zeros((corrected_margins.shape[0], 3), dtype=corrected_margins.dtype, device=corrected_margins.device)
            native[:, 1] = 0.5
            return None, native
        m._recovery_option_compatibility_signature = MethodType(fake_signature, m)
        out = _run(m)
        assert out is not None
        p = torch.softmax(captured['root_logits'], dim=-1)
        delta = captured['corrected'] - margins.unsqueeze(0)
        weighted_translation = (p.unsqueeze(-1) * delta).sum(dim=1)
        assert torch.allclose(weighted_translation, torch.zeros_like(weighted_translation), atol=2e-7)
        assert delta.abs().max().item() > 0.0
