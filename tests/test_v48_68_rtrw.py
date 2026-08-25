from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from ocrap.cli.train import _absolute_feasibility_bce, _semantic_witness_checkpoint_feature_contract
from ocrap.models.data import (
    DIRECT_ROBUST_TRUST_RECOVERY_WITNESS_FEATURE_SCHEMA,
    OPTION_FEATURE_DIM,
    direct_semantic_recovery_witness_features_from_sample,
    option_features_from_sample,
)
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout():
    return FlatFeatureLayout(feature_max_agents=2)


def _sample(*, agent_accel_x: float = 0.0, privileged: bool = False):
    ego = np.zeros(9, np.float32); ego[6] = 4.0; ego[7] = 4.8; ego[8] = 2.0
    states = np.zeros((10, 9), np.float32)
    states[:, 0] = np.arange(1, 11) * 0.4; states[:, 6] = 4.0; states[:, 7] = 4.8; states[:, 8] = 2.0
    controls = np.zeros((9, 4), np.float32)
    hist = np.zeros((1, 2, 16), np.float32)
    hist[0, 1, 0] = 18.0
    hist[0, 1, 5] = float(agent_accel_x)
    hist[0, 1, 10] = 4.8; hist[0, 1, 11] = 2.0
    d = {
        'ego_state': ego, 'prefix_states': states, 'prefix_controls': controls,
        'agent_history': hist, 'agent_valid': np.asarray([[1, 1]], bool),
        'recovery_modes': np.asarray(['stop', 'lateral_escape'], object),
        'recovery_params': np.asarray([[-5., 5., 0.], [3.5, 5., 1.5]], np.float32),
        'option_valid': np.asarray([1, 1], bool), 'prefix_macro_id': 0, 'prefix_macro_name': 'candidate',
        'prefix_param': np.zeros(0, np.float32), 'utility': 0., 'feasible': 1., 'hard_violation': 0., 'harm_proxy': 0.,
    }
    if privileged:
        d.update({'m_star': np.ones((3, 2), np.float32) * 99., 'root_future_signature': np.ones((3, 8), np.float32) * 77.,
                  'r_dep_star': np.float32(-999.), 'bucket_id': np.int64(2)})
    return d


def _cfg(*, fidelity=False, robust=False):
    return {
        'sample_rate_hz': 10.0, 'recovery_horizon_s': 4.0, 'prefix_horizon_s': 1.0, 'route_dev_max_m': 2.5,
        'control_limits': {'a_max': 3.0, 'a_min': -6.0, 'delta_max': 0.55, 'j_max': 6.0, 'steer_rate_max': 0.5},
        'model': {
            'feature_max_agents': 2,
            'direct_recovery_semantic_witness_route_alignment': True,
            'direct_recovery_semantic_witness_reentry_alignment': True,
            'direct_recovery_semantic_witness_control_projection': True,
            'direct_recovery_semantic_witness_boundary_transport': False,
            'direct_recovery_semantic_witness_projection_fidelity_weighting': fidelity,
            'direct_recovery_semantic_witness_robust_occupancy': robust,
        },
        'default_available_distance_m': 60.0,
    }


def _model(*, fidelity=False, robust=False):
    L = _layout()
    return OCRAPModel(
        input_dim=L.total_dim, num_roots=3, num_options=2, d_model=16, d_obs=8,
        encoder_type='structured_transformer', feature_layout=asdict(L), num_layers=1, num_heads=4, dropout=0.0,
        option_feature_dim=OPTION_FEATURE_DIM, direct_recovery_value_head=True,
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_active_set_alignment=True,
        direct_recovery_semantic_witness_path_stop_alignment=False,
        direct_recovery_semantic_witness_classlocal_transport=False,
        direct_recovery_semantic_witness_route_alignment=True,
        direct_recovery_semantic_witness_reentry_alignment=True,
        direct_recovery_semantic_witness_control_projection=True,
        direct_recovery_semantic_witness_boundary_transport=False,
        direct_recovery_semantic_witness_projection_fidelity_weighting=fidelity,
        direct_recovery_semantic_witness_robust_occupancy=robust,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _opt(batch=1):
    z = torch.from_numpy(option_features_from_sample(_sample())).float()
    return z.unsqueeze(0).repeat(batch, 1, 1)


def _manual_features(batch=1, *, raw_control=-0.7615942):
    f = torch.full((batch, 2, 14), 0.5, dtype=torch.float32)
    f[..., 4] = raw_control
    f[..., 8] = 0.2; f[..., 9] = 0.2; f[..., 11] = 0.0; f[..., 12] = 0.6; f[..., 13] = 0.6
    return f


def _force_support_and_margins(m, margins: torch.Tensor):
    m.root_logit_head.forward = MethodType(lambda self, z: torch.zeros((*z.shape[:-1], 1), device=z.device, dtype=z.dtype), m.root_logit_head)
    m.obs_embed_head.forward = MethodType(lambda self, z: torch.zeros((*z.shape[:-1], 8), device=z.device, dtype=z.dtype), m.obs_embed_head)
    def mf(self, z):
        vals = margins.to(device=z.device, dtype=z.dtype)
        if vals.ndim == 2:
            vals = vals.unsqueeze(0)
        return vals.expand(z.shape[0], -1, -1).unsqueeze(-1)
    m.margin_head.forward = MethodType(mf, m.margin_head)


def test_rtrw_schema4_checkpoint_contract():
    base = {
        'direct_recovery_absolute_semantic_witness_correction': True,
        'direct_recovery_semantic_witness_route_alignment': True,
        'direct_recovery_semantic_witness_reentry_alignment': True,
        'direct_recovery_semantic_witness_control_projection': True,
    }
    assert _semantic_witness_checkpoint_feature_contract(base) == (3, 'projected_boundary_common_executable_recovery_witness')
    assert _semantic_witness_checkpoint_feature_contract({**base,
        'direct_recovery_semantic_witness_projection_fidelity_weighting': True}) == (
        4, 'robust_trust_projected_recovery_witness')
    assert _semantic_witness_checkpoint_feature_contract({**base,
        'direct_recovery_semantic_witness_robust_occupancy': True}) == (
        4, 'robust_trust_projected_recovery_witness')
    assert DIRECT_ROBUST_TRUST_RECOVERY_WITNESS_FEATURE_SCHEMA == 4


def test_rtrw_robust_occupancy_is_observation_only_and_conservative_to_cv():
    d = _sample(agent_accel_x=-5.0)
    cv = direct_semantic_recovery_witness_features_from_sample(d, _cfg(), num_options=2)
    robust = direct_semantic_recovery_witness_features_from_sample(d, _cfg(robust=True), num_options=2)
    privileged = direct_semantic_recovery_witness_features_from_sample(
        _sample(agent_accel_x=-5.0, privileged=True), _cfg(robust=True), num_options=2)
    assert cv.shape == robust.shape == (2, 14)
    # Coordinate 0 is min-clearance reserve.  The robust set contains CV plus
    # the observed-current-acceleration continuation, so it cannot be safer.
    assert np.all(robust[:, 0] <= cv[:, 0] + 1e-7)
    assert np.any(robust[:, 0] < cv[:, 0] - 1e-4)
    assert np.array_equal(robust, privileged), 'robust occupancy must not read teacher/future fields'


def test_rtrw_fidelity_keeps_projected_realization_but_softly_discounts_rescue():
    base = torch.full((3, 2), -0.25, dtype=torch.float32)
    q = _model(fidelity=False).eval(); t = _model(fidelity=True).eval()
    _force_support_and_margins(q, base); _force_support_and_margins(t, base)
    x = torch.zeros((1, _layout().total_dim)); feat = _manual_features(raw_control=-float(np.tanh(1.0)))
    rv = torch.ones((1, 3), dtype=torch.bool); ov = torch.ones((1, 2), dtype=torch.bool)
    out_q = q._direct_semantic_witness_absolute_feasibility(q._scene_tokens(x), x, _opt(), feat, root_valid=rv, option_valid=ov)
    out_t = t._direct_semantic_witness_absolute_feasibility(t._scene_tokens(x), x, _opt(), feat, root_valid=rv, option_valid=ov)
    assert out_q is not None and out_t is not None
    # Projection means raw control severity is not a hard physical veto.
    assert torch.allclose(out_q[4], out_t[4], atol=0, rtol=0)
    assert torch.all(out_t[4] > 0)
    # atanh(-tanh(1))=-1 -> fidelity=1/(1+1)=0.5.
    assert torch.allclose(out_t[5], 0.5 * out_q[5], atol=1e-5, rtol=1e-5)


def test_rtrw_zero_new_flags_are_execution_exact_v4867_q_features():
    d = _sample(agent_accel_x=-5.0)
    q = direct_semantic_recovery_witness_features_from_sample(d, _cfg(), num_options=2)
    q2 = direct_semantic_recovery_witness_features_from_sample(d, _cfg(fidelity=False, robust=False), num_options=2)
    assert np.array_equal(q, q2)


def test_rtrw_bce_gradient_remains_two_shared_gains():
    torch.manual_seed(4868)
    m = _model(fidelity=True, robust=True).train()
    _force_support_and_margins(m, torch.full((3, 2), -0.3))
    for n, p in m.named_parameters():
        p.requires_grad_(n == 'direct_absolute_semantic_witness_gain')
    x = torch.randn((4, _layout().total_dim))
    out = m._direct_semantic_witness_absolute_feasibility(
        m._scene_tokens(x), x, _opt(4), _manual_features(4),
        root_valid=torch.ones((4, 3), dtype=torch.bool), option_valid=torch.ones((4, 2), dtype=torch.bool))
    assert out is not None
    loss = _absolute_feasibility_bce(
        {'direct_recovery_absolute_feasibility_logit': out[0]},
        {'r_dep_star': torch.tensor([-.5, .5, -.5, .5]), 'is_nominal': torch.zeros(4),
         'bucket_id': torch.tensor([1, 1, 2, 2]), 'time_index': torch.arange(4)})
    loss.backward(); g = m.direct_absolute_semantic_witness_gain.grad
    assert g is not None and torch.isfinite(g).all() and torch.any(g != 0)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 2


def test_rtrw_checkpoint_schema4_and_flags_roundtrip(tmp_path: Path):
    from ocrap.models.inference import load_model_bundle
    m = _model(fidelity=True, robust=True).eval(); L = _layout()
    model_cfg = {
        'transformer_layers': 1, 'transformer_heads': 4, 'dropout': 0.0,
        'encoder_type': 'structured_transformer', 'option_feature_dim': OPTION_FEATURE_DIM,
        'direct_recovery_value_head': True, 'direct_recovery_absolute_semantic_witness_correction': True,
        'direct_recovery_semantic_witness_active_set_alignment': True, 'direct_recovery_semantic_witness_path_stop_alignment': False,
        'direct_recovery_semantic_witness_classlocal_transport': False, 'direct_recovery_semantic_witness_route_alignment': True,
        'direct_recovery_semantic_witness_reentry_alignment': True, 'direct_recovery_semantic_witness_control_projection': True,
        'direct_recovery_semantic_witness_boundary_transport': False,
        'direct_recovery_semantic_witness_projection_fidelity_weighting': True,
        'direct_recovery_semantic_witness_robust_occupancy': True,
        'direct_recovery_evidence_native_certificate_preservation': True,
    }
    ckpt = {
        'model_state': m.state_dict(), 'input_dim': L.total_dim, 'num_roots': 3, 'num_options': 2, 'd_model': 16, 'd_obs': 8,
        'tau_obs': 1.0, 'encoder_type': 'structured_transformer', 'feature_layout': asdict(L), 'd_signature': 0,
        'd_future_signature': 0, 'option_feature_dim': OPTION_FEATURE_DIM, **model_cfg,
        'direct_recovery_absolute_semantic_witness_feature_schema': 4,
        'direct_recovery_absolute_semantic_witness_feature_source': 'robust_trust_projected_recovery_witness',
        'cfg': {'sample_rate_hz': 10.0, 'recovery_horizon_s': 4.0, 'model': model_cfg, 'runtime': {'device': 'cpu'}},
    }
    p = tmp_path / 'rtrw.pt'; torch.save(ckpt, p); b = load_model_bundle(p)
    assert b.model.direct_recovery_semantic_witness_projection_fidelity_weighting is True
    assert b.model.direct_recovery_semantic_witness_robust_occupancy is True
    assert b.model.direct_recovery_semantic_witness_control_projection is True
    assert b.model.direct_recovery_semantic_witness_boundary_transport is False
