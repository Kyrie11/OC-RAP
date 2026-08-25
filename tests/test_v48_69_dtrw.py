from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
import runpy
from types import MethodType

import numpy as np
import pytest
import torch

from ocrap.cli.train import _semantic_witness_checkpoint_feature_contract
from ocrap.models.data import (
    DIRECT_DEMAND_TEMPERED_RECOVERY_WITNESS_FEATURE_SCHEMA,
    OPTION_FEATURE_DIM,
    direct_semantic_recovery_witness_features_from_sample,
    option_features_from_sample,
)
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout():
    return FlatFeatureLayout(feature_max_agents=2)


def _sample():
    ego = np.zeros(9, np.float32); ego[6] = 4.0; ego[7] = 4.8; ego[8] = 2.0
    states = np.zeros((10, 9), np.float32)
    states[:, 0] = np.arange(1, 11) * 0.4; states[:, 6] = 4.0; states[:, 7] = 4.8; states[:, 8] = 2.0
    controls = np.zeros((9, 4), np.float32)
    hist = np.zeros((1, 2, 16), np.float32)
    hist[0, 1, 0] = 18.0; hist[0, 1, 10] = 4.8; hist[0, 1, 11] = 2.0
    return {
        'ego_state': ego, 'prefix_states': states, 'prefix_controls': controls,
        'agent_history': hist, 'agent_valid': np.asarray([[1, 1]], bool),
        'recovery_modes': np.asarray(['stop', 'lateral_escape'], object),
        'recovery_params': np.asarray([[-5., 5., 0.], [3.5, 5., 1.5]], np.float32),
        'option_valid': np.asarray([1, 1], bool), 'prefix_macro_id': 0,
        'prefix_macro_name': 'candidate', 'prefix_param': np.zeros(0, np.float32),
        'utility': 0., 'feasible': 1., 'hard_violation': 0., 'harm_proxy': 0.,
    }


def _cfg(*, demand=False):
    return {
        'sample_rate_hz': 10.0, 'recovery_horizon_s': 4.0, 'prefix_horizon_s': 1.0,
        'route_dev_max_m': 2.5,
        'control_limits': {'a_max': 3.0, 'a_min': -6.0, 'delta_max': 0.55, 'j_max': 6.0, 'steer_rate_max': 0.5},
        'model': {
            'feature_max_agents': 2,
            'direct_recovery_semantic_witness_route_alignment': True,
            'direct_recovery_semantic_witness_reentry_alignment': True,
            'direct_recovery_semantic_witness_control_projection': True,
            'direct_recovery_semantic_witness_boundary_transport': False,
            'direct_recovery_semantic_witness_projection_fidelity_weighting': True,
            'direct_recovery_semantic_witness_demand_normalized_fidelity': demand,
            'direct_recovery_semantic_witness_robust_occupancy': False,
        },
        'default_available_distance_m': 60.0,
    }


def _model(*, demand=False):
    L = _layout()
    return OCRAPModel(
        input_dim=L.total_dim, num_roots=3, num_options=2, d_model=16, d_obs=8,
        encoder_type='structured_transformer', feature_layout=asdict(L), num_layers=1,
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
        direct_recovery_semantic_witness_demand_normalized_fidelity=demand,
        direct_recovery_semantic_witness_robust_occupancy=False,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _opt(batch=1):
    z = torch.from_numpy(option_features_from_sample(_sample())).float()
    return z.unsqueeze(0).repeat(batch, 1, 1)


def _force_support_and_margins(m):
    m.root_logit_head.forward = MethodType(
        lambda self, z: torch.zeros((*z.shape[:-1], 1), device=z.device, dtype=z.dtype),
        m.root_logit_head,
    )
    m.obs_embed_head.forward = MethodType(
        lambda self, z: torch.zeros((*z.shape[:-1], 8), device=z.device, dtype=z.dtype),
        m.obs_embed_head,
    )
    def mf(self, z):
        vals = torch.zeros((1, 3, 2), device=z.device, dtype=z.dtype)
        return vals.expand(z.shape[0], -1, -1).unsqueeze(-1)
    m.margin_head.forward = MethodType(mf, m.margin_head)


def _features():
    f = torch.full((1, 2, 14), 0.6, dtype=torch.float32)
    # Raw desired-command violation: atanh(h_control)=-1 for both options.
    f[..., 4] = -float(np.tanh(1.0))
    # Keep path-preservation and active-set tails feasible/inactive.
    f[..., 8] = 0.3; f[..., 9] = 0.3; f[..., 11] = 0.0
    f[..., 12] = 0.6; f[..., 13] = 0.6
    # Option 0: zero observed clearance demand: atanh(gain)==atanh(terminal).
    f[0, 0, 1] = float(np.tanh(1.0)); f[0, 0, 2] = float(np.tanh(1.0))
    # Option 1: one normalized unit of observed clearance demand.
    f[0, 1, 1] = float(np.tanh(0.2)); f[0, 1, 2] = float(np.tanh(1.2))
    return f


def test_v4869_schema5_checkpoint_contract():
    base = {
        'direct_recovery_absolute_semantic_witness_correction': True,
        'direct_recovery_semantic_witness_route_alignment': True,
        'direct_recovery_semantic_witness_reentry_alignment': True,
        'direct_recovery_semantic_witness_control_projection': True,
        'direct_recovery_semantic_witness_projection_fidelity_weighting': True,
    }
    assert _semantic_witness_checkpoint_feature_contract(base) == (4, 'robust_trust_projected_recovery_witness')
    assert _semantic_witness_checkpoint_feature_contract({
        **base, 'direct_recovery_semantic_witness_demand_normalized_fidelity': True,
    }) == (5, 'demand_tempered_projected_recovery_witness')
    assert DIRECT_DEMAND_TEMPERED_RECOVERY_WITNESS_FEATURE_SCHEMA == 5


def test_v4869_feature_side_channel_is_byte_identical_to_v4868_t():
    a = direct_semantic_recovery_witness_features_from_sample(_sample(), _cfg(demand=False), num_options=2)
    b = direct_semantic_recovery_witness_features_from_sample(_sample(), _cfg(demand=True), num_options=2)
    assert a.shape == b.shape == (2, 14)
    assert np.array_equal(a, b)


def test_v4869_zero_demand_is_exact_v4868_fidelity_and_demand_tempers_only_urgent_option():
    t = _model(demand=False).eval(); d = _model(demand=True).eval()
    _force_support_and_margins(t); _force_support_and_margins(d)
    x = torch.zeros((1, _layout().total_dim)); feat = _features()
    rv = torch.ones((1, 3), dtype=torch.bool); ov = torch.ones((1, 2), dtype=torch.bool)
    ot = t._direct_semantic_witness_absolute_feasibility(t._scene_tokens(x), x, _opt(), feat, root_valid=rv, option_valid=ov)
    od = d._direct_semantic_witness_absolute_feasibility(d._scene_tokens(x), x, _opt(), feat, root_valid=rv, option_valid=ov)
    assert ot is not None and od is not None
    # Physical witness sign/set is untouched.
    assert torch.equal(ot[4] > 0, od[4] > 0)
    # Equal margins/roots make frozen common support 1 before fidelity.  T gives
    # 1/(1+1)=0.5.  v48.69 option 0 has zero demand -> exact T.  Option 1 has
    # demand=1 -> (1+1)/(1+1+1)=2/3.
    assert torch.allclose(ot[5], torch.full_like(ot[5], 0.5), atol=2e-5, rtol=2e-5)
    assert torch.allclose(od[5][..., 0], ot[5][..., 0], atol=2e-5, rtol=2e-5)
    assert torch.allclose(od[5][..., 1], torch.full_like(od[5][..., 1], 2.0/3.0), atol=2e-5, rtol=2e-5)
    assert torch.all(od[5] >= ot[5])


def test_v4869_demand_flag_requires_projection_and_fidelity():
    L = _layout()
    with pytest.raises(ValueError, match='demand-normalized projection fidelity'):
        OCRAPModel(
            input_dim=L.total_dim, num_roots=3, num_options=2, d_model=16, d_obs=8,
            encoder_type='structured_transformer', feature_layout=asdict(L), num_layers=1,
            num_heads=4, dropout=0.0, option_feature_dim=OPTION_FEATURE_DIM,
            direct_recovery_value_head=True,
            direct_recovery_absolute_semantic_witness_correction=True,
            direct_recovery_semantic_witness_control_projection=True,
            direct_recovery_semantic_witness_projection_fidelity_weighting=False,
            direct_recovery_semantic_witness_demand_normalized_fidelity=True,
        )


def test_v4869_checkpoint_schema5_and_flag_roundtrip(tmp_path: Path):
    from ocrap.models.inference import load_model_bundle
    m = _model(demand=True).eval(); L = _layout()
    model_cfg = {
        'transformer_layers': 1, 'transformer_heads': 4, 'dropout': 0.0,
        'encoder_type': 'structured_transformer', 'option_feature_dim': OPTION_FEATURE_DIM,
        'direct_recovery_value_head': True,
        'direct_recovery_absolute_semantic_witness_correction': True,
        'direct_recovery_semantic_witness_active_set_alignment': True,
        'direct_recovery_semantic_witness_path_stop_alignment': False,
        'direct_recovery_semantic_witness_classlocal_transport': False,
        'direct_recovery_semantic_witness_route_alignment': True,
        'direct_recovery_semantic_witness_reentry_alignment': True,
        'direct_recovery_semantic_witness_control_projection': True,
        'direct_recovery_semantic_witness_boundary_transport': False,
        'direct_recovery_semantic_witness_projection_fidelity_weighting': True,
        'direct_recovery_semantic_witness_demand_normalized_fidelity': True,
        'direct_recovery_semantic_witness_robust_occupancy': False,
        'direct_recovery_evidence_native_certificate_preservation': True,
    }
    ckpt = {
        'model_state': m.state_dict(), 'input_dim': L.total_dim, 'num_roots': 3,
        'num_options': 2, 'd_model': 16, 'd_obs': 8, 'tau_obs': 1.0,
        'encoder_type': 'structured_transformer', 'feature_layout': asdict(L),
        'd_signature': 0, 'd_future_signature': 0, 'option_feature_dim': OPTION_FEATURE_DIM,
        **model_cfg,
        'direct_recovery_absolute_semantic_witness_feature_schema': 5,
        'direct_recovery_absolute_semantic_witness_feature_source': 'demand_tempered_projected_recovery_witness',
        'cfg': {'sample_rate_hz': 10.0, 'recovery_horizon_s': 4.0,
                'model': model_cfg, 'runtime': {'device': 'cpu'}},
    }
    p = tmp_path / 'dtrw.pt'; torch.save(ckpt, p)
    b = load_model_bundle(p)
    assert b.model.direct_recovery_semantic_witness_projection_fidelity_weighting is True
    assert b.model.direct_recovery_semantic_witness_demand_normalized_fidelity is True
    assert b.model.direct_recovery_semantic_witness_robust_occupancy is False


def test_v4869_runner_freezes_rejected_v4868_branches():
    text = (Path(__file__).resolve().parents[1] / 'scripts' / 'run_v48_69_dcp_drfc_bcde_rifa_dtrw_two_gpu.sh').read_text()
    assert 'SEMANTIC_WITNESS_PROJECTION_FIDELITY=true' in text
    assert 'SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY=true' in text
    assert 'SEMANTIC_WITNESS_ROBUST_OCCUPANCY=false' in text
    assert 'SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false' in text
    assert 'PROPOSAL_TOP_K=5' in text
    assert 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' in text
    assert 'v48.69-DCP-DRFC-BCDE-RIFA-OC-DTRW' in text


def test_v4869_pipeline_checker_requires_v4868_validated_branch():
    text = (Path(__file__).resolve().parents[1] / 'tools' / 'check_v48_69_pipeline_complete.py').read_text()
    assert "engineering_version')=='v48.68.0-OC-RTRW'" in text
    assert "projection_fidelity_mechanism_gate') is True" in text
    assert "robust_occupancy_mechanism_gate') is False" in text
    assert "EXPECTED_SCHEMA=5" in text
    assert "EXPECTED_SOURCE='demand_tempered_projected_recovery_witness'" in text
    assert "'engineering_version':'v48.69.1-OC-DTRW-ENGFIX'" in text
    assert "demand trust audit invalid or row alignment incomplete" in text



def test_v4869_runner_embedded_python_is_syntax_valid():
    """Regression for the v48.69 factor-contract heredoc SyntaxError."""
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_v48_69_dcp_drfc_bcde_rifa_dtrw_two_gpu.sh'
    text = script.read_text()
    blocks = re.findall(r"<<'PY2'\n(.*?)\nPY2", text, flags=re.S)
    assert len(blocks) >= 2
    for i, source in enumerate(blocks):
        compile(source, f'{script.name}:PY2[{i}]', 'exec')


def test_v4869_truth_debt_keeps_exact_zero_teacher_boundary():
    mod = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools' / 'audit_v48_69_truth_debt.py'))
    row = {
        'teacher_candidate_r_dep': 0.0,
        'teacher_adv': 0.0,
        'teacher_harmful': False,
        'semantic_best_common_viability': -0.1,
        'absolute_feasibility_pass': False,
    }
    out = mod['summ']([row])
    assert out['teacher_feasible']['rows'] == 1


def test_v4869_demand_audit_fails_closed_on_row_set_mismatch():
    mod = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools' / 'audit_v48_69_demand_trust.py'))
    key = ('scene-1', 1.0, 'fold-0', 'cand-0')
    row = {
        'teacher_candidate_r_dep': 0.1,
        'teacher_adv': 0.02,
        'teacher_harmful': False,
        'semantic_best_common_viability': 0.2,
        'semantic_max_common_support': 0.8,
        'absolute_feasibility_pass': False,
    }
    out = mod['summarize']({key: row}, {})
    assert out['row_set_equal'] is False
    assert out['positive_certificate_set_equal'] is False
    assert out['missing_in_D_count'] == 1

def test_v4869_changelog_records_v4868_branch_decision():
    text = (Path(__file__).resolve().parents[1] / 'ALGORITHM_CHANGELOG.md').read_text()
    assert text.startswith('## v48.69 — OC-DTRW')
    assert 'T_FIDELITY is a clean mechanism GO' in text
    assert 'U_OCCUPANCY is a mechanism STOP' in text
    assert 'do not use a hard minimum over CV and current-acceleration occupancy' in text
