from __future__ import annotations
from dataclasses import asdict
from types import MethodType
from pathlib import Path
import torch

from ocrap.models.data import OPTION_FEATURE_DIM
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _layout():
    return FlatFeatureLayout(feature_max_agents=2)


def _model(*, classlocal: bool):
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
        direct_recovery_semantic_witness_classlocal_transport=classlocal,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _force_distinguishable_root_preferences(m: OCRAPModel) -> None:
    # Equal root mass; every root is observationally distinguishable.  Root 0
    # prefers option 0, roots 1/2 prefer option 1.  This is the case where the
    # paper permits different post-prefix observation classes to choose different
    # recovery options, so a candidate-global option correction is over-coupled.
    m.root_logit_head.forward = MethodType(
        lambda self, z: torch.zeros((*z.shape[:-1], 1), device=z.device, dtype=z.dtype),
        m.root_logit_head,
    )

    def obs(self, z):
        B, K, _ = z.shape
        base = torch.zeros((B, K, 8), device=z.device, dtype=z.dtype)
        base[:, 0, 0] = -20.0
        base[:, 1, 0] = 0.0
        base[:, 2, 0] = 20.0
        return base

    m.obs_embed_head.forward = MethodType(obs, m.obs_embed_head)

    def margins(self, z):
        vals = torch.tensor(
            [[[0.20, -1.00], [-1.00, 0.20], [-1.00, 0.20]]],
            device=z.device,
            dtype=z.dtype,
        ).expand(z.shape[0], -1, -1)
        return vals.unsqueeze(-1)

    m.margin_head.forward = MethodType(margins, m.margin_head)


def _option_features(batch=1):
    # One stop and one lateral-escape option; only mode one-hots are required by
    # the semantics witness.  The remaining option feature coordinates stay zero.
    x = torch.zeros((batch, 2, OPTION_FEATURE_DIM), dtype=torch.float32)
    x[:, 0, 0] = 1.0
    x[:, 1, 2] = 1.0
    return x


def _positive_semantic_features(batch=1):
    # [hmin,hterm,hgain,hstopLegacy,hctrl,hstabMin,hstabTerm,hstabGain,
    #  hclearFloor,hstabFloor,pathStop,stabActive].  Both options are physically
    # viable; stability is observably inactive, and v48.65 keeps legacy stopping.
    f = torch.full((batch, 2, 12), 0.5, dtype=torch.float32)
    f[..., 8] = 0.2
    f[..., 9] = 0.2
    f[..., 11] = 0.0
    return f


def test_clrw_zero_gain_is_exact_v4864_native_boundary():
    torch.manual_seed(4865)
    m = _model(classlocal=True).eval()
    _force_distinguishable_root_preferences(m)
    L = _layout()
    x = torch.randn((2, L.total_dim))
    mem = m._scene_tokens(x)
    rv = torch.ones((2, 3), dtype=torch.bool)
    ov = torch.ones((2, 2), dtype=torch.bool)
    _sig, native = m._direct_recovery_option_compatibility_evidence(
        mem, x, _option_features(2), root_valid=rv, option_valid=ov
    )
    out = m._direct_semantic_witness_absolute_feasibility(
        mem, x, _option_features(2), _positive_semantic_features(2),
        root_valid=rv, option_valid=ov,
    )
    assert out is not None
    assert torch.equal(out[3], torch.zeros(2))
    assert torch.allclose(out[1], native[:, 1], atol=0.0, rtol=0.0)
    assert out[12] is not None and out[13] is not None and out[14] is not None


def test_clrw_respects_distinguishable_observation_class_option_choice():
    torch.manual_seed(4865)
    L = _layout()
    x = torch.zeros((1, L.total_dim))
    rv = torch.ones((1, 3), dtype=torch.bool)
    ov = torch.ones((1, 2), dtype=torch.bool)
    feat = _positive_semantic_features(1)
    opt = _option_features(1)

    global_m = _model(classlocal=False).eval()
    local_m = _model(classlocal=True).eval()
    _force_distinguishable_root_preferences(global_m)
    _force_distinguishable_root_preferences(local_m)
    with torch.no_grad():
        global_m.direct_absolute_semantic_witness_gain[:] = torch.tensor([1.0, 0.0])
        local_m.direct_absolute_semantic_witness_gain[:] = torch.tensor([1.0, 0.0])
    gout = global_m._direct_semantic_witness_absolute_feasibility(
        global_m._scene_tokens(x), x, opt, feat, root_valid=rv, option_valid=ov
    )
    lout = local_m._direct_semantic_witness_absolute_feasibility(
        local_m._scene_tokens(x), x, opt, feat, root_valid=rv, option_valid=ov
    )
    assert gout is not None and lout is not None
    # Each distinguishable class can put unit local support on its own preferred
    # option.  The candidate-global v48.64 support cannot do that simultaneously.
    assert lout[14] is not None and float(lout[14].item()) > float(gout[9].item())
    assert float(lout[1].item()) > float(gout[1].item())


def test_clrw_gradients_remain_only_two_shared_gains():
    torch.manual_seed(4865)
    m = _model(classlocal=True).train()
    _force_distinguishable_root_preferences(m)
    for n, p in m.named_parameters():
        p.requires_grad_(n == "direct_absolute_semantic_witness_gain")
    L = _layout()
    x = torch.randn((4, L.total_dim))
    out = m._direct_semantic_witness_absolute_feasibility(
        m._scene_tokens(x), x, _option_features(4), _positive_semantic_features(4),
        root_valid=torch.ones((4, 3), dtype=torch.bool),
        option_valid=torch.ones((4, 2), dtype=torch.bool),
    )
    assert out is not None
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out[0], labels)
    loss.backward()
    g = m.direct_absolute_semantic_witness_gain.grad
    assert g is not None and torch.isfinite(g).all() and torch.any(g != 0)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 2


def test_clrw_checkpoint_roundtrip_preserves_classlocal_flag(tmp_path):
    from ocrap.models.inference import load_model_bundle
    m = _model(classlocal=True).eval()
    L = _layout()
    ckpt = {
        'model_state': m.state_dict(), 'input_dim': L.total_dim, 'num_roots': 3,
        'num_options': 2, 'd_model': 16, 'd_obs': 8, 'tau_obs': 1.0,
        'encoder_type': 'structured_transformer', 'feature_layout': asdict(L),
        'd_signature': 0, 'd_future_signature': 0, 'option_feature_dim': OPTION_FEATURE_DIM,
        'direct_recovery_value_head': True,
        'direct_recovery_absolute_semantic_witness_correction': True,
        'direct_recovery_semantic_witness_active_set_alignment': True,
        'direct_recovery_semantic_witness_path_stop_alignment': False,
        'direct_recovery_semantic_witness_classlocal_transport': True,
        'direct_recovery_absolute_semantic_witness_feature_schema': 1,
        'direct_recovery_absolute_semantic_witness_feature_source': 'semantics_aligned_common_executable_recovery_witness',
        'direct_recovery_evidence_native_certificate_preservation': True,
        'cfg': {'sample_rate_hz': 10.0, 'recovery_horizon_s': 4.0,
                'model': {'transformer_layers': 1, 'transformer_heads': 4, 'dropout': 0.0,
                          'encoder_type': 'structured_transformer', 'option_feature_dim': OPTION_FEATURE_DIM,
                          'direct_recovery_value_head': True,
                          'direct_recovery_absolute_semantic_witness_correction': True,
                          'direct_recovery_semantic_witness_active_set_alignment': True,
                          'direct_recovery_semantic_witness_path_stop_alignment': False,
                          'direct_recovery_semantic_witness_classlocal_transport': True,
                          'direct_recovery_evidence_native_certificate_preservation': True},
                'runtime': {'device': 'cpu'}},
    }
    p = tmp_path / 'clrw.pt'; torch.save(ckpt, p)
    b = load_model_bundle(p)
    assert b.model.direct_recovery_semantic_witness_classlocal_transport is True
    assert b.model.direct_recovery_semantic_witness_path_stop_alignment is False


def test_clrw_teacher_truth_audit_detects_classlocal_sign_need(tmp_path):
    import json, subprocess, sys
    import numpy as np
    from ocrap.algorithms.ocmero import oc_mero
    root = tmp_path / 'near_contact'; samples = root / 'samples'; samples.mkdir(parents=True)
    m = np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    p = np.asarray([0.5, 0.5], dtype=np.float32)
    c = np.eye(2, dtype=np.float32)
    r = oc_mero(m, p, c, alpha=.2, beta=.2, option_valid=np.ones(2,bool), root_valid=np.ones(2,bool), top_m=8)
    np.savez_compressed(samples/'a.npz', m_star=m, root_probs=p, c_star=c,
                        root_valid=np.ones(2,bool), option_valid=np.ones(2,bool), r_dep_star=np.float32(r.r_dep))
    out = tmp_path / 'audit.json'
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(repo/'tools/audit_v48_65_teacher_certificate_semantics.py'),
                    '--root', str(root), '--output', str(out)], cwd=repo, check=True)
    d = json.loads(out.read_text())
    assert d['valid'] is True and d['test_roots_read'] is False and d['dataset_reconstruction'] is False
    assert d['overall']['teacher_feasible_classlocal_required_for_sign_fraction'] == 1.0
    assert d['overall']['max_r_dep_recompute_abs_error'] <= 1e-6


def test_clrw_shell_and_factor_contracts_are_fail_closed():
    root = Path(__file__).resolve().parents[1]
    train = (root/'scripts/train_ocrap_v48_trac_sr.sh').read_text()
    adapt = (root/'scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh').read_text()
    launch = (root/'scripts/run_v48_65_dcp_drfc_bcde_rifa_clrw_two_gpu.sh').read_text()
    assert 'direct_recovery_semantic_witness_classlocal_transport' in train
    assert 'SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT' in adapt
    for z in ('SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=true',
              'SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false',
              'EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_semantic_witness_gain',
              'MAX_EVIDENCE_CALIBRATOR_PARAMS=2', 'PROPOSAL_TOP_K=5',
              'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5', 'L_CLASSLOCAL', 'M_Main_OCCLRW',
              'audit_v48_65_teacher_certificate_semantics.py'):
        assert z in launch
    assert 'J_PATHSTOP' in launch  # documented as intentionally not carried into Main
    vi = (root/'tools/check_v48_65_variant_isolation.py').read_text()
    pc = (root/'tools/check_v48_65_pipeline_complete.py').read_text()
    for checker in (vi, pc):
        assert "'model_v48_trac_sr'/'train_summary.json'" in checker
        assert 'TRAINING_COMPLETE.json' in checker and 'EVIDENCE_CORRECTION_COMPLETE.json' in checker
