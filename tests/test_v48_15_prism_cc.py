from __future__ import annotations

from pathlib import Path

import torch

from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).parents[1]


def _model(*, calibrator: bool) -> OCRAPModel:
    return OCRAPModel(
        input_dim=12,
        num_roots=2,
        num_options=3,
        d_model=8,
        d_obs=4,
        encoder_type="mlp",
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_output="score",
        direct_recovery_relative_features_include_absolute=False,
        direct_recovery_set_tournament=True,
        direct_recovery_set_tournament_hidden=16,
        direct_recovery_set_tournament_heads=2,
        direct_recovery_set_tournament_dropout=0.0,
        direct_recovery_set_tournament_replace_base=True,
        direct_recovery_delta_head=True,
        direct_recovery_delta_regime_experts=True,
        direct_recovery_delta_policy_features=True,
        direct_recovery_delta_hidden=16,
        direct_recovery_delta_dropout=0.0,
        direct_recovery_delta_mode="ordinal_evidence",
        direct_recovery_evidence_calibrator=calibrator,
        direct_recovery_evidence_calibrator_hidden=8,
        direct_recovery_evidence_calibrator_scale=0.3,
    ).eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(4815)
    x = torch.randn(6, 12)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    buckets = torch.tensor([1, 1, 1, 2, 2, 2])
    return x, groups, nominal, buckets


def test_zero_initialized_evidence_calibrator_exactly_preserves_source_outputs() -> None:
    torch.manual_seed(15)
    source = _model(calibrator=False)
    with torch.no_grad():
        for adapter in source.direct_delta_adapters or []:
            adapter[-1].weight.normal_(0.0, 0.15)
            adapter[-1].bias.copy_(torch.tensor([0.25, -0.40]))
    corrected = _model(calibrator=True)
    current = corrected.state_dict()
    compatible = {k: v for k, v in source.state_dict().items() if k in current and current[k].shape == v.shape}
    corrected.load_state_dict(compatible, strict=False)
    x, groups, nominal, buckets = _inputs()
    with torch.no_grad():
        base = source(x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True)
        out = corrected(x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True)
    assert torch.count_nonzero(out["direct_recovery_evidence_calibrator_residual"]) == 0
    assert torch.allclose(base["direct_recovery_evidence_benefit_logit"], out["direct_recovery_evidence_benefit_logit"], atol=1e-8)
    assert torch.allclose(base["direct_recovery_evidence_harm_logit"], out["direct_recovery_evidence_harm_logit"], atol=1e-8)


def test_evidence_calibrator_is_tiny_regime_specific_and_bounded() -> None:
    model = _model(calibrator=True)
    assert model.direct_evidence_calibrators is not None
    params = sum(p.numel() for p in model.direct_evidence_calibrators.parameters())
    assert 0 < params < 2000
    with torch.no_grad():
        for adapter in model.direct_evidence_calibrators:
            adapter[-1].weight.fill_(2.0)
            adapter[-1].bias.fill_(2.0)
    x, groups, nominal, buckets = _inputs()
    with torch.no_grad():
        out = model(x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True)
    residual = out["direct_recovery_evidence_calibrator_residual"]
    assert residual.shape == (6, 2)
    assert float(residual.abs().max()) <= 0.300001


def test_certificate_controller_distinguishes_artifact_failure_from_gate_failure() -> None:
    text = (ROOT / "scripts" / "calibrate_v48_14_certificate_pool.sh").read_text()
    assert 'local variant="$1"\n  local gpu="$2"\n  local run=' in text
    assert "CALIBRATION_FAILED.json" in text
    assert "raise SystemExit(30)" in text
    assert 'VARIANTS="${VARIANTS:-balanced,precision}"' in text


def test_safe_probe_requires_real_bucket_targets_and_does_not_force_test_split() -> None:
    runner = (ROOT / "src" / "ocrap" / "simulation" / "closed_loop_runner.py").read_text()
    shell = (ROOT / "scripts" / "run_ocrap_v48_trac_sr.sh").read_text()
    assert "require_bucket_targets" in runner
    assert "no targets matched" in runner
    assert '--set closed_loop.bucket_split="${SAFE_BUCKET_SPLIT:-}"' in shell
    assert "--set closed_loop.require_bucket_targets=true" in shell
