from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def _model(*, benefit: bool = False, harm: bool = False) -> OCRAPModel:
    return OCRAPModel(
        input_dim=FlatFeatureLayout().total_dim,
        num_roots=3,
        num_options=4,
        d_model=12,
        d_obs=6,
        encoder_type="structured_transformer",
        num_layers=1,
        num_heads=3,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_pooling="candidate_concat_raw",
        direct_recovery_delta_head=True,
        direct_recovery_delta_mode="ordinal_evidence",
        direct_recovery_delta_regime_experts=True,
        direct_recovery_delta_policy_features=True,
        direct_recovery_evidence_calibrator=True,
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_source="physical_interaction",
        direct_recovery_evidence_interaction_hidden=16,
        direct_recovery_evidence_interaction_dropout=0.0,
        direct_recovery_evidence_dual_interaction_bridge=True,
        direct_recovery_evidence_postprefix_obs_transport_benefit=benefit,
        direct_recovery_evidence_postprefix_obs_transport_harm=harm,
        direct_recovery_evidence_postprefix_obs_transport_scale=1.0,
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=5,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=False,
        direct_recovery_evidence_admission_prior_mode="joint_reserve",
        direct_recovery_evidence_reserve_factor_alignment=True,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_reliability="1,1,1,0,0",
    )


def _batch():
    torch.manual_seed(4843)
    x = torch.randn(8, FlatFeatureLayout().total_dim)
    groups = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    return x, groups, nominal


def test_poet_is_exact_shared_baseline_at_zero_initialisation() -> None:
    torch.manual_seed(4843)
    base = _model().eval()
    torch.manual_seed(4843)
    poet = _model(benefit=True, harm=True).eval()
    missing, unexpected = poet.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {
        "direct_evidence_postprefix_obs_transport_benefit.weight",
        "direct_evidence_postprefix_obs_transport_harm.weight",
    }
    x, groups, nominal = _batch()
    with torch.no_grad():
        out_a = base(x, bucket_id=torch.ones(8, dtype=torch.long), group_index=groups, is_nominal=nominal, direct_only=True)
        out_d = poet(x, bucket_id=torch.ones(8, dtype=torch.long), group_index=groups, is_nominal=nominal, direct_only=True)
    for key in (
        "direct_recovery_evidence_benefit_logit",
        "direct_recovery_evidence_component_harm_logits",
        "direct_recovery_opportunity_logit",
        "direct_recovery_harm_logit",
    ):
        assert torch.equal(out_a[key], out_d[key])


def test_poet_signature_is_candidate_relative_bounded_and_nominal_zero() -> None:
    model = _model(benefit=True, harm=True).eval()
    x, groups, nominal = _batch()
    with torch.no_grad():
        out = model(x, bucket_id=torch.ones(8, dtype=torch.long), group_index=groups, is_nominal=nominal, direct_only=True)
    absolute = out["direct_recovery_evidence_postprefix_obs_signature"]
    relative = out["direct_recovery_evidence_postprefix_obs_signature_relative"]
    assert absolute.shape == (8, 4)
    assert relative.shape == (8, 4)
    assert torch.isfinite(absolute).all() and torch.isfinite(relative).all()
    assert bool(torch.all((absolute >= 0.0) & (absolute <= 1.0)))
    assert torch.equal(relative[nominal.bool()], torch.zeros_like(relative[nominal.bool()]))
    # Random candidate-conditioned prefix features should normally alter at least
    # one post-prefix structural coordinate relative to the nominal candidate.
    assert relative[~nominal.bool()].abs().sum() > 0


def test_poet_task_injection_is_separable_and_regime_free() -> None:
    torch.manual_seed(7)
    harm_only = _model(harm=True).eval()
    x, groups, nominal = _batch()
    # Make the zero-init structural projection visible without changing the base OCAF.
    with torch.no_grad():
        harm_only.direct_evidence_postprefix_obs_transport_harm.weight.fill_(0.25)
        out = harm_only(x, bucket_id=torch.ones(8, dtype=torch.long), group_index=groups, is_nominal=nominal, direct_only=True)
    transport = out["direct_recovery_evidence_postprefix_obs_harm_transport"]
    assert "direct_recovery_evidence_postprefix_obs_benefit_transport" not in out
    assert torch.equal(transport[nominal.bool()], torch.zeros_like(transport[nominal.bool()]))
    assert transport[~nominal.bool()].abs().sum() > 0
    # The model flag/API contains no regime-specific transport path.
    assert not hasattr(harm_only, "direct_evidence_postprefix_obs_transport_regime")


def test_poet_requires_dual_physical_interaction_context() -> None:
    kwargs = dict(
        input_dim=FlatFeatureLayout().total_dim,
        num_roots=2,
        num_options=3,
        d_model=8,
        d_obs=4,
        encoder_type="structured_transformer",
        num_layers=1,
        num_heads=2,
        direct_recovery_value_head=True,
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_source="physical_interaction",
        direct_recovery_evidence_postprefix_obs_transport_harm=True,
    )
    with pytest.raises(ValueError, match="dual OCAF"):
        OCRAPModel(**kwargs)


def test_poet_wrappers_are_clean_two_by_two_and_disable_v4842_negative_arms() -> None:
    arm = (ROOT / "scripts/run_v48_43_poet_ablation_arm.sh").read_text()
    parallel = (ROOT / "scripts/run_v48_43_poet_2x2_parallel.sh").read_text()
    assert "EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false" in arm
    assert "EVIDENCE_RANK_BENEFIT_SKIP=false" in arm
    assert "EVIDENCE_DUAL_INTERACTION_BRIDGE=true" in arm
    assert "EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=true" in arm
    assert "EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=true" in arm
    for token in ("A)", "B)", "C)", "D)"):
        assert token in arm
    assert "MAX_PARALLEL_ARMS" in parallel
    assert "REGIME_CONDITIONING=true" not in arm


def test_poet_flags_are_cache_checkpoint_and_contract_bound() -> None:
    files = [
        ROOT / "src/ocrap/cli/train.py",
        ROOT / "src/ocrap/models/inference.py",
        ROOT / "scripts/train_ocrap_v48_trac_sr.sh",
        ROOT / "scripts/adapt_ocrap_v48_36_ocaf_variant.sh",
        ROOT / "scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh",
        ROOT / "tools/check_v48_36_ocaf_model_contract.py",
        ROOT / "tools/check_v48_36_ocaf_training_contract.py",
    ]
    text = "\n".join(p.read_text() for p in files)
    for token in (
        "postprefix_obs_transport_benefit",
        "postprefix_obs_transport_harm",
        "postprefix_obs_transport_scale",
    ):
        assert token in text


def test_poet_transport_receives_gradient_while_structural_teacher_is_detached() -> None:
    model = _model(benefit=True, harm=True).train()
    x, groups, nominal = _batch()
    out = model(x, bucket_id=torch.ones(8, dtype=torch.long), group_index=groups, is_nominal=nominal, direct_only=True)
    loss = (
        out["direct_recovery_evidence_postprefix_obs_benefit_transport"].sum()
        + out["direct_recovery_evidence_postprefix_obs_harm_transport"].sum()
    )
    loss.backward()
    bgrad = model.direct_evidence_postprefix_obs_transport_benefit.weight.grad
    hgrad = model.direct_evidence_postprefix_obs_transport_harm.weight.grad
    assert bgrad is not None and torch.isfinite(bgrad).all() and bgrad.abs().sum() > 0
    assert hgrad is not None and torch.isfinite(hgrad).all() and hgrad.abs().sum() > 0
    # POET treats the already-trained post-prefix latent observation model as a
    # frozen structural teacher during evidence adaptation.
    assert all(param.grad is None for param in model.root_logit_head.parameters())
    assert all(param.grad is None for param in model.obs_embed_head.parameters())


def test_poet_prefixes_are_approved_by_stage_transfer_contract() -> None:
    text = (ROOT / "tools/check_v48_36_stage_transfer.py").read_text()
    assert '"direct_evidence_postprefix_obs_transport_benefit"' in text
    assert '"direct_evidence_postprefix_obs_transport_harm"' in text
