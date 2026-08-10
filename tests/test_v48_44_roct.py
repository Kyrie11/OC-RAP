from __future__ import annotations

from pathlib import Path

import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def _model(*, benefit: bool = False, deployability: bool = False) -> OCRAPModel:
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
        direct_recovery_evidence_roct_benefit=benefit,
        direct_recovery_evidence_roct_deployability=deployability,
        direct_recovery_evidence_roct_scale=3.0,
        direct_recovery_evidence_roct_alpha=0.2,
        direct_recovery_evidence_roct_beta=0.2,
        direct_recovery_evidence_roct_top_m=8,
        direct_recovery_evidence_roct_option_temperature=0.35,
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
    torch.manual_seed(4844)
    x = torch.randn(8, FlatFeatureLayout().total_dim)
    groups = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    root_valid = torch.ones((8, 3), dtype=torch.bool)
    option_valid = torch.ones((8, 4), dtype=torch.bool)
    return x, groups, nominal, root_valid, option_valid


def _forward(model: OCRAPModel):
    x, groups, nominal, root_valid, option_valid = _batch()
    out = model(
        x,
        bucket_id=torch.ones(8, dtype=torch.long),
        group_index=groups,
        is_nominal=nominal,
        direct_only=True,
        root_valid=root_valid,
        option_valid=option_valid,
    )
    return out, nominal


def test_roct_is_exact_shared_baseline_at_zero_initialisation() -> None:
    torch.manual_seed(4844)
    base = _model().eval()
    torch.manual_seed(4844)
    roct = _model(benefit=True, deployability=True).eval()
    missing, unexpected = roct.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {
        "direct_evidence_roct_benefit.weight",
        "direct_evidence_roct_deployability.weight",
    }
    with torch.no_grad():
        out_a, _ = _forward(base)
        out_d, _ = _forward(roct)
    for key in (
        "direct_recovery_evidence_benefit_logit",
        "direct_recovery_evidence_component_harm_logits",
        "direct_recovery_evidence_effective_component_harm_logits",
        "direct_recovery_opportunity_logit",
        "direct_recovery_harm_logit",
    ):
        assert torch.equal(out_a[key], out_d[key])


def test_roct_signature_is_bounded_candidate_relative_and_nominal_zero() -> None:
    model = _model(benefit=True, deployability=True).eval()
    with torch.no_grad():
        out, nominal = _forward(model)
    absolute = out["direct_recovery_evidence_roct_signature"]
    relative = out["direct_recovery_evidence_roct_signature_relative"]
    assert absolute.shape == (8, 4)
    assert relative.shape == (8, 4)
    assert torch.isfinite(absolute).all() and torch.isfinite(relative).all()
    assert bool(torch.all((absolute >= 0.0) & (absolute <= 1.0)))
    assert torch.equal(relative[nominal.bool()], torch.zeros_like(relative[nominal.bool()]))
    assert relative[~nominal.bool()].abs().sum() > 0


def test_roct_deployability_side_is_component_local() -> None:
    torch.manual_seed(11)
    base = _model().eval()
    torch.manual_seed(11)
    roct = _model(deployability=True).eval()
    roct.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        roct.direct_evidence_roct_deployability.weight.fill_(0.4)
        out_a, _ = _forward(base)
        out_b, nominal = _forward(roct)
    a = out_a["direct_recovery_evidence_component_harm_logits"]
    b = out_b["direct_recovery_evidence_component_harm_logits"]
    delta = b - a
    assert torch.equal(delta[:, 0], torch.zeros_like(delta[:, 0]))
    assert delta[:, 1][~nominal.bool()].abs().sum() > 0
    assert torch.equal(delta[:, 2:], torch.zeros_like(delta[:, 2:]))
    assert torch.equal(
        out_a["direct_recovery_evidence_benefit_logit"],
        out_b["direct_recovery_evidence_benefit_logit"],
    )
    correction = out_b["direct_recovery_evidence_roct_deployability_correction"]
    assert torch.equal(correction[nominal.bool()], torch.zeros_like(correction[nominal.bool()]))
    assert float(correction.abs().max()) <= 3.0 + 1e-6


def test_roct_benefit_side_does_not_rotate_harm_components() -> None:
    torch.manual_seed(12)
    base = _model().eval()
    torch.manual_seed(12)
    roct = _model(benefit=True).eval()
    roct.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        roct.direct_evidence_roct_benefit.weight.fill_(0.4)
        out_a, _ = _forward(base)
        out_c, nominal = _forward(roct)
    assert torch.equal(
        out_a["direct_recovery_evidence_component_harm_logits"],
        out_c["direct_recovery_evidence_component_harm_logits"],
    )
    assert (
        out_c["direct_recovery_evidence_benefit_logit"]
        - out_a["direct_recovery_evidence_benefit_logit"]
    )[~nominal.bool()].abs().sum() > 0
    correction = out_c["direct_recovery_evidence_roct_benefit_correction"]
    assert torch.equal(correction[nominal.bool()], torch.zeros_like(correction[nominal.bool()]))
    assert float(correction.abs().max()) <= 3.0 + 1e-6


def test_roct_adapters_receive_gradient_but_structural_teacher_is_detached() -> None:
    model = _model(benefit=True, deployability=True).train()
    out, _ = _forward(model)
    loss = (
        out["direct_recovery_evidence_roct_benefit_correction"].sum()
        + out["direct_recovery_evidence_roct_deployability_correction"].sum()
    )
    loss.backward()
    bgrad = model.direct_evidence_roct_benefit.weight.grad
    dgrad = model.direct_evidence_roct_deployability.weight.grad
    assert bgrad is not None and torch.isfinite(bgrad).all() and bgrad.abs().sum() > 0
    assert dgrad is not None and torch.isfinite(dgrad).all() and dgrad.abs().sum() > 0
    assert all(param.grad is None for param in model.root_logit_head.parameters())
    assert all(param.grad is None for param in model.obs_embed_head.parameters())
    assert all(param.grad is None for param in model.margin_head.parameters())


def test_roct_wrappers_are_clean_unified_two_by_two() -> None:
    arm = (ROOT / "scripts/run_v48_44_roct_ablation_arm.sh").read_text()
    parallel = (ROOT / "scripts/run_v48_44_roct_2x2_parallel.sh").read_text()
    assert "EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false" in arm
    assert "EVIDENCE_RANK_BENEFIT_SKIP=false" in arm
    assert "EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=false" in arm
    assert "EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=false" in arm
    assert "EVIDENCE_ROCT_DEPLOYABILITY=true" in arm
    assert "EVIDENCE_ROCT_BENEFIT=true" in arm
    assert "MAX_PARALLEL_ARMS" in parallel
    for forbidden in ("REGIME_CONDITIONING=true", "NEAR_ROCT", "CONTACT_ROCT", "SAFE_ROCT"):
        assert forbidden not in arm


def test_roct_is_cache_checkpoint_contract_and_stage_transfer_bound() -> None:
    files = [
        ROOT / "src/ocrap/cli/train.py",
        ROOT / "src/ocrap/models/inference.py",
        ROOT / "scripts/train_ocrap_v48_trac_sr.sh",
        ROOT / "scripts/adapt_ocrap_v48_36_ocaf_variant.sh",
        ROOT / "scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh",
        ROOT / "tools/check_v48_36_ocaf_model_contract.py",
        ROOT / "tools/check_v48_36_ocaf_training_contract.py",
        ROOT / "tools/check_v48_36_stage_transfer.py",
    ]
    text = "\n".join(p.read_text() for p in files)
    for token in (
        "roct_benefit",
        "roct_deployability",
        "roct_scale",
        "roct_alpha",
        "roct_beta",
        "roct_top_m",
        "roct_option_temperature",
    ):
        assert token in text
    stage = (ROOT / "tools/check_v48_36_stage_transfer.py").read_text()
    assert '"direct_evidence_roct_benefit"' in stage
    assert '"direct_evidence_roct_deployability"' in stage
