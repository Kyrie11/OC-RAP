from __future__ import annotations

from pathlib import Path

import torch

from ocrap.algorithms.ocmero import torch_oc_mero
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def _model() -> OCRAPModel:
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
        direct_recovery_evidence_roct_benefit=True,
        direct_recovery_evidence_roct_deployability=True,
        direct_recovery_evidence_roct_scale=3.0,
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


def _full_witness_loss(model: OCRAPModel) -> torch.Tensor:
    torch.manual_seed(4845)
    b, k, l = 6, 3, 4
    x = torch.randn(b, FlatFeatureLayout().total_dim)
    root_valid = torch.ones((b, k), dtype=torch.bool)
    option_valid = torch.ones((b, l), dtype=torch.bool)
    out = model(x, root_valid=root_valid, option_valid=option_valid)
    root_p = torch.softmax(out["root_logits"], dim=-1)
    r_dep, r_orc, gap, q = torch_oc_mero(
        out["margins"], root_p, out["c_star"], alpha=0.2, beta=0.2,
        root_valid=root_valid, option_valid=option_valid, top_m=8,
    )
    # A compact stand-in for the exact train-time witness objectives.  It must
    # propagate only through the explicitly unfrozen witness heads below.
    # The real SOWR stage also keeps the explicit root-assignment and
    # observation-kernel losses.  Those are essential because lower-tail/top-m
    # OC-MERO operations alone need not provide dense gradients to root/obs.
    return (r_dep.square().mean() + 0.25 * r_orc.square().mean()
            + 0.25 * gap.square().mean() + 0.5 * q.square().mean()
            + 0.25 * out["root_logits"].square().mean()
            + 0.50 * out["c_star"].square().mean())


def test_sowr_margin_witness_update_is_head_only() -> None:
    model = _model().train()
    prefixes = ("root_logit_head", "margin_head")
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(prefixes))
    loss = _full_witness_loss(model)
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters()}
    assert any(g is not None and g.abs().sum() > 0 for n, g in grads.items() if n.startswith("margin_head"))
    assert any(g is not None and g.abs().sum() > 0 for n, g in grads.items() if n.startswith("root_logit_head"))
    assert all(g is None for n, g in grads.items() if n.startswith("encoder."))
    assert all(g is None for n, g in grads.items() if n.startswith("obs_embed_head"))
    assert all(g is None for n, g in grads.items() if n.startswith("direct_evidence_"))


def test_sowr_observation_update_is_head_only() -> None:
    model = _model().train()
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("obs_embed_head"))
    loss = _full_witness_loss(model)
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters()}
    assert any(g is not None and g.abs().sum() > 0 for n, g in grads.items() if n.startswith("obs_embed_head"))
    assert all(g is None for n, g in grads.items() if n.startswith("encoder."))
    assert all(g is None for n, g in grads.items() if n.startswith("margin_head"))
    assert all(g is None for n, g in grads.items() if n.startswith("root_logit_head"))


def test_sowr_2x2_holds_roct_and_shared_policy_fixed() -> None:
    arm = (ROOT / "scripts/run_v48_45_sowr_ablation_arm.sh").read_text()
    parallel = (ROOT / "scripts/run_v48_45_sowr_2x2_parallel.sh").read_text()
    assert "EVIDENCE_ROCT_BENEFIT=true" in arm
    assert "EVIDENCE_ROCT_DEPLOYABILITY=true" in arm
    assert 'EVIDENCE_ROCT_SCALE="${EVIDENCE_ROCT_SCALE:-3.0}"' in arm
    assert "PROPOSAL_TOP_K=5" in arm
    assert "V4845_SOWR_MARGIN_WITNESS=1" in arm
    assert "V4845_SOWR_OBS_KERNEL=1" in arm
    for token in ("A)", "B)", "C)", "D)"):
        assert token in arm
    for forbidden in ("REGIME_CONDITIONING=true", "NEAR_SOWR", "CONTACT_SOWR", "SAFE_SOWR"):
        assert forbidden not in arm
    assert "MAX_PARALLEL_ARMS" in parallel


def test_sowr_stage_uses_exact_existing_teacher_losses_and_no_encoder_finetune() -> None:
    stage = (ROOT / "scripts/adapt_ocrap_v48_45_sowr_stage.sh").read_text()
    assert 'prefixes="root_logit_head,margin_head"' in stage
    assert 'prefixes+="obs_embed_head"' in stage
    assert "DIRECT_ONLY_FAST_PATH=false" in stage
    assert "LOSS_MARGIN=" in stage and "LOSS_OBS=" in stage
    assert "LOSS_OPTION_Q=1.50" in stage
    assert "LOSS_OPTION_ADMISSION=1.00" in stage
    assert "LOSS_OPTION_BEST=1.00" in stage
    assert "DIRECT_VALUE_WEIGHT=0" in stage
    assert "ENCODER_LR_SCALE=0" in stage
    assert "SKIP_POST_TRAIN_CALIBRATION=1" in stage
    assert "regime_id_exposed" in stage
    assert "encoder" not in 'prefixes="root_logit_head,margin_head"'


def test_sowr_source_checkpoint_is_bound_into_factor_cache_contract() -> None:
    variant = (ROOT / "scripts/adapt_ocrap_v48_36_ocaf_variant.sh").read_text()
    assert 'SOURCE_CKPT="$WITNESS_RUN/model_v48_sowr/best.pt"' in variant
    assert '--source-checkpoint "$SOURCE_CKPT"' in variant
    assert 'sowr_margin_witness=${V4845_SOWR_MARGIN_WITNESS:-0}' in variant
    assert 'sowr_obs_kernel=${V4845_SOWR_OBS_KERNEL:-0}' in variant


def test_legacy_direct_training_defaults_are_unchanged() -> None:
    train = (ROOT / "scripts/train_ocrap_v48_trac_sr.sh").read_text()
    assert 'training.direct_only_fast_path="${DIRECT_ONLY_FAST_PATH:-true}"' in train
    assert 'loss_weights.margin="${LOSS_MARGIN:-0}"' in train
    assert 'loss_weights.obs="${LOSS_OBS:-0}"' in train
    assert 'loss_weights.option_q="${LOSS_OPTION_Q:-0}"' in train
    assert 'loss_weights.direct_recovery_value="${DIRECT_VALUE_WEIGHT:-10.0}"' in train


def test_sowr_comparator_is_development_certificate_only() -> None:
    text = (ROOT / "tools/compare_v48_45_sowr_2x2.py").read_text()
    assert '"test_roots_read": False' in text
    assert "SOWR_COMPLETE.json" in text
    assert "direct_value_risk_" in text
    assert "dev_diagnostic_" in text
    assert "test_" not in text.replace('"test_roots_read": False', "")
