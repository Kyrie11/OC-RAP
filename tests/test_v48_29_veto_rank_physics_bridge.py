from __future__ import annotations

from pathlib import Path

import torch

from ocrap.evaluation.baselines import _bucket_aliases
from ocrap.models.ocrap import OCRAPModel
from ocrap.simulation.closed_loop_runner import (
    _bucket_gamma_aliases,
    _gamma_for_bucket,
    _is_post_contact_bucket_name,
)
from ocrap.utils.regimes import bucket_aliases, canonical_regime_name

ROOT = Path(__file__).resolve().parents[1]


def _model(prior_mode: str = "benefit_only") -> OCRAPModel:
    return OCRAPModel(
        input_dim=12, num_roots=2, num_options=3, d_model=8, d_obs=4,
        encoder_type="mlp", num_layers=1, num_heads=2, dropout=0.0,
        direct_recovery_value_head=True, direct_recovery_value_output="score",
        direct_recovery_relative_features_include_absolute=False,
        direct_recovery_set_tournament=True, direct_recovery_set_tournament_hidden=16,
        direct_recovery_set_tournament_heads=2, direct_recovery_set_tournament_dropout=0.0,
        direct_recovery_set_tournament_replace_base=True,
        direct_recovery_delta_head=True, direct_recovery_delta_regime_experts=True,
        direct_recovery_delta_policy_features=True, direct_recovery_delta_hidden=16,
        direct_recovery_delta_dropout=0.0, direct_recovery_delta_mode="ordinal_evidence",
        direct_recovery_evidence_calibrator=True, direct_recovery_evidence_calibrator_hidden=12,
        direct_recovery_evidence_calibrator_scale=0.75,
        direct_recovery_evidence_calibrator_mode="dual_tail_context",
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_detach=True,
        direct_recovery_evidence_calibrator_context_source="tournament",
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=5,
        direct_recovery_evidence_component_scale=6.0,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_consensus_disagreement_penalty=0.15,
        direct_recovery_evidence_admission_head=True,
        direct_recovery_evidence_admission_scale=2.0,
        direct_recovery_evidence_admission_bounded=True,
        direct_recovery_evidence_admission_prior_mode=prior_mode,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_prior_logit=-2.0,
    ).eval()


def test_provenance_prefixed_near_and_contact_buckets_resolve() -> None:
    near = "evidence_adapt_dev_near_contact"
    contact = "evidence_adapt_dev_contact"
    assert canonical_regime_name(near) == "near_contact"
    assert canonical_regime_name(contact) == "contact"
    assert "near_contact" in bucket_aliases(near)
    assert "contact" in bucket_aliases(contact)
    assert _bucket_aliases(near) == bucket_aliases(near)
    assert _bucket_gamma_aliases(contact) == bucket_aliases(contact)
    assert not _is_post_contact_bucket_name(near)
    assert _is_post_contact_bucket_name(contact)



def test_provenance_prefixed_bucket_loads_calibrated_gamma() -> None:
    cfg = {"selection": {"gamma_rec_by_bucket": {"near_contact": 0.197, "contact": 0.162}}}
    assert _gamma_for_bucket(0.0, cfg, "evidence_adapt_dev_near_contact") == 0.197
    assert _gamma_for_bucket(0.0, cfg, "evidence_adapt_dev_contact") == 0.162

def test_benefit_only_admission_prior_is_supported_and_persisted() -> None:
    model = _model("benefit_only")
    assert model.direct_recovery_evidence_admission_prior_mode == "benefit_only"
    try:
        _model("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid admission prior mode must fail closed")


def test_benefit_only_mode_removes_soft_risk_double_penalty() -> None:
    source = (ROOT / "src" / "ocrap" / "models" / "ocrap.py").read_text()
    block = source[source.index('if self.direct_recovery_evidence_admission_prior_mode == "benefit_only"'):]
    assert "admission_prior = unified_benefit_logit.detach()" in block[:1000]
    assert "softplus(unified_harm_logit.detach())" not in block[:1000]


def test_safe_hard_negative_objective_uses_deployed_score() -> None:
    source = (ROOT / "src" / "ocrap" / "models" / "losses.py").read_text()
    assert "ordinal_evidence_safe_hard_negative_weight" in source
    block = source[source.index("v48.29 VETO-RANK"):]
    assert "deployed_safe_utility[best_safe]" in block[:2200]
    assert "hard_negative" in block[:2200]
    assert "deployed_safe_utility.new_zeros" in block[:2200]


def test_shadow_defaults_to_fast_physics_and_checks_runtime_contract() -> None:
    shadow = (ROOT / "scripts" / "run_v48_29_dev_shadow_closed_loop.sh").read_text()
    assert 'SHADOW_LABEL_MODE="${SHADOW_LABEL_MODE:-fast}"' in shadow
    assert 'SHADOW_AUDIT_LABELS="${SHADOW_AUDIT_LABELS:-0}"' in shadow
    assert "check_v48_29_shadow_runtime_contract.py" in shadow
    checker = (ROOT / "tools" / "check_v48_29_shadow_runtime_contract.py").read_text()
    assert "gamma_rec" in checker
    assert "post_contact_target" in checker
    assert "contact_anchor_step" in checker


def test_all_eight_v48_29_ablations_run_concurrently() -> None:
    text = (ROOT / "scripts" / "run_v48_29_parallel_ablations.sh").read_text()
    for name in (
        "A_risk_centered_reference",
        "B_veto_decoupled",
        "C_add_safe_hard_negative",
        "D_add_frontier_to_hard_negative",
    ):
        assert name in text
    assert "max_concurrent_tasks':8" in text
    assert 'run_task "$group" balanced "$GPU0" &' in text
    assert 'run_task "$group" precision "$GPU1" &' in text


def test_v48_29_main_contract_is_bounded_benefit_only() -> None:
    main = (ROOT / "scripts" / "run_v48_29_veto_rank_physics_dedicated.sh").read_text()
    assert "--expect-admission-bounded true" in main
    assert "--expect-admission-prior-mode benefit_only" in main
    staged = (ROOT / "scripts" / "adapt_ocrap_v48_29_veto_rank_variant.sh").read_text()
    assert "EVIDENCE_ADMISSION_BOUNDED=true" in staged
    assert "EVIDENCE_ADMISSION_PRIOR_MODE:-benefit_only" in staged
