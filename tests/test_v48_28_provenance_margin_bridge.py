from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from ocrap.data.waymax_loader import (
    _decode_scenario_id,
    _infer_womd_source_role,
    _legacy_scenario_id_from_state,
)
from ocrap.models.ocrap import OCRAPModel
from ocrap.simulation.closed_loop_runner import (
    _canonical_womd_scene_id,
    _legacy_waymax_source_index,
    _source_role_from_pattern,
)

ROOT = Path(__file__).resolve().parents[1]


def _model(scale: float = 6.0) -> OCRAPModel:
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
        direct_recovery_evidence_component_scale=scale,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_consensus_disagreement_penalty=0.15,
        direct_recovery_evidence_admission_head=True,
        direct_recovery_evidence_admission_scale=1.0,
        direct_recovery_evidence_admission_bounded=True,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_prior_logit=-2.0,
    ).eval()


def test_official_womd_scenario_id_decoder_supports_bytes_and_uint8() -> None:
    expected = "abc012ff"
    assert _decode_scenario_id(np.asarray(expected.encode(), dtype="S8")) == expected
    assert _decode_scenario_id(np.frombuffer(expected.encode(), dtype=np.uint8)) == expected


def test_womd_validation_and_interactive_roles_are_not_interchangeable() -> None:
    standard = "/womd/validation/validation_tfexample.tfrecord@150"
    interactive = "/womd/validation_interactive/validation_interactive_tfexample.tfrecord@150"
    assert _infer_womd_source_role(standard) == "validation"
    assert _source_role_from_pattern(standard) == "validation"
    assert _infer_womd_source_role(interactive) == "validation_interactive"
    assert _source_role_from_pattern(interactive) == "validation_interactive"


def test_legacy_scene_id_keeps_source_index_only_as_migration_key() -> None:
    assert _canonical_womd_scene_id("waymax_deadbeef__wx00011519") == "waymax_deadbeef"
    assert _legacy_waymax_source_index("waymax_deadbeef__wx00011519") == 11519
    assert _legacy_waymax_source_index("officialhex") is None


def test_component_harm_range_can_express_strong_harmful_probability() -> None:
    model = _model(scale=6.0)
    max_logit = model.direct_recovery_evidence_component_prior_logit + model.direct_recovery_evidence_component_scale
    min_logit = model.direct_recovery_evidence_component_prior_logit - model.direct_recovery_evidence_component_scale
    assert max_logit > 0.0
    assert min_logit < -4.0
    assert torch.sigmoid(torch.tensor(max_logit)).item() > 0.95


def test_factor_checkpoint_metric_depends_on_supervised_factor_loss() -> None:
    source = (ROOT / "src" / "ocrap" / "cli" / "train.py").read_text()
    assert 'epoch_metrics["direct_factor_supervised_risk"]' in source
    block = source[source.index('epoch_metrics["direct_factor_supervised_risk"]'):]
    assert 'loss_direct_recovery_value' in block[:900]
    staged = (ROOT / "scripts" / "adapt_ocrap_v48_28_provenance_margin_variant.sh").read_text()
    assert "BEST_METRIC=direct_factor_supervised_risk" in staged
    assert "EVALUATE_INITIAL_CHECKPOINT=false" in staged


def test_v48_28_model_contract_persists_component_scale() -> None:
    required = {"direct_recovery_evidence_component_scale"}
    for path in (ROOT / "src" / "ocrap" / "cli" / "train.py", ROOT / "src" / "ocrap" / "models" / "inference.py"):
        tree = ast.parse(path.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "OCRAPModel"]
        assert calls
        assert required <= {kw.arg for kw in calls[-1].keywords if kw.arg}
    audit = (ROOT / "tools" / "check_v48_28_model_contract.py").read_text()
    assert "--expect-component-scale" in audit


def test_dev_shadow_defaults_to_standard_validation_and_fails_closed() -> None:
    runner = (ROOT / "scripts" / "run_ocrap_v48_trac_sr.sh").read_text()
    shadow = (ROOT / "scripts" / "run_v48_28_dev_shadow_closed_loop.sh").read_text()
    assert 'shadow_womd_source="$WOMD_VAL"' in runner
    assert 'closed_loop.require_bucket_targets=true' in runner
    assert 'closed_loop.raw_max_scenarios=${DEV_SHADOW_RAW_MAX_SCENARIOS:-0}' in runner
    assert "audit_v48_28_shadow_provenance.py" in shadow
    assert "SHADOW_PROVENANCE_AUDIT.json" in shadow
    assert (ROOT / "scripts" / "repair_v48_27_dev_shadow_with_v48_28.sh").is_file()


def test_official_id_loader_follows_waymax_custom_loader_contract() -> None:
    source = (ROOT / "src" / "ocrap" / "data" / "waymax_loader.py").read_text()
    assert 'features["scenario/id"] = tf.io.FixedLenFeature([1], tf.string)' in source
    assert "tf.io.parse_example(serialized, features)" in source
    assert 'processed["scenario/id"]' in source
    assert 'retain_official_scenario_id' in source


def test_all_eight_ablations_are_launched_concurrently() -> None:
    text = (ROOT / "scripts" / "run_v48_28_parallel_ablations.sh").read_text()
    for name in (
        "A_three_factor_wide_range",
        "B_five_factor_old_range",
        "C_five_factor_wide_range_regression",
        "D_add_listwise_frontier",
    ):
        assert name in text
    assert "max_concurrent_tasks':8" in text
    assert 'run_task "$group" balanced "$GPU0" &' in text
    assert 'run_task "$group" precision "$GPU1" &' in text
    assert 'NUM_WORKERS="${NUM_WORKERS:-1}"' in text
