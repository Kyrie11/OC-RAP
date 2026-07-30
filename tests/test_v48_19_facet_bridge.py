from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from ocrap.algorithms.evidence_targets import (
    ComponentVetoTolerances,
    component_veto_margin_numpy,
    component_veto_margin_torch,
    component_veto_soft_target,
)
from ocrap.cli.train import _finalize_direct_policy_stats
from ocrap.evaluation.certificate_stats import certificate_support_feasibility, wilson_interval, wilson_z
from ocrap.models.ocrap import OCRAPModel


def _facet_model(*, shared: bool) -> OCRAPModel:
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
        direct_recovery_evidence_calibrator=True,
        direct_recovery_evidence_calibrator_hidden=12,
        direct_recovery_evidence_calibrator_scale=0.5,
        direct_recovery_evidence_calibrator_mode="dual_tail_context",
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_detach=True,
        direct_recovery_evidence_calibrator_context_source="tournament",
        direct_recovery_evidence_calibrator_shared=shared,
        direct_recovery_evidence_calibrator_regime_scale=0.25,
    ).eval()


def _inputs():
    torch.manual_seed(4819)
    return (
        torch.randn(6, 12),
        torch.tensor([[0], [0], [0], [1], [1], [1]]),
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        torch.tensor([1, 1, 1, 2, 2, 2]),
    )


def test_component_veto_can_overlap_total_benefit() -> None:
    # Total PCD can improve through deployability while DRS regresses enough to
    # trigger a non-compensatory safety veto.
    margin = component_veto_margin_numpy(
        candidate_drs=0.75, nominal_drs=0.90,
        candidate_r_dep=2.5, nominal_r_dep=-1.0,
        candidate_gap=0.05, nominal_gap=0.30,
        tolerances=ComponentVetoTolerances(drs=0.05),
    )
    assert margin > 0.0


def test_component_veto_unchanged_components_are_not_harmful() -> None:
    margin = component_veto_margin_numpy(
        candidate_drs=0.8, nominal_drs=0.8,
        candidate_r_dep=0.4, nominal_r_dep=0.4,
        candidate_gap=0.2, nominal_gap=0.2,
        candidate_hard=0.0, nominal_hard=0.0,
        candidate_harm_proxy=0.0, nominal_harm_proxy=0.0,
    )
    assert margin < 0.0
    target = component_veto_soft_target(torch.tensor([margin]))
    assert float(target.item()) < 0.5


def test_component_veto_numpy_torch_parity() -> None:
    kwargs = dict(
        candidate_drs=0.72, nominal_drs=0.80,
        candidate_r_dep=0.4, nominal_r_dep=0.6,
        candidate_gap=0.3, nominal_gap=0.2,
        candidate_hard=0.1, nominal_hard=0.0,
        candidate_harm_proxy=0.0, nominal_harm_proxy=0.0,
    )
    expected = component_veto_margin_numpy(**kwargs)
    actual = component_veto_margin_torch(**{
        k: torch.tensor([v], dtype=torch.float32) for k, v in kwargs.items()
    })
    assert np.isclose(float(actual.item()), expected, atol=1e-6)


def test_shared_calibrator_is_identity_initialized_bounded_and_cross_regime() -> None:
    model = _facet_model(shared=True)
    assert model.direct_evidence_shared_calibrator is not None
    x, groups, nominal, buckets = _inputs()
    with torch.no_grad():
        initial = model(x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True)
    assert torch.count_nonzero(initial["direct_recovery_evidence_calibrator_residual"]) == 0
    with torch.no_grad():
        model.direct_evidence_shared_calibrator[-1].bias.fill_(10.0)
        for adapter in model.direct_evidence_calibrators:
            adapter[-1].bias.zero_()
        shifted = model(x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True)
    residual = shifted["direct_recovery_evidence_calibrator_residual"]
    assert float(residual.abs().max()) <= 0.500001
    # The shared correction moves recovery rows from both regimes identically;
    # nominal rows are still pinned later in the evidence logits.
    assert bool((residual[buckets == 1].abs() > 0).any())
    assert bool((residual[buckets == 2].abs() > 0).any())


def test_certificate_preflight_detects_v4818_near_impossibility() -> None:
    # Historical v48.18 used the central two-sided 90% critical value while
    # reporting directional LCB90/UCB90 metrics. Its Near fit specification is
    # impossible even for an oracle ranking.
    impossible = certificate_support_feasibility(
        num_groups=127, num_opportunities=8, min_selected=12,
        min_precision_lcb=0.50, max_harmful_selected_ucb=0.22,
        max_harmful_group_ucb=0.12, bound_type="two_sided",
    )
    assert not impossible["feasible"]
    assert impossible["optimistic_best_precision_lcb"] < 0.50

    # v48.19 pre-registers directional one-sided 90% bounds. Fit needs ten
    # selections because only eight opportunities exist; verify can retain the
    # original minimum of eight.
    fit = certificate_support_feasibility(
        num_groups=127, num_opportunities=8, min_selected=10,
        min_precision_lcb=0.50, max_harmful_selected_ucb=0.22,
        max_harmful_group_ucb=0.12, confidence_level=0.90, bound_type="one_sided",
    )
    verify = certificate_support_feasibility(
        num_groups=163, num_opportunities=6, min_selected=8,
        min_precision_lcb=0.40, max_harmful_selected_ucb=0.25,
        max_harmful_group_ucb=0.14, confidence_level=0.90, bound_type="one_sided",
    )
    assert fit["feasible"] and fit["first_feasible_selected"] == 10
    assert verify["feasible"] and verify["first_feasible_selected"] == 8
    assert np.isclose(wilson_z(confidence_level=0.90, bound_type="one_sided"), 1.2815515655)
    assert wilson_interval(6, 8, confidence_level=0.90, bound_type="one_sided")[0] > 0.40


def _stats(recall_hits: float, harmful: float, false: float) -> dict[str, float]:
    stats: dict[str, float] = {}
    for regime in ("near", "contact"):
        stats[f"group_count_{regime}"] = 40.0
        stats[f"positive_count_{regime}"] = 10.0
        stats[f"positive_admission_hit_{regime}"] = recall_hits
        stats[f"certificate_positive_regret_sum_{regime}"] = 1.0
        stats[f"admitted_harmful_{regime}"] = harmful
        stats[f"false_intervention_{regime}"] = false
    return stats


def test_facet_metric_prefers_controlled_cross_regime_recall_over_all_abstain() -> None:
    cfg = {
        "direct_policy_metric_min_positive_recall": 0.20,
        "direct_policy_metric_cross_regime_min_recall": 0.20,
        "direct_policy_metric_facet_min_recall": 0.20,
    }
    abstain = _finalize_direct_policy_stats(_stats(0.0, 0.0, 0.0), cfg)
    controlled = _finalize_direct_policy_stats(_stats(3.0, 1.0, 2.0), cfg)
    unsafe = _finalize_direct_policy_stats(_stats(8.0, 16.0, 24.0), cfg)
    assert controlled["direct_facet_selection_risk"] < abstain["direct_facet_selection_risk"]
    assert unsafe["direct_facet_selection_risk"] > controlled["direct_facet_selection_risk"]


def test_teacher_index_contract_detects_manifest_or_tolerance_drift(tmp_path) -> None:
    import argparse
    import hashlib
    import importlib.util
    import json

    tool = Path(__file__).resolve().parents[1] / "tools" / "check_v48_19_target_support.py"
    spec = importlib.util.spec_from_file_location("check_v48_19_target_support", tool)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    near = tmp_path / "near"
    contact = tmp_path / "contact"
    near.mkdir(); contact.mkdir()
    (near / "manifest.csv").write_text("path,scene_id\na,near-1\n", encoding="utf-8")
    (contact / "manifest.csv").write_text("path,scene_id\nb,contact-1\n", encoding="utf-8")

    def record(root):
        manifest = root / "manifest.csv"
        return {
            "root": str(root.resolve()),
            "manifest": str(manifest.resolve()),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }

    summary = {
        "index_contract": {
            "dataset_roots": [str(near.resolve()), str(contact.resolve())],
            "dataset_manifests": [record(near), record(contact)],
            "alpha": 0.2,
            "beta": 0.2,
            "top_m": 8,
            "positive_gain": 0.015,
            "deployable_macro_ids": [2, 3, 5, 6, 7],
            "component_harm_tolerances": {
                "drs": 0.05,
                "deployability_gate": 0.05,
                "gap_discount": 0.05,
                "hard_violation": 0.05,
                "harm_proxy": 0.05,
            },
        }
    }
    args = argparse.Namespace(
        expected_dataset=f"{near},{contact}", alpha=0.2, beta=0.2, top_m=8,
        positive_gain=0.015, deployable_macro_ids="2,3,5,6,7",
        component_harm_drs_tolerance=0.05,
        component_harm_dep_tolerance=0.05,
        component_harm_gap_tolerance=0.05,
        component_harm_hard_tolerance=0.05,
        component_harm_proxy_tolerance=0.05,
    )
    assert module._contract_audit(args, summary)["valid"]

    (near / "manifest.csv").write_text("path,scene_id\na,near-2\n", encoding="utf-8")
    drift = module._contract_audit(args, summary)
    assert not drift["valid"]
    assert "index contract mismatch: dataset_manifests" in drift["failures"]

    (near / "manifest.csv").write_text("path,scene_id\na,near-1\n", encoding="utf-8")
    args.component_harm_hard_tolerance = 0.0
    drift = module._contract_audit(args, summary)
    assert not drift["valid"]
    assert "index contract mismatch: component_harm_tolerances.hard_violation" in drift["failures"]
