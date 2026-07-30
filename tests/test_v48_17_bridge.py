from __future__ import annotations

import torch

from ocrap.cli.train import SceneTimeBatchSampler
from ocrap.models.ocrap import OCRAPModel
from ocrap.simulation.closed_loop_runner import _route_progression_from_trace


def _bridge_model(*, calibrator: bool) -> OCRAPModel:
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
        direct_recovery_evidence_calibrator_scale=0.75,
        direct_recovery_evidence_calibrator_mode="simplex_context",
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_detach=True,
    ).eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(4817)
    x = torch.randn(6, 12)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    buckets = torch.tensor([1, 1, 1, 2, 2, 2])
    return x, groups, nominal, buckets


def test_simplex_context_zero_initialization_preserves_source_evidence() -> None:
    torch.manual_seed(17)
    source = _bridge_model(calibrator=False)
    with torch.no_grad():
        for adapter in source.direct_delta_adapters or []:
            adapter[-1].weight.normal_(0.0, 0.12)
            adapter[-1].bias.copy_(torch.tensor([0.20, -0.35]))
    corrected = _bridge_model(calibrator=True)
    current = corrected.state_dict()
    compatible = {
        key: value
        for key, value in source.state_dict().items()
        if key in current and current[key].shape == value.shape
    }
    corrected.load_state_dict(compatible, strict=False)
    x, groups, nominal, buckets = _inputs()
    with torch.no_grad():
        base = source(
            x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True
        )
        out = corrected(
            x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True
        )
    assert torch.count_nonzero(out["direct_recovery_evidence_calibrator_residual"]) == 0
    assert torch.allclose(
        base["direct_recovery_evidence_benefit_logit"],
        out["direct_recovery_evidence_benefit_logit"],
        atol=1.0e-6,
    )
    assert torch.allclose(
        base["direct_recovery_evidence_harm_logit"],
        out["direct_recovery_evidence_harm_logit"],
        atol=1.0e-6,
    )


def test_simplex_context_calibrator_is_small_bounded_and_normalized() -> None:
    model = _bridge_model(calibrator=True)
    assert model.direct_evidence_calibrators is not None
    params = sum(parameter.numel() for parameter in model.direct_evidence_calibrators.parameters())
    assert 0 < params < 20_000
    with torch.no_grad():
        for adapter in model.direct_evidence_calibrators:
            adapter[-1].weight.fill_(1.5)
            adapter[-1].bias.copy_(torch.tensor([1.0, -0.5, 0.75]))
    x, groups, nominal, buckets = _inputs()
    with torch.no_grad():
        out = model(
            x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True
        )
    residual = out["direct_recovery_evidence_calibrator_residual"]
    probabilities = out["direct_recovery_evidence_class_probabilities"]
    assert residual.shape == (6, 3)
    assert out["direct_recovery_evidence_calibrator_input"].shape[-1] > 4
    assert float(residual.abs().max()) <= 0.750001
    assert torch.all(probabilities >= 0.0)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(6), atol=1.0e-6)


def test_stratified_group_sampler_interleaves_all_evidence_states() -> None:
    # One sample per group makes the emitted batch labels directly observable.
    groups = [[i] for i in range(6)]
    strata = [2, 2, 0, 0, 1, 1]
    sampler = SceneTimeBatchSampler(
        groups,
        batch_size=3,
        replacement=True,
        group_strata=strata,
        stratified=True,
        stratum_fractions={2: 1 / 3, 0: 1 / 3, 1: 1 / 3},
        shuffle_within_group=False,
    )
    torch.manual_seed(7)
    batches = list(iter(sampler))
    assert len(batches) == 2
    for batch in batches:
        assert {strata[index] for index in batch} == {0, 1, 2}


def test_route_progression_fallback_uses_signed_route_arc_length() -> None:
    route = torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]).numpy()
    forward = _route_progression_from_trace([[1.0, 0.2], [7.0, -0.1]], route)
    backward = _route_progression_from_trace([[7.0, 0.0], [2.0, 0.0]], route)
    assert forward is not None and abs(forward - 6.0) < 1.0e-4
    assert backward is not None and abs(backward + 5.0) < 1.0e-4
