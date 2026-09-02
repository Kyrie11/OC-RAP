from __future__ import annotations
from dataclasses import asdict
import torch
from ocrap.models.data import OPTION_FEATURE_DIM
from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel

def layout(): return FlatFeatureLayout(feature_max_agents=2)
def model(signed=False):
 L=layout(); return OCRAPModel(input_dim=L.total_dim,num_roots=3,num_options=2,d_model=16,d_obs=8,encoder_type='structured_transformer',feature_layout=asdict(L),num_layers=1,num_heads=4,dropout=0.0,option_feature_dim=OPTION_FEATURE_DIM,direct_recovery_absolute_semantic_witness_correction=True,direct_recovery_semantic_witness_active_set_alignment=True,direct_recovery_semantic_witness_path_stop_alignment=False,direct_recovery_semantic_witness_route_alignment=True,direct_recovery_semantic_witness_reentry_alignment=True,direct_recovery_semantic_witness_control_projection=True,direct_recovery_semantic_witness_root_tail_source=True,direct_recovery_semantic_witness_tail_localization=True,direct_recovery_semantic_witness_structured_tail_field=True,direct_recovery_semantic_witness_signed_tail_channels=signed)

def test_v4882_zero_init_and_shapes():
 a=model(False); b=model(True)
 assert a.direct_absolute_root_tail_source_scale is None and b.direct_absolute_root_tail_source_scale is None
 assert tuple(a.direct_absolute_structured_tail_field_weight.shape)==(1,16)
 assert tuple(b.direct_absolute_structured_tail_field_weight.shape)==(2,16)
 assert torch.count_nonzero(a.direct_absolute_structured_tail_field_weight)==0
 assert torch.count_nonzero(b.direct_absolute_structured_tail_field_weight)==0

def test_v4882_signed_requires_structured():
 L=layout()
 try:
  OCRAPModel(input_dim=L.total_dim,num_roots=3,num_options=2,d_model=16,d_obs=8,encoder_type='structured_transformer',feature_layout=asdict(L),direct_recovery_absolute_semantic_witness_correction=True,direct_recovery_semantic_witness_root_tail_source=True,direct_recovery_semantic_witness_signed_tail_channels=True)
 except ValueError as e: assert 'signed tail channels' in str(e)
 else: raise AssertionError('expected fail closed')



def test_v4882_replacement_sampler_never_duplicates_group_inside_minibatch(monkeypatch):
    from ocrap.cli.train import SceneTimeBatchSampler

    # Force replacement order to repeat group 0 before the nominal batch is full.
    def fake_multinomial(weights, num_samples, replacement):
        assert replacement is True
        return torch.tensor([0, 0, 1], dtype=torch.long)[:num_samples]

    monkeypatch.setattr(torch, 'multinomial', fake_multinomial)
    sampler = SceneTimeBatchSampler(
        groups=[[0, 1, 2], [3, 4, 5]],
        batch_size=9,
        replacement=True,
        shuffle_within_group=False,
        stratified=False,
    )
    batches = list(iter(sampler))
    # The repeated draw is preserved across the epoch, but never coalesced with
    # another copy of the same scene-time candidate set in a single minibatch.
    assert batches == [[0, 1, 2], [0, 1, 2]]
    assert all(len(batch) == len(set(batch)) for batch in batches)


def test_v4882_group_sampler_keeps_oversized_group_atomic():
    from ocrap.cli.train import SceneTimeBatchSampler
    sampler = SceneTimeBatchSampler(
        groups=[list(range(6)), [6, 7]],
        batch_size=4,
        replacement=False,
        shuffle_within_group=False,
        shuffle_groups=False,
    )
    batches = list(iter(sampler))
    assert batches[0] == list(range(6))
    assert batches[1] == [6, 7]


def test_v4882_strict_group_index_preflight_rejects_bad_nominal_count(tmp_path):
    import json
    from types import SimpleNamespace
    from ocrap.cli.train import _make_group_batch_sampler

    paths = [tmp_path / f'c{i}.npz' for i in range(2)]
    for path in paths:
        path.touch()
    rows = [
        {'path': str(paths[0]), 'scene': 's', 'time': 1, 'bucket': 1, 'nominal': True},
        {'path': str(paths[1]), 'scene': 's', 'time': 1, 'bucket': 1, 'nominal': True},
    ]
    index = tmp_path / 'index.jsonl'
    index.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    cfg = {'training': {
        'group_batching': True,
        'group_index_path': str(index),
        'group_batching_replacement': True,
        'direct_value_strict_shape_contract': True,
        'artifact_sampler_weight': 0.0,
        'negative_deployable_sampler_weight': 0.0,
        'safe_positive_sampler_weight': 0.0,
        'regime_balance_power': 0.0,
    }}
    try:
        _make_group_batch_sampler(SimpleNamespace(paths=paths), cfg, batch_size=8)
    except RuntimeError as exc:
        assert 'exactly one nominal in the source group index' in str(exc)
    else:
        raise AssertionError('expected strict group-index preflight failure')
