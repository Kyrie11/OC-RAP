from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_selected_waymax_iterator_delays_state_materialization(monkeypatch):
    import ocrap.data.waymax_loader as wl

    examples = [
        {'scenario/id': f's{i}', 'i': i}
        for i in range(5)
    ]
    converted: list[int] = []

    class FakeDataLoader:
        @staticmethod
        def preprocess_serialized_womd_data(x, config=None):
            return x

        @staticmethod
        def get_data_generator(dataset_cfg, parse, postprocess):
            for x in examples:
                yield postprocess(parse(x))

    class FakeFactories:
        @staticmethod
        def simulator_state_from_womd_dict(example, include_sdc_paths=True):
            converted.append(int(example['i']))
            return SimpleNamespace(num_objects=4)

    monkeypatch.setattr(wl, '_apply_jax_env', lambda cfg: None)
    monkeypatch.setattr(wl, '_require_waymax', lambda: (None, None, None, FakeDataLoader, FakeFactories))
    monkeypatch.setattr(wl, '_make_dataset_config', lambda patterns, cfg: SimpleNamespace(max_num_objects=4))
    monkeypatch.setattr(wl, '_scenario_identity_from_payload', lambda payload, i, state, cfg: (f'saved{i}', f'base{i}', f'legacy{i}'))
    monkeypatch.setattr(wl, '_infer_womd_source_role', lambda p: 'validation')
    monkeypatch.setattr(wl, '_paths_to_waymax_path', lambda p: str(p))

    def fake_raw(state, saved_id, i, cfg):
        return SimpleNamespace(metadata={'_waymax_scenario_index': int(i)})

    monkeypatch.setattr(wl, 'raw_scenario_from_waymax_state', fake_raw)
    cfg = {'waymax': {'retain_official_scenario_id': False, 'dataloader_include_sdc_paths': True}}
    rows = list(wl.iter_waymax_womd_scenarios_selected('fake@1', [1, 3], cfg))
    assert converted == [1, 3]
    assert [r.metadata['_waymax_scenario_index'] for r in rows] == [1, 3]


def test_v4891_audit_config_skips_metadata_only_future_metrics(monkeypatch, tmp_path: Path):
    mod = _load_tool('v4891_sidecar_perf', 'tools/build_v48_91_common_exogenous_physical_sidecar.py')
    monkeypatch.setattr(mod, '_origin_replay_metadata', lambda path, idx: {})
    monkeypatch.setattr(mod, '_find_summary', lambda path: None)
    sample = {
        '__path__': str(tmp_path / 'x__wx00000001_t0001_a00.npz'),
        'source_scenario_index': np.asarray(1),
        'womd_source_pattern': np.asarray('/raw/validation.tfrecord@150'),
        'future_sources': np.asarray(['replay', 'reactive', 'targeted']),
        'm_star': np.zeros((8, 12), dtype=np.float32),
        'agent_history': np.zeros((10, 64, 16), dtype=np.float32),
    }
    base = {'waymax': {'compute_future_metrics': True, 'use_jit_scan_rollouts': False}}
    cfg, _, _ = mod._config_for_sample(sample, base, resolved_pattern='/raw/validation.tfrecord@150', resolved_index=1)
    assert cfg['waymax']['compute_future_metrics'] is False
    assert cfg['waymax']['use_jit_scan_rollouts'] is True
    assert cfg['waymax']['cache_env_objects'] is True
    assert cfg['waymax']['cache_postprefix_rollouts'] is True
    assert cfg['waymax']['cache_teacher_metric_rollouts'] is True


def test_v4891_legacy_path_sharding_key():
    mod = _load_tool('v4891_sidecar_shard', 'tools/build_v48_91_common_exogenous_physical_sidecar.py')
    assert mod._legacy_source_index_from_path('waymax_x__wx00012658_t0029_a05.npz') == 12658
    assert mod._legacy_source_index_from_path('no_source.npz') == -1
    assert 12658 % 2 == 0 and 12659 % 2 == 1


def test_v4891_merge_parts_is_disjoint_and_deterministic(tmp_path: Path):
    parts = []
    sums = []
    for w, sample in [(0, '/tmp/b.npz'), (1, '/tmp/a.npz')]:
        part = tmp_path / f'p{w}.jsonl.gz'
        with gzip.open(part, 'wt', encoding='utf-8') as f:
            f.write(json.dumps({'valid': True, 'sample_path': sample, 'future_probability_max_abs_error': 0.0, 'active_root_margin_max_abs_error': 0.0}) + '\n')
        summ = tmp_path / f's{w}.json'
        summ.write_text(json.dumps({
            'valid': True, 'attribution_ready': True,
            'engineering_version': 'v48.91.3-OC-CEPMI-REPLAYFIX',
            'requested_samples': 1, 'valid_samples': 1, 'labeled_candidate_pairs': 1,
            'worker_partition': {'num_workers': 2, 'worker_index': w},
            'replay_provenance_resolution_counts': {'index:legacy_wx_migration_key': 1},
            'resolved_source_index_min': 10 + w, 'resolved_source_index_max': 10 + w,
            'performance': {'total_seconds': 2.0 + w, 'sample_replay_accumulated_seconds': 1.0, 'target_source_indices': 1, 'history_cache_hits': 0},
            'legacy_provenance_migration_used': True,
        }))
        parts.append(part); sums.append(summ)
    out = tmp_path / 'out.jsonl.gz'; summary = tmp_path / 'out.summary.json'
    cmd = [sys.executable, str(ROOT / 'tools/merge_v48_91_common_exogenous_physical_sidecar_parts.py')]
    for p in parts: cmd += ['--part', str(p)]
    for s in sums: cmd += ['--part-summary', str(s)]
    cmd += ['--output', str(out), '--summary', str(summary)]
    subprocess.run(cmd, check=True, cwd=ROOT)
    doc = json.loads(summary.read_text())
    assert doc['valid'] and doc['requested_samples'] == 2
    with gzip.open(out, 'rt', encoding='utf-8') as f:
        rows = [json.loads(x) for x in f]
    assert [Path(r['sample_path']).name for r in rows] == ['a.npz', 'b.npz']



def test_v4891_exogenous_class_key_ignores_waymax_metric_metadata():
    from ocrap.v48_90_partition_transport import future_class_keys
    base = {
        'future_sources': np.asarray(['reactive']),
        'future_metadata': json.dumps([{'rollout_variant': 'shared', 'waymax_metrics': {'overlap': 1.0}}]),
        'future_probs': np.asarray([1.0], dtype=np.float32),
        'future_valid': np.asarray([1.0], dtype=np.float32),
        'root_assignments': np.asarray([0], dtype=np.int64),
    }
    alt = dict(base)
    alt['future_metadata'] = json.dumps([{'rollout_variant': 'shared'}])
    k0, u0, _ = future_class_keys(base, exogenous=True)
    k1, u1, _ = future_class_keys(alt, exogenous=True)
    assert k0 == k1
    assert np.array_equal(u0, u1)

def test_v4891_runner_exposes_safe_parallel_replay_controls():
    text = (ROOT / 'scripts/run_v48_91_dcp_drfc_bcde_rifa_cepmi.sh').read_text()
    assert 'V4891_REPLAY_WORKERS' in text
    assert 'V4891_PROGRESS_EVERY' in text
    assert 'GPU0' in text and 'GPU1' in text
    assert 'merge_v48_91_common_exogenous_physical_sidecar_parts.py' in text
