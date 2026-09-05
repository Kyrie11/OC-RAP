#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

ENGINEERING_VERSION = 'v48.91.2-OC-CEPMI-PERF'


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', action='append', type=Path, required=True, help='partial sidecar .jsonl.gz; repeat')
    ap.add_argument('--part-summary', action='append', type=Path, required=True, help='matching partial summary; repeat')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--summary', type=Path, required=True)
    a = ap.parse_args()
    if len(a.part) != len(a.part_summary):
        raise SystemExit('part / part-summary count mismatch')
    n = len(a.part)
    summaries = [json.loads(p.read_text(encoding='utf-8')) for p in a.part_summary]
    errors: list[str] = []
    indices: list[int] = []
    provenance: dict[str, int] = defaultdict(int)
    all_rows: dict[str, dict[str, Any]] = {}
    for path, sp, s in zip(a.part, a.part_summary, summaries):
        if not s.get('valid') or not s.get('attribution_ready'):
            errors.append(f'invalid part summary {sp}')
        wp = s.get('worker_partition') or {}
        if int(wp.get('num_workers', -1)) != n:
            errors.append(f'worker count mismatch {sp}: {wp}')
        indices.append(int(wp.get('worker_index', -1)))
        if str(s.get('engineering_version')) != ENGINEERING_VERSION:
            errors.append(f'engineering version mismatch {sp}: {s.get("engineering_version")}')
        for k, v in (s.get('replay_provenance_resolution_counts') or {}).items():
            provenance[str(k)] += int(v)
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                key = str(Path(r.get('sample_path', '')).resolve())
                if not key:
                    errors.append(f'row without sample_path in {path}')
                    continue
                if key in all_rows:
                    errors.append(f'duplicate sample across parts: {key}')
                    continue
                all_rows[key] = r
    if sorted(indices) != list(range(n)):
        errors.append(f'worker indices incomplete: {sorted(indices)} expected={list(range(n))}')

    requested = sum(int(s.get('requested_samples', 0)) for s in summaries)
    valid_rows = sum(1 for r in all_rows.values() if r.get('valid'))
    if len(all_rows) != requested:
        errors.append(f'merged row count {len(all_rows)} != requested_samples {requested}')
    if valid_rows != requested:
        errors.append(f'merged valid row count {valid_rows} != requested_samples {requested}')

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.output, 'wt', encoding='utf-8') as f:
        for key in sorted(all_rows):
            f.write(json.dumps(all_rows[key], sort_keys=True) + '\n')

    perf = [s.get('performance') or {} for s in summaries]
    finite_probs = [s.get('max_future_probability_error') for s in summaries if s.get('max_future_probability_error') is not None]
    finite_roots = [s.get('max_active_root_margin_error') for s in summaries if s.get('max_active_root_margin_error') is not None]
    source_min = [s.get('resolved_source_index_min') for s in summaries if s.get('resolved_source_index_min') is not None]
    source_max = [s.get('resolved_source_index_max') for s in summaries if s.get('resolved_source_index_max') is not None]
    summary = {
        'schema': 'ocrap-v48.91-common-exogenous-future-physical-sidecar-v1',
        'engineering_version': ENGINEERING_VERSION,
        'valid': not errors,
        'attribution_ready': not errors,
        'errors': errors[:100],
        'requested_samples': requested,
        'valid_samples': valid_rows,
        'labeled_candidate_pairs': sum(int(s.get('labeled_candidate_pairs', 0)) for s in summaries),
        'output': str(a.output.resolve()),
        'output_sha256': _sha(a.output),
        'max_future_probability_error': max(finite_probs) if finite_probs else None,
        'max_active_root_margin_error': max(finite_roots) if finite_roots else None,
        'replay_config': summaries[0].get('replay_config') if summaries else None,
        'womd_source_pattern_override': summaries[0].get('womd_source_pattern_override') if summaries else None,
        'replay_provenance_resolution_counts': dict(sorted(provenance.items())),
        'legacy_provenance_migration_used': any(bool(s.get('legacy_provenance_migration_used')) for s in summaries),
        'resolved_source_index_min': min(source_min) if source_min else None,
        'resolved_source_index_max': max(source_max) if source_max else None,
        'worker_partition': {
            'num_workers': n,
            'worker_indices': sorted(indices),
            'rule': 'source_scenario_index_mod_num_workers',
            'merged': True,
        },
        'performance': {
            'parallel_workers': n,
            'worker_total_seconds': [float(x.get('total_seconds', 0.0)) for x in perf],
            'parallel_wall_proxy_seconds': max([float(x.get('total_seconds', 0.0)) for x in perf], default=0.0),
            'sum_worker_replay_seconds': sum(float(x.get('sample_replay_accumulated_seconds', 0.0)) for x in perf),
            'target_source_indices_sum': sum(int(x.get('target_source_indices', 0)) for x in perf),
            'history_cache_hits': sum(int(x.get('history_cache_hits', 0)) for x in perf),
            'sparse_source_iterator': True,
            'metadata_only_future_metrics_skipped': True,
        },
        'part_sidecars': [str(p.resolve()) for p in a.part],
        'part_summaries': [str(p.resolve()) for p in a.part_summary],
        'canonical_npz_modified': False,
        'planner_parameters_trained': 0,
        'teacher_labels_changed': False,
        'teacher_metadata_input_to_model': False,
        'dataset_reconstruction': False,
        'dataset_reselection': False,
        'test_roots_read': False,
        'purpose': 'merged execution-equivalent parallel replay sidecar; canonical dataset remains untouched',
    }
    a.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'valid': summary['valid'], 'parts': n, 'requested_samples': requested, 'errors': len(errors), 'performance': summary['performance']}), flush=True)
    return 0 if summary['valid'] else 30


if __name__ == '__main__':
    raise SystemExit(main())
