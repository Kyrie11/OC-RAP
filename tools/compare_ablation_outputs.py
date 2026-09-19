#!/usr/bin/env python3
"""Fail-closed scientific equivalence check for two completed ablation output roots.

Ignore *only* timing, resume bookkeeping and provenance in the top-level
summary. Compare every scientific scene field exactly, including decisions
when full scene detail is stored. All arms, variants and regimes must match.
Never write to either experiment directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# Keys with scientifically relevant information in the aggregate JSON.
AGGREGATE_KEYS = {
    'num_scenes', 'num_decisions', 'num_metric_steps', 'metrics_valid',
    'runtime_contract', 'waymax_metrics', 'macro_counts',
    'selection_reason_counts', 'active_regime_counts',
    'intervention_episode_count', 'intervention_scene_rate',
    'label_modes', 'gamma_rec',
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=True)


def _scene_key(scene: dict) -> str:
    # Contact can have several targets in one scenario. A scene ID alone is NOT
    # a sufficient identity; the locked target and its frozen time are required.
    return _canonical((scene.get('target_key'), scene.get('scene_id'),
                       scene.get('target_time_index')))


def _scenes(path: Path, summary: dict) -> dict[str, dict]:
    embedded = summary.get('scenes')
    if isinstance(embedded, list):
        records = embedded
    else:
        journal = path.with_suffix(path.suffix + '.scenes.jsonl')
        if not journal.is_file():
            raise ValueError(f'neither embedded scenes nor scene journal is available: {path}')
        records = []
        with journal.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records.append(record.get('scene', record))
    result = {}
    for scene in records:
        if not isinstance(scene, dict):
            raise ValueError(f'non-object scene in {path}')
        key = _scene_key(scene)
        if key in result:
            raise ValueError(f'duplicate target/scene/time key {key} in {path}')
        # The duration/latency of executing a policy is intentionally excluded;
        # every other field, including action and exact evaluation metrics, stays.
        result[key] = {name: value for name, value in scene.items() if name != 'timing'}
    if len(result) != int(summary.get('num_scenes', -1)):
        raise ValueError(f'scene count differs from aggregate in {path}: '
                         f'{len(result)} vs {summary.get("num_scenes")}')
    return result


def _aggregate(summary: dict) -> dict:
    return {key: value for key, value in summary.items()
            if key in AGGREGATE_KEYS or key.startswith(('closed_loop_', 'intervention_', 'macro_'))}


def compare_pair(before: Path, after: Path) -> list[str]:
    b, a = json.loads(before.read_text(encoding='utf-8')), json.loads(after.read_text(encoding='utf-8'))
    differences = []
    b_agg, a_agg = _aggregate(b), _aggregate(a)
    for field in sorted(set(b_agg) | set(a_agg)):
        if _canonical(b_agg.get(field)) != _canonical(a_agg.get(field)):
            differences.append(f'aggregate.{field}')
    b_scene, a_scene = _scenes(before, b), _scenes(after, a)
    if set(b_scene) != set(a_scene):
        differences.append(f'target_set: before={len(b_scene)} after={len(a_scene)}; '
                           f'only_before={len(set(b_scene)-set(a_scene))}; '
                           f'only_after={len(set(a_scene)-set(b_scene))}')
    for key in sorted(set(b_scene) & set(a_scene)):
        old, new = b_scene[key], a_scene[key]
        if _canonical(old) != _canonical(new):
            changed = [field for field in sorted(set(old) | set(new))
                       if _canonical(old.get(field)) != _canonical(new.get(field))]
            differences.append(f'scene={key}: changed={changed[:12]}')
            if len(differences) > 30:
                differences.append('... more differences omitted')
                break
    return differences


def _artifacts(root: Path, include_latency: bool) -> dict[Path, Path]:
    found = {}
    for path in sorted(root.rglob('closed_loop_ocrap.json')):
        relative = path.relative_to(root)
        if not include_latency and 'latency_isolated' in relative.parts:
            continue
        if '_shared_preflight_' in str(relative):
            continue
        found[relative] = path
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before', required=True, type=Path, help='Unmodified original completed OUT_ROOT')
    ap.add_argument('--after', required=True, type=Path, help='Optimized completed OUT_ROOT')
    ap.add_argument('--include-latency', action='store_true',
                    help='Compare scientific outputs in latency replays too (timings excluded)')
    args = ap.parse_args()
    before, after = _artifacts(args.before, args.include_latency), _artifacts(args.after, args.include_latency)
    errors = []
    if not before or not after:
        errors.append(f'empty artifact set: before={len(before)}, after={len(after)}')
    missing_after = sorted(set(before) - set(after))
    missing_before = sorted(set(after) - set(before))
    if missing_after or missing_before:
        errors.append(f'inconsistent job set: only_before={missing_after[:15]}; '
                      f'only_after={missing_before[:15]}')
    for relative in sorted(set(before) & set(after)):
        try:
            differences = compare_pair(before[relative], after[relative])
        except (OSError, ValueError, TypeError) as exc:
            differences = [f'could not compare: {exc}']
        if differences:
            errors.extend(f'{relative}: {text}' for text in differences)
        else:
            print(f'[SAME] {relative}', flush=True)
    if errors:
        print(f'[FAIL] {len(errors)} difference(s); scientific equivalence NOT established:', file=sys.stderr)
        for difference in errors[:50]:
            print(f'  {difference}', file=sys.stderr)
        return 1
    print(f'[PASS] Exact scientific outputs agree for {len(before)} paired artifacts. '
          'Timing, resume bookkeeping and non-scientific run provenance excluded.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
