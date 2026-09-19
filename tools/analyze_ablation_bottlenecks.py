#!/usr/bin/env python3
"""Report measured per-ablation time without touching any scientific artifact.

Separate unmeasured scene time (e.g. preparation/contact prelude) from job time
outside scenes (checkpoint, WOMD scan, JAX compilation, journaling). Neither is
misrepresented as an isolated planner-latency measurement. Resume/reuse phases
are labeled and their phase duration is *not* interpreted as a fresh full run.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

PARTS = (
    'state_history', 'candidate_features', 'policy_selection',
    'teacher_labels', 'audit_labels', 'waymax_step_metrics',
)
FIELDS = (
    'arm', 'variant', 'regime', 'run_kind', 'status', 'scenes', 'decisions',
    'phase_wall_s', 'scene_wall_sum_s', 'outside_scene_s',
    'unmeasured_inside_scene_s', *[f'{p}_s' for p in PARTS],
    'dominant_measured_component', 'outside_scene_fraction',
    'timing_contract', 'artifact',
)


def _duration(phase: dict) -> float | None:
    try:
        start = datetime.fromisoformat(str(phase['started_at']).replace('Z', '+00:00'))
        end = datetime.fromisoformat(str(phase['ended_at']).replace('Z', '+00:00'))
        return max(0.0, (end - start).total_seconds())
    except (KeyError, ValueError, TypeError):
        return None


def collect(root: Path, full_root: Path, latency_root: Path | None = None) -> list[dict]:
    rows = []
    phase_paths = set(root.glob('*/**/*.phase.json')) | set(full_root.glob('*/*.phase.json'))
    for phase_path in sorted(phase_paths):
        regime = phase_path.name.removesuffix('.phase.json')
        variant_root = phase_path.parent
        arm = variant_root.parent.name
        if arm in {'latency_isolated', '_shared_preflight_submission_ablation_metrics'}:
            continue
        if not (variant_root / regime / 'closed_loop_ocrap.json').is_file():
            continue
        artifact = variant_root / regime / 'closed_loop_ocrap.json'
        try:
            phase = json.loads(phase_path.read_text(encoding='utf-8'))
            result = json.loads(artifact.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        timing = result.get('timing') or {}
        parts = timing.get('totals_s') or {}
        scene = float(timing.get('scene_wall_sum_s') or 0.)
        elapsed = _duration(phase)
        # A reused artifact's phase covers a fast validation, not its execution.
        outside = None if elapsed is None or elapsed < scene else elapsed - scene
        inside = max(0., scene - sum(float(v or 0.) for v in parts.values()))
        dominant = max(PARTS, key=lambda k: float(parts.get(k, 0.) or 0.))
        row = {
            'arm': '_native_full_reference' if variant_root.parent == full_root else arm,
            'variant': variant_root.name,
            'regime': regime,
            'run_kind': 'accuracy',
            'status': phase.get('status'),
            'scenes': int(result.get('num_scenes') or 0),
            'decisions': int(result.get('num_decisions') or 0),
            'phase_wall_s': elapsed,
            'scene_wall_sum_s': scene,
            'outside_scene_s': outside,
            'unmeasured_inside_scene_s': inside,
            'dominant_measured_component': dominant,
            'outside_scene_fraction': (outside / elapsed if outside is not None and elapsed else None),
            'timing_contract': timing.get('execution_contract'),
            'artifact': str(artifact),
        }
        row.update({f'{part}_s': float(parts.get(part, 0.) or 0.) for part in PARTS})
        rows.append(row)
    # Isolated latency runs have no accuracy phase JSON. Their own timing fields
    # remain useful, but an outside-scene wall estimate would be fabricated.
    if latency_root is not None and latency_root.is_dir():
        for artifact in sorted(latency_root.rglob('closed_loop_ocrap.json')):
            relative = artifact.relative_to(latency_root)
            if len(relative.parts) != 4:
                continue
            arm, variant, regime, _ = relative.parts
            try:
                result = json.loads(artifact.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            timing = result.get('timing') or {}
            parts = timing.get('totals_s') or {}
            scene = float(timing.get('scene_wall_sum_s') or 0.)
            row = {
                'arm': arm, 'variant': variant, 'regime': regime,
                'run_kind': 'isolated_latency', 'status': 'artifact_present',
                'scenes': int(result.get('num_scenes') or 0),
                'decisions': int(result.get('num_decisions') or 0),
                'phase_wall_s': None,
                'scene_wall_sum_s': scene, 'outside_scene_s': None,
                'unmeasured_inside_scene_s': max(0., scene - sum(float(v or 0.) for v in parts.values())),
                'dominant_measured_component': max(PARTS, key=lambda k: float(parts.get(k, 0.) or 0.)),
                'outside_scene_fraction': None,
                'timing_contract': timing.get('execution_contract'),
                'artifact': str(artifact),
            }
            row.update({f'{part}_s': float(parts.get(part, 0.) or 0.) for part in PARTS})
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--full-root', type=Path)
    parser.add_argument('--latency-root', type=Path,
                        help='Also include isolated latency artifacts without fabricating phase wall time')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    full = args.full_root or args.root / '_native_full_reference'
    rows = collect(args.root, full, args.latency_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'[Ablation timing] {len(rows)} completed/reused artifact(s) -> {args.output}', flush=True)
    if rows:
        groups = {}
        for row in rows:
            bucket = (row['run_kind'], row['regime'])
            totals = groups.setdefault(bucket, {key: 0. for key in PARTS})
            for key in PARTS:
                totals[key] += row[f'{key}_s']
        for (kind, regime), parts in sorted(groups.items()):
            print(f'[Ablation timing] {kind}/{regime} measured components (sum over jobs): '
                  + ', '.join(f'{key}={value:.1f}s' for key, value in sorted(
                      parts.items(), key=lambda kv: -kv[1])), flush=True)


if __name__ == '__main__':
    main()
