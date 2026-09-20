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
    'deployed_planner_s_per_decision', 'completion_fraction',
    'timing_contract', 'artifact',
)


def _duration(phase: dict) -> float | None:
    try:
        start = datetime.fromisoformat(str(phase['started_at']).replace('Z', '+00:00'))
        end = datetime.fromisoformat(str(phase['ended_at']).replace('Z', '+00:00'))
        return max(0.0, (end - start).total_seconds())
    except (KeyError, ValueError, TypeError):
        return None



def _journal_summary(journal: Path) -> dict | None:
    """Aggregate timing from an append-only scene journal for in-progress jobs."""
    scenes: dict[str, dict] = {}
    try:
        with journal.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                scene = raw.get('scene', raw) if isinstance(raw, dict) else None
                if not isinstance(scene, dict):
                    continue
                target = str(scene.get('target_key') or '').strip()
                if target:
                    key = 'target:' + target
                else:
                    key = 'scene:' + str(scene.get('scene_id') or len(scenes))
                scenes[key] = scene
    except (OSError, ValueError, TypeError):
        return None
    if not scenes:
        return None
    rows = list(scenes.values())
    totals = {part: 0.0 for part in PARTS}
    wall = 0.0
    decisions = 0
    contracts = set()
    for scene in rows:
        decisions += int(scene.get('num_decisions') or 0)
        timing = scene.get('timing') or {}
        wall += float(timing.get('wall_s') or 0.0)
        parts = timing.get('totals_s') or {}
        for part in PARTS:
            totals[part] += float(parts.get(part) or 0.0)
        contracts.add(str(timing.get('execution_contract') or 'unspecified'))
    return {
        'num_scenes': len(rows),
        'num_decisions': decisions,
        'timing': {
            'scene_wall_sum_s': wall,
            'totals_s': totals,
            'execution_contract': ','.join(sorted(contracts)),
        },
    }


def _progress_fraction(artifact: Path, completed: int) -> float | None:
    progress = Path(str(artifact) + '.progress.json')
    try:
        doc = json.loads(progress.read_text(encoding='utf-8'))
        total = int(doc.get('total_rollouts') or doc.get('total') or 0)
        if total > 0:
            return min(1.0, max(0.0, float(completed) / total))
    except (OSError, ValueError, TypeError):
        pass
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
        artifact = variant_root / regime / 'closed_loop_ocrap.json'
        journal = Path(str(artifact) + '.scenes.jsonl')
        try:
            phase = json.loads(phase_path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if artifact.is_file():
            try:
                result = json.loads(artifact.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                continue
            result_source = 'final_artifact'
        else:
            result = _journal_summary(journal)
            if result is None:
                continue
            result_source = 'partial_journal'
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
            'status': (str(phase.get('status') or 'unknown') if result_source == 'final_artifact'
                       else f"{phase.get('status') or 'unknown'}_partial_journal"),
            'scenes': int(result.get('num_scenes') or 0),
            'decisions': int(result.get('num_decisions') or 0),
            'phase_wall_s': elapsed,
            'scene_wall_sum_s': scene,
            'outside_scene_s': outside,
            'unmeasured_inside_scene_s': inside,
            'dominant_measured_component': dominant,
            'outside_scene_fraction': (outside / elapsed if outside is not None and elapsed else None),
            'deployed_planner_s_per_decision': (
                sum(float(parts.get(k, 0.) or 0.) for k in ('state_history','candidate_features','policy_selection'))
                / max(int(result.get('num_decisions') or 0), 1)
            ),
            'completion_fraction': _progress_fraction(artifact, int(result.get('num_scenes') or 0)),
            'timing_contract': timing.get('execution_contract'),
            'artifact': str(artifact if result_source == 'final_artifact' else journal),
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
                'deployed_planner_s_per_decision': (
                    sum(float(parts.get(k, 0.) or 0.) for k in ('state_history','candidate_features','policy_selection'))
                    / max(int(result.get('num_decisions') or 0), 1)
                ),
                'completion_fraction': 1.0,
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
    print(f'[Ablation timing] {len(rows)} final/reused/in-progress result(s) -> {args.output}', flush=True)
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
