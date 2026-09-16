#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.config.yaml_io import load_config
from ocrap.data.waymax_loader import iter_waymax_womd_scenarios_selected
from ocrap.simulation.closed_loop_runner import _load_closed_loop_targets, _source_role_from_pattern


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description='Freeze the observation-legal target cohort before any planner is run.')
    ap.add_argument('--bucket-dataset', required=True)
    ap.add_argument('--womd-pattern', required=True)
    ap.add_argument('--bucket-split', default='test')
    ap.add_argument('--max-targets-per-scene', type=int, default=1)
    ap.add_argument('--max-bucket-targets', type=int, default=0)
    ap.add_argument('--expected-womd-role', default='validation')
    ap.add_argument('--config', default=None)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault('closed_loop', {})
    cfg['closed_loop'].update({
        'bucket_split': args.bucket_split,
        'max_targets_per_scene': int(args.max_targets_per_scene),
        'max_bucket_targets': int(args.max_bucket_targets),
        'target_keys_file': '',
        'require_target_keys': False,
        'require_observation_legal_route': True,
    })
    cfg.setdefault('waymax', {})
    cfg['waymax'].update({
        'dataloader_include_sdc_paths': True,
        'allow_logged_sdc_route_fallback': False,
        'retain_official_scenario_id': True,
    })
    cfg['scenario_start_index'] = 0
    cfg['scenario_stride'] = 1
    cfg['_selected_replay_progress_every'] = int(cfg.get('_selected_replay_progress_every', 0) or 0)

    targets = _load_closed_loop_targets(args.bucket_dataset, cfg)
    if not targets:
        raise SystemExit('no bucket targets loaded')
    source_role = _source_role_from_pattern(args.womd_pattern)
    if args.expected_womd_role and args.expected_womd_role != 'auto' and source_role != args.expected_womd_role:
        raise SystemExit(f'WOMD role mismatch: expected {args.expected_womd_role}, got {source_role}')
    source_indices = [int(t['source_scenario_index']) for t in targets]
    if any(i < 0 for i in source_indices):
        raise SystemExit('all final locked targets must have nonnegative source_scenario_index provenance')

    diag: dict[str, Any] = {}
    yielded: set[int] = set()
    for raw in iter_waymax_womd_scenarios_selected(
        args.womd_pattern,
        source_indices,
        parser_cfg=cfg,
        skip_observation_legal_route_unavailable=True,
        eligibility_diagnostics=diag,
    ):
        yielded.add(int((raw.metadata or {}).get('_waymax_scenario_index', -1)))

    excluded_rows = list(diag.get('route_ineligible', []) or [])
    excluded_indices = {
        int(r['source_scenario_index']) for r in excluded_rows
        if r.get('source_scenario_index') is not None and int(r['source_scenario_index']) >= 0
    }
    requested_indices = set(source_indices)
    unresolved = sorted(requested_indices - yielded - excluded_indices)
    if unresolved:
        raise SystemExit(f'final target lock scan did not resolve {len(unresolved)} source indices; first={unresolved[:10]}')

    eligible = [t for t in targets if int(t['source_scenario_index']) not in excluded_indices]
    excluded_targets = [t for t in targets if int(t['source_scenario_index']) in excluded_indices]
    doc = {
        'schema': 'ocrap-observation-legal-target-lock-v1',
        'status': 'OBSERVATION_LEGAL_TARGET_LOCK_COMPLETE',
        'bucket_dataset': str(Path(args.bucket_dataset).resolve()),
        'bucket_split': args.bucket_split,
        'womd_pattern': args.womd_pattern,
        'womd_source_role': source_role,
        'route_contract': {
            'route_source': 'womd_v1_3_1_sdc_paths_connectivity_only',
            'require_observation_legal_route': True,
            'allow_logged_sdc_route_fallback': False,
            'eligibility_policy': 'strict_observation_legal_target_exclusion_v1',
            'exclusion_is_method_independent': True,
        },
        'requested_target_count': len(targets),
        'num_target_keys': len(eligible),
        'target_keys': [str(t['target_key']) for t in eligible],
        'excluded_target_count': len(excluded_targets),
        'excluded_target_keys': [str(t['target_key']) for t in excluded_targets],
        'excluded_route_details': excluded_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'observation_legal_target_lock',
        'output': str(args.output),
        'requested': len(targets),
        'eligible': len(eligible),
        'excluded': len(excluded_targets),
        'sha256': _sha256(args.output),
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
