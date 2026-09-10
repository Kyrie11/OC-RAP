#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ocrap.data.build.diagnose import diagnose_dataset


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return {'_read_error': str(exc), '_path': str(path)}
    return value if isinstance(value, dict) else {'_value': value}


def _manifest_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {'exists': False}
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    def counts(key: str) -> dict[str, int]:
        c = Counter(str(r.get(key, '') or '') for r in rows)
        return {k: int(v) for k, v in sorted(c.items())}
    scene_key = 'original_scenario_id' if 'original_scenario_id' in fields else 'scene_id'
    scenes = {str(r.get(scene_key, '') or '') for r in rows}
    scenes.discard('')
    groups = {
        (str(r.get(scene_key, '') or ''), str(r.get('time_index', '') or ''))
        for r in rows
        if str(r.get(scene_key, '') or '')
    }
    return {
        'exists': True,
        'path': str(path.resolve()),
        'sha256': _sha256(path),
        'rows': len(rows),
        'columns': fields,
        'unique_scenes': len(scenes),
        'unique_scene_time_groups': len(groups),
        'split_counts': counts('split_id') if 'split_id' in fields else {},
        'womd_source_role_counts': counts('womd_source_role') if 'womd_source_role' in fields else {},
        'scenario_id_source_counts': counts('scenario_id_source') if 'scenario_id_source' in fields else {},
    }


def _json_artifact(path: Path) -> dict[str, Any]:
    return {
        'exists': path.is_file(),
        'path': str(path.resolve()),
        'sha256': _sha256(path),
        'content': _read_json(path),
    }


def _construction_snapshot(root: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    resume_path = root / 'resume_contract.json'
    summary_path = root / 'dataset_summary.json'
    resume = _read_json(resume_path)
    summary = _read_json(summary_path)
    semantic = (resume or {}).get('semantic_config') if isinstance(resume, dict) else None
    cfg = semantic if isinstance(semantic, dict) else None
    snap = {
        'resume_contract': _json_artifact(resume_path),
        'dataset_summary': _json_artifact(summary_path),
        'merged_dataset_summary': _json_artifact(root / 'merged_dataset_summary.json'),
        'scene_filter_provenance': _json_artifact(root / 'scene_filter_provenance.json'),
        'womd_replay_contract': _json_artifact(root / 'womd_replay_contract.json'),
        'manifest_repair_report': _json_artifact(root / 'manifest_repair_v48.json'),
        'manifest': _manifest_summary(root / 'manifest.csv'),
    }
    if isinstance(cfg, dict):
        snap['reconstructed_build_parameters'] = {
            'evidence_class': 'resume_contract.semantic_config',
            'womd_patterns': cfg.get('womd_patterns'),
            'data_source': cfg.get('data_source'),
            'simulation_backend': cfg.get('simulation_backend'),
            'split': cfg.get('split'),
            'num_candidate_prefixes': cfg.get('num_candidate_prefixes'),
            'num_reactive_futures': cfg.get('num_reactive_futures'),
            'num_targeted_futures': cfg.get('num_targeted_futures'),
            'targeted_future_kinds': cfg.get('targeted_future_kinds'),
            'num_roots': cfg.get('num_roots'),
            'num_recovery_options': cfg.get('num_recovery_options'),
            'max_times_per_scenario': cfg.get('max_times_per_scenario'),
            'max_biased_times_per_scenario': cfg.get('max_biased_times_per_scenario'),
            # These top-level scan controls are intentionally excluded from the
            # resume semantic fingerprint in current builders. Keep the fields
            # here for legacy contracts that may contain them, but do not silently
            # claim exact recovery when they are absent.
            'scenario_start_index': cfg.get('scenario_start_index'),
            'scenario_stride': cfg.get('scenario_stride'),
            'scenario_worker_index': cfg.get('scenario_worker_index'),
            'max_scenarios': cfg.get('max_scenarios'),
            'dataset_quality': cfg.get('dataset_quality'),
            'artifact': cfg.get('artifact'),
            'waymax': cfg.get('waymax'),
            'regime_thresholds': cfg.get('regime_thresholds'),
            'ocmero': cfg.get('ocmero'),
        }
    snap['parameter_recovery_notes'] = {
        'semantic_config_status': (
            'exact_dataset_resume_contract' if isinstance(cfg, dict) else 'missing_or_legacy'
        ),
        'scan_scope_status': (
            'dataset_summary_evidence_available' if isinstance(summary, dict) else 'missing_or_legacy'
        ),
        'important_limitation': (
            'Current builders intentionally exclude top-level max_scenarios/scenario_start_index/'
            'scenario_stride/scenario_worker_index from resume semantic fingerprints. Recover those '
            'from dataset_summary/merge/filter provenance when available; otherwise report unknown '
            'rather than guessing.'
        ),
    }
    if isinstance(summary, dict):
        snap['scan_scope_evidence'] = {
            'evidence_class': 'dataset_summary',
            'raw_scenarios_seen': summary.get('raw_scenarios_seen'),
            'scenario_start_index': summary.get('scenario_start_index'),
            'scenario_stride': summary.get('scenario_stride'),
            'scenario_worker_index': summary.get('scenario_worker_index'),
            'source_max_scenarios': summary.get('source_max_scenarios'),
            'scene_time_groups': summary.get('scene_time_groups'),
            'unique_raw_scene_ids': summary.get('unique_raw_scene_ids'),
            'generation': summary.get('generation'),
            'dataset_quality': summary.get('dataset_quality'),
            'artifact': summary.get('artifact'),
            'waymax': summary.get('waymax'),
            'regime_thresholds': summary.get('regime_thresholds'),
        }
    return snap, cfg


def main() -> int:
    ap = argparse.ArgumentParser(description='Read-only OC-RAP dataset property + construction-provenance audit.')
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-samples', type=int, default=None,
                    help='Optional sample cap for a quick scan. Omit for paper-grade full scan.')
    args = ap.parse_args()

    root = args.dataset.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f'dataset directory not found: {root}')
    construction, cfg = _construction_snapshot(root)
    diagnostics = diagnose_dataset(root, output=None, max_samples=args.max_samples, cfg=cfg)
    out = {
        'schema': 'ocrap-dataset-properties-audit-v1',
        'dataset': str(root),
        'read_only': True,
        'sample_scan': 'full' if args.max_samples is None else f'first_{args.max_samples}',
        'construction': construction,
        'diagnostics': diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'dataset': root.name,
        'samples': diagnostics.get('num_samples'),
        'scenes': diagnostics.get('num_scenes'),
        'failures': len(diagnostics.get('failures') or []),
        'warnings': len(diagnostics.get('warnings') or []),
        'womd_source_role_counts': (construction.get('manifest') or {}).get('womd_source_role_counts', {}),
        'output': str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
