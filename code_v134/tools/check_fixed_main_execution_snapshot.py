#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description='Fail closed if immutable V48.124 execution snapshot changed.')
    ap.add_argument('--repo', type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    manifest = repo / 'EXECUTION_SNAPSHOT.json'
    if not manifest.is_file():
        print(json.dumps({'valid': False, 'errors': ['missing_execution_snapshot_manifest']}))
        return 30
    doc = json.loads(manifest.read_text(encoding='utf-8'))
    errors: list[str] = []
    if doc.get('schema') != 'ocrap-v48.124-execution-snapshot-v1' or doc.get('valid') is not True:
        errors.append('snapshot_contract')
    files = doc.get('files') or {}
    for rel, rec in files.items():
        p = repo / rel
        if not p.is_file():
            errors.append(f'missing:{rel}')
            continue
        expected_size = (rec or {}).get('size')
        if expected_size is None or p.stat().st_size != int(expected_size) or sha(p) != str((rec or {}).get('sha256') or ''):
            errors.append(f'sha:{rel}')
    current = {p.relative_to(repo).as_posix() for p in repo.rglob('*') if p.is_file() and p.name != 'EXECUTION_SNAPSHOT.json' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix not in {'.pyc', '.pyo'}}
    expected = set(files)
    extra = sorted(current - expected)
    missing = sorted(expected - current)
    if extra:
        errors.extend(f'extra:{x}' for x in extra)
    if missing:
        errors.extend(f'missing_manifested:{x}' for x in missing)
    print(json.dumps({'valid': not errors, 'errors': errors, 'num_files': len(files), 'tree_sha256': doc.get('tree_sha256')}))
    return 0 if not errors else 30


if __name__ == '__main__':
    raise SystemExit(main())
