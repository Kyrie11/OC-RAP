#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

EXCLUDED_DIRS = {'.git', '.pytest_cache', '__pycache__', 'runs'}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ignored(_dir: str, names: list[str]) -> set[str]:
    out: set[str] = set()
    for name in names:
        if name in EXCLUDED_DIRS or Path(name).suffix in EXCLUDED_SUFFIXES:
            out.add(name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='Create immutable source snapshot for V48.124 fixed-Main execution.')
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()

    repo = args.repo.resolve()
    out = args.output.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo, out, ignore=ignored)

    files: dict[str, dict[str, object]] = {}
    for p in sorted(x for x in out.rglob('*') if x.is_file() and x.name != 'EXECUTION_SNAPSHOT.json'):
        rel = p.relative_to(out).as_posix()
        files[rel] = {'sha256': sha(p), 'size': p.stat().st_size}
    digest = hashlib.sha256()
    for rel, rec in files.items():
        digest.update(rel.encode('utf-8')); digest.update(b'\0')
        digest.update(str(rec['sha256']).encode('ascii')); digest.update(b'\0')
        digest.update(str(rec['size']).encode('ascii')); digest.update(b'\n')
    doc = {
        'schema': 'ocrap-v48.124-execution-snapshot-v1',
        'run_instance_id': args.run_id,
        'source_repo': str(repo),
        'snapshot_repo': str(out),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'num_files': len(files),
        'tree_sha256': digest.hexdigest(),
        'files': files,
        'valid': True,
    }
    (out / 'EXECUTION_SNAPSHOT.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'valid': True, 'snapshot_repo': str(out), 'tree_sha256': doc['tree_sha256'], 'num_files': len(files)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
