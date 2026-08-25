#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import ocrap
import ocrap.cli.train as tr
import ocrap.models.data as data_mod
import ocrap.models.inference as inference_mod
import ocrap.models.ocrap as model_mod


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    expected_modules = {
        'ocrap': (repo / 'src/ocrap/__init__.py').resolve(),
        'train': (repo / 'src/ocrap/cli/train.py').resolve(),
        'data': (repo / 'src/ocrap/models/data.py').resolve(),
        'model': (repo / 'src/ocrap/models/ocrap.py').resolve(),
        'inference': (repo / 'src/ocrap/models/inference.py').resolve(),
    }
    actual_modules = {
        'ocrap': Path(ocrap.__file__).resolve(),
        'train': Path(tr.__file__).resolve(),
        'data': Path(data_mod.__file__).resolve(),
        'model': Path(model_mod.__file__).resolve(),
        'inference': Path(inference_mod.__file__).resolve(),
    }
    errors: list[str] = []
    for key, expected in expected_modules.items():
        actual = actual_modules[key]
        if actual != expected:
            errors.append(f'imported {key} module mismatch: {actual} != {expected}')

    base = {
        'direct_recovery_absolute_semantic_witness_correction': True,
        'direct_recovery_semantic_witness_route_alignment': True,
        'direct_recovery_semantic_witness_reentry_alignment': True,
        'direct_recovery_semantic_witness_control_projection': True,
        'direct_recovery_semantic_witness_projection_fidelity_weighting': True,
        'direct_recovery_semantic_witness_robust_occupancy': False,
        'direct_recovery_semantic_witness_soft_occupancy_disagreement': True,
    }
    contracts = {}
    for name, demand in [('E70_OCCSOFT', False), ('G70_Main_OCDOTW', True)]:
        c = tr._semantic_witness_checkpoint_feature_contract(
            {**base, 'direct_recovery_semantic_witness_demand_normalized_fidelity': demand}
        )
        contracts[name] = {'schema': c[0], 'source': c[1]}
        if c != (6, 'demand_occupancy_tempered_projected_recovery_witness'):
            errors.append(f'{name} serializer contract mismatch: {c}')

    module_sha256 = {key: sha(path) for key, path in actual_modules.items()}
    valid = not errors
    doc = {
        'schema': 'ocrap-v48.70-runtime-code-contract-v2',
        'valid': valid,
        'repo': str(repo),
        'imported_modules': {k: str(v) for k, v in actual_modules.items()},
        'expected_modules': {k: str(v) for k, v in expected_modules.items()},
        'module_sha256': module_sha256,
        'contracts': contracts,
        'python_executable': sys.executable,
        'errors': errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'event': 'v48_70_runtime_code_contract', 'valid': valid, 'output': str(args.output)}))
    return 0 if valid else 30


if __name__ == '__main__':
    raise SystemExit(main())
