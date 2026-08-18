#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description='Authorize byte/protocol-compatible reuse of v48.51-B as the v48.52 A reference.')
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--protocol-seal', type=Path, required=True)
    ap.add_argument('--source-run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    errors: list[str] = []
    required = {
        'status': args.run / 'AUTHORITATIVE_RUN_STATUS.json',
        'factor': args.run / 'V48_51_FACTOR_CONTRACT.json',
        'attempt': args.run / 'ATTEMPT_STARTED.json',
        'source': args.run / 'SOURCE_CHECKPOINT_CONTRACT.json',
        'gate': args.run / 'GATE_SPEC.json',
    }
    docs: dict[str, dict[str, Any]] = {}
    for name, path in required.items():
        if not path.is_file():
            errors.append(f'missing_{name}:{path}')
            continue
        try:
            docs[name] = _json(path)
        except Exception as exc:
            errors.append(f'unreadable_{name}:{exc!r}')

    if not args.protocol_seal.is_file():
        errors.append(f'missing_current_protocol_seal:{args.protocol_seal}')
        current_protocol_sha = None
    else:
        current_protocol_sha = _sha256(args.protocol_seal)

    current_source_sha: dict[str, str | None] = {}
    for variant in ('balanced', 'precision'):
        ckpt = args.source_run / 'candidates' / variant / 'model_v48_trac_sr' / 'best.pt'
        if not ckpt.is_file():
            errors.append(f'missing_current_source_{variant}:{ckpt}')
            current_source_sha[variant] = None
        else:
            current_source_sha[variant] = _sha256(ckpt)

    status = docs.get('status', {})
    checks = status.get('checks') or {}
    if not (
        status.get('valid') is True
        and status.get('pipeline_valid') is True
        and int(status.get('authoritative_exit_code', 99)) in (0, 20)
        and checks.get('certificate_executed', status.get('certificate_executed', True)) is True
        and checks.get('gate_evaluated', status.get('gate_evaluated', True)) is True
        and status.get('test_roots_read') is False
    ):
        errors.append('historical_status_not_authoritative_pipeline_valid_rc0_or20')

    factor = docs.get('factor', {})
    factor_ok = (
        factor.get('version') == 'v48.51-DCP-DRFC-BCDE'
        and factor.get('arm') == 'B'
        and factor.get('training_option_execution_semantics') == 'observation_class'
        and factor.get('evaluation_option_execution_semantics') == 'observation_class'
        and factor.get('native_certificate_preservation') is True
        and factor.get('recovery_frontier_calibration') is True
        and factor.get('native_margin_complete_preservation') is False
        and factor.get('native_advantage_preservation') is True
        and factor.get('decision_equivalent_frontier') is False
        and factor.get('boundary_complete_frontier') is True
        and factor.get('native_exact_advantage_preservation') is False
        and factor.get('native_boundary_complete_advantage_preservation') is False
        and factor.get('new_tuned_thresholds') is False
        and factor.get('strategy_regime_conditioning') is False
        and factor.get('test_roots_read') is False
        and int(factor.get('proposal_top_k', -1)) == 5
    )
    if not factor_ok:
        errors.append('historical_factor_contract_is_not_v48_51_B_reference')

    attempt = docs.get('attempt', {})
    if current_protocol_sha is not None and attempt.get('protocol_seal_sha256') != current_protocol_sha:
        errors.append('protocol_seal_sha_mismatch')
    if attempt.get('test_roots_read') is not False:
        errors.append('historical_attempt_test_roots_read')

    source = docs.get('source', {})
    source_checks = source.get('checks') or {}
    if source.get('valid') is not True or source.get('test_roots_read') is not False:
        errors.append('historical_source_contract_invalid')
    for variant in ('balanced', 'precision'):
        hist = (source_checks.get(variant) or {}).get('sha256')
        if current_source_sha.get(variant) is not None and hist != current_source_sha[variant]:
            errors.append(f'source_checkpoint_sha_mismatch:{variant}')

    gate = docs.get('gate', {})
    protocol = gate.get('protocol') or {}
    if protocol.get('strategy_regime_conditioning') is not False:
        errors.append('historical_gate_regime_conditioning')
    if protocol.get('test_roots_read') is not False:
        errors.append('historical_gate_test_roots_read')
    policy = protocol.get('policy') or {}
    if policy.get('option_execution_semantics') != 'observation_class' or int(policy.get('proposal_top_k', -1)) != 5:
        errors.append('historical_gate_policy_semantics_mismatch')

    witness_checks: dict[str, Any] = {}
    for variant in ('balanced', 'precision'):
        stage_path = args.run / 'candidates' / variant / 'v48_47_recovery_frontier' / 'V48_47_WITNESS_STAGE.json'
        if not stage_path.is_file():
            errors.append(f'missing_historical_witness_stage:{variant}')
            witness_checks[variant] = None
            continue
        try:
            stage = _json(stage_path)
        except Exception as exc:
            errors.append(f'unreadable_historical_witness_stage:{variant}:{exc!r}')
            witness_checks[variant] = None
            continue
        ok = (
            stage.get('stage') == 'frontier'
            and stage.get('option_execution_semantics') == 'observation_class'
            and stage.get('boundary_complete_frontier') is True
            and stage.get('decision_equivalent_frontier') is False
            and bool(stage.get('physical_teacher_sign_alignment', False)) is False
        )
        witness_checks[variant] = {
            'valid_reference_stage': ok,
            'boundary_complete_frontier': stage.get('boundary_complete_frontier'),
            'physical_teacher_sign_alignment': bool(stage.get('physical_teacher_sign_alignment', False)),
        }
        if not ok:
            errors.append(f'historical_witness_stage_not_qproxy_bcfc:{variant}')

    out = {
        'event': 'v48_52_reference_reuse_contract',
        'version': 'v48.52-DCP-DRFC-BCDE-PSA',
        'reference_run': str(args.run.resolve(strict=False)),
        'reference_version': 'v48.51-B-BC-FC',
        'current_protocol_seal_sha256': current_protocol_sha,
        'current_source_checkpoint_sha256': current_source_sha,
        'historical_authoritative_exit_code': status.get('authoritative_exit_code'),
        'witness_checks': witness_checks,
        'valid': not errors,
        'errors': errors,
        'strategy_regime_conditioning': False,
        'test_roots_read': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return 0 if not errors else 4


if __name__ == '__main__':
    raise SystemExit(main())
