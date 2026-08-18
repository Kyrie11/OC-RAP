#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _identity(run: Path) -> dict[str, Any]:
    attempt=_json(run/'ATTEMPT_STARTED.json'); source=_json(run/'SOURCE_CHECKPOINT_CONTRACT.json'); gate=_json(run/'GATE_SPEC.json')
    checks=source.get('checks') or {}
    protocol=gate.get('protocol') or {}
    return {
        'protocol_seal_sha256':attempt.get('protocol_seal_sha256'),
        'source_checkpoint_sha256':{v:(checks.get(v) or {}).get('sha256') for v in ('balanced','precision')},
        'gate_protocol':protocol,
    }


def main() -> int:
    ap=argparse.ArgumentParser(description='Fail-closed reuse of v48.52 A/B as v48.53 CSE X-axis references.')
    ap.add_argument('--a', type=Path, required=True)
    ap.add_argument('--b', type=Path, required=True)
    ap.add_argument('--protocol-seal', type=Path, required=True)
    ap.add_argument('--source-run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    errors: list[str]=[]
    current_protocol_sha=_sha256(args.protocol_seal) if args.protocol_seal.is_file() else None
    if current_protocol_sha is None: errors.append('missing_current_protocol_seal')
    current_source={}
    for v in ('balanced','precision'):
        p=args.source_run/'candidates'/v/'model_v48_trac_sr'/'best.pt'
        current_source[v]=_sha256(p) if p.is_file() else None
        if current_source[v] is None: errors.append(f'missing_current_source:{v}')

    docs={}
    identities={}
    for name, run in (('A',args.a),('B',args.b)):
        try:
            status=_json(run/'AUTHORITATIVE_RUN_STATUS.json')
            factor=_json(run/'V48_52_FACTOR_CONTRACT.json')
            ident=_identity(run)
        except Exception as exc:
            errors.append(f'unreadable_{name}:{exc!r}'); continue
        identities[name]=ident
        checks=status.get('checks') or {}
        if not (status.get('pipeline_valid') is True and int(status.get('authoritative_exit_code',99)) in (0,20)
                and checks.get('certificate_executed',status.get('certificate_executed',True)) is True
                and checks.get('gate_evaluated',status.get('gate_evaluated',True)) is True
                and status.get('test_roots_read') is False):
            errors.append(f'{name}_status_invalid')
        teacher=(name=='B')
        factor_ok=(
            factor.get('version')=='v48.52-DCP-DRFC-BCDE-PSA'
            and factor.get('arm')==name
            and bool(factor.get('physical_teacher_sign_alignment',False)) is teacher
            and factor.get('boundary_complete_frontier') is True
            and factor.get('native_advantage_preservation') is True
            and factor.get('native_exact_advantage_preservation') is False
            and factor.get('native_boundary_complete_advantage_preservation') is False
            and factor.get('student_sign_coordinate')=='hard_qbest_ge_zero_root_mass_exact_pcd'
            and factor.get('strategy_regime_conditioning') is False
            and factor.get('test_roots_read') is False
            and int(factor.get('proposal_top_k',-1))==5
        )
        if not factor_ok: errors.append(f'{name}_factor_invalid')
        if current_protocol_sha is not None and ident.get('protocol_seal_sha256')!=current_protocol_sha:
            errors.append(f'{name}_protocol_seal_sha_mismatch')
        for v in ('balanced','precision'):
            if current_source[v] is not None and ident.get('source_checkpoint_sha256',{}).get(v)!=current_source[v]:
                errors.append(f'{name}_source_checkpoint_sha_mismatch:{v}')
        protocol=ident.get('gate_protocol') or {}; policy=protocol.get('policy') or {}
        if protocol.get('strategy_regime_conditioning') is not False or protocol.get('test_roots_read') is not False:
            errors.append(f'{name}_gate_seal_invalid')
        if policy.get('option_execution_semantics')!='observation_class' or int(policy.get('proposal_top_k',-1))!=5:
            errors.append(f'{name}_policy_semantics_invalid')
        witness={}
        for v in ('balanced','precision'):
            p=run/'candidates'/v/'v48_47_recovery_frontier'/'V48_47_WITNESS_STAGE.json'
            if not p.is_file(): errors.append(f'{name}_missing_witness:{v}'); continue
            d=_json(p); expected_teacher=teacher
            ok=(d.get('boundary_complete_frontier') is True and d.get('decision_equivalent_frontier') is False
                and d.get('option_execution_semantics')=='observation_class'
                and bool(d.get('physical_teacher_sign_alignment',False)) is expected_teacher
                and bool(d.get('physical_student_sign_alignment',False)) is False)
            witness[v]={'valid':ok,'physical_teacher_sign_alignment':bool(d.get('physical_teacher_sign_alignment',False)),
                        'physical_student_sign_alignment':bool(d.get('physical_student_sign_alignment',False))}
            if not ok: errors.append(f'{name}_witness_invalid:{v}')
        docs[name]={'run':str(run.resolve(strict=False)),'authoritative_exit_code':status.get('authoritative_exit_code'),'factor':factor,'witness_checks':witness}

    if identities.get('A') and identities.get('B') and identities['A']!=identities['B']:
        errors.append('A_B_identity_mismatch')
    out={'event':'v48_53_ab_reference_reuse_contract','version':'v48.53-DCP-DRFC-BCDE-CSE',
         'references':docs,'current_protocol_seal_sha256':current_protocol_sha,'current_source_checkpoint_sha256':current_source,
         'valid':not errors,'errors':errors,'strategy_regime_conditioning':False,'test_roots_read':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return 0 if not errors else 4

if __name__=='__main__': raise SystemExit(main())
