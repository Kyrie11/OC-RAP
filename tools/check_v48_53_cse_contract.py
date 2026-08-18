#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _bool_text(x: str) -> bool:
    s = str(x).strip().lower()
    if s in {'1', 'true', 'yes', 'on'}:
        return True
    if s in {'0', 'false', 'no', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'expected boolean, got {x!r}')


def main() -> int:
    ap = argparse.ArgumentParser(description='Fail-closed v48.53 CSE witness contract check.')
    ap.add_argument('--run', type=Path, required=True, help='Variant run root, e.g. candidates/balanced')
    ap.add_argument('--expect-teacher-physical', type=_bool_text, required=True)
    ap.add_argument('--expect-student-physical', type=_bool_text, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    stage_root = args.run / 'v48_47_recovery_frontier'
    stage_path = stage_root / 'V48_47_WITNESS_STAGE.json'
    complete_path = stage_root / 'V48_47_WITNESS_COMPLETE.json'
    ckpt_path = stage_root / 'model_v48_47_witness' / 'best.pt'
    errors: list[str] = []
    stage: dict[str, Any] = {}
    complete: dict[str, Any] = {}
    training: dict[str, Any] = {}

    for p, name in ((stage_path, 'stage_contract'), (complete_path, 'stage_complete'), (ckpt_path, 'witness_checkpoint')):
        if not p.is_file():
            errors.append(f'missing_{name}:{p}')

    if stage_path.is_file():
        try:
            stage = json.loads(stage_path.read_text(encoding='utf-8'))
        except Exception as exc:
            errors.append(f'stage_contract_unreadable:{exc!r}')
    if complete_path.is_file():
        try:
            complete = json.loads(complete_path.read_text(encoding='utf-8'))
        except Exception as exc:
            errors.append(f'stage_complete_unreadable:{exc!r}')
    if ckpt_path.is_file():
        try:
            payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            cfg = payload.get('cfg') if isinstance(payload, dict) else None
            cfg = cfg if isinstance(cfg, dict) else {}
            training = cfg.get('training') if isinstance(cfg.get('training'), dict) else {}
        except Exception as exc:
            errors.append(f'witness_checkpoint_unreadable:{exc!r}')

    teacher_expected = bool(args.expect_teacher_physical)
    student_expected = bool(args.expect_student_physical)
    teacher_coordinate = (
        'q_selected_mstar_physical_drs_exact_pcd' if teacher_expected else 'q_hard_proxy_drs_exact_pcd'
    )
    student_coordinate = (
        'q_selected_predicted_margin_physical_drs_exact_pcd'
        if student_expected else 'hard_qbest_ge_zero_root_mass_exact_pcd'
    )
    checks = {
        'stage_frontier': stage.get('stage') == 'frontier',
        'option_execution_observation_class': stage.get('option_execution_semantics') == 'observation_class',
        'boundary_complete_frontier': stage.get('boundary_complete_frontier') is True,
        'old_decision_equivalent_frontier_disabled': stage.get('decision_equivalent_frontier') is False,
        'physical_teacher_sign_alignment': bool(stage.get('physical_teacher_sign_alignment', False)) is teacher_expected,
        'physical_student_sign_alignment': bool(stage.get('physical_student_sign_alignment', False)) is student_expected,
        'teacher_sign_coordinate': stage.get('teacher_sign_coordinate') == teacher_coordinate,
        'student_sign_coordinate': stage.get('student_sign_coordinate') == student_coordinate,
        'frontier_order_coordinate': stage.get('frontier_order_coordinate') == 'smooth_boundary_drs_smooth_pcd',
        'native_downstream_transports_isolated': (
            stage.get('native_certificate_preservation') is False
            and stage.get('native_margin_complete_preservation') is False
            and stage.get('native_advantage_preservation') is False
            and stage.get('native_exact_advantage_preservation') is False
            and stage.get('native_boundary_complete_advantage_preservation') is False
        ),
        'checkpoint_boundary_complete_frontier': bool(training.get('recovery_frontier_boundary_complete', False)) is True,
        'checkpoint_physical_teacher_sign_alignment': bool(training.get('recovery_frontier_physical_teacher_sign_alignment', False)) is teacher_expected,
        'checkpoint_physical_student_sign_alignment': bool(training.get('recovery_frontier_physical_student_sign_alignment', False)) is student_expected,
        'checkpoint_option_execution_observation_class': training.get('option_execution_semantics') == 'observation_class',
    }
    for name, ok in checks.items():
        if not ok:
            errors.append(f'failed:{name}')

    if ckpt_path.is_file() and complete:
        actual_sha = _sha256(ckpt_path)
        if complete.get('checkpoint_sha256') != actual_sha:
            errors.append('witness_checkpoint_sha_mismatch')
    else:
        actual_sha = None

    doc = {
        'event': 'v48_53_cse_contract',
        'version': 'v48.53-DCP-DRFC-BCDE-CSE',
        'run': str(args.run),
        'expect_physical_teacher_sign_alignment': teacher_expected,
        'expect_physical_student_sign_alignment': student_expected,
        'checks': checks,
        'teacher_sign_coordinate': stage.get('teacher_sign_coordinate'),
        'student_sign_coordinate': stage.get('student_sign_coordinate'),
        'frontier_order_coordinate': stage.get('frontier_order_coordinate'),
        'witness_checkpoint': str(ckpt_path),
        'witness_checkpoint_sha256': actual_sha,
        'valid': not errors,
        'errors': errors,
        'strategy_regime_conditioning': False,
        'test_roots_read': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return 0 if not errors else 4


if __name__ == '__main__':
    raise SystemExit(main())
