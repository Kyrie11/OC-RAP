#!/usr/bin/env python3
"""Fail-closed reuse of an identical v48.47 witness stage.

Used only to avoid recomputing the B observation stage in D/Main.  Reuse is
accepted iff source checkpoint, input/group contracts, and all decision-observation
hyperparameters match exactly.  The copied checkpoint bytes are therefore the exact
B artifact, not a retrained approximation.
"""
from __future__ import annotations
import argparse, hashlib, json, math, shutil
from pathlib import Path


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def close(a: object, b: float) -> bool:
    try: return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
    except Exception: return False


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-run', type=Path, required=True)
    ap.add_argument('--destination-run', type=Path, required=True)
    ap.add_argument('--expected-source', type=Path, required=True)
    ap.add_argument('--expected-stage', required=True)
    ap.add_argument('--expected-train-mix', required=True)
    ap.add_argument('--expected-val-mix', required=True)
    ap.add_argument('--expected-group-index', type=Path, required=True)
    ap.add_argument('--expected-epochs', type=int, required=True)
    ap.add_argument('--expected-obs-loss-weight', type=float, required=True)
    ap.add_argument('--expected-conflict-scale', type=float, required=True)
    ap.add_argument('--expected-conflict-temperature', type=float, required=True)
    ap.add_argument('--expected-max-weight', type=float, required=True)
    args=ap.parse_args()
    src=args.source_run.resolve(); dst=args.destination_run.resolve()
    complete=src/'V48_47_WITNESS_COMPLETE.json'; contract=src/'V48_47_WITNESS_STAGE.json'
    ckpt=src/'model_v48_47_witness'/'best.pt'
    iso=src/'V48_47_STAGE_ISOLATION_CONTRACT.json'
    required=(complete,contract,ckpt,iso,args.expected_source,args.expected_group_index)
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise SystemExit('v48.47 witness reuse missing files: '+', '.join(missing))
    d=json.loads(complete.read_text()); c=json.loads(contract.read_text())
    checks={
      'stage': d.get('stage')==args.expected_stage==c.get('stage'),
      'source_sha': d.get('source_sha256')==sha(args.expected_source),
      'checkpoint_sha': d.get('checkpoint_sha256')==sha(ckpt),
      'semantics': d.get('option_execution_semantics')=='observation_class' and c.get('option_execution_semantics')=='observation_class',
      'no_regime': c.get('strategy_regime_conditioning') is False and d.get('strategy_regime_conditioning') is False,
      'no_test': c.get('test_roots_read') is False and d.get('test_roots_read') is False,
      'train_mix': c.get('train_mix')==args.expected_train_mix,
      'val_mix': c.get('val_mix')==args.expected_val_mix,
      'group_sha': c.get('group_index_sha256')==sha(args.expected_group_index),
      'epochs': int(c.get('epochs',-1))==args.expected_epochs,
      'obs_loss': close(c.get('loss_obs'),args.expected_obs_loss_weight),
      'frontier_off': close(c.get('loss_recovery_frontier'),0.0),
      'decision_weighted_obs': c.get('decision_weighted_observation_loss') is True,
      'witness_fast_path': c.get('witness_fast_path')==args.expected_stage,
      'frozen_modules_eval': c.get('frozen_modules_eval') is True,
      'obs_conflict_scale': close(c.get('obs_conflict_scale'),args.expected_conflict_scale),
      'obs_conflict_temperature': close(c.get('obs_conflict_temperature'),args.expected_conflict_temperature),
      'obs_max_weight': close(c.get('obs_max_weight'),args.expected_max_weight),
    }
    # The stage contract in v48.47 records the core stage settings.  The remaining
    # weighting hyperparameters are published in the train summary's resolved config
    # when available; require them when present and never accept a mismatch.
    summary=src/'model_v48_47_witness'/'train_summary.json'
    if summary.is_file():
        sd=json.loads(summary.read_text())
        cfg=sd.get('config') or sd.get('resolved_config') or {}
        tc=(cfg.get('training') or {}) if isinstance(cfg,dict) else {}
        if 'witness_fast_path' in tc: checks['summary_witness_fast_path']=str(tc.get('witness_fast_path'))==args.expected_stage
        if 'frozen_modules_eval' in tc: checks['summary_frozen_modules_eval']=bool(tc.get('frozen_modules_eval')) is True
        expected={
          'decision_weighted_obs_conflict_scale':args.expected_conflict_scale,
          'decision_weighted_obs_temperature':args.expected_conflict_temperature,
          'decision_weighted_obs_max_weight':args.expected_max_weight,
        }
        for k,v in expected.items():
            if k in tc: checks[k]=close(tc[k],v)
    bad=[k for k,v in checks.items() if not v]
    if bad:
        raise SystemExit('v48.47 witness reuse contract mismatch: '+','.join(bad))
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src,dst,copy_function=shutil.copy2)
    audit={
      'event':'v48_47_witness_reuse','source_run':str(src),'destination_run':str(dst),
      'checkpoint_sha256':sha(dst/'model_v48_47_witness'/'best.pt'),'checks':checks,'valid':True,
    }
    (dst/'V48_47_WITNESS_REUSE.json').write_text(json.dumps(audit,indent=2)+'\n')
    return 0

if __name__=='__main__': raise SystemExit(main())
