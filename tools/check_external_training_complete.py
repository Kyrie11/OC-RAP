#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import yaml

def nested(cfg,*keys,default=None):
    cur=cfg
    for k in keys:
        if not isinstance(cur,dict):return default
        cur=cur.get(k)
    return default if cur is None else cur

def main()->int:
    ap=argparse.ArgumentParser(description='Return success when a learned external baseline completed the configured epoch budget.')
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--require-deployable-contract',action='store_true');ap.add_argument('--require-implementation-version',default=None)
    a=ap.parse_args()
    if not(a.checkpoint.is_file() and a.summary.is_file() and a.config.is_file()):return 1
    try:
        cfg=yaml.safe_load(a.config.read_text()) or {}; summary=json.loads(a.summary.read_text())
        expected=int(nested(cfg,'external_baselines','training','epochs',default=0) or 0)
        completed=int(summary.get('epochs_completed',0)); requested=int(summary.get('epochs_requested',0))
        if expected<=0 or completed<expected or requested<expected:return 1
    except Exception:return 1
    # Reuse the authoritative checkpoint validator rather than duplicating tensor/contract checks.
    import subprocess,sys
    cmd=[sys.executable,'tools/validate_external_checkpoint.py','--checkpoint',str(a.checkpoint)]
    if a.require_deployable_contract:cmd.append('--require-deployable-contract')
    if a.require_implementation_version:cmd += ['--require-implementation-version',a.require_implementation_version]
    return 0 if subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 else 1
if __name__=='__main__':raise SystemExit(main())
