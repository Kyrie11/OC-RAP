#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

def read(path: Path) -> dict[str, Any] | None:
    try:
        value=json.loads(path.read_text(encoding='utf-8')); return value if isinstance(value,dict) else None
    except Exception:return None

def _same_path(a: str|None,b: str|None)->bool:
    if a in {None,''} or b in {None,''}: return a in {None,''} and b in {None,''}
    try:return Path(str(a)).expanduser().resolve()==Path(str(b)).expanduser().resolve()
    except Exception:return str(a)==str(b)

def _collect_keys(value: Any, out:set[str])->None:
    if isinstance(value,str):
        x=value.strip();
        if x.startswith('target:'): x=x[len('target:'):]
        if x: out.add(x)
    elif isinstance(value,list):
        for x in value:_collect_keys(x,out)
    elif isinstance(value,dict):
        for k in ('target_key','resume_key'):
            if value.get(k): _collect_keys(value[k],out); return
        for k in ('target_keys','selected','items','scenes'):
            if k in value:_collect_keys(value[k],out)

def _target_keys(path:Path)->set[str]:
    out:set[str]=set(); text=path.read_text(encoding='utf-8')
    try:_collect_keys(json.loads(text),out)
    except json.JSONDecodeError:
        for line in text.splitlines():
            line=line.strip()
            if not line or line.startswith('#'):continue
            try:_collect_keys(json.loads(line),out)
            except json.JSONDecodeError:_collect_keys(line,out)
    return out

def _journal_keys(path:Path)->set[str]:
    out:set[str]=set()
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            row=json.loads(line); scene=row.get('scene',row) if isinstance(row,dict) else {}
            if isinstance(scene,dict):
                key=str(scene.get('target_key') or '').strip()
                if key: out.add(key)
    except Exception:return set()
    return out

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--method',default=None)
    ap.add_argument('--bucket-dataset',default=None); ap.add_argument('--checkpoint',default=None); ap.add_argument('--dependency',action='append',default=[])
    ap.add_argument('--quiet',action='store_true'); ap.add_argument('--require-scenes',action='store_true'); ap.add_argument('--target-keys-file',type=Path,default=None)
    ap.add_argument('--require-latency-contract',default=None)
    a=ap.parse_args(); p=a.output; prog=read(p.with_suffix(p.suffix+'.progress.json')); result=read(p); journal=p.with_suffix(p.suffix+'.scenes.jsonl'); errors=[]
    complete=bool(result and prog and prog.get('status')=='complete' and journal.is_file())
    if not complete:errors.append('missing_result_progress_or_journal')
    if complete and result.get('bucket_target_count') not in (None,0):
        complete=int(result.get('num_scenes') or 0)==int(result.get('bucket_target_count') or 0)
        if not complete:errors.append('scene_count_does_not_match_bucket_target_count')
    if result and prog:
        rf=str(result.get('run_fingerprint','') or ''); pf=str(prog.get('run_fingerprint','') or '')
        if rf and pf and rf!=pf: complete=False; errors.append('result_progress_fingerprint_mismatch')
    if result and a.method is not None and str(result.get('method','')).lower()!=str(a.method).lower(): complete=False; errors.append('method_mismatch')
    if result and a.bucket_dataset is not None and not _same_path(result.get('bucket_dataset'),a.bucket_dataset): complete=False; errors.append('bucket_dataset_mismatch')
    if result and a.require_scenes:
        scenes=result.get('scenes'); n=int(result.get('num_scenes') or 0)
        if not bool(result.get('scenes_embedded')) or not isinstance(scenes,list) or len(scenes)!=n: complete=False; errors.append('embedded_scenes_required_but_missing_or_incomplete')
    expected_keys:set[str]=set(); observed_keys:set[str]=set()
    if a.target_keys_file is not None:
        if not a.target_keys_file.is_file(): complete=False; errors.append('target_keys_file_missing')
        else:
            expected_keys=_target_keys(a.target_keys_file); observed_keys=_journal_keys(journal) if journal.is_file() else set()
            if not expected_keys: complete=False; errors.append('target_keys_file_empty')
            elif observed_keys != expected_keys:
                complete=False; errors.append('target_key_set_mismatch')
            if result and int(result.get('num_scenes') or 0) != len(expected_keys): complete=False; errors.append('scene_count_does_not_match_target_lock')
            if result and int(result.get('bucket_target_count') or 0) not in (0,len(expected_keys)): complete=False; errors.append('bucket_target_count_does_not_match_target_lock')
    if result and a.require_latency_contract:
        got=str(((result.get('timing') or {}).get('execution_contract') or ''))
        if got != str(a.require_latency_contract): complete=False; errors.append(f'latency_contract_mismatch:{got}')
    deps=[Path(x) for x in a.dependency]
    if a.checkpoint:deps.append(Path(a.checkpoint))
    if p.is_file():
        try:
            out_ns=p.stat().st_mtime_ns
            for dep in deps:
                if not dep.exists():complete=False;errors.append(f'missing_dependency:{dep}')
                elif dep.stat().st_mtime_ns>out_ns:complete=False;errors.append(f'dependency_newer_than_output:{dep}')
        except OSError as exc:complete=False;errors.append(f'freshness_check_failed:{exc}')
    doc={'event':'closed_loop_artifact_check','output':str(p),'complete':complete,'result_exists':p.is_file(),'journal_exists':journal.is_file(),'progress_status':prog.get('status') if prog else None,'num_scenes':result.get('num_scenes') if result else None,'bucket_target_count':result.get('bucket_target_count') if result else None,'run_fingerprint':result.get('run_fingerprint') if result else None,'target_lock_count':len(expected_keys) if expected_keys else None,'observed_target_count':len(observed_keys) if a.target_keys_file is not None else None,'errors':errors}
    if not a.quiet:print(json.dumps(doc,ensure_ascii=False))
    return 0 if complete else 1
if __name__=='__main__':raise SystemExit(main())
