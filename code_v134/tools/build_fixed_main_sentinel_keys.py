#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict:
    x=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(x,dict): raise ValueError(path)
    return x


def _journal_scenes(path: Path) -> tuple[list[dict[str, Any]], str]:
    journal=path.with_suffix(path.suffix+'.scenes.jsonl')
    if not journal.is_file():
        return [], 'none'
    scenes=[]; seen=set(); fps=set()
    with journal.open(encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: rec=json.loads(line)
            except json.JSONDecodeError: continue
            if not isinstance(rec,dict): continue
            scene=rec.get('scene',rec)
            if not isinstance(scene,dict): continue
            fp=str(rec.get('run_fingerprint') or '')
            if fp: fps.add(fp)
            key=str(scene.get('target_key') or '')
            if key and key not in seen:
                seen.add(key); scenes.append(scene)
    if len(fps)>1:
        raise ValueError(f'multiple journal fingerprints for {path}: {sorted(fps)}')
    return scenes, 'journal'


def scenes(path: Path, doc: dict) -> tuple[list[dict[str, Any]], str]:
    vals=doc.get('scenes')
    if isinstance(vals,list) and vals:
        return [x for x in vals if isinstance(x,dict)], 'embedded'
    return _journal_scenes(path)


def keys(vals: list[dict[str, Any]]) -> set[str]:
    return {str(s.get('target_key')) for s in vals if s.get('target_key')}


def main() -> int:
    ap=argparse.ArgumentParser()
    for regime in ('safe','near','contact'):
        for variant in ('nominal','balanced','precision'):
            ap.add_argument(f'--{variant}-{regime}', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--key-dir', type=Path, required=True)
    args=ap.parse_args()
    out={'schema':'ocrap-v48.124-fixed-main-sentinel-v1','valid':True,'errors':[],'regimes':{}}
    args.key_dir.mkdir(parents=True,exist_ok=True)
    for regime in ('safe','near','contact'):
        docs={v:load(getattr(args,f'{v}_{regime}')) for v in ('nominal','balanced','precision')}
        loaded={v:scenes(getattr(args,f'{v}_{regime}'),d) for v,d in docs.items()}
        ksets={v:keys(pair[0]) for v,pair in loaded.items()}
        sources={v:pair[1] for v,pair in loaded.items()}
        common=set.intersection(*ksets.values()) if ksets else set()
        same=ksets['nominal']==ksets['balanced']==ksets['precision']
        if not common or not same:
            out['valid']=False; out['errors'].append(f'{regime}_target_key_mismatch_or_empty')
            target=None
        else:
            target=sorted(common)[0]
            kp=args.key_dir/f'{regime}.json'
            kp.write_text(json.dumps({'target_keys':[target]},indent=2)+'\n',encoding='utf-8')
        out['regimes'][regime]={
            'same_target_keys':same,
            'num_nominal':len(ksets['nominal']),
            'num_balanced':len(ksets['balanced']),
            'num_precision':len(ksets['precision']),
            'scene_sources':sources,
            'sentinel_target_key':target,
        }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'valid':out['valid'],'output':str(args.output)}))
    return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
