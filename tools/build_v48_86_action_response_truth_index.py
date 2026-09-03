#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, os, time
from collections import defaultdict
from pathlib import Path
from typing import Any

POSITIVE_GAIN=0.015
DEPLOYABLE_MACROS={2,3,5,6,7}


def _rows(paths:list[Path])->list[dict[str,Any]]:
    out=[]
    for path in paths:
        with path.open('r',encoding='utf-8') as f:
            for ln,line in enumerate(f,1):
                line=line.strip()
                if not line: continue
                r=json.loads(line)
                r['_source']=str(path)
                out.append(r)
    return out


def _sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def _path(r:dict[str,Any], key:str)->str:
    v=str(r.get(key,'')).strip()
    return str(Path(v).expanduser().resolve()) if v else ''


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--absolute-index',type=Path,required=True)
    ap.add_argument('--pcd-index',type=Path,action='append',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--summary',type=Path,required=True)
    a=ap.parse_args(); t0=time.perf_counter()
    abs_rows=_rows([a.absolute_index]); pcd_rows=_rows(a.pcd_index)
    pcd={_path(r,'path'):r for r in pcd_rows}
    if len(pcd)!=len(pcd_rows): raise SystemExit('duplicate path in PCD index')
    groups=defaultdict(list); missing=[]
    for r in abs_rows:
        sp=_path(r,'sample_path'); pr=pcd.get(sp)
        if pr is None:
            missing.append(sp); continue
        role=str(r.get('dataset_role',''))
        key=(role,str(r.get('scene_id','')),int(r.get('time_index',-1)))
        groups[key].append((r,pr))
    if missing: raise SystemExit(f'action-response truth PCD coverage missing={len(missing)} first={missing[0]!r}')
    out=[]; role_stats=defaultdict(lambda:{'rows':0,'response_informative':0,'safe_positive':0,'component_harmful':0,'deployable':0,'finite_response':0,'positive_response_lower':0,'negative_response_upper':0})
    bad_groups=[]
    for key,pairs in groups.items():
        noms=[z for z in pairs if bool(z[1].get('nominal',False))]
        if len(noms)!=1:
            bad_groups.append((key,len(noms))); continue
        ar0,pr0=noms[0]
        nlo=float(ar0.get('physical_lower',-1e6)); nhi=float(ar0.get('physical_upper',1e6)); ninf=bool(ar0.get('informative',False))
        npc=float(pr0.get('teacher_pcd',float('nan')))
        if not math.isfinite(npc): bad_groups.append((key,'nominal_teacher_pcd_nonfinite')); continue
        for ar,pr in pairs:
            sp=_path(ar,'sample_path'); nominal=bool(pr.get('nominal',False)); role=str(ar.get('dataset_role',''))
            lo=0.0 if nominal else float(ar.get('physical_lower',-1e6))-nhi
            hi=0.0 if nominal else float(ar.get('physical_upper',1e6))-nlo
            informative=bool(nominal or (ninf and bool(ar.get('informative',False)) and math.isfinite(lo) and math.isfinite(hi) and lo<=hi))
            adv=float(pr.get('teacher_pcd',npc))-npc
            harmful=bool(pr.get('component_harmful',False))
            macro=int(pr.get('macro',-1))
            deployable=bool((not nominal) and macro in DEPLOYABLE_MACROS)
            beneficial=bool(adv>=POSITIVE_GAIN)
            safe_positive=bool((not nominal) and deployable and beneficial and not harmful)
            rec={
                'schema':'ocrap-v48.86-action-response-truth-v1','valid':True,'sample_path':sp,
                'dataset_role':role,'scene_id':str(ar.get('scene_id','')),'time_index':int(ar.get('time_index',-1)),
                'candidate_index':int(ar.get('candidate_index',pr.get('candidate',-1))),'nominal':nominal,'macro':macro,
                'response_informative':informative,'response_lower':float(lo),'response_upper':float(hi),
                'teacher_adv':float(adv),'component_harmful':harmful,'safe_positive':safe_positive,'deployable':deployable,
                'physical_candidate_lower':float(ar.get('physical_lower',-1e6)),'physical_candidate_upper':float(ar.get('physical_upper',1e6)),
                'physical_nominal_lower':nlo,'physical_nominal_upper':nhi,
            }
            out.append(rec); st=role_stats[role]; st['rows']+=1; st['response_informative']+=int(informative); st['safe_positive']+=int(safe_positive); st['component_harmful']+=int(harmful and deployable); st['deployable']+=int(deployable)
            finite=abs(lo)<9e5 and abs(hi)<9e5; st['finite_response']+=int(finite); st['positive_response_lower']+=int(lo>0); st['negative_response_upper']+=int(hi<0)
    if bad_groups: raise SystemExit(f'invalid PCD group nominal contract count={len(bad_groups)} first={bad_groups[0]!r}')
    if len(out)!=len(abs_rows): raise SystemExit(f'action-response truth row mismatch output={len(out)} abs={len(abs_rows)}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8') as f:
        for r in out:f.write(json.dumps(r,sort_keys=True)+'\n')
    roles={}
    for role,st in sorted(role_stats.items()):
        roles[role]={**st,'informative_fraction':float(st['response_informative']/max(st['rows'],1))}
    summary={'schema':'ocrap-v48.86-action-response-truth-summary-v1','engineering_version':'v48.86.0-OC-CRSC','valid':True,'rows':len(out),'roles':roles,'positive_gain':POSITIVE_GAIN,'deployable_macros':sorted(DEPLOYABLE_MACROS),'absolute_index':str(a.absolute_index),'absolute_index_sha256':_sha(a.absolute_index),'pcd_indices':[str(p) for p in a.pcd_index],'pcd_sha256':{str(p):_sha(p) for p in a.pcd_index},'output':str(a.output),'output_sha256':_sha(a.output),'teacher_metadata_input_to_model':False,'dataset_reconstruction':False,'elapsed_seconds':float(time.perf_counter()-t0)}
    a.summary.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'valid':True,'rows':len(out),'output':str(a.output)}));return 0
if __name__=='__main__':raise SystemExit(main())
