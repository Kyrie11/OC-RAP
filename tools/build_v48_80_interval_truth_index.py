#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from ocrap.data.build.diagnose import iter_sample_paths
from ocrap.data.serialization import load_npz_selected
from ocrap.v48_80_interval_truth_contract import nested_tail_physical_interval

KEYS=frozenset({'scene_id','time_index','candidate_index','split_id','m_star','root_probs','root_valid','c_star','option_valid','root_assignments','future_metadata','recovery_modes','r_dep_star'})
def parse(x):
    role,raw=x.split('=',1); p=Path(raw).expanduser().resolve()
    if not p.is_dir(): raise argparse.ArgumentTypeError(x)
    return role,p
def scalar(d,k,z):
    try:return np.asarray(d.get(k,z)).item()
    except Exception:return z
def one(role,p,a,b,m,t):
    d=load_npz_selected(p,KEYS); r=nested_tail_physical_interval(d,alpha=a,beta=b,top_m=m,recompute_tolerance=t).to_dict()
    r.update(sample_path=str(p.resolve()),dataset_role=role,scene_id=str(scalar(d,'scene_id',p.stem)),time_index=int(scalar(d,'time_index',-1)),candidate_index=int(scalar(d,'candidate_index',-1)),split_id=str(scalar(d,'split_id','')))
    return r
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',action='append',required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--summary',type=Path,required=True); ap.add_argument('--alpha',type=float,default=.2); ap.add_argument('--beta',type=float,default=.2); ap.add_argument('--top-m',type=int,default=8); ap.add_argument('--recompute-tolerance',type=float,default=1e-5); ap.add_argument('--workers',type=int,default=8); a=ap.parse_args()
    roots=[parse(x) for x in a.root]; entries=[]; seen=set()
    for role,root in roots:
        for p in iter_sample_paths(root):
            rp=p.resolve();
            if rp in seen: raise SystemExit(f'duplicate sample {rp}')
            seen.add(rp); entries.append((role,rp))
    t0=time.perf_counter(); fn=lambda z:one(z[0],z[1],a.alpha,a.beta,a.top_m,a.recompute_tolerance)
    if a.workers>1:
        with ThreadPoolExecutor(max_workers=a.workers,thread_name_prefix='v4880-interval') as ex: rows=list(ex.map(fn,entries))
    else: rows=list(map(fn,entries))
    bad=[r for r in rows if not r['valid']]
    if bad: raise SystemExit(f'invalid recomputations={len(bad)}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
    by={}
    for role,_ in roots:
        rr=[r for r in rows if r['dataset_role']==role]; inf=[r for r in rr if r['informative']]; exact=[r for r in rr if r['exact_physical']]
        by[role]={'rows':len(rr),'informative_rows':len(inf),'informative_fraction':len(inf)/max(1,len(rr)),'exact_physical_rows':len(exact),'two_sided_rows':sum(r['lower_finite'] and r['upper_finite'] for r in inf),'lower_only_rows':sum(r['lower_finite'] and not r['upper_finite'] for r in inf),'upper_only_rows':sum(r['upper_finite'] and not r['lower_finite'] for r in inf)}
    sha=hashlib.sha256(a.output.read_bytes()).hexdigest(); doc={'schema':'ocrap-v48.80-interval-truth-index-summary-v1','engineering_version':'v48.80.0-OC-PISTC','valid':True,'rows':len(rows),'roles':by,'output':str(a.output.resolve()),'output_sha256':sha,'build_seconds':time.perf_counter()-t0,'dataset_reconstruction':False,'teacher_labels_changed':False,'teacher_future_input_to_model':False,'test_roots_read':False}
    a.summary.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n'); print(json.dumps(doc))
if __name__=='__main__':main()
