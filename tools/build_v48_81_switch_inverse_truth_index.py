#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from ocrap.v48_81_switch_inverse_truth_contract import nested_tail_switch_inverse_interval

def parse(x):
    a,b=x.split('=',1);return a,Path(b).expanduser().resolve()
def iter_sample_paths(root:Path):
    m=root/'manifest.csv'
    with m.open(newline='') as f:
        for r in csv.DictReader(f):
            q=r.get('path') or r.get('sample_path') or r.get('file')
            if q:
                p=Path(q);yield p if p.is_absolute() else (root/p).resolve()
def one(role,p,alpha,beta,top_m,tol):
    with np.load(p,allow_pickle=False) as z:s={k:z[k] for k in z.files}
    r=nested_tail_switch_inverse_interval(s,alpha=alpha,beta=beta,top_m=top_m,recompute_tolerance=tol).to_dict()
    r.update(dataset_role=role,sample_path=str(p.resolve()),scene_id=str(np.asarray(s.get('scene_id','')).reshape(-1)[0]),time_index=int(np.asarray(s.get('time_index',-1)).reshape(-1)[0]),candidate_index=int(np.asarray(s.get('candidate_index',-1)).reshape(-1)[0]),split_id=str(np.asarray(s.get('split_id','')).reshape(-1)[0]))
    return r
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',action='append',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--alpha',type=float,default=.2);ap.add_argument('--beta',type=float,default=.2);ap.add_argument('--top-m',type=int,default=8);ap.add_argument('--recompute-tolerance',type=float,default=1e-5);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
    roots=[parse(x) for x in a.root];entries=[];seen=set()
    for role,root in roots:
        for p in iter_sample_paths(root):
            rp=p.resolve()
            if rp in seen:raise SystemExit(f'duplicate sample {rp}')
            seen.add(rp);entries.append((role,rp))
    t0=time.perf_counter();fn=lambda z:one(z[0],z[1],a.alpha,a.beta,a.top_m,a.recompute_tolerance)
    with ThreadPoolExecutor(max_workers=max(1,a.workers),thread_name_prefix='v4881-inverse') as ex:rows=list(ex.map(fn,entries))
    bad=[r for r in rows if not r['valid']]
    if bad:raise SystemExit(f'invalid recomputations={len(bad)}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
    by={}
    for role,_ in roots:
        rr=[r for r in rows if r['dataset_role']==role];inf=[r for r in rr if r['informative']];exact=[r for r in rr if r['exact_physical']]; widths=[float(r['interval_width']) for r in inf if r.get('interval_width') is not None]
        by[role]={'rows':len(rr),'informative_rows':len(inf),'informative_fraction':len(inf)/max(1,len(rr)),'exact_physical_rows':len(exact),'two_sided_rows':sum(r['lower_finite'] and r['upper_finite'] for r in inf),'lower_only_rows':sum(r['lower_finite'] and not r['upper_finite'] for r in inf),'upper_only_rows':sum(r['upper_finite'] and not r['lower_finite'] for r in inf),'median_finite_interval_width':float(np.median(widths)) if widths else None,'mean_exact_cell_fraction':float(np.mean([r['exact_cell_fraction'] for r in rr])) if rr else 0.0,'mean_inactive_structural_cell_fraction':float(np.mean([r['inactive_structural_cell_fraction'] for r in rr])) if rr else 0.0}
    sha=hashlib.sha256(a.output.read_bytes()).hexdigest();doc={'schema':'ocrap-v48.81-switch-inverse-truth-index-summary-v1','engineering_version':'v48.81.0-OC-SITC','valid':True,'rows':len(rows),'roles':by,'output':str(a.output.resolve()),'output_sha256':sha,'build_seconds':time.perf_counter()-t0,'dataset_reconstruction':False,'teacher_labels_changed':False,'teacher_future_input_to_model':False,'test_roots_read':False}
    a.summary.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps(doc))
if __name__=='__main__':main()
