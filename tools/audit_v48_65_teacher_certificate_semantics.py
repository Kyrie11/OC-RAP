#!/usr/bin/env python3
"""Audit the existing teacher certificate without rebuilding any dataset.

v48.65 uses this as a read-only truth-contract diagnostic.  It never rewrites
NPZs/manifests and never touches test roots.  The audit asks two questions that
became material after v48.64:
  1) does stored r_dep_star equal exact OC-MERO recomputation from stored m_star;
  2) among teacher-feasible samples, how often do distinguishable observation
     classes select different recovery options (per-class choice) instead of one
     globally shared option being positive for every valid class.
It also reports structural-margin exact-value occupancy and r_dep plateaus so the
paper does not silently describe a continuous signed-min teacher if the released
labels contain structural post-processing plateaus.
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np

# Keep this read-only audit runnable both from the launcher and as a standalone
# tool/test.  The official launcher already exports PYTHONPATH; this bootstrap
# removes a packaging/test-harness dependency without changing any algorithm or
# experimental output semantics.
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
for _p in (str(_SRC), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ocrap.algorithms.ocmero import oc_mero
from ocrap.algorithms.lcv import weighted_lcvar
from ocrap.models.data import iter_sample_paths_many


def scalar(d, key, default=np.nan):
    a=np.asarray(d.get(key, default)); return a.item() if a.shape==() else default

def norm_p(p, rv):
    x=np.asarray(p,dtype=np.float64).reshape(-1)
    v=np.asarray(rv,dtype=bool).reshape(-1)[:x.size]
    x=np.where(v,np.clip(x,0,None),0.0); z=x.sum()
    return x/z if z>1e-12 else np.where(v,1.0/max(1,v.sum()),0.0)

def one(path:Path, alpha:float,beta:float,top_m:int):
    with np.load(path,allow_pickle=True) as z:
        d={k:z[k] for k in z.files}
    required=('m_star','root_probs','c_star','root_valid','option_valid','r_dep_star')
    missing=[k for k in required if k not in d]
    if missing:
        raise ValueError(f'missing teacher-certificate fields: {missing}')
    m=np.asarray(d['m_star'],dtype=np.float64)
    p=np.asarray(d['root_probs'],dtype=np.float64)
    c=np.asarray(d['c_star'],dtype=np.float64)
    rv=np.asarray(d['root_valid'],dtype=bool)
    ov=np.asarray(d['option_valid'],dtype=bool)
    res=oc_mero(m,p,c,alpha=alpha,beta=beta,option_valid=ov,root_valid=rv,use_lcvar=True,use_obs_kernel=True,top_m=top_m)
    stored=float(scalar(d,'r_dep_star',np.nan))
    if not math.isfinite(stored):
        raise ValueError('non-finite r_dep_star')
    pn=norm_p(p,rv)
    valid_root=rv[:m.shape[0]] & (pn>0)
    q=res.q
    best=res.best_option
    positive_class_mass=float((pn*(res.r_per_root>=0)).sum())
    selected=[int(best[i]) for i in range(len(best)) if valid_root[i]]
    diversity=len(set(selected)) if selected else 0
    # Counterfactual information pattern: one option forced globally across
    # observation classes.  Actual OC-MERO maximizes option per observation
    # class before the outer lower-tail aggregation.  The gap below directly
    # measures whether candidate-global option transport is too restrictive.
    global_scores=[]
    for l in range(m.shape[1]):
        if l>=len(ov) or not ov[l]:
            continue
        global_scores.append(float(weighted_lcvar(q[:,l], pn, alpha)))
    global_shared_score=max(global_scores) if global_scores else -1e9
    vals=m[np.isfinite(m)&(m>-1e8)]
    exact={str(x):int(np.sum(np.isclose(vals,x,rtol=0,atol=1e-7))) for x in (0.6,0.9,-0.8)}
    return {
      'stored_r_dep':stored,'recomputed_r_dep':float(res.r_dep),'abs_error':abs(stored-float(res.r_dep)),
      'teacher_feasible':stored>=0,'positive_class_mass':positive_class_mass,
      'selected_option_diversity':diversity,'global_shared_option_score':global_shared_score,
      'classlocal_minus_global_shared_score':float(res.r_dep-global_shared_score),
      'classlocal_required_for_sign':bool(stored>=0 and global_shared_score<0),
      'r_dep_eq_0p5':abs(stored-0.5)<=1e-7,'exact_margin_counts':exact,
      'margin_count':int(vals.size),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',action='append',required=True,help='Existing train/dev/calibration/certificate root; never test')
    ap.add_argument('--alpha',type=float,default=.2); ap.add_argument('--beta',type=float,default=.2); ap.add_argument('--top-m',type=int,default=8)
    ap.add_argument('--max-samples-per-root',type=int,default=0,help='0 = all; deterministic prefix only for debugging')
    ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    rows=[]; byroot={}; errors=[]
    for raw in a.root:
        root=Path(raw).resolve()
        # Dataset roots are explicit launcher arguments; fail closed on a
        # test-like root name without rejecting unrelated parent directories
        # (e.g. pytest temporary paths in the synthetic regression test).
        if 'test' in root.name.lower():
            raise SystemExit(f'refusing test-like root: {root}')
        paths=list(iter_sample_paths_many(root))
        if a.max_samples_per_root>0: paths=paths[:a.max_samples_per_root]
        rr=[]
        for p in paths:
            try: rr.append(one(p,a.alpha,a.beta,a.top_m))
            except Exception as e: errors.append({'path':str(p),'error':repr(e)})
        rows.extend(rr); byroot[str(root)]=rr
    def summarize(rs):
        if not rs:return {'samples':0}
        feas=[r for r in rs if r['teacher_feasible']]
        mc=Counter(); totalm=0
        for r in rs:
            mc.update(r['exact_margin_counts']); totalm+=r['margin_count']
        return {
          'samples':len(rs),'teacher_feasible_samples':len(feas),
          'max_r_dep_recompute_abs_error':max(r['abs_error'] for r in rs),
          'r_dep_integrity_fraction_le_1e_6':float(np.mean([r['abs_error']<=1e-6 for r in rs])),
          'r_dep_eq_0p5_fraction':float(np.mean([r['r_dep_eq_0p5'] for r in rs])),
          'teacher_feasible_positive_class_mass_mean':float(np.mean([r['positive_class_mass'] for r in feas])) if feas else None,
          'teacher_feasible_multi_option_class_fraction':float(np.mean([r['selected_option_diversity']>1 for r in feas])) if feas else None,
          'teacher_feasible_global_shared_score_mean':float(np.mean([r['global_shared_option_score'] for r in feas])) if feas else None,
          'teacher_feasible_classlocal_minus_global_shared_score_mean':float(np.mean([r['classlocal_minus_global_shared_score'] for r in feas])) if feas else None,
          'teacher_feasible_classlocal_required_for_sign_fraction':float(np.mean([r['classlocal_required_for_sign'] for r in feas])) if feas else None,
          'exact_margin_value_counts':dict(mc),'finite_margin_count':totalm,
          'exact_margin_value_fraction':{k:(float(v/totalm) if totalm else None) for k,v in mc.items()},
        }
    overall=summarize(rows)
    # Exact OC-MERO integrity is the only fail-closed condition.  Diversity and
    # plateaus are scientific diagnostics, not reasons to rewrite labels.
    valid=bool(not errors and overall.get('samples',0)>0 and overall.get('max_r_dep_recompute_abs_error',1.0)<=1e-5)
    doc={'schema':'ocrap-v48.65-teacher-certificate-semantics-audit-v1','valid':valid,'errors':errors,
         'test_roots_read':False,'read_only_existing_dataset':True,'dataset_reconstruction':False,
         'ocmero':{'alpha':a.alpha,'beta':a.beta,'top_m':a.top_m},'overall':overall,
         'by_root':{k:summarize(v) for k,v in byroot.items()},
         'interpretation':{
           'classlocal_transport_gap':'actual OC-MERO permits per-observation-class option choice; global_shared_option_score forces one option across classes. A positive classlocal-minus-global gap or sign flip supports class-local correction transport.',
           'plateau_warning':'exact 0.5 r_dep or exact 0.6/0.9/-0.8 margin occupancy is reported only as a truth-contract diagnostic; it is not edited or distilled into the deployed model',
         }}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_65_teacher_certificate_semantics_audit','valid':valid,'output':str(a.output)}))
    raise SystemExit(0 if valid else 30)
if __name__=='__main__': main()
