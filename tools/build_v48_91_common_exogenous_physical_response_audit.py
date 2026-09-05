#!/usr/bin/env python3
from __future__ import annotations
import argparse,gzip,hashlib,json,math
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np

from ocrap.v48_91_common_exogenous_physical_margin import ENGINEERING_VERSION, audit_future_physical_response
from tools.build_v48_89_root_correspondence_audit import _auc,_load_sample,_quantiles


def _sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def _load_sidecar(p:Path)->dict[str,dict[str,Any]]:
 out={}
 with gzip.open(p,'rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line)
   key=str(Path(r['sample_path']).resolve())
   if key in out: raise ValueError(f'duplicate sidecar sample {key}')
   out[key]=r
 return out

def _matrix(row:dict[str,Any],field:str)->np.ndarray:
 F=int(row['future_count']);L=int(row['option_count']);m=np.full((F,L),np.nan,dtype=np.float64)
 for k,v in row[field].items(): m[:,int(k)]=np.asarray(v,dtype=np.float64)
 return m

@lru_cache(maxsize=16384)
def _load_sample_cached(path_text:str):
 return _load_sample(Path(path_text))

def _mean(rows,field):
 v=[float(r[field]) for r in rows if r.get(field) is not None and math.isfinite(float(r[field]))]
 return float(np.mean(v)) if v else None

def _macro_auc(rows,field):
 num=0.0;den=0
 by=defaultdict(list)
 for r in rows: by[int(r.get('macro',-1))].append(r)
 for g in by.values():
  p=np.asarray([float(r[field]) for r in g if r.get('safe_positive')],dtype=np.float64)
  n=np.asarray([float(r[field]) for r in g if r.get('teacher_harmful')],dtype=np.float64)
  if not len(p) or not len(n): continue
  num+=float((p[:,None]>n[None,:]).sum()+0.5*(p[:,None]==n[None,:]).sum()); den+=len(p)*len(n)
 return float(num/den) if den else None

def _top1(rows,field):
 groups=defaultdict(list)
 for r in rows: groups[(r['scene_id'],int(r['time_index']))].append(r)
 gs=[g for g in groups.values() if any(r.get('safe_positive') for r in g)]
 if not gs:return None,None,None,0
 acc=[];chance=[]
 for g in gs:
  s=np.asarray([float(r[field]) for r in g]); mx=float(np.max(s)); idx=np.where(np.abs(s-mx)<=1e-12)[0]
  acc.append(np.mean([bool(g[i].get('safe_positive')) for i in idx])); chance.append(np.mean([bool(r.get('safe_positive')) for r in g]))
 a=float(np.mean(acc));c=float(np.mean(chance));return a,c,a-c,len(gs)

def _summ(rows:list[dict[str,Any]])->dict[str,Any]:
 valid=[r for r in rows if r.get('valid')]; safe=[r for r in valid if r.get('safe_positive')]; harm=[r for r in valid if r.get('teacher_harmful')]
 out={'rows':len(rows),'valid_rows':len(valid),'safe_positive_rows':len(safe),'harmful_rows':len(harm)}
 for f in ('common_exogenous_tail_coverage','response_informative_mass','response_sign_identifiable_mass','response_point_identifiable_mass','duplicate_physical_homogeneity_mass_candidate','duplicate_physical_homogeneity_mass_nominal','future_tail_reconstruction_error'):
  out[f]=_quantiles([float(r[f]) for r in valid])
 field='signed_response_score'
 out['response_safe_positive_mean']=_mean(safe,field);out['response_harmful_mean']=_mean(harm,field)
 out['response_safe_vs_harmful_auc']=_auc([True]*len(safe)+[False]*len(harm),[float(r[field]) for r in safe]+[float(r[field]) for r in harm]) if safe and harm else None
 out['response_macro_stratified_auc']=_macro_auc(valid,field)
 a,c,l,n=_top1(valid,field);out['response_top1_accuracy']=a;out['response_top1_chance']=c;out['response_top1_lift']=l;out['powered_safe_positive_groups']=n
 return out

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--v48-90-audit',type=Path,required=True);ap.add_argument('--sidecar',type=Path,required=True);ap.add_argument('--sidecar-summary',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--alpha',type=float,default=.2);ap.add_argument('--beta',type=float,default=.2);ap.add_argument('--intra-root-alpha',type=float,default=.2);ap.add_argument('--top-m',type=int,default=8);args=ap.parse_args()
 ss=json.loads(args.sidecar_summary.read_text());
 if not(ss.get('valid') and ss.get('attribution_ready') and ss.get('dataset_reconstruction') is False and ss.get('dataset_reselection') is False): raise SystemExit('invalid V48.91 sidecar summary')
 if str(ss.get('engineering_version')) != ENGINEERING_VERSION: raise SystemExit(f"sidecar summary engineering_version={ss.get('engineering_version')!r} != {ENGINEERING_VERSION!r}")
 sc=_load_sidecar(args.sidecar); out=[];errors=[]
 matrix_cache={}
 def get_matrix(path_text,field):
  key=(path_text,field)
  m=matrix_cache.get(key)
  if m is None:
   m=_matrix(sc[path_text],field); matrix_cache[key]=m
  return m
 with args.v48_90_audit.open(encoding='utf-8') as f:
  for line in f:
   base=json.loads(line)
   if not(base.get('valid') and base.get('label_available')): continue
   cp=str(Path(base['sample_path']).resolve());npth=str(Path(base['nominal_sample_path']).resolve())
   if cp not in sc or npth not in sc: errors.append(f'missing sidecar pair {cp} / {npth}');continue
   if not(sc[cp].get('valid') and sc[npth].get('valid')): errors.append(f'invalid sidecar pair {cp} / {npth}');continue
   try:
    cs=_load_sample_cached(cp);ns=_load_sample_cached(npth);cst=get_matrix(cp,'m_future_structural');nst=get_matrix(npth,'m_future_structural');cph=get_matrix(cp,'m_future_physical');nph=get_matrix(npth,'m_future_physical')
    m=audit_future_physical_response(cs,ns,cst,nst,cph,nph,alpha=args.alpha,beta=args.beta,intra_root_alpha=args.intra_root_alpha,top_m=args.top_m).to_dict()
    m.update(schema='ocrap-v48.91-common-exogenous-future-physical-response-row-v1',engineering_version=ENGINEERING_VERSION,dataset_role=base['dataset_role'],scene_id=base['scene_id'],time_index=base['time_index'],candidate_index=base['candidate_index'],sample_path=cp,nominal_sample_path=npth,teacher_adv=base['teacher_adv'],teacher_harmful=base['teacher_harmful'],teacher_feasible=base['teacher_feasible'],safe_positive=base['safe_positive'],macro=base['macro'],partition_stability=base['exogenous_tail_partition_stability'],v48_90_signed_response_score=base['exogenous_transport_signed_response_score'],planner_parameters_trained=0,teacher_metadata_input_to_model=False,dataset_reconstruction=False,dataset_reselection=False)
    if not m['valid']:errors.append(f'invalid physical response role={base["dataset_role"]} key={(base["scene_id"],base["time_index"],base["candidate_index"])}: {m["error"]}')
    out.append(m)
   except Exception as exc:errors.append(f'{cp}: {exc}')
 args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('w',encoding='utf-8') as f:
  for r in out:f.write(json.dumps(r,sort_keys=True)+'\n')
 roles=sorted(set(r['dataset_role'] for r in out))
 summ={'schema':'ocrap-v48.91-common-exogenous-future-physical-response-summary-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors[:100],'rows':len(out),'roles':{role:_summ([r for r in out if r['dataset_role']==role]) for role in roles},'sidecar':str(args.sidecar.resolve()),'sidecar_sha256':_sha(args.sidecar),'sidecar_summary_sha256':_sha(args.sidecar_summary),'output':str(args.output.resolve()),'output_sha256':_sha(args.output),'planner_parameters_trained':0,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'dataset_reconstruction':False,'dataset_reselection':False,'test_roots_read':False,'boundary_transport':False,'regime_conditioning':False}
 args.summary.write_text(json.dumps(summ,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':summ['valid'],'rows':len(out),'errors':len(errors)}));return 0 if summ['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
