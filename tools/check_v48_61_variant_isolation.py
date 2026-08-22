#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(p:Path)->dict:return json.loads(p.read_text(encoding='utf-8'))
def env(p:Path):
    out={}; dup={}
    for raw in p.read_text(encoding='utf-8').splitlines():
        s=raw.strip()
        if not s or s.startswith('#') or '=' not in s: continue
        k,v=s.split('=',1); k=k.strip(); v=v.strip()
        if k in out: dup.setdefault(k,[out[k]]).append(v)
        out[k]=v
    return out,dup
def resolved(x):return str(Path(x).expanduser().resolve(strict=False))

def main()->int:
    ap=argparse.ArgumentParser(description='v48.61 ERWF balanced/precision provenance isolation')
    ap.add_argument('--reference-run',type=Path,required=True); ap.add_argument('--reference-contract',type=Path,required=True)
    ap.add_argument('--erwf-run',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); errors=[]; rc=load(a.reference_contract)
    if not rc.get('valid'): errors.append('reference reuse contract invalid')
    snap=rc.get('reference_candidate_checkpoint_sha256') or {}; variants={}; learned_paths=[]
    for v in ('balanced','precision'):
        ref=a.reference_run/'candidates'/v/'model_v48_trac_sr'/'best.pt'
        dst=a.erwf_run/'candidates'/v/'model_v48_trac_sr'/'best.pt'
        summ=a.erwf_run/'candidates'/v/'model_v48_trac_sr'/'train_summary.json'
        state=a.erwf_run/'candidates'/v/'V48_61_STAGE_I_STATE_ISOLATION.json'
        pol=a.erwf_run/'candidates'/v/'POLICY_CONTRACT.env'
        req=[ref,dst,summ,state,pol]; miss=[str(p) for p in req if not p.is_file()]
        if miss: errors.append(f'{v}: missing {miss}'); variants[v]={'valid':False,'missing':miss}; continue
        sd=load(summ); st=load(state); pe,dups=env(pol)
        critical={'ABSOLUTE_FEASIBILITY_MODE','ABSOLUTE_FEASIBILITY_THRESHOLD','SELECTION_SEMANTICS'}
        dupcrit=sorted(k for k in dups if k in critical)
        refsha=sha(ref); trainable=list(sd.get('trainable_param_prefixes') or [])
        ok={
          'reference_snapshot_matches':str(snap.get(v,''))==refsha,
          'init_checkpoint_matches':resolved(sd.get('init_checkpoint',''))==resolved(ref),
          'output_checkpoint_matches':resolved(sd.get('checkpoint',''))==resolved(dst),
          'trainable_prefix_exact':trainable==['direct_absolute_executable_witness_weight'],
          'stage_i_isolation':bool(st.get('valid')) and bool(st.get('stage_i_bitwise_identity')) and bool(st.get('only_executable_witness_weight_added')),
          'feature_contract':bool(st.get('executable_witness_feature_contract_valid')),
          'policy_mode':pe.get('ABSOLUTE_FEASIBILITY_MODE')=='learned',
          'policy_threshold':pe.get('ABSOLUTE_FEASIBILITY_THRESHOLD')=='0.5',
          'policy_order':pe.get('SELECTION_SEMANTICS')=='rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank',
          'no_duplicate_critical_keys':not dupcrit,
        }
        valid=all(ok.values())
        if not valid: errors.append(f'{v}: isolation/provenance failed: {[k for k,x in ok.items() if not x]}')
        variants[v]={'valid':valid,'checks':ok,'reference_checkpoint':resolved(ref),'reference_sha256':refsha,
                     'erwf_checkpoint':resolved(dst),'trainable_param_prefixes':trainable,'duplicate_critical_keys':dupcrit}
        learned_paths.append(resolved(dst))
    distinct=len(learned_paths)==2 and len(set(learned_paths))==2
    if not distinct: errors.append('balanced/precision ERWF checkpoint paths not distinct')
    valid=not errors and distinct and all(x.get('valid') for x in variants.values())
    doc={'schema':'ocrap-v48.61-erwf-variant-isolation-v1','valid':valid,'reference_run':resolved(a.reference_run),
         'erwf_run':resolved(a.erwf_run),'variants':variants,'distinct_erwf_checkpoint_paths':distinct,'errors':errors,'test_roots_read':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_61_erwf_variant_isolation','valid':valid,'output':str(a.output)})); return 0 if valid else 30
if __name__=='__main__': raise SystemExit(main())
