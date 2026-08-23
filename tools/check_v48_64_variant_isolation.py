#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def env(p):
 out={};dup={}
 for raw in Path(p).read_text().splitlines():
  z=raw.strip()
  if not z or z.startswith('#') or '=' not in z:continue
  k,v=z.split('=',1)
  if k in out:dup.setdefault(k,[out[k]]).append(v)
  out[k]=v
 return out,dup
def res(x):return str(Path(x).expanduser().resolve(strict=False))
def b(x):return str(x).strip().lower() in {'1','true','yes','on'}

def main()->int:
 ap=argparse.ArgumentParser(description='v48.64 OC-SARW balanced/precision provenance isolation')
 ap.add_argument('--reference-run',type=Path,required=True);ap.add_argument('--reference-contract',type=Path,required=True);ap.add_argument('--sarw-run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--active-set',default='true');ap.add_argument('--path-stop',default='true');a=ap.parse_args();errors=[];rc=load(a.reference_contract);snap=rc.get('reference_candidate_checkpoint_sha256') or {};variants={};learn=[]
 if not rc.get('valid'):errors.append('reference contract invalid')
 for v in ('balanced','precision'):
  base=a.sarw_run/'candidates'/v
  ref=a.reference_run/'candidates'/v/'model_v48_trac_sr'/'best.pt'
  dst=base/'model_v48_trac_sr'/'best.pt'
  # Canonical train summary location produced by train_ocrap_v48_trac_sr.sh.
  summ=base/'model_v48_trac_sr'/'train_summary.json'
  state=base/'V48_64_STAGE_I_STATE_ISOLATION.json';pol=base/'POLICY_CONTRACT.env'
  complete=base/'TRAINING_COMPLETE.json';evidence_complete=base/'EVIDENCE_CORRECTION_COMPLETE.json'
  miss=[str(p) for p in (ref,dst,summ,state,pol,complete,evidence_complete) if not p.is_file()]
  if miss:errors.append(f'{v}: missing {miss}');variants[v]={'valid':False,'missing':miss};continue
  sd,st=load(summ),load(state);tc=load(complete);ec=load(evidence_complete);pe,dups=env(pol);trainable=list(sd.get('trainable_param_prefixes') or []);refsha=sha(ref);dstsha=sha(dst)
  ck=sd.get('checkpoint')
  checks={'reference_snapshot_matches':str(snap.get(v,''))==refsha,'init_checkpoint_matches':res(sd.get('init_checkpoint',''))==res(ref),'output_checkpoint_matches':res(ck or '')==res(dst),
          'trainable_prefix_exact':trainable==['direct_absolute_semantic_witness_gain'],'stage_i_isolation':bool(st.get('valid')) and bool(st.get('stage_i_bitwise_identity')) and bool(st.get('only_semantic_witness_gain_added')),
          'feature_contract':bool(st.get('semantic_witness_feature_contract_valid')),'factor_flags':bool(st.get('factor_flags_valid')) and bool(st.get('active_set_alignment'))==b(a.active_set) and bool(st.get('path_stop_alignment'))==b(a.path_stop),
          'policy_mode':pe.get('ABSOLUTE_FEASIBILITY_MODE')=='learned','policy_threshold':pe.get('ABSOLUTE_FEASIBILITY_THRESHOLD')=='0.5','policy_order':pe.get('SELECTION_SEMANTICS')=='rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank',
          'no_duplicate_critical_keys':not any(k in {'ABSOLUTE_FEASIBILITY_MODE','ABSOLUTE_FEASIBILITY_THRESHOLD','SELECTION_SEMANTICS'} for k in dups),
          'training_complete_checkpoint_matches':res(tc.get('checkpoint',''))==res(dst),
          'training_complete_sha_matches':str(tc.get('checkpoint_sha256',''))==dstsha,
          'training_complete_epoch_matches':tc.get('best_epoch')==sd.get('best_epoch') and tc.get('epochs_completed')==sd.get('epochs_completed'),
          'evidence_complete_checkpoint_matches':res(ec.get('checkpoint',''))==res(dst),
          'evidence_complete_sha_matches':str(ec.get('checkpoint_sha256',''))==dstsha,
          'evidence_complete_source_matches':res(ec.get('source_checkpoint',''))==res(ref),
          'evidence_complete_trainable_exact':list(ec.get('trainable_prefixes') or [])==['direct_absolute_semantic_witness_gain'] and int(ec.get('trainable_state_params',-1))==2,
          'evidence_complete_no_regime_input':ec.get('regime_id_exposed_to_evidence_model') is False and ec.get('test_roots_read') is False}
  ok=all(checks.values());
  if not ok:errors.append(f'{v}: failed {[k for k,x in checks.items() if not x]}')
  variants[v]={'valid':ok,'checks':checks,'trainable_param_prefixes':trainable,'reference_sha256':refsha,'sarw_checkpoint_sha256':dstsha,'sarw_checkpoint':res(dst),'train_summary':res(summ)};learn.append(res(dst))
 distinct=len(learn)==2 and len(set(learn))==2
 if not distinct:errors.append('balanced/precision SARW checkpoints not distinct')
 valid=not errors and distinct and all(x.get('valid') for x in variants.values())
 doc={'schema':'ocrap-v48.64-sarw-variant-isolation-v1','valid':valid,'reference_run':res(a.reference_run),'sarw_run':res(a.sarw_run),'active_set_alignment':b(a.active_set),'path_stop_alignment':b(a.path_stop),'variants':variants,'distinct_sarw_checkpoint_paths':distinct,'errors':errors,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_64_sarw_variant_isolation','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
