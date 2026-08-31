#!/usr/bin/env python3
from __future__ import annotations
import os as _v48_74_os
_v48_74_os.environ.setdefault("OCRAP_V48_74_SIGNED_VIABILITY", "1")
import argparse,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def env(p):
 out={}
 for raw in Path(p).read_text().splitlines():
  z=raw.strip()
  if z and not z.startswith('#') and '=' in z:k,v=z.split('=',1);out[k]=v
 return out
def res(x):return str(Path(x).expanduser().resolve(strict=False))
def b(x):return str(x).strip().lower() in {'1','true','yes','on'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference-run',type=Path,required=True);ap.add_argument('--reference-contract',type=Path,required=True);ap.add_argument('--svbw-run',type=Path,required=True);ap.add_argument('--anchor',required=True);ap.add_argument('--response',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errors=[];rc=load(a.reference_contract);snap=rc.get('reference_candidate_checkpoint_sha256') or {};variants={};learn=[]
 if not rc.get('valid'):errors.append('reference contract invalid')
 for v in ('balanced','precision'):
  base=a.svbw_run/'candidates'/v;ref=a.reference_run/'candidates'/v/'model_v48_trac_sr'/'best.pt';dst=base/'model_v48_trac_sr'/'best.pt';summ=base/'model_v48_trac_sr'/'train_summary.json';state=base/'V48_74_STAGE_I_STATE_ISOLATION.json';pol=base/'POLICY_CONTRACT.env';complete=base/'TRAINING_COMPLETE.json';ecp=base/'EVIDENCE_CORRECTION_COMPLETE.json';miss=[str(p) for p in (ref,dst,summ,state,pol,complete,ecp) if not p.is_file()]
  if miss:errors.append(f'{v}: missing {miss}');variants[v]={'valid':False};continue
  sd,st,tc,ec=load(summ),load(state),load(complete),load(ecp);pe=env(pol);refsha=sha(ref);dstsha=sha(dst);flags=st.get('factor_flags') or {};trainable=list(sd.get('trainable_param_prefixes') or [])
  checks={'reference_snapshot_matches':str(snap.get(v,''))==refsha,'init_checkpoint_matches':res(sd.get('init_checkpoint',''))==res(ref),'output_checkpoint_matches':res(sd.get('checkpoint',''))==res(dst),'trainable_prefix_exact':trainable==['direct_absolute_semantic_witness_gain'],'stage_i_isolation':bool(st.get('valid')) and bool(st.get('stage_i_bitwise_identity')) and bool(st.get('only_semantic_witness_gain_added')),'feature_contract':bool(st.get('semantic_witness_feature_contract_valid')),'factor_flags':bool(st.get('factor_flags_valid')) and flags.get('control_projection') is True and flags.get('boundary_transport') is False and flags.get('projection_fidelity') is True and flags.get('demand_normalized_fidelity') is False and flags.get('robust_occupancy') is False and flags.get('soft_occupancy_disagreement') is False and flags.get('boundary_localized_occupancy_trust') is False and flags.get('history_occupancy_reachability') is False and flags.get('interaction_box_support') is True and flags.get('interaction_hull_support') is True and flags.get('interaction_anchor_support')==b(a.anchor) and flags.get('interaction_response_support')==b(a.response),'policy_mode':pe.get('ABSOLUTE_FEASIBILITY_MODE')=='learned','policy_threshold':pe.get('ABSOLUTE_FEASIBILITY_THRESHOLD')=='0.5','policy_order':pe.get('SELECTION_SEMANTICS')=='rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank','training_complete_sha_matches':str(tc.get('checkpoint_sha256',''))==dstsha,'evidence_complete_sha_matches':str(ec.get('checkpoint_sha256',''))==dstsha,'evidence_complete_source_matches':res(ec.get('source_checkpoint',''))==res(ref),'evidence_complete_trainable_exact':list(ec.get('trainable_prefixes') or [])==['direct_absolute_semantic_witness_gain'] and int(ec.get('trainable_state_params',-1))==2,'evidence_complete_no_regime_input':ec.get('regime_id_exposed_to_evidence_model') is False and ec.get('test_roots_read') is False}
  ok=all(checks.values());
  if not ok:errors.append(f'{v}: failed {[k for k,z in checks.items() if not z]}')
  variants[v]={'valid':ok,'checks':checks,'reference_sha256':refsha,'svbw_checkpoint_sha256':dstsha,'semantic_witness_gain':st.get('effective_clamped_semantic_witness_gain')};learn.append(res(dst))
 distinct=len(learn)==2 and len(set(learn))==2
 if not distinct:errors.append('balanced/precision SVBW checkpoints not distinct')
 valid=not errors and all(z.get('valid') for z in variants.values()) and distinct;doc={'schema':'ocrap-v48.74-svbw-variant-isolation-v1','valid':valid,'reference_run':res(a.reference_run),'svbw_run':res(a.svbw_run),'active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,'route_alignment':True,'reentry_alignment':True,'control_projection':True,'boundary_transport':False,'projection_fidelity':True,'demand_normalized_fidelity':False,'robust_occupancy':False,'soft_occupancy_disagreement':False,'boundary_localized_occupancy_trust':False,'history_occupancy_reachability':False,'interaction_box_support':True,'interaction_hull_support':True,'interaction_anchor_support':b(a.anchor),'interaction_response_support':b(a.response),'variants':variants,'distinct_svbw_checkpoint_paths':distinct,'errors':errors,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_74_svbw_variant_isolation','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
