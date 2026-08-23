#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
RIFA='rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256();f=Path(p).open('rb')
 with f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(description='v48.62 OC-CWRF attribution-ready engineering sentinel')
 ap.add_argument('--reference-contract',type=Path,required=True);ap.add_argument('--v61-complete',type=Path,required=True);ap.add_argument('--occwrf-run',type=Path,required=True);ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args();errors=[];hashes={};vi=a.occwrf_run/'V48_62_VARIANT_ISOLATION.json';terminal=a.occwrf_run/'dedicated_recalibration_status.json';factor=a.occwrf_run/'V48_62_FACTOR_CONTRACT.json';top=[a.reference_contract,a.v61_complete,a.feasibility_audit,a.comparison,vi,terminal,factor]
 for p in top:
  if not p.is_file():errors.append(f'missing {p}')
 if a.reference_contract.is_file() and not load(a.reference_contract).get('valid'):errors.append('reference contract invalid')
 if a.v61_complete.is_file():
  d=load(a.v61_complete)
  if not (d.get('valid') and d.get('attribution_ready') and d.get('engineering_version')=='v48.61.0-ERWF' and not d.get('test_roots_read')):errors.append('V48.61 prerequisite invalid')
 if vi.is_file() and not load(vi).get('valid'):errors.append('OC-CWRF variant/state isolation invalid')
 if factor.is_file():
  fd=load(factor);req={'trainable_parameters':2,'threshold':.5,'threshold_search':False,'regime_id_input':False,'proposal_top_k':5,'proposal_expansion':False,'centering':False,'test_roots_read':False,'common_witness_feature_schema':1,'common_witness_feature_source':'observation_consistent_option_resolved_finite_time_recovery_witness','teacher_future_input':False,'relative_score_intervention':False}
  for k,v in req.items():
   if fd.get(k)!=v:errors.append(f'factor contract mismatch {k}={fd.get(k)!r}')
 if terminal.is_file():
  td=load(terminal);codes=td.get('controller_exit_codes') or {}
  if not (td.get('certificate_executed') and td.get('gate_evaluated') and all(int(codes.get(v,-1)) in (0,20) for v in ('balanced','precision')) and not td.get('test_roots_read')):errors.append(f'invalid OC-CWRF terminal status {codes}')
 for v in ('balanced','precision'):
  base=a.occwrf_run/'candidates'/v;cal=base/'calibration';req=[base/'POLICY_CONTRACT.env',base/'V48_62_STAGE_I_STATE_ISOLATION.json',cal/'METRIC_CALIBRATION_CONTRACT.json',cal/'CERTIFICATE_CALIBRATION_COMPLETE.json',cal/'dev_diagnostic_near_v48.json',cal/'dev_diagnostic_contact_v48.json',cal/'dev_diagnostic_near_v48.proposal_rows.jsonl',cal/'dev_diagnostic_contact_v48.proposal_rows.jsonl',cal/'direct_value_risk_near_v48.json',cal/'direct_value_risk_contact_v48.json',cal/'direct_value_risk_near_v48.proposal_rows.jsonl',cal/'direct_value_risk_contact_v48.proposal_rows.jsonl']
  miss=[str(p) for p in req if not p.is_file() or p.stat().st_size==0]
  if miss:errors.append(f'{v}: missing/empty {miss}')
  metric=cal/'METRIC_CALIBRATION_CONTRACT.json'
  if metric.is_file():
   md=load(metric);sc=md.get('selection_contract') or {}
   if not (md.get('valid') and sc.get('mode')=='learned' and sc.get('mode_valid') and sc.get('threshold_valid') and sc.get('selection_semantics_valid') and sc.get('expected_selection_semantics')==RIFA and not md.get('test_roots_read')):errors.append(f'{v}: metric contract invalid')
  st=base/'V48_62_STAGE_I_STATE_ISOLATION.json'
  if st.is_file():
   sd=load(st)
   if not sd.get('valid'):errors.append(f'{v}: state isolation invalid')
   if not sd.get('common_witness_feature_contract_valid'):errors.append(f'{v}: common-witness feature contract invalid')
 for p in top:
  if p.is_file():hashes[str(p)]=sha(p)
 valid=not errors;doc={'schema':'ocrap-v48.62-occwrf-pipeline-complete-v1','valid':valid,'attribution_ready':valid,'algorithm_version':'v48.62-DCP-DRFC-BCDE-RIFA-OC-CWRF','engineering_version':'v48.62.0-OC-CWRF','errors':errors,'artifact_sha256':hashes,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_62_occwrf_pipeline_complete','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
