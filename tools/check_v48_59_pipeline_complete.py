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
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--reference-contract',type=Path,required=True);ap.add_argument('--v58-complete',type=Path,required=True)
    ap.add_argument('--orfc-run',type=Path,required=True);ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    errors=[]; hashes={}; top=[a.reference_contract,a.v58_complete,a.feasibility_audit,a.comparison,a.orfc_run/'V48_59_VARIANT_ISOLATION.json']
    for p in top:
        if not p.is_file():errors.append(f'missing {p}')
    if a.reference_contract.is_file() and not load(a.reference_contract).get('valid'):errors.append('reference contract invalid')
    if a.v58_complete.is_file() and not (load(a.v58_complete).get('valid') and load(a.v58_complete).get('attribution_ready')):errors.append('V48.58 prerequisite package invalid')
    vi=a.orfc_run/'V48_59_VARIANT_ISOLATION.json'
    if vi.is_file() and not load(vi).get('valid'):errors.append('ORFC variant/state isolation invalid')
    terminal=a.orfc_run/'dedicated_recalibration_status.json';factor=a.orfc_run/'V48_59_FACTOR_CONTRACT.json'
    for p in (terminal,factor):
        if not p.is_file():errors.append(f'missing {p}')
    if terminal.is_file():
        td=load(terminal);codes=td.get('controller_exit_codes') or {}
        if not (td.get('certificate_executed') and td.get('gate_evaluated') and all(int(codes.get(v,-1)) in (0,20) for v in ('balanced','precision')) and not td.get('test_roots_read')):errors.append(f'invalid ORFC terminal status {codes}')
    for v in ('balanced','precision'):
        base=a.orfc_run/'candidates'/v;cal=base/'calibration'
        req=[base/'POLICY_CONTRACT.env',base/'V48_59_STAGE_I_STATE_ISOLATION.json',cal/'METRIC_CALIBRATION_CONTRACT.json',cal/'CERTIFICATE_CALIBRATION_COMPLETE.json',
             cal/'dev_diagnostic_near_v48.json',cal/'dev_diagnostic_contact_v48.json',cal/'dev_diagnostic_near_v48.proposal_rows.jsonl',cal/'dev_diagnostic_contact_v48.proposal_rows.jsonl',
             cal/'direct_value_risk_near_v48.json',cal/'direct_value_risk_contact_v48.json',cal/'direct_value_risk_near_v48.proposal_rows.jsonl',cal/'direct_value_risk_contact_v48.proposal_rows.jsonl']
        miss=[str(p) for p in req if not p.is_file() or p.stat().st_size==0]
        if miss:errors.append(f'{v}: missing/empty {miss}')
        metric=cal/'METRIC_CALIBRATION_CONTRACT.json'
        if metric.is_file():
            md=load(metric);sc=md.get('selection_contract') or {}
            if not (md.get('valid') and sc.get('mode')=='learned' and sc.get('mode_valid') and sc.get('threshold_valid') and sc.get('selection_semantics_valid') and sc.get('expected_selection_semantics')==RIFA and not md.get('test_roots_read')):errors.append(f'{v}: metric contract invalid')
        st=base/'V48_59_STAGE_I_STATE_ISOLATION.json'
        if st.is_file() and not load(st).get('valid'):errors.append(f'{v}: state isolation invalid')
    for p in top+[terminal,factor]:
        if p.is_file():hashes[str(p)]=sha(p)
    valid=not errors;doc={'schema':'ocrap-v48.59-orfc-pipeline-complete-v1','valid':valid,'attribution_ready':valid,'algorithm_version':'v48.59-DCP-DRFC-BCDE-RIFA-ORFC',
      'engineering_version':'v48.59.0-ORFC','errors':errors,'artifact_sha256':hashes,'test_roots_read':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_59_pipeline_complete','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
