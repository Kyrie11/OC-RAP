#!/usr/bin/env python3
"""Final fail-closed completeness check for the v48.58 RIFA attribution package."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

RIFA_ORDER="rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank"


def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--reference-contract",type=Path,required=True)
    ap.add_argument("--native-run",type=Path,required=True)
    ap.add_argument("--learned-run",type=Path,required=True)
    ap.add_argument("--feasibility-audit",type=Path,required=True)
    ap.add_argument("--comparison",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    errors=[]; hashes={}; run_checks={}
    required_top=[a.reference_contract,a.feasibility_audit,a.comparison,a.learned_run/'V48_58_VARIANT_ISOLATION.json']
    for p in required_top:
        if not p.is_file(): errors.append(f"missing top-level artifact: {p}")
    if a.reference_contract.is_file() and not bool(load(a.reference_contract).get('valid')):
        errors.append("reference reuse contract invalid")
    if (a.learned_run/'V48_58_VARIANT_ISOLATION.json').is_file() and not bool(load(a.learned_run/'V48_58_VARIANT_ISOLATION.json').get('valid')):
        errors.append("parallel variant/state isolation invalid")
    for arm,run,mode in (("B",a.native_run,"native"),("C_Main",a.learned_run,"learned")):
        arm_doc={}; run_checks[arm]=arm_doc
        terminal=run/'dedicated_recalibration_status.json'
        factor=run/'V48_58_FACTOR_CONTRACT.json'
        for p in (terminal,factor):
            if not p.is_file(): errors.append(f"{arm}: missing {p}")
        if terminal.is_file():
            td=load(terminal); codes=td.get('controller_exit_codes') or {}
            terminal_ok=(bool(td.get('certificate_executed')) and bool(td.get('gate_evaluated')) and
                         all(int(codes.get(v,-1)) in (0,20) for v in ('balanced','precision')) and not bool(td.get('test_roots_read')))
            arm_doc['terminal_status_valid']=terminal_ok
            if not terminal_ok: errors.append(f"{arm}: invalid terminal certificate status {codes}")
        for variant in ('balanced','precision'):
            base=run/'candidates'/variant
            cal=base/'calibration'
            required=[
                base/'POLICY_CONTRACT.env', cal/'METRIC_CALIBRATION_CONTRACT.json', cal/'CERTIFICATE_CALIBRATION_COMPLETE.json',
                cal/'dev_diagnostic_near_v48.json', cal/'dev_diagnostic_contact_v48.json',
                cal/'dev_diagnostic_near_v48.proposal_rows.jsonl', cal/'dev_diagnostic_contact_v48.proposal_rows.jsonl',
                cal/'direct_value_risk_near_v48.json', cal/'direct_value_risk_contact_v48.json',
                cal/'direct_value_risk_near_v48.proposal_rows.jsonl', cal/'direct_value_risk_contact_v48.proposal_rows.jsonl',
                cal/'calibration_safe_v48.json', cal/'calibration_near_v48.json', cal/'calibration_contact_v48.json',
            ]
            missing=[str(p) for p in required if not p.is_file() or p.stat().st_size==0]
            if missing: errors.append(f"{arm}/{variant}: missing/empty artifacts: {missing}")
            vdoc={'missing':missing}; arm_doc[variant]=vdoc
            metric=cal/'METRIC_CALIBRATION_CONTRACT.json'
            if metric.is_file():
                md=load(metric); sc=md.get('selection_contract') or {}
                metric_ok=(bool(md.get('valid')) and sc.get('mode')==mode and bool(sc.get('mode_valid')) and
                           bool(sc.get('threshold_valid')) and bool(sc.get('selection_semantics_valid')) and
                           sc.get('expected_selection_semantics')==RIFA_ORDER and not bool(md.get('test_roots_read')))
                vdoc['metric_contract_valid']=metric_ok
                if not metric_ok: errors.append(f"{arm}/{variant}: metric/selection contract invalid")
            cert=cal/'CERTIFICATE_CALIBRATION_COMPLETE.json'
            if cert.is_file():
                cd=load(cert); cert_ok=(bool(cd.get('certificate_executed')) and bool(cd.get('gate_evaluated')) and bool(cd.get('certificate_data_valid')) and not bool(cd.get('test_roots_read')))
                vdoc['certificate_artifacts_valid']=cert_ok
                if not cert_ok: errors.append(f"{arm}/{variant}: certificate calibration incomplete")
            if arm=='C_Main':
                state=base/'V48_58_STAGE_I_STATE_ISOLATION.json'
                if not state.is_file() or not bool(load(state).get('valid')):
                    errors.append(f"{arm}/{variant}: Stage-I isolation missing/invalid")
        for p in [terminal,factor]:
            if p.is_file(): hashes[str(p)]=sha(p)
    for p in required_top:
        if p.is_file(): hashes[str(p)]=sha(p)
    valid=not errors
    doc={
        'schema':'ocrap-v48.58.2-pipeline-complete-v1','valid':valid,'algorithm_version':'v48.58-DCP-DRFC-BCDE-RIFA',
        'engineering_version':'v48.58.2-RIFA-SELECTION-CONTRACT-HOTFIX','attribution_ready':valid,
        'run_checks':run_checks,'errors':errors,'artifact_sha256':hashes,'test_roots_read':False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'event':'v48_58_pipeline_complete','valid':valid,'output':str(a.output)}))
    return 0 if valid else 30

if __name__=='__main__': raise SystemExit(main())
