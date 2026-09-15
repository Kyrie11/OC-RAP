#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path

FINAL='OC-RAP-v48.124.10.5-ALL-STATE-SUPPORT-LOCALIZATION.json'
MANIFEST='OC-RAP-v48.124.10.5-result-bundle-manifest.json'
REQUIRED=[
    FINAL, MANIFEST,
    'audit/balanced/closed_loop_ocrap.json',
    'audit/precision/closed_loop_ocrap.json',
    'provenance/runtime_source_snapshot_manifest.json',
    'provenance/current_checkpoint_contract.json',
    'provenance/route_audit.json',
    'reference/OC-RAP-v48.124.10.4-PCD-ORACLE-CEILING.json',
    'reference/reference/v48124102/base/balanced/closed_loop_ocrap.json',
    'reference/reference/v48124102/base/precision/closed_loop_ocrap.json',
    'reference/reference/v48124102/nominal/near/closed_loop_nominal.json',
    'reference/reference/v48124102/provenance/frozen_checkpoint_contract.json',
    'reference/reference/v48124102/support/near_dataset_support.json',
]

def sha(raw:bytes)->str: return hashlib.sha256(raw).hexdigest()

def main()->int:
    ap=argparse.ArgumentParser(description='Verify and materialize the attribution-ready V48.124.10.5 support-localization evidence.')
    ap.add_argument('--support-results-zip',required=True)
    ap.add_argument('--output-dir',required=True)
    a=ap.parse_args(); src=Path(a.support_results_zip).resolve(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    if not src.is_file(): raise SystemExit(f'missing 10.5 results zip: {src}')
    with zipfile.ZipFile(src) as z:
        names=set(z.namelist())
        if MANIFEST not in names: raise SystemExit('10.5 manifest missing')
        m=json.loads(z.read(MANIFEST)); rows=m.get('files') or {}
        if not(m.get('complete_exit_zero') is True and int(m.get('pipeline_exit_code',-1))==0):
            raise SystemExit('10.5 pipeline not exit-zero')
        for rel,meta in rows.items():
            if rel not in names: raise SystemExit(f'10.5 manifest member missing: {rel}')
            raw=z.read(rel)
            if sha(raw)!=str((meta or {}).get('sha256','')): raise SystemExit(f'10.5 manifest SHA mismatch: {rel}')
            if len(raw)!=int((meta or {}).get('size',-1)): raise SystemExit(f'10.5 manifest size mismatch: {rel}')
        miss=[r for r in REQUIRED if r not in names]
        if miss: raise SystemExit(f'10.5 required files missing: {miss}')
        adj=json.loads(z.read(FINAL))
        expected_branch='ABSOLUTE_ADMISSION_BLOCKS_ALL_NONTRIGGER_POSITIVE_PCD_OPPORTUNITIES_ON_FROZEN_INTERVENTION_COHORT'
        expected_next='audit_absolute_admission_false_negative_semantics_on_missed_nontrigger_opportunities_no_threshold_sweep'
        if not(adj.get('valid') and adj.get('attribution_ready') and adj.get('algorithm_modified') is False):
            raise SystemExit('10.5 attribution contract invalid')
        if adj.get('reference_status')!='PCD_ORACLE_CEILING_STOP': raise SystemExit(f'unexpected 10.5 reference status: {adj.get("reference_status")}')
        if adj.get('diagnostic_branch')!=expected_branch: raise SystemExit(f'unexpected 10.5 branch: {adj.get("diagnostic_branch")}')
        if adj.get('next_branch')!=expected_next: raise SystemExit(f'unexpected 10.5 next branch: {adj.get("next_branch")}')
        for rel in REQUIRED:
            dst=out/rel; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(z.read(rel))
    doc={
        'schema':'ocrap-v48.124.10.6-nonfloor-admission-screen-reference-v1','valid':True,'attribution_ready':True,
        'support_results_zip':str(src),'support_results_zip_sha256':sha(src.read_bytes()),
        'manifest_files_verified':len(rows),'reference_status_10_5':adj.get('diagnostic_branch'),'reference_next_branch_10_5':adj.get('next_branch'),
    }
    (out/'REFERENCE_CONTRACT.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(doc,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
