#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path

REQUIRED = [
    'OC-RAP-v48.124.10.4-PCD-ORACLE-CEILING.json',
    'OC-RAP-v48.124.10.4-result-bundle-manifest.json',
    'provenance/runtime_source_snapshot_manifest.json',
    'provenance/current_checkpoint_contract.json',
    'reference/v48124102/OC-RAP-v48.124.10.2-observation-legal-near-adjudication.json',
    'reference/v48124102/base/balanced/closed_loop_ocrap.json',
    'reference/v48124102/base/precision/closed_loop_ocrap.json',
    'reference/v48124102/cohort/balanced_keys.json',
    'reference/v48124102/cohort/precision_keys.json',
    'reference/v48124102/nominal/near/closed_loop_nominal.json',
    'reference/v48124102/provenance/frozen_checkpoint_contract.json',
    'reference/v48124102/support/near_dataset_support.json',
]

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--ceiling-results-zip', required=True)
    ap.add_argument('--output-dir', required=True)
    a=ap.parse_args(); src=Path(a.ceiling_results_zip).resolve(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    if not src.is_file(): raise SystemExit(f'missing 10.4 result zip: {src}')
    with zipfile.ZipFile(src) as z:
        names=set(z.namelist())
        mname='OC-RAP-v48.124.10.4-result-bundle-manifest.json'
        if mname not in names: raise SystemExit('10.4 manifest missing')
        m=json.loads(z.read(mname)); rows=m.get('files') or {}
        if not (m.get('complete_exit_zero') is True and int(m.get('pipeline_exit_code',-1))==0):
            raise SystemExit('10.4 pipeline not exit-zero')
        for rel,meta in rows.items():
            if rel not in names: raise SystemExit(f'10.4 manifest member missing: {rel}')
            raw=z.read(rel)
            if sha(raw)!=str((meta or {}).get('sha256','')): raise SystemExit(f'10.4 manifest SHA mismatch: {rel}')
            if len(raw)!=int((meta or {}).get('size',-1)): raise SystemExit(f'10.4 manifest size mismatch: {rel}')
        miss=[r for r in REQUIRED if r not in names]
        if miss: raise SystemExit(f'10.4 required reference files missing: {miss}')
        adj=json.loads(z.read('OC-RAP-v48.124.10.4-PCD-ORACLE-CEILING.json'))
        if not(adj.get('valid') and adj.get('attribution_ready') and adj.get('status')=='PCD_ORACLE_CEILING_STOP' and adj.get('go') is False):
            raise SystemExit(f'10.4 did not establish attribution-ready oracle ceiling STOP: {adj.get("status")}')
        expected_next='close_relative_only_repair_as_insufficient_keep_relative_abstention_nonharm_guard_and_localize_candidate_action_realization_or_base_trigger_support_no_threshold_or_relative_head_sweep'
        if adj.get('next_branch')!=expected_next: raise SystemExit(f'unexpected 10.4 next branch: {adj.get("next_branch")}')
        for rel in REQUIRED:
            dst=out/rel; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(z.read(rel))
    doc={
        'schema':'ocrap-v48.124.10.5-all-state-support-reference-v1',
        'valid':True,
        'ceiling_results_zip':str(src),
        'ceiling_results_zip_sha256':sha(src.read_bytes()),
        'manifest_files_verified':len(rows),
        'reference_status_10_4':adj.get('status'),
        'reference_next_branch_10_4':adj.get('next_branch'),
    }
    (out/'REFERENCE_CONTRACT.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(doc,indent=2,sort_keys=True))
if __name__=='__main__': main()
