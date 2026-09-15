#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path

REQUIRED_102 = [
    'OC-RAP-v48.124.10.2-observation-legal-near-adjudication.json',
    'OC-RAP-v48.124.10.2-result-bundle-manifest.json',
    'nominal/near/closed_loop_nominal.json',
    'base/balanced/closed_loop_ocrap.json',
    'base/precision/closed_loop_ocrap.json',
    'cohort/balanced_keys.json',
    'cohort/precision_keys.json',
    'provenance/frozen_checkpoint_contract.json',
    'provenance/observation_legal_runtime_contract.json',
    'provenance/runtime_source_snapshot_manifest.json',
    'provenance/all_route_audit.json',
    'support/near_dataset_support.json',
]
REQUIRED_103 = [
    'OC-RAP-v48.124.10.3-CANDIDATE-QUALITY-AUDIT.json',
    'OC-RAP-v48.124.10.3-result-bundle-manifest.json',
    'provenance/candidate_quality_runtime_contract.json',
    'provenance/current_checkpoint_contract.json',
    'provenance/route_audit.json',
    'provenance/runtime_source_snapshot_manifest.json',
]

def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def _verify(zf: zipfile.ZipFile, manifest_name: str) -> tuple[dict, int]:
    names=set(zf.namelist())
    if manifest_name not in names:
        raise SystemExit(f'missing manifest {manifest_name}')
    m=json.loads(zf.read(manifest_name))
    if not (m.get('complete_exit_zero') is True and int(m.get('pipeline_exit_code', -1)) == 0):
        raise SystemExit(f'pipeline not exit-zero for {manifest_name}')
    rows=m.get('files') or {}
    for rel,meta in rows.items():
        if rel not in names: raise SystemExit(f'manifest member missing: {rel}')
        raw=zf.read(rel)
        if _sha(raw)!=str((meta or {}).get('sha256','')): raise SystemExit(f'manifest SHA mismatch: {rel}')
        if len(raw)!=int((meta or {}).get('size',-1)): raise SystemExit(f'manifest size mismatch: {rel}')
    return m,len(rows)

def _extract(zf: zipfile.ZipFile, required: list[str], out: Path, prefix: str) -> None:
    names=set(zf.namelist()); missing=[x for x in required if x not in names]
    if missing: raise SystemExit(f'{prefix} missing required files: {missing}')
    for rel in required:
        dst=out/prefix/rel; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(zf.read(rel))

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--near-results-zip',required=True)
    ap.add_argument('--candidate-results-zip',required=True)
    ap.add_argument('--output-dir',required=True)
    a=ap.parse_args(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    p102=Path(a.near_results_zip).resolve(); p103=Path(a.candidate_results_zip).resolve()
    if not p102.is_file(): raise SystemExit(f'missing 10.2 Near result zip: {p102}')
    if not p103.is_file(): raise SystemExit(f'missing 10.3 candidate result zip: {p103}')
    with zipfile.ZipFile(p102) as z:
        m102,n102=_verify(z,'OC-RAP-v48.124.10.2-result-bundle-manifest.json')
        _extract(z,REQUIRED_102,out,'v48124102')
        adj=json.loads(z.read('OC-RAP-v48.124.10.2-observation-legal-near-adjudication.json'))
        if not(adj.get('valid') and adj.get('attribution_ready') and adj.get('status')=='OBSERVATION_LEGAL_NEAR_SYSTEM_AXIS_STOP'):
            raise SystemExit('10.2 reference is not the attribution-ready Near STOP branch')
    with zipfile.ZipFile(p103) as z:
        m103,n103=_verify(z,'OC-RAP-v48.124.10.3-result-bundle-manifest.json')
        _extract(z,REQUIRED_103,out,'v48124103')
        adj103=json.loads(z.read('OC-RAP-v48.124.10.3-CANDIDATE-QUALITY-AUDIT.json'))
        if not(adj103.get('valid') and adj103.get('attribution_ready')):
            raise SystemExit('10.3 candidate audit is not attribution-ready')
        if adj103.get('diagnostic_branch')!='relative_evidence_alignment_bottleneck_better_admitted_teacher_pcd_candidates_exist_but_relative_head_does_not_support_them':
            raise SystemExit(f'unexpected 10.3 diagnostic branch: {adj103.get("diagnostic_branch")}')
    doc={
        'schema':'ocrap-v48.124.10.4-pcd-oracle-reference-v1','valid':True,
        'near_results_zip':str(p102),'near_results_zip_sha256':_sha(p102.read_bytes()),'near_manifest_files_verified':n102,
        'candidate_results_zip':str(p103),'candidate_results_zip_sha256':_sha(p103.read_bytes()),'candidate_manifest_files_verified':n103,
        'reference_status_10_2':adj.get('status'),'reference_branch_10_3':adj103.get('diagnostic_branch'),
    }
    (out/'REFERENCE_CONTRACT.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps(doc,indent=2,sort_keys=True))
if __name__=='__main__': main()
