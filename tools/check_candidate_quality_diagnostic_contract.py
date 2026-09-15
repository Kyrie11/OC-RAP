#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

FROZEN_SCIENTIFIC_FILES = [
    "src/ocrap/planning/selector.py",
    "src/ocrap/evaluation/baselines.py",
    "configs/v48_124_10_near_relative_delta.yaml",
    "configs/v48_124_10_near_nested_evidence.yaml",
]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--reference-source-manifest',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    repo=Path(args.repo).resolve(); ref=json.loads(Path(args.reference_source_manifest).read_text(encoding='utf-8'))
    ref_files=ref.get('files') or {}
    errors=[]; checks={}
    for rel in FROZEN_SCIENTIFIC_FILES:
        p=repo/rel; meta=ref_files.get(rel)
        actual=sha(p) if p.is_file() else None; expected=(meta or {}).get('sha256')
        ok=bool(actual and expected and actual==expected)
        checks[rel]={'actual_sha256':actual,'expected_sha256':expected,'match':ok}
        if not ok: errors.append(f'frozen scientific source mismatch: {rel}')
    out={
      'schema':'ocrap-v48.124.10.3-candidate-quality-runtime-contract-v1',
      'engineering_version':'v48.124.10.3-CANDIDATE-QUALITY-DIAGNOSTIC',
      'scientific_version':'v48.124-OC-FMSA',
      'algorithm_modified':False,
      'diagnostic_instrumentation_only':True,
      'valid':not errors,'attribution_ready':not errors,'errors':errors,'frozen_scientific_files':checks,
      'diagnostic_contract':{
        'teacher_labels_after_action_selection_only':True,
        'audit_intervention_only':True,
        'audit_candidate_scope':'all',
        'audit_records_not_policy_inputs':True,
        'publication_evidence':False,
      }
    }
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2,sort_keys=True))
    if errors: raise SystemExit(30)
if __name__=='__main__': main()
