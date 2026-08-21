#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference-run',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args(); r=args.reference_run
    common_required=[
        'SOURCE_CHECKPOINT_CONTRACT.json','dedicated_protocol_audit.json',
        'AUTHORITATIVE_RUN_STATUS.json','DATASET_ROOT_CONTRACT.json',
        'evidence_adapt_teacher_pcd_index.jsonl','evidence_adapt_teacher_pcd_index_summary.json',
        'evidence_adapt_dev_teacher_pcd_index.jsonl','evidence_adapt_dev_teacher_pcd_index_summary.json',
    ]
    factor_name = (
        'V48_56_FACTOR_CONTRACT.json' if (r/'V48_56_FACTOR_CONTRACT.json').is_file()
        else 'V48_57_FACTOR_CONTRACT.json' if (r/'V48_57_FACTOR_CONTRACT.json').is_file()
        else ''
    )
    required=common_required + ([factor_name] if factor_name else [])
    missing=[x for x in common_required if not (r/x).is_file()]
    if not factor_name:
        missing.append('V48_56_FACTOR_CONTRACT.json|V48_57_FACTOR_CONTRACT.json')
    checks={"reference_exists":r.is_dir(),"required_files_present":not missing}
    errors=[]
    if missing: errors.append({'missing':missing})
    factor=source=protocol=status=dataset={}
    if not missing:
        factor=_json(r/factor_name)
        source=_json(r/'SOURCE_CHECKPOINT_CONTRACT.json')
        protocol=_json(r/'dedicated_protocol_audit.json')
        status=_json(r/'AUTHORITATIVE_RUN_STATUS.json')
        dataset=_json(r/'DATASET_ROOT_CONTRACT.json')
        if factor_name == 'V48_56_FACTOR_CONTRACT.json':
            expected={
                'arm':'A','factor_x_deployability_zero_boundary':False,'factor_y_gap_ordinal_only':False,
                'native_dep_boundary_aligned':False,'native_certificate_preservation':True,
                'native_advantage_preservation':True,'boundary_complete_frontier':True,
                'root_logit_recalibration':False,'strategy_regime_conditioning':False,'proposal_top_k':5,
                'test_roots_read':False,
            }
        else:
            expected={
                'arm':'A','factor_common_measure_root_invariance':False,
                'native_dep_boundary_aligned':False,'gap_ordinal_only':False,
                'native_certificate_preservation':True,'native_advantage_preservation':True,
                'boundary_complete_frontier':True,'root_logit_recalibration':False,
                'strategy_regime_conditioning':False,'proposal_top_k':5,'test_roots_read':False,
            }
        checks['exact_v48_56_A_semantics']=all(factor.get(k)==v for k,v in expected.items())
        checks['factor_contract_source']=factor_name
        checks['source_contract_valid']=bool(source.get('valid')) and not bool(source.get('test_roots_read'))
        checks['protocol_valid']=bool(protocol.get('valid'))
        checks['dataset_contract_valid']=bool(dataset.get('valid')) and not bool(dataset.get('test_roots_read'))
        checks['pipeline_valid']=bool(status.get('pipeline_valid')) and int(status.get('authoritative_exit_code',99)) in {0,20}
        overlaps=protocol.get('scene_overlaps',{}) or {}
        checks['scene_disjoint']=all(int(v)==0 for v in overlaps.values())
        for name in ('balanced','precision'):
            c=(source.get('checks',{}) or {}).get(name,{})
            checks[f'source_{name}_hash_valid']=bool(c.get('manifest_hash_match')) and bool(c.get('nonempty'))
        if not checks['exact_v48_56_A_semantics']:
            errors.append({'factor_contract_mismatch':{k:{'expected':v,'actual':factor.get(k)} for k,v in expected.items() if factor.get(k)!=v}})
    valid=all(bool(v) for k,v in checks.items() if k != 'factor_contract_source')
    doc={
        'schema':'ocrap-v48.58-reference-reuse-contract-v1','valid':valid,
        'reference_run':str(r.resolve(strict=False)),'checks':checks,'errors':errors,
        'reference_artifact_sha256':({x:_sha(r/x) for x in required if (r/x).is_file()}),
        'source_checkpoint_sha256':{
            k:v.get('sha256') for k,v in ((source.get('checks',{}) or {}).items()) if isinstance(v,dict)
        },
        'reuse_rationale':'v48.58 holds Stage-I exactly at validated v48.56-A semantics; only the new lexicographic absolute-feasibility stage is intervened on',
        'test_roots_read':False,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'event':'v48_58_reference_reuse_check','valid':valid,'output':str(args.output)}))
    return 0 if valid else 30
if __name__=='__main__': raise SystemExit(main())
