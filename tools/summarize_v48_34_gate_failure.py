#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as f:
        data=json.load(f)
    if not isinstance(data,dict): raise ValueError(path)
    return data


def _nearest(doc: dict[str, Any]) -> dict[str, Any] | None:
    rows=doc.get('near_miss_frontier') or []
    if not rows: return None
    return min(rows,key=lambda r: float(r.get('constraint_deficit',1e9)))


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    variants={}
    overall=[]
    for variant in ('balanced','precision'):
        cal=args.run/'candidates'/variant/'calibration'
        if not cal.exists(): continue
        regimes={}
        for regime in ('near','contact'):
            cert=_read(cal/f'direct_value_risk_{regime}_v48.json')
            dev=_read(cal/f'dev_frozen_rule_{regime}_v48.json')
            oracle=((cert.get('proposal_constrained_oracle_gate') or {}).get('verify') or {})
            nearest=_nearest(dev)
            kind=str(cert.get('rejection_kind') or '')
            layer=(
                'proposal_or_contract' if not bool(oracle.get('feasible',False)) else
                'development_rule_fit' if not bool(dev.get('valid_for_deployment',False)) else
                'certificate_generalization' if not bool(cert.get('valid_for_deployment',False)) else
                'passed'
            )
            regimes[regime]={
                'failure_layer':layer,
                'rejection_kind':kind,
                'proposal_oracle_feasible':bool(oracle.get('feasible',False)),
                'proposal_safe_positive_groups':oracle.get('proposal_safe_positive_groups'),
                'development_rule_valid':bool(dev.get('valid_for_deployment',False)),
                'development_nearest_rule':nearest,
                'certificate_verify':cert.get('verify'),
                'candidate_positive_auc':cert.get('candidate_positive_auc'),
                'legacy_evidence_only_safe_positive_auc':cert.get('legacy_evidence_only_top1_safe_positive_auc', cert.get('proposal_evidence_top1_safe_positive_auc')),
                'legacy_evidence_only_harm_auc':cert.get('proposal_evidence_top1_harm_auc'),
                'legacy_evidence_only_correlation':cert.get('legacy_evidence_only_top1_correlation', cert.get('proposal_evidence_top1_correlation')),
                'exact_eligible_safe_positive_auc':cert.get('proposal_exact_eligible_top1_safe_positive_auc'),
                'exact_eligible_harm_auc':cert.get('proposal_exact_eligible_top1_harm_auc'),
                'exact_eligible_correlation':cert.get('proposal_exact_eligible_top1_correlation'),
                'exact_eligible_selected_count':cert.get('proposal_exact_eligible_selected_count'),
                'exact_eligible_abstention_rate':cert.get('proposal_exact_eligible_abstention_rate'),
            }
            overall.append(layer)
        variants[variant]=regimes
    result={
        'event':'v48_34_gate_failure_decomposition',
        'run':str(args.run),
        'gate_passed':bool(overall) and all(x=='passed' for x in overall),
        'dominant_failure_layer':(
            'proposal_or_contract' if 'proposal_or_contract' in overall else
            'development_rule_fit' if 'development_rule_fit' in overall else
            'certificate_generalization' if 'certificate_generalization' in overall else
            'passed' if overall else 'missing_results'
        ),
        'variants':variants,
        'interpretation':{
            'proposal_or_contract':'Frozen proposal/label/gate contract is not mathematically feasible.',
            'development_rule_fit':'Oracle support exists, but no threshold rule satisfies adaptation-dev joint safety/coverage constraints.',
            'certificate_generalization':'A dev-valid frozen rule failed independent certificate verification.',
            'passed':'All registered regimes passed.',
        },
        'test_roots_read':False,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    return 0

if __name__=='__main__': raise SystemExit(main())
