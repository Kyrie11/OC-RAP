#!/usr/bin/env python3
"""Compare v48.56-A reference against v48.57 CMRI on dev/certificate only.

The comparator is deliberately single-axis: component labels/roles, source
checkpoints, protocol, gate, proposal top-k and thresholds must stay fixed.  It
also ingests the root-source decomposition audits so a downstream metric gain
cannot be called CMRI evidence unless the proposed source mechanism moved in the
predicted direction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    d=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(d,dict): raise TypeError(path)
    return d


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def _f(x: Any) -> float:
    try:
        y=float(x); return y if math.isfinite(y) else float('nan')
    except Exception: return float('nan')


def _med(xs: list[float]) -> float | None:
    ys=[x for x in xs if math.isfinite(x)]
    return None if not ys else float(median(ys))


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()


def _run_identity(run: Path) -> dict[str, Any]:
    src=_json(run/'SOURCE_CHECKPOINT_CONTRACT.json')
    gate=_json(run/'GATE_SPEC.json')
    data=_json(run/'DATASET_ROOT_CONTRACT.json')
    checks=src.get('checks') or {}
    protocol=gate.get('protocol') or {}
    manifests={str(d.get('role')): d.get('manifest_sha256') for d in (protocol.get('datasets') or []) if isinstance(d,dict) and d.get('role')}
    return {
        'source_run_resolved':src.get('source_run_resolved'),
        'source_checkpoint_sha256':{v:(checks.get(v) or {}).get('sha256') for v in ('balanced','precision')},
        'gate_protocol_sha256':gate.get('protocol_sha256'),
        'gate_dataset_manifest_sha256':manifests,
        'dataset_root_contract_valid':bool(data.get('valid')),
        'test_roots_read':bool(src.get('test_roots_read')) or bool(data.get('test_roots_read')) or bool(protocol.get('test_roots_read')),
    }


def _cert(run: Path, regime: str) -> dict[str, Any]:
    cal=run/'candidates'/'precision'/'calibration'
    d=_json(cal/f'direct_value_risk_{regime}_v48.json')
    s=d.get('verify') if isinstance(d.get('verify'),dict) else d.get('all',{})
    return {
        'candidate_safe_positive_auc':d.get('candidate_safe_positive_auc'),
        'candidate_harm_auc':d.get('candidate_harm_auc'),
        'proposal_safe_positive_auc':d.get('proposal_evidence_top1_safe_positive_auc'),
        'proposal_harm_auc':d.get('proposal_evidence_top1_harm_auc'),
        'selected':s.get('num_selected'),
        'positive_selected':s.get('num_positive_selected'),
        'harmful_selected':s.get('num_harmful_selected'),
        'positive_recall':s.get('positive_recall'),
        'precision':s.get('precision'),
        'harmful_selected_ucb90':s.get('harmful_selected_ucb90',s.get('harmful_selected_ucb')),
        'valid_for_deployment':d.get('valid_for_deployment'),
        'rejection_kind':d.get('rejection_kind'),
    }


def _proposal_geometry(run: Path, regime: str, positive_gain: float=0.015) -> dict[str, Any]:
    p=run/'candidates'/'precision'/'calibration'/f'direct_value_risk_{regime}_v48.proposal_rows.jsonl'
    rows=_rows(p)
    safe=[r for r in rows if _f(r.get('teacher_adv'))>=positive_gain and not bool(r.get('teacher_harmful',False))]
    harmful=[r for r in rows if bool(r.get('teacher_harmful',False))]
    def coord(subset: list[dict[str,Any]], idx:int)->list[float]:
        out=[]
        for r in subset:
            v=r.get('predicted_native_pair_margins')
            if isinstance(v,list) and idx<len(v) and math.isfinite(_f(v[idx])): out.append(_f(v[idx]))
        return out
    out={'safe_positive_n':len(safe),'harmful_n':len(harmful),'coordinates':{}}
    for idx,name in enumerate(('drs','deployability','gap_quality')):
        sp=coord(safe,idx); hp=coord(harmful,idx)
        out['coordinates'][name]={
            'safe_positive_false_veto_fraction':None if not sp else sum(x>0 for x in sp)/len(sp),
            'safe_positive_margin_median':_med(sp),
            'harmful_false_safe_fraction':None if not hp else sum(x<=0 for x in hp)/len(hp),
            'harmful_margin_median':_med(hp),
        }
    out['safe_positive_native_smooth_adv_margin_median']=_med([_f(r.get('native_smooth_adv_margin')) for r in safe])
    out['safe_positive_pred_adv_median']=_med([_f(r.get('pred_adv')) for r in safe])
    out['safe_positive_pred_adv_nonnegative_fraction']=(None if not safe else sum(_f(r.get('pred_adv'))>=0 for r in safe)/len(safe))
    return out


def _dev(run: Path, regime: str, positive_gain: float=0.015) -> dict[str, Any]:
    cal=run/'candidates'/'precision'/'calibration'
    d=_json(cal/f'dev_diagnostic_{regime}_v48.json')
    s=d.get('fit') if isinstance(d.get('fit'),dict) else d.get('all',{})
    rows=_rows(cal/f'dev_diagnostic_{regime}_v48.proposal_rows.jsonl')
    safe=[r for r in rows if _f(r.get('teacher_adv'))>=positive_gain and not bool(r.get('teacher_harmful',False))]
    return {
        'selected':s.get('num_selected'),'positive_selected':s.get('num_positive_selected'),
        'harmful_selected':s.get('num_harmful_selected'),'positive_recall':s.get('positive_recall'),
        'precision':s.get('precision'),'harmful_selected_ucb90':s.get('harmful_selected_ucb90',s.get('harmful_selected_ucb')),
        'safe_positive_n':len(safe),
        'safe_positive_opportunity_median':_med([_f(r.get('opportunity')) for r in safe]),
        'safe_positive_pred_adv_median':_med([_f(r.get('pred_adv')) for r in safe]),
        'safe_positive_pred_adv_nonnegative_fraction':None if not safe else sum(_f(r.get('pred_adv'))>=0 for r in safe)/len(safe),
        'safe_positive_harm_median':_med([_f(r.get('harm')) for r in safe]),
    }


def _status(run: Path) -> dict[str,Any]:
    d=_json(run/'AUTHORITATIVE_RUN_STATUS.json')
    return {'pipeline_valid':bool(d.get('pipeline_valid')),'authoritative_exit_code':int(d.get('authoritative_exit_code',99))}


def _factor_b(run: Path) -> dict[str,Any]:
    d=_json(run/'V48_57_FACTOR_CONTRACT.json')
    required={
        'arm':'B','factor_common_measure_root_invariance':True,'root_head_retrained':False,
        'root_logit_recalibration':False,'new_learned_parameters':False,
        'native_dep_boundary_aligned':False,'gap_ordinal_only':False,
        'boundary_complete_frontier':True,'strategy_regime_conditioning':False,
        'proposal_top_k':5,'test_roots_read':False,
    }
    return {'valid':all(d.get(k)==v for k,v in required.items()),'checks':{k:{'expected':v,'actual':d.get(k)} for k,v in required.items()},'contract':d}


def _audit_summary(path: Path | None) -> dict[str,Any]:
    if path is None or not path.is_file(): return {'available':False}
    d=_json(path)
    out={'available':True,'checkpoint_sha256':d.get('checkpoint_sha256'),'model_cmri':d.get('model_common_measure_root_mass_enabled'),'buckets':{}}
    for reg,b in (d.get('buckets') or {}).items():
        rr=b.get('relative_r_dep') or {}; pos=b.get('teacher_positive_gain_cohort') or {}; drift=b.get('root_source_drift') or {}; elig=b.get('cmri_eligibility') or {}; align=b.get('root_slot_alignment_diagnostic') or {}
        out['buckets'][reg]={
            'cmri_group_coverage':elig.get('group_coverage'),
            'root_support_identity_rate':elig.get('pair_root_support_identity_rate'),
            'root_signature_nearest_slot_identity_rate_median':(align.get('root_signature_nearest_slot_identity_rate') or {}).get('median'),
            'root_future_signature_nearest_slot_identity_rate_median':(align.get('root_future_signature_nearest_slot_identity_rate') or {}).get('median'),
            'predicted_root_drift_js_median':(drift.get('predicted_candidate_to_nominal_js') or {}).get('median'),
            'teacher_root_drift_js_median':(drift.get('teacher_candidate_to_nominal_js') or {}).get('median'),
            'excess_root_drift_js_median':(drift.get('excess_predicted_minus_teacher_js') or {}).get('median'),
            'legacy_abs_error_median':(rr.get('legacy_abs_error') or {}).get('median'),
            'common_abs_error_median':(rr.get('common_measure_abs_error') or {}).get('median'),
            'common_abs_error_gain_median':(rr.get('common_measure_abs_error_gain_positive_is_better') or {}).get('median'),
            'legacy_sign_accuracy':rr.get('legacy_sign_accuracy'),'common_sign_accuracy':rr.get('common_measure_sign_accuracy'),
            'legacy_safe_positive_auc':rr.get('legacy_safe_positive_auc'),'common_safe_positive_auc':rr.get('common_measure_safe_positive_auc'),
            'teacher_positive_n':pos.get('n'),'legacy_positive_capture':pos.get('legacy_capture_rate'),'common_positive_capture':pos.get('common_measure_capture_rate'),
            'deployed_measure_contract_error_js_median':(drift.get('deployed_recovery_measure_to_nominal_js') or {}).get('median'),
            'native_dep_common_abs_error_median':(drift.get('native_dep_vs_recomputed_common_r_dep_abs_error') or {}).get('median'),
        }
    return out


def _delta(a: Any,b: Any)->float|None:
    aa,bb=_f(a),_f(b)
    return None if not (math.isfinite(aa) and math.isfinite(bb)) else bb-aa


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference-a',required=True,type=Path)
    ap.add_argument('--b',required=True,type=Path)
    ap.add_argument('--audit-a',type=Path)
    ap.add_argument('--audit-b',type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    A,B=args.reference_a,args.b
    ident_a,ident_b=_run_identity(A),_run_identity(B)
    attribution={
        'source_checkpoint_equal':ident_a['source_checkpoint_sha256']==ident_b['source_checkpoint_sha256'],
        'gate_protocol_equal':ident_a['gate_protocol_sha256']==ident_b['gate_protocol_sha256'],
        'dataset_manifests_equal':ident_a['gate_dataset_manifest_sha256']==ident_b['gate_dataset_manifest_sha256'],
        'dataset_contract_valid':ident_a['dataset_root_contract_valid'] and ident_b['dataset_root_contract_valid'],
        'no_test_roots':not ident_a['test_roots_read'] and not ident_b['test_roots_read'],
        'a_status':_status(A),'b_status':_status(B),'b_factor':_factor_b(B),
    }
    attribution['valid']=all([
        attribution['source_checkpoint_equal'],attribution['gate_protocol_equal'],attribution['dataset_manifests_equal'],
        attribution['dataset_contract_valid'],attribution['no_test_roots'],attribution['a_status']['pipeline_valid'],
        attribution['b_status']['pipeline_valid'],attribution['a_status']['authoritative_exit_code'] in {0,20},
        attribution['b_status']['authoritative_exit_code'] in {0,20},attribution['b_factor']['valid']
    ])
    regimes={}
    for reg in ('near','contact'):
        ca,cb=_cert(A,reg),_cert(B,reg); da,db=_dev(A,reg),_dev(B,reg); ga,gb=_proposal_geometry(A,reg),_proposal_geometry(B,reg)
        regimes[reg]={
            'certificate':{'A':ca,'B_CMRI':cb,'delta_B_minus_A':{k:_delta(ca.get(k),cb.get(k)) for k in ('candidate_safe_positive_auc','candidate_harm_auc','proposal_safe_positive_auc','proposal_harm_auc','positive_recall','precision','harmful_selected_ucb90')}},
            'development':{'A':da,'B_CMRI':db,'delta_B_minus_A':{k:_delta(da.get(k),db.get(k)) for k in ('positive_recall','precision','harmful_selected_ucb90','safe_positive_opportunity_median','safe_positive_pred_adv_median','safe_positive_pred_adv_nonnegative_fraction','safe_positive_harm_median')}},
            'native_geometry':{'A':ga,'B_CMRI':gb},
        }
    audit_a=_audit_summary(args.audit_a); audit_b=_audit_summary(args.audit_b)
    # Do not reduce the decision to one scalar.  These booleans are screening
    # aids and keep the preregistered evidence hierarchy explicit.
    deployment_nonworse={}
    for reg in ('near','contact'):
        a=regimes[reg]['certificate']['A']; b=regimes[reg]['certificate']['B_CMRI']
        ar,br=_f(a.get('positive_recall')),_f(b.get('positive_recall')); au,bu=_f(a.get('harmful_selected_ucb90')),_f(b.get('harmful_selected_ucb90'))
        deployment_nonworse[reg]=bool(math.isfinite(ar) and math.isfinite(br) and br>=ar and math.isfinite(au) and math.isfinite(bu) and bu<=au)
    source_basis = audit_b if audit_b.get('available') else audit_a
    source_go={}
    for reg in ('near','contact'):
        x=(source_basis.get('buckets') or {}).get(reg,{})
        gain=_f(x.get('common_abs_error_gain_median')); rawcap=_f(x.get('legacy_positive_capture')); cmcap=_f(x.get('common_positive_capture')); cov=_f(x.get('cmri_group_coverage'))
        source_go[reg]=bool(math.isfinite(gain) and gain>0 and math.isfinite(rawcap) and math.isfinite(cmcap) and cmcap>=rawcap and math.isfinite(cov) and cov>=0.5)
    doc={
        'schema':'ocrap-v48.57-cmri-single-axis-comparison-v1','reference_a':str(A.resolve(strict=False)),'b':str(B.resolve(strict=False)),
        'attribution_contract':attribution,'regimes':regimes,'root_source_audit':{'A_counterfactual_substitution':audit_a,'B_deployment_projection':audit_b,'screening_basis':('B' if audit_b.get('available') else 'A')},
        'screening':{
            'source_mechanism_go_by_regime':source_go,'deployment_recall_risk_nonworse_by_regime':deployment_nonworse,
            'cmri_absorption_candidate':bool(attribution['valid'] and all(source_go.values()) and all(deployment_nonworse.values())),
            'centering_authorized':False,
            'centering_rule':'only reconsider after a genuine Near+Contact upstream/source Pareto improvement; then verify opportunity/pred_adv remains the residual systematic negative bias',
            'stop_rule':'if support/slot alignment is weak, teacher root drift is comparable, source substitution fails, or B gains no Near+Contact source/deployment Pareto, STOP CMRI/common-measure family; do not retune root logits or add thresholds',
        },
        'test_roots_read':False,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'event':'v48_57_cmri_compare_complete','valid':attribution['valid'],'output':str(args.output)},ensure_ascii=False))
    return 0 if attribution['valid'] else 30

if __name__=='__main__': raise SystemExit(main())
