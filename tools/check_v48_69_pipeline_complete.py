#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

RIFA='rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank'
EXPECTED_SCHEMA=5
EXPECTED_SOURCE='demand_tempered_projected_recovery_witness'

def load(p:Path): return json.loads(p.read_text())
def sha(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(1<<20),b''): h.update(z)
    return h.hexdigest()

def check_run(run:Path, errors:list[str], hashes:dict[str,str]):
    vi=run/'V48_69_VARIANT_ISOLATION.json'
    factor=run/'V48_69_FACTOR_CONTRACT.json'
    terminal=run/'dedicated_recalibration_status.json'
    for p in (vi,factor,terminal):
        if not p.is_file(): errors.append(f'missing {p}')
    if vi.is_file():
        d=load(vi)
        if not d.get('valid'): errors.append(f'{run.name}: variant isolation invalid')
        req={'active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,'route_alignment':True,'reentry_alignment':True,'control_projection':True,'boundary_transport':False,'projection_fidelity':True,'demand_normalized_fidelity':True,'robust_occupancy':False,'test_roots_read':False}
        for k,v in req.items():
            if d.get(k)!=v: errors.append(f'{run.name}: variant {k} mismatch {d.get(k)!r}!={v!r}')
    if factor.is_file():
        d=load(factor)
        req={
            'trainable_parameters':2,'threshold':.5,'threshold_search':False,'regime_id_input':False,
            'proposal_top_k':5,'proposal_expansion':False,'test_roots_read':False,
            'semantic_witness_feature_schema':EXPECTED_SCHEMA,'semantic_witness_feature_source':EXPECTED_SOURCE,
            'active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,
            'route_alignment':True,'reentry_alignment':True,'control_projection':True,
            'boundary_transport':False,'projection_fidelity':True,'demand_normalized_fidelity':True,
            'robust_occupancy':False,'relative_score_intervention':False,'teacher_future_input':False,
        }
        for k,v in req.items():
            if d.get(k)!=v: errors.append(f'{run.name}: factor {k} mismatch {d.get(k)!r}!={v!r}')
    for v in ('balanced','precision'):
        base=run/'candidates'/v; cal=base/'calibration'
        req=[
            base/'model_v48_trac_sr'/'best.pt',base/'model_v48_trac_sr'/'train_summary.json',
            base/'TRAINING_COMPLETE.json',base/'EVIDENCE_CORRECTION_COMPLETE.json',
            base/'V48_69_STAGE_I_STATE_ISOLATION.json',base/'POLICY_CONTRACT.env',
            cal/'METRIC_CALIBRATION_CONTRACT.json',cal/'dev_diagnostic_near_v48.proposal_rows.jsonl',
            cal/'dev_diagnostic_contact_v48.proposal_rows.jsonl',cal/'direct_value_risk_near_v48.proposal_rows.jsonl',
            cal/'direct_value_risk_contact_v48.proposal_rows.jsonl',
        ]
        miss=[str(p) for p in req if not p.is_file() or p.stat().st_size==0]
        if miss: errors.append(f'{run.name}/{v}: missing/empty {miss}')
        stp=base/'V48_69_STAGE_I_STATE_ISOLATION.json'
        if stp.is_file():
            st=load(stp)
            if not (st.get('valid') and st.get('stage_i_bitwise_identity') and st.get('semantic_witness_feature_contract_valid') and st.get('factor_flags_valid')):
                errors.append(f'{run.name}/{v}: state isolation invalid')
            if not (st.get('semantic_witness_feature_schema')==EXPECTED_SCHEMA and st.get('semantic_witness_feature_source')==EXPECTED_SOURCE):
                errors.append(f'{run.name}/{v}: state feature contract mismatch')
        mp=cal/'METRIC_CALIBRATION_CONTRACT.json'
        if mp.is_file():
            md=load(mp); sc=md.get('selection_contract') or {}
            if not (md.get('valid') and sc.get('mode')=='learned' and sc.get('threshold_valid') and sc.get('selection_semantics_valid') and sc.get('expected_selection_semantics')==RIFA and not md.get('test_roots_read')):
                errors.append(f'{run.name}/{v}: metric calibration contract invalid')
    for p in (vi,factor,terminal):
        if p.is_file(): hashes[str(p)]=sha(p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference-contract',type=Path,required=True)
    ap.add_argument('--v68-complete',type=Path,required=True)
    ap.add_argument('--v68-comparison',type=Path,required=True)
    ap.add_argument('--dtrw-run',type=Path,required=True)
    ap.add_argument('--feasibility-audit',type=Path,required=True)
    ap.add_argument('--demand-audit',type=Path,required=True)
    ap.add_argument('--truth-audit',type=Path,required=True)
    ap.add_argument('--comparison',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); errors=[]; hashes={}
    tops=(a.reference_contract,a.v68_complete,a.v68_comparison,a.feasibility_audit,a.demand_audit,a.truth_audit,a.comparison)
    for p in tops:
        if not p.is_file(): errors.append(f'missing {p}')
    if a.reference_contract.is_file() and not load(a.reference_contract).get('valid'): errors.append('reference contract invalid')
    if a.v68_complete.is_file():
        d=load(a.v68_complete)
        if not (d.get('valid') and d.get('attribution_ready') and d.get('engineering_version')=='v48.68.0-OC-RTRW' and not d.get('test_roots_read')):
            errors.append('V48.68 prerequisite invalid')
    if a.v68_comparison.is_file():
        d=load(a.v68_comparison); pr=d.get('preregistered_decision') or {}
        if not (pr.get('status')=='STOP' and pr.get('projection_fidelity_mechanism_gate') is True and pr.get('robust_occupancy_mechanism_gate') is False):
            errors.append('V48.68 scientific branch prerequisite invalid')
    check_run(a.dtrw_run,errors,hashes)
    for p in tops:
        if p.is_file(): hashes[str(p)]=sha(p)
    valid=not errors
    doc={
        'schema':'ocrap-v48.69-dtrw-pipeline-complete-v1','valid':valid,'attribution_ready':valid,
        'algorithm_version':'v48.69-DCP-DRFC-BCDE-RIFA-OC-DTRW','engineering_version':'v48.69.0-OC-DTRW',
        'errors':errors,'artifact_sha256':hashes,
        'factorial_arms':{'D69_Main_OCDTRW':str(a.dtrw_run),'historical_T68':'v48.68 T_FIDELITY','historical_Q67':'v48.67 Q_CTRLPROJ','historical_P66':'v48.66 OC-ACRW Main'},
        'test_roots_read':False,'dataset_reconstruction':False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_69_dtrw_pipeline_complete','valid':valid,'output':str(a.output)}))
    return 0 if valid else 30
if __name__=='__main__': raise SystemExit(main())
