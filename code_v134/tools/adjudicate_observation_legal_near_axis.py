#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
from ocrap.audits.fixed_main_stability import NEAR_BENEFIT, NEAR_HARD_NO_HARM, _gate_benefit, _gate_no_harm, target_keys

ENGINEERING_VERSION="v48.124.10.2-OBSERVATION-LEGAL-PROVENANCE-ENGFIX"
SECONDARY={"closed_loop_bounded_NUP":"higher","route_progression_m":"higher","near_contact_exposure_duration_s":"lower"}

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def gate(comps: dict[str,dict[str,Any]]):
    rows={}; common=None
    for v in ('balanced','precision'):
        h=_gate_no_harm(comps[v],NEAR_HARD_NO_HARM); b=_gate_benefit(comps[v],NEAR_BENEFIT); s=_gate_no_harm(comps[v],SECONDARY)
        bs=set(b['beneficial_metrics']); common=bs if common is None else common & bs
        rows[v]={"primary_near_go":bool(h['go'] and b['go']),"hard_no_harm":h,"benefit":b,"secondary_system_no_harm":s}
    common=common or set(); primary=bool(all(rows[v]['primary_near_go'] for v in rows) and common); sec=all(rows[v]['secondary_system_no_harm']['go'] for v in rows)
    return {"primary_near_gate_go":primary,"common_beneficial_metrics":sorted(common),"secondary_system_no_harm_go":bool(sec),"promotion_ready":bool(primary and sec),"variants":rows}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runtime-contract',required=True); ap.add_argument('--route-audit',required=True); ap.add_argument('--nominal-full',required=True)
    for v in ('balanced','precision'):
        ap.add_argument(f'--base-{v}-full',required=True); ap.add_argument(f'--base-{v}-vs-nominal',required=True)
    for arm in ('delta','nested'):
        for v in ('balanced','precision'):
            ap.add_argument(f'--{arm}-{v}-full',required=True); ap.add_argument(f'--{arm}-{v}-vs-nominal',required=True); ap.add_argument(f'--{arm}-{v}-vs-base',required=True)
    ap.add_argument('--output',required=True); a=ap.parse_args(); errors=[]
    runtime=load(a.runtime_contract); route=load(a.route_audit); nominal=load(a.nominal_full); nkeys=target_keys(nominal)
    if not(runtime.get('valid') and runtime.get('attribution_ready') and runtime.get('engineering_version')==ENGINEERING_VERSION): errors.append('runtime_contract_invalid')
    if not route.get('valid'): errors.append('observation_legal_route_audit_invalid')
    if len(nkeys)!=250: errors.append(f'nominal_population:{len(nkeys)}')
    base_comp={}; base_full={}
    for v in ('balanced','precision'):
        bf=load(getattr(a,f'base_{v}_full')); bc=load(getattr(a,f'base_{v}_vs_nominal')); base_full[v]=bf; base_comp[v]=bc
        if target_keys(bf)!=nkeys: errors.append(f'base_target_mismatch:{v}')
        if int(bc.get('num_paired_scenes') or -1)!=250: errors.append(f'base_pair_coverage:{v}')
    arms={"base":{"gate":gate(base_comp)}}
    expected={'delta':'lcb_constrained_relative_delta','nested':'lcb_constrained_nested_evidence'}
    for arm in ('delta','nested'):
        comps={}; variants={}
        for v in ('balanced','precision'):
            full=load(getattr(a,f'{arm}_{v}_full')); cn=load(getattr(a,f'{arm}_{v}_vs_nominal')); cb=load(getattr(a,f'{arm}_{v}_vs_base'))
            if target_keys(full)!=nkeys: errors.append(f'{arm}_target_mismatch:{v}')
            if int(cn.get('num_paired_scenes') or -1)!=250 or int(cb.get('num_paired_scenes') or -1)!=250: errors.append(f'{arm}_pair_coverage:{v}')
            scfg=full.get('selector_config') or {}
            selector=str(scfg.get('ocrap_selector') or '')
            if selector!=expected[arm]: errors.append(f'{arm}_selector:{v}:{selector}')
            try:
                if abs(float(scfg.get('rifa_relative_min_advantage'))-0.0)>1e-12: errors.append(f'{arm}_relative_min_advantage:{v}')
                if arm=='nested':
                    if int(scfg.get('rifa_relative_proposal_top_k'))!=5: errors.append(f'nested_topk:{v}')
                    if abs(float(scfg.get('rifa_relative_opportunity_threshold'))-0.65)>1e-12: errors.append(f'nested_opportunity:{v}')
                    if abs(float(scfg.get('rifa_relative_harm_threshold'))-0.30)>1e-12: errors.append(f'nested_harm:{v}')
            except Exception:
                errors.append(f'{arm}_relative_metadata_contract:{v}')
            recon=full.get('exact_monotone_subset_reconstruction') or {}
            if not recon.get('internal_diagnostic_only'): errors.append(f'{arm}_reconstruction_contract:{v}')
            comps[v]=cn; variants[v]={"selector":selector,"comparison_vs_nominal":cn,"comparison_vs_base":cb,"intervention_rate":full.get('intervention_rate')}
        arms[arm]={"gate":gate(comps),"variants":variants}
    if errors:
        status='OBSERVATION_LEGAL_NEAR_ATTRIBUTION_NOT_ENTERED'; promoted=None; next_branch='fix_engineering_only'
    elif arms['base']['gate']['promotion_ready']:
        status='OBSERVATION_LEGAL_BASE_MAIN_NEAR_GO'; promoted='base'; next_branch='base_main_near_closes_under_legal_route_run_fresh_safe_and_contact_before_any_deployed_freeze'
    elif arms['delta']['gate']['promotion_ready']:
        status='OBSERVATION_LEGAL_RELATIVE_SIGN_GO'; promoted='delta'; next_branch='run_fresh_full_250_near_relative_sign_confirmation_then_safe_contact_before_freeze'
    elif arms['nested']['gate']['promotion_ready']:
        status='OBSERVATION_LEGAL_NESTED_EVIDENCE_GO'; promoted='nested'; next_branch='run_fresh_full_250_near_nested_confirmation_then_safe_contact_before_freeze'
    else:
        status='OBSERVATION_LEGAL_NEAR_SYSTEM_AXIS_STOP'; promoted=None; next_branch='keep_recovery_mechanism_frozen_audit_absolute_admitted_candidate_action_quality_no_v48_125_threshold_capacity_or_retraining_sweep'
    out={"schema":"ocrap-v48.124.10.2-observation-legal-near-axis-v1","engineering_version":ENGINEERING_VERSION,"scientific_version":"v48.124-OC-FMSA","valid":not errors,"attribution_ready":not errors,"errors":errors,"status":status,"promoted_arm":promoted,"next_branch":next_branch,"arms":arms,
         "freeze_rule":"No deployed-Main freeze is authorized by reconstructed diagnostic evidence alone. A promoted delta/nested arm requires a fresh full 250-scene Near confirmation; any Near GO then requires observation-legal Safe and common-anchor Contact confirmation under the same frozen selector."}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({"valid":out['valid'],"status":status,"promoted_arm":promoted,"next_branch":next_branch},indent=2)); return 0 if not errors else 30
if __name__=='__main__': raise SystemExit(main())
