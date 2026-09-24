#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from ocrap.audits.fixed_main_stability import NEAR_BENEFIT,NEAR_HARD_NO_HARM,_gate_benefit,_gate_no_harm,target_keys
SECONDARY={"closed_loop_bounded_NUP":"higher","route_progression_m":"higher","near_contact_exposure_duration_s":"lower"}
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--selector',required=True); ap.add_argument('--nominal-full',required=True); ap.add_argument('--balanced-full',required=True); ap.add_argument('--precision-full',required=True); ap.add_argument('--balanced-vs-nominal',required=True); ap.add_argument('--precision-vs-nominal',required=True); ap.add_argument('--route-audit',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    errors=[]; n=load(a.nominal_full); route=load(a.route_audit); nkeys=target_keys(n)
    if len(nkeys)!=250: errors.append(f'nominal_population:{len(nkeys)}')
    if not route.get('valid'): errors.append('route_audit_invalid')
    rows={}; common=None
    for v in ('balanced','precision'):
        full=load(getattr(a,f'{v}_full')); comp=load(getattr(a,f'{v}_vs_nominal'))
        if target_keys(full)!=nkeys: errors.append(f'target_mismatch:{v}')
        if int(comp.get('num_paired_scenes') or -1)!=250: errors.append(f'pair_coverage:{v}')
        got=str((full.get('selector_config') or {}).get('ocrap_selector') or '')
        if got!=a.selector: errors.append(f'selector:{v}:{got}')
        h=_gate_no_harm(comp,NEAR_HARD_NO_HARM); b=_gate_benefit(comp,NEAR_BENEFIT); s=_gate_no_harm(comp,SECONDARY)
        bs=set(b['beneficial_metrics']); common=bs if common is None else common & bs
        rows[v]={"hard_no_harm":h,"benefit":b,"secondary_system_no_harm":s,"primary_near_go":bool(h['go'] and b['go'])}
    common=common or set(); go=bool(not errors and all(r['primary_near_go'] and r['secondary_system_no_harm']['go'] for r in rows.values()) and common)
    out={"schema":"ocrap-fresh-near-confirmation-v1","valid":not errors,"attribution_ready":not errors,"selector":a.selector,"go":go,"status":"FRESH_FULL_250_NEAR_GO" if go else ("FRESH_FULL_250_NEAR_STOP" if not errors else "FRESH_NEAR_ATTRIBUTION_NOT_ENTERED"),"common_beneficial_metrics":sorted(common),"errors":errors,"variants":rows,"next_branch":"run_observation_legal_safe_and_common_anchor_contact_under_same_frozen_selector_before_deployed_freeze" if go else "do_not_freeze_deployed_main_keep_recovery_mechanism_frozen_and_continue_near_system_axis_diagnosis"}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n'); print(json.dumps({"valid":out['valid'],"status":out['status'],"next_branch":out['next_branch']},indent=2)); return 0 if not errors else 30
if __name__=='__main__': raise SystemExit(main())
