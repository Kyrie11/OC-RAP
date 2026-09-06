#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any, Iterable

from ocrap.v48_94_support_reserve_admission import ENGINEERING_VERSION

ROLES=("dev_near","dev_contact","certificate_near","certificate_contact")
VARIANTS=("balanced","precision")
ROLE_FILES={
 "dev_near":"dev_diagnostic_near_v48.proposal_rows.jsonl",
 "dev_contact":"dev_diagnostic_contact_v48.proposal_rows.jsonl",
 "certificate_near":"direct_value_risk_near_v48.proposal_rows.jsonl",
 "certificate_contact":"direct_value_risk_contact_v48.proposal_rows.jsonl",
}

def load_jsonl(p:Path)->list[dict[str,Any]]:
    out=[]
    with p.open() as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try: out.append(json.loads(line))
                except Exception as e: raise ValueError(f"invalid JSONL {p}:{i}: {e}")
    return out

def key_prop(r:dict[str,Any])->tuple[str,int,int]:
    return (str(r["scene"]),int(r["time"]),int(r["candidate"]))

def key_v93(r:dict[str,Any])->tuple[str,int,int]:
    return (str(r["scene_id"]),int(r["time_index"]),int(r["candidate_index"]))

def auc(pos:Iterable[float], neg:Iterable[float])->float|None:
    p=[float(x) for x in pos if math.isfinite(float(x))]
    n=[float(x) for x in neg if math.isfinite(float(x))]
    if not p or not n: return None
    wins=0.0
    for x in p:
        for y in n:
            wins += 1.0 if x>y else 0.5 if x==y else 0.0
    return wins/(len(p)*len(n))

def rate(vals:list[bool])->float|None:
    return sum(vals)/len(vals) if vals else None

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--l80-run',type=Path,required=True)
    ap.add_argument('--v93-audit',type=Path,required=True)
    ap.add_argument('--main-run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()

    labels_by_role:dict[str,dict[tuple[str,int,int],dict[str,Any]]]={}
    for r in load_jsonl(a.v93_audit):
        role=str(r['dataset_role'])
        if role not in ROLES: continue
        k=key_v93(r)
        if k in labels_by_role.setdefault(role,{}): raise ValueError(f"duplicate V48.93 key {role} {k}")
        labels_by_role[role][k]=r

    errors=[]; cells={}; unique_mode={role:[] for role in ROLES}
    for variant in VARIANTS:
        cells[variant]={}
        newroot=a.main_run/'candidates'/variant/'evaluation'
        oldroot=a.l80_run/'candidates'/variant/'calibration'
        for role in ROLES:
            name=ROLE_FILES[role]
            np=newroot/name; op=oldroot/name
            if not np.is_file(): errors.append(f"missing new proposal {np}"); continue
            if not op.is_file(): errors.append(f"missing historical L80 proposal {op}"); continue
            newrows=load_jsonl(np); oldrows=load_jsonl(op)
            nk={key_prop(r) for r in newrows}; ok={key_prop(r) for r in oldrows}
            proposal_identity=(nk==ok)
            if not proposal_identity:
                errors.append(f"proposal set changed {variant}/{role}: new={len(nk)} old={len(ok)} new_only={len(nk-ok)} old_only={len(ok-nk)}")
            joined=[]
            lab=labels_by_role[role]
            for r in newrows:
                k=key_prop(r)
                if k not in lab:
                    errors.append(f"V48.93 label missing {variant}/{role}/{k}")
                    continue
                y=lab[k]
                cc=r.get('native_candidate_certificate'); nc=r.get('native_nominal_certificate')
                if not(isinstance(cc,list) and len(cc)>=4 and isinstance(nc,list) and len(nc)>=4):
                    errors.append(f"native certificate missing {variant}/{role}/{k}"); continue
                if r.get('support_reserve_state') not in {'support_establishment','reserve_debt'}:
                    errors.append(f"support-reserve state missing {variant}/{role}/{k}"); continue
                joined.append((r,y,cc,nc))
            if len(joined)!=len(newrows):
                errors.append(f"join incomplete {variant}/{role}: {len(joined)}/{len(newrows)}")

            def rows_where(pred): return [(r,y,cc,nc) for r,y,cc,nc in joined if pred(y)]
            sp=rows_where(lambda y: bool(y.get('safe_positive')))
            hm=rows_where(lambda y: bool(y.get('teacher_harmful')))
            tf=rows_where(lambda y: bool(y.get('teacher_feasible')))
            ti=rows_where(lambda y: not bool(y.get('teacher_feasible')))
            def ns(x): return float(x[2][1])
            def ss(x): return float(x[0]['support_reserve_score'])
            def npass(x): return ns(x)>=0.5
            def spass(x): return bool(x[0]['absolute_feasibility_pass'])
            mode_rows=[]
            for r,y,cc,nc in sp:
                m=str(y.get('mediation_mode'))
                expected={'drs_activation':'support_establishment','deployability_gain':'reserve_debt'}.get(m)
                if expected is not None:
                    good=str(r['support_reserve_state'])==expected
                    mode_rows.append(good); unique_mode[role].append((key_prop(r),good))
            cell={
                'proposal_set_identity':proposal_identity,
                'rows':len(joined),'safe_positive_rows':len(sp),'harmful_rows':len(hm),
                'teacher_feasible_rows':len(tf),'teacher_infeasible_rows':len(ti),
                'mode_labeled_safe_positive_rows':len(mode_rows),
                'mode_observability_accuracy':rate(mode_rows),
                'safe_vs_harm_auc_native':auc(map(ns,sp),map(ns,hm)),
                'safe_vs_harm_auc_srca':auc(map(ss,sp),map(ss,hm)),
                'teacher_feasible_auc_native':auc(map(ns,tf),map(ns,ti)),
                'teacher_feasible_auc_srca':auc(map(ss,tf),map(ss,ti)),
                'safe_positive_pass_native':rate([npass(x) for x in sp]),
                'safe_positive_pass_srca':rate([spass(x) for x in sp]),
                'harmful_pass_native':rate([npass(x) for x in hm]),
                'harmful_pass_srca':rate([spass(x) for x in hm]),
                'teacher_infeasible_pass_native':rate([npass(x) for x in ti]),
                'teacher_infeasible_pass_srca':rate([spass(x) for x in ti]),
            }
            for base,new,key in [
                ('safe_vs_harm_auc_native','safe_vs_harm_auc_srca','safe_vs_harm_auc_delta'),
                ('teacher_feasible_auc_native','teacher_feasible_auc_srca','teacher_feasible_auc_delta'),
                ('safe_positive_pass_native','safe_positive_pass_srca','safe_positive_pass_delta'),
                ('harmful_pass_native','harmful_pass_srca','harmful_pass_delta'),
                ('teacher_infeasible_pass_native','teacher_infeasible_pass_srca','teacher_infeasible_pass_delta')]:
                cell[key]=None if cell[base] is None or cell[new] is None else float(cell[new]-cell[base])
            cells[variant][role]=cell

    # Deduplicate the balanced/precision copies for the state-observability gate.
    mode_by_role={}
    for role, vals in unique_mode.items():
        d={}
        for k,g in vals:
            if k in d and d[k]!=g: errors.append(f"mode-observability mismatch across variants {role}/{k}")
            d[k]=g
        mode_by_role[role]={'rows':len(d),'accuracy':rate(list(d.values()))}

    out={
      'schema':'ocrap-v48.94-srca-audit-v1','engineering_version':ENGINEERING_VERSION,
      'valid':not errors,'attribution_ready':not errors,'errors':errors,
      'experiment_type':'fixed_zero_parameter_support_reserve_complementarity_absolute_source',
      'planner_parameters_trained':0,'dataset_reconstruction':False,'dataset_reselection':False,
      'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,
      'boundary_transport':False,'relative_ranker_modified':False,'regime_conditioning':False,
      'proposal_identity_required':True,'v48_93_labels_reused':True,
      'mode_observability_by_role':mode_by_role,'cells':cells,'test_roots_read':False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':out['valid'],'errors':errors[:3],'mode':mode_by_role}))
    return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
