#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from collections import Counter
from pathlib import Path
import numpy as np

BASE_LIMITING = ("clearance", "stopping", "control", "stability")
KINDS = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
VARIANTS = ("balanced", "precision")


def read_rows(path: Path):
    rows=[]
    if not path.is_file(): return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def auc(labels, scores):
    y=np.asarray(labels,dtype=bool); s=np.asarray(scores,dtype=float); p=s[y]; n=s[~y]
    if not len(p) or not len(n): return None
    return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))


def mean(rows,key):
    vals=[]
    for r in rows:
        try:
            v=float(r[key])
            if math.isfinite(v): vals.append(v)
        except Exception: pass
    return float(np.mean(vals)) if vals else None


def frac(rows,pred):
    return float(sum(bool(pred(r)) for r in rows)/len(rows)) if rows else None


def barrier_names(arm: str):
    names=list(BASE_LIMITING)
    if arm in {"P66_OCACRW", "Q67_CTRLPROJ", "T68_FIDELITY", "D69_DTRW", "E70_OCCSOFT", "G70_OCDOTW", "H71_BOUNDARY_LOCAL", "J71_HISTORY_TUBE", "K71_OCBORW"}: names.append("route")
    if arm in {"P66_OCACRW", "Q67_CTRLPROJ", "T68_FIDELITY", "D69_DTRW", "E70_OCCSOFT", "G70_OCDOTW", "H71_BOUNDARY_LOCAL", "J71_HISTORY_TUBE", "K71_OCBORW"}: names.append("reentry_persistence")
    return names


def semantic_summary(rows, names):
    if not rows or not any(r.get("semantic_positive_option_count") is not None for r in rows): return None
    counts=Counter(); bars=[[] for _ in names]
    positive=[]
    for r in rows:
        try:
            positive.append(float(r.get("semantic_best_common_viability"))>0.0)
        except Exception: positive.append(False)
        x=r.get("semantic_limiting_constraint")
        if x is not None:
            i=int(x); counts[names[i] if 0<=i<len(names) else f"unknown_{i}"]+=1
        b=r.get("semantic_best_barriers")
        if isinstance(b,list):
            for j in range(min(len(b),len(names))):
                try:
                    v=float(b[j])
                    if math.isfinite(v): bars[j].append(v)
                except Exception: pass
    return {
        "rows":len(rows),
        "any_positive_common_option_fraction":float(np.mean(positive)) if positive else None,
        "universal_failure_fraction":frac(rows,lambda r:float(r.get("semantic_universal_failure") or 0)>0),
        "best_common_viability_mean":mean(rows,"semantic_best_common_viability"),
        "max_common_support_mean":mean(rows,"semantic_max_common_support"),
        "limiting_constraint_counts":dict(counts),
        "limiting_constraint_fractions":{k:float(v/len(rows)) for k,v in counts.items()},
        "best_option_barrier_means":{names[j]:(float(np.mean(bars[j])) if bars[j] else None) for j in range(len(names))},
    }


def summarize(rows, arm, positive_gain=.015):
    if not rows: return {"rows":0}
    safe=[r for r in rows if float(r.get("teacher_adv",-1e9))>=positive_gain and not bool(r.get("teacher_harmful",False))]
    harm=[r for r in rows if bool(r.get("teacher_harmful",False))]
    feas=[r for r in rows if float(r["teacher_candidate_r_dep"])>=0]
    inf=[r for r in rows if float(r["teacher_candidate_r_dep"])<0]
    labels=[float(r["teacher_candidate_r_dep"])>=0 for r in rows]
    probs=[r.get("absolute_feasibility_probability") for r in rows]
    have=all(x is not None and math.isfinite(float(x)) for x in probs)
    cert=[r for r in rows if r.get("semantic_best_common_viability") is not None and float(r["semantic_best_common_viability"])>0]
    cert_feas=[r for r in cert if float(r["teacher_candidate_r_dep"])>=0]
    cert_inf=[r for r in cert if float(r["teacher_candidate_r_dep"])<0]
    safe_cert=[r for r in safe if r.get("semantic_best_common_viability") is not None and float(r["semantic_best_common_viability"])>0]
    cert_probs=[float(r["absolute_feasibility_probability"]) for r in cert if r.get("absolute_feasibility_probability") is not None and math.isfinite(float(r["absolute_feasibility_probability"]))]
    cert_labels=[float(r["teacher_candidate_r_dep"])>=0 for r in cert if r.get("absolute_feasibility_probability") is not None and math.isfinite(float(r["absolute_feasibility_probability"]))]
    cert_pass=[r for r in cert if r.get("absolute_feasibility_probability") is not None and float(r["absolute_feasibility_probability"])>=.5]
    safe_base4=[]
    for r in safe:
        b=r.get("semantic_best_barriers")
        if isinstance(b,list) and len(b)>=4:
            safe_base4.append(all(float(x)>0.0 for x in b[:4]))
    d={
        "rows":len(rows),"groups":len({(r.get("scene"),r.get("time"),r.get("fold")) for r in rows}),
        "safe_positive_rows":len(safe),"harmful_rows":len(harm),
        "teacher_infeasible_count":len(inf),"teacher_feasible_count":len(feas),
        "positive_certificate_rows":len(cert),
        "positive_certificate_teacher_feasible_precision":float(len(cert_feas)/len(cert)) if cert else None,
        "positive_certificate_teacher_infeasible_fraction":float(len(cert_inf)/len(cert)) if cert else None,
        "positive_certificate_probability_auc_for_teacher_feasibility":auc(cert_labels,cert_probs) if cert_probs else None,
        "positive_certificate_pass_rows":len(cert_pass),
        "positive_certificate_pass_teacher_feasible_precision":(sum(float(r["teacher_candidate_r_dep"])>=0 for r in cert_pass)/len(cert_pass) if cert_pass else None),
        "candidate_deviation_auc_for_teacher_feasibility":auc(labels,[-float(r.get("deviation",0.0)) for r in rows]),
        "safe_positive_positive_certificate_rows":len(safe_cert),
        "safe_positive_positive_certificate_fraction":float(len(safe_cert)/len(safe)) if safe else None,
        "safe_positive_pass_given_positive_certificate":frac(safe_cert,lambda r:float(r["absolute_feasibility_probability"])>=.5),
        "safe_positive_base4_positive_fraction":float(np.mean(safe_base4)) if safe_base4 else None,
    }
    if have:
        ps=[float(x) for x in probs]
        d.update({
            "absolute_feasibility_auc":auc(labels,ps),
            "absolute_feasibility_accuracy_at_0_5":float(np.mean([(p>=.5)==y for p,y in zip(ps,labels)])),
            "safe_positive_pass_fraction":frac(safe,lambda r:float(r["absolute_feasibility_probability"])>=.5),
            "harmful_pass_fraction":frac(harm,lambda r:float(r["absolute_feasibility_probability"])>=.5),
            "teacher_infeasible_pass_fraction":frac(inf,lambda r:float(r["absolute_feasibility_probability"])>=.5),
            "teacher_feasible_reject_fraction":frac(feas,lambda r:float(r["absolute_feasibility_probability"])<.5),
        })
    subs={"safe_positive":safe,"harmful":harm,"teacher_feasible":feas,"teacher_infeasible":inf}
    names=barrier_names(arm)
    d["semantic_coverage_diagnostics"]={n:semantic_summary(z,names) for n,z in subs.items()}
    return d


def paths(run:Path,v:str):
    b=run/'candidates'/v/'calibration'
    return {
        'dev_near':b/'dev_diagnostic_near_v48.proposal_rows.jsonl',
        'dev_contact':b/'dev_diagnostic_contact_v48.proposal_rows.jsonl',
        'certificate_near':b/'direct_value_risk_near_v48.proposal_rows.jsonl',
        'certificate_contact':b/'direct_value_risk_contact_v48.proposal_rows.jsonl',
    }


def main():
    ap=argparse.ArgumentParser(description='v48.71 OC-DOTW feasibility-role and active-demand witness audit')
    ap.add_argument('--arm',action='append',required=True); ap.add_argument('--variant',action='append',default=[])
    ap.add_argument('--positive-gain',type=float,default=.015); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    out={}
    for spec in a.arm:
        name,raw=spec.split('=',1); run=Path(raw); out[name]={}
        for v in a.variant or list(VARIANTS): out[name][v]={k:summarize(read_rows(p),name,a.positive_gain) for k,p in paths(run,v).items()}
    doc={
        'schema':'ocrap-v48.71-dotw-feasibility-role-audit-v1','positive_gain':a.positive_gain,'arms':out,
        'interpretation_contract':{
            'teacher_boundary':'candidate R_dep >= 0','stage_ii_threshold':0.5,
            'safe_positive':'teacher_adv >= positive_gain AND not teacher_harmful','no_threshold_search':True,
            'test_roots_read':False,'dataset_reconstruction':False,
            'semantic_diagnostics':'diagnostic only; positive-certificate precision measures whether the observable witness is trustworthy, not a learned target',
            'barrier_order_by_arm':{k:barrier_names(k) for k in out},
        },
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_71_dotw_feasibility_role_audit','output':str(a.output)}))

if __name__=='__main__': main()
