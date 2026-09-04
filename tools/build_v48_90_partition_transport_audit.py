#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from ocrap.v48_90_partition_transport import audit_partition_transport_pair
from tools.build_v48_89_root_correspondence_audit import (
    ROLE_FILES,
    _auc,
    _iter_manifest,
    _labels,
    _load_sample,
    _parse_root,
    _quantiles,
    _scalar,
    _sha,
)


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    vals=[float(r[field]) for r in rows if r.get(field) is not None and math.isfinite(float(r[field]))]
    return float(np.mean(vals)) if vals else None


def _macro_stratified_auc(rows: list[dict[str, Any]], score_field: str) -> float | None:
    # Exact pairwise AUC using only safe-positive/harmful pairs within the same
    # macro type.  This prevents a branch-stability score from looking useful
    # merely because different macro families have different base rates.
    numer=0.0
    denom=0
    by_macro: dict[int,list[dict[str,Any]]]=defaultdict(list)
    for r in rows:
        by_macro[int(r.get("macro",-1))].append(r)
    for g in by_macro.values():
        pos=[float(r[score_field]) for r in g if r.get("safe_positive") and math.isfinite(float(r[score_field]))]
        neg=[float(r[score_field]) for r in g if r.get("teacher_harmful") and math.isfinite(float(r[score_field]))]
        if not pos or not neg:
            continue
        p=np.asarray(pos,dtype=np.float64)[:,None]
        n=np.asarray(neg,dtype=np.float64)[None,:]
        numer += float((p>n).sum()+0.5*(p==n).sum())
        denom += int(p.size*n.size)
    return float(numer/denom) if denom else None


def _top1(rows: list[dict[str,Any]], score_field: str) -> tuple[float|None,float|None,float|None,int]:
    groups: dict[tuple[str,int],list[dict[str,Any]]]=defaultdict(list)
    for r in rows:
        groups[(str(r["scene_id"]),int(r["time_index"]))].append(r)
    powered=[g for g in groups.values() if any(bool(r.get("safe_positive")) for r in g)]
    if not powered:
        return None,None,None,0
    # Tie-aware expected top-1 accuracy.  A deterministic first-row tie break can
    # create candidate-order artefacts when a structural score has many exact
    # ties (notably partition stability at 1.0).  Treat all maximizers as
    # equally likely so the metric depends only on the score, not row order.
    acc_terms=[]
    for g in powered:
        scores=np.asarray([float(r[score_field]) for r in g],dtype=np.float64)
        m=float(np.max(scores))
        tied=[r for r,s in zip(g,scores) if abs(float(s)-m)<=1e-12]
        acc_terms.append(sum(bool(r.get("safe_positive")) for r in tied)/len(tied))
    acc=float(np.mean(acc_terms))
    chance=float(np.mean([sum(bool(r.get("safe_positive")) for r in g)/len(g) for g in powered]))
    return acc,chance,acc-chance,len(powered)


def _summarize_role(rows: list[dict[str,Any]]) -> dict[str,Any]:
    valid=[r for r in rows if r.get("valid")]
    labeled=[r for r in valid if r.get("label_available")]
    safe=[r for r in labeled if r.get("safe_positive")]
    harmful=[r for r in labeled if r.get("teacher_harmful")]

    qfields=(
        "recipe_shared_mass_candidate","recipe_shared_mass_nominal","recipe_matched_transport_mass",
        "recipe_unresolved_semantic_mass_candidate","recipe_unresolved_semantic_mass_nominal",
        "exchangeable_duplicate_mass_candidate","exchangeable_duplicate_mass_nominal",
        "duplicate_root_homogeneity_mass_candidate","duplicate_root_homogeneity_mass_nominal",
        "exogenous_shared_mass_candidate","exogenous_shared_mass_nominal","exogenous_matched_transport_mass",
        "exogenous_unresolved_mass_candidate","exogenous_unresolved_mass_nominal",
        "recipe_tail_transport_coverage","recipe_tail_transport_purity","recipe_tail_partition_stability",
        "exogenous_tail_transport_coverage","exogenous_tail_transport_purity","exogenous_tail_partition_stability",
        "exogenous_tail_split_merge_mass","exogenous_tail_unmatched_mass",
        "exogenous_transport_sign_identifiable_mass","exogenous_transport_informative_response_mass",
        "exogenous_transport_point_identifiable_mass",
    )
    out: dict[str,Any]={
        "rows":len(rows),"valid_rows":len(valid),"valid_fraction":len(valid)/max(1,len(rows)),
        "labeled_rows":len(labeled),"label_coverage":len(labeled)/max(1,len(valid)),
        "safe_positive_rows":len(safe),"harmful_rows":len(harmful),
    }
    for f in qfields:
        out[f]=_quantiles([float(r[f]) for r in valid])

    def auc_for(field: str) -> float|None:
        if not safe or not harmful:
            return None
        return _auc([True]*len(safe)+[False]*len(harmful),
                    [float(r[field]) for r in safe]+[float(r[field]) for r in harmful])

    out["partition_stability_safe_positive_mean"]=_mean(safe,"exogenous_tail_partition_stability")
    out["partition_stability_harmful_mean"]=_mean(harmful,"exogenous_tail_partition_stability")
    out["partition_stability_safe_vs_harmful_auc"]=auc_for("exogenous_tail_partition_stability")
    out["partition_stability_macro_stratified_auc"]=_macro_stratified_auc(labeled,"exogenous_tail_partition_stability")
    pacc,pchance,plift,pn=_top1(labeled,"exogenous_tail_partition_stability")
    out["partition_stability_top1_accuracy"]=pacc
    out["partition_stability_top1_chance"]=pchance
    out["partition_stability_top1_lift"]=plift
    out["powered_safe_positive_groups"]=pn

    out["response_safe_positive_mean"]=_mean(safe,"exogenous_transport_signed_response_score")
    out["response_harmful_mean"]=_mean(harmful,"exogenous_transport_signed_response_score")
    out["response_safe_vs_harmful_auc"]=auc_for("exogenous_transport_signed_response_score")
    racc,rchance,rlift,_=_top1(labeled,"exogenous_transport_signed_response_score")
    out["response_top1_accuracy"]=racc
    out["response_top1_chance"]=rchance
    out["response_top1_lift"]=rlift

    # Descriptive, threshold-free retention structure.
    out["safe_positive_full_partition_stability_fraction"]=(
        float(np.mean([float(r["exogenous_tail_partition_stability"])>=1.0-1e-6 for r in safe])) if safe else None
    )
    out["harmful_full_partition_stability_fraction"]=(
        float(np.mean([float(r["exogenous_tail_partition_stability"])>=1.0-1e-6 for r in harmful])) if harmful else None
    )
    return out


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",action="append",required=True,help="role=/path/to/dataset")
    ap.add_argument("--proposal-run",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--summary",type=Path,required=True)
    ap.add_argument("--alpha",type=float,default=0.2)
    ap.add_argument("--beta",type=float,default=0.2)
    ap.add_argument("--top-m",type=int,default=8)
    args=ap.parse_args()

    roots=[_parse_root(x) for x in args.root]
    roles=[r for r,_ in roots]
    if len(roots)!=len(ROLE_FILES) or set(roles)!=set(ROLE_FILES) or len(set(roles))!=len(roles):
        raise SystemExit(f"all roles required exactly once: {sorted(ROLE_FILES)}")
    proposal=args.proposal_run.expanduser().resolve()
    labels,label_identity=_labels(proposal)

    samples_by_role: dict[str,list[dict[str,Any]]]=defaultdict(list)
    seen:set[Path]=set()
    for role,root in roots:
        for path in _iter_manifest(root):
            if path in seen:
                raise SystemExit(f"duplicate sample path {path}")
            seen.add(path)
            samples_by_role[role].append(_load_sample(path))

    t0=time.perf_counter(); out_rows=[]; errors=[]
    for role in ROLE_FILES:
        groups: dict[tuple[str,int],list[dict[str,Any]]]=defaultdict(list)
        for s in samples_by_role[role]:
            groups[(s["__scene__"],s["__time__"])].append(s)
        matched=set()
        for key,group in groups.items():
            noms=[s for s in group if s["__nominal__"] or s["__candidate__"]==0]
            if len(noms)!=1:
                errors.append(f"role={role} group={key} nominal_count={len(noms)}")
                continue
            nominal=noms[0]
            for candidate in group:
                if candidate is nominal: continue
                rec=audit_partition_transport_pair(candidate,nominal,alpha=args.alpha,beta=args.beta,top_m=args.top_m).to_dict()
                lk=(candidate["__scene__"],candidate["__time__"],candidate["__candidate__"])
                lab=labels[role].get(lk)
                if lab is not None:
                    matched.add(lk)
                    sample_r=float(_scalar(candidate,"r_dep_star",float("nan")))
                    if not math.isfinite(sample_r) or abs(sample_r-float(lab["teacher_candidate_r_dep"]))>1e-6:
                        errors.append(f"proposal/dataset teacher R_dep mismatch role={role} key={lk}")
                    sample_macro=int(_scalar(candidate,"prefix_macro_type_id",_scalar(candidate,"prefix_macro_id",-1)))
                    if sample_macro!=int(lab["macro"]):
                        errors.append(f"proposal/dataset macro mismatch role={role} key={lk}")
                rec.update(
                    schema="ocrap-v48.90-counterfactual-equivalence-partition-transport-row-v1",
                    engineering_version="v48.90.0-OC-CEPT",
                    dataset_role=role,
                    sample_path=candidate["__path__"],nominal_sample_path=nominal["__path__"],
                    scene_id=candidate["__scene__"],time_index=candidate["__time__"],candidate_index=candidate["__candidate__"],
                    label_available=lab is not None,
                    proposal_label_variants=list(lab.get("_v4889_label_variants",[])) if lab else [],
                    teacher_adv=float(lab.get("teacher_adv",float("nan"))) if lab else None,
                    teacher_harmful=bool(lab.get("teacher_harmful",False)) if lab else None,
                    teacher_feasible=(float(lab.get("teacher_candidate_r_dep",-1.0))>=0.0) if lab else None,
                    safe_positive=(float(lab.get("teacher_adv",-1.0))>=0.015 and not bool(lab.get("teacher_harmful",False))) if lab else None,
                    macro=int(lab.get("macro",-1)) if lab else None,
                    teacher_metadata_input_to_model=False,dataset_reconstruction=False,planner_parameters_trained=0,
                )
                if not rec["valid"]:
                    errors.append(f"invalid pair role={role} key={lk}: {rec.get('error')}")
                out_rows.append(rec)
        missing=set(labels[role])-matched
        if missing:
            errors.append(f"proposal labels missing from dataset role={role}: count={len(missing)} preview={sorted(missing)[:5]}")

    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8") as f:
        for r in out_rows: f.write(json.dumps(r,sort_keys=True)+"\n")
    summary={
        "schema":"ocrap-v48.90-counterfactual-equivalence-partition-transport-summary-v1",
        "engineering_version":"v48.90.0-OC-CEPT",
        "valid":not errors,"attribution_ready":not errors,"errors":errors[:100],"rows":len(out_rows),
        "roles":{role:_summarize_role([r for r in out_rows if r["dataset_role"]==role]) for role in ROLE_FILES},
        "label_identity":label_identity,
        "output":str(args.output.resolve()),"output_sha256":_sha(args.output),
        "proposal_run":str(proposal),
        "proposal_run_role":"teacher/safe-positive labels only; same V48.89.1 union cohort; no model input",
        "alpha":args.alpha,"beta":args.beta,"top_m":args.top_m,
        "elapsed_seconds":time.perf_counter()-t0,
        "teacher_labels_changed":False,"teacher_metadata_input_to_model":False,"dataset_reconstruction":False,
        "test_roots_read":False,"planner_parameters_trained":0,"boundary_transport":False,"regime_conditioning":False,
    }
    args.summary.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"valid":summary["valid"],"rows":len(out_rows),"errors":len(errors),"summary":str(args.summary)}))
    return 0 if summary["valid"] else 30

if __name__=="__main__":
    raise SystemExit(main())
