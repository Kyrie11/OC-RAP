#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

KINDS = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
VARIANTS = ("balanced", "precision")

def load(p: Path) -> dict: return json.loads(p.read_text(encoding="utf-8"))

def dep_block(d: dict) -> tuple[str, dict]:
    phase = "fit" if bool(d.get("development_fit_only")) else "verify"
    return phase, (d.get(phase) or {})

def metric(run: Path, variant: str, kind: str) -> dict[str, Any]:
    fn = {
        "dev_near": "dev_diagnostic_near_v48.json",
        "dev_contact": "dev_diagnostic_contact_v48.json",
        "certificate_near": "direct_value_risk_near_v48.json",
        "certificate_contact": "direct_value_risk_contact_v48.json",
    }[kind]
    p = run / "candidates" / variant / "calibration" / fn
    if not p.is_file(): return {"missing": str(p)}
    d = load(p); phase, dep = dep_block(d)
    dk=("num_groups","num_selected","selection_rate","num_positive_selected","positive_recall","precision",
        "precision_wilson_lcb90","num_harmful_selected","harmful_selected_rate","harmful_selected_ucb90",
        "harmful_group_exposure","harmful_group_exposure_ucb90","num_opportunities")
    qk=("candidate_safe_positive_auc","candidate_harm_auc","candidate_pred_teacher_correlation",
        "candidate_rank_teacher_correlation","proposal_evidence_top1_correlation","proposal_evidence_top1_safe_positive_auc",
        "proposal_evidence_top1_harm_auc","proposal_deployed_rule_abstention_rate","proposal_deployed_rule_top1_safe_positive_auc",
        "proposal_top_k","proposal_positive_group_count","proposal_oracle_best_hit_rate_positive_groups",
        "proposal_any_positive_hit_rate_positive_groups")
    return {"path":str(p),"phase":phase,"valid_for_deployment":d.get("valid_for_deployment"),
        "rejection_kind":d.get("rejection_kind"),"absolute_feasibility_mode":d.get("absolute_feasibility_mode"),
        "absolute_feasibility_threshold":d.get("absolute_feasibility_threshold"),
        "deployment":{k:dep.get(k) for k in dk if k in dep},
        "ranking_and_selector_diagnostics":{k:d.get(k) for k in qk if k in d},
        "proposal_constrained_oracle_gate":d.get("proposal_constrained_oracle_gate"),
        "proposal_support_curve":d.get("proposal_support_curve")}

def audit_metric(audit:dict, arm:str, variant:str, kind:str, key:str):
    return (((audit.get("arms") or {}).get(arm) or {}).get(variant) or {}).get(kind,{}).get(key)

def main()->None:
    ap=argparse.ArgumentParser(description="v48.61 ERWF controlled source attribution")
    for n in ("a","b","c","d","e","f"): ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--feasibility-audit",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    x=ap.parse_args(); audit=load(x.feasibility_audit)
    arms={"A":x.a,"B_native":x.b,"C_AFE":x.c,"D_ORFC":x.d,"E_CPHR":x.e,"F_Main_ERWF":x.f}
    deltas={}; auc_go=True; meaningful_count=0
    for v in VARIANTS:
        deltas[v]={}
        for k in KINDS:
            b=audit_metric(audit,"B_native",v,k,"absolute_feasibility_auc")
            f=audit_metric(audit,"F_ERWF",v,k,"absolute_feasibility_auc")
            delta=(None if b is None or f is None else float(f)-float(b))
            deltas[v][k]={"F_minus_B_absolute_feasibility_auc":delta}
            if delta is None or delta <= 0.0: auc_go=False
            if delta is not None and delta >= 0.01: meaningful_count += 1
    # Pre-registered strong GO: no split may regress and most splits must show a nontrivial gain.
    auc_go = bool(auc_go and meaningful_count >= 6)
    harm_checks=[]
    for v in VARIANTS:
        for k in KINDS:
            f_inf=audit_metric(audit,"F_ERWF",v,k,"teacher_infeasible_pass_fraction")
            f_harm=audit_metric(audit,"F_ERWF",v,k,"harmful_pass_fraction")
            legacy=[]
            for arm in ("C_AFE","D_ORFC","E_CPHR"):
                z=audit_metric(audit,arm,v,k,"teacher_infeasible_pass_fraction")
                if z is not None: legacy.append(float(z))
            # F must not merely reproduce the ~0.4-0.55 permissive operating-point shift.
            cap=min(legacy)-0.05 if legacy else 0.35
            ok=(f_inf is not None and float(f_inf) <= max(0.0,cap))
            harm_checks.append({"variant":v,"split":k,"F_teacher_infeasible_pass_fraction":f_inf,
                                "F_harmful_pass_fraction":f_harm,"legacy_best_minus_0p05_cap":max(0.0,cap),"valid":ok})
    harmful_go=all(z["valid"] for z in harm_checks)
    scientific_go=bool(auc_go and harmful_go)
    doc={
      "schema":"ocrap-v48.61-erwf-comparison-v1",
      "arms":{name:{v:{k:metric(run,v,k) for k in KINDS} for v in VARIANTS} for name,run in arms.items()},
      "feasibility_role_audit":audit,
      "source_deltas":deltas,
      "preregistered_decision":{
        "status":"GO" if scientific_go else "STOP",
        "all_8_F_minus_B_auc_positive_and_at_least_6_ge_0p01":auc_go,
        "harmful_infeasible_pass_materially_below_C_D_E":harmful_go,
        "harmful_checks":harm_checks,
      },
      "attribution_order":[
        "F-B absolute-source ordering (primary)",
        "F-E isolates option-resolved recovery continuation from candidate-level static CPHR",
        "F-D isolates candidate x option continuation witness from context-free option bias",
        "state/variant isolation and fixed top-5 proposal contract",
        "F-A deployment propagation only after source evidence",
      ],
      "scientific_contract":{
        "primary_hypothesis":"recoverability is identified by an executable candidate x recovery-option continuation witness, not by a candidate-level static headroom scalar correction",
        "GO_requires":[
          "F-B absolute feasibility AUC improves in all balanced/precision x dev/certificate x Near/Contact cells, with >=0.01 gain in at least six of eight",
          "teacher-infeasible/harmful pass is materially below the permissive AFE/ORFC/CPHR family rather than only moving the 0.5 operating point",
          "only six shared bounded non-negative witness weights train; Stage-I stays bitwise frozen",
          "fixed top-5, threshold 0.5, RIFA order, no regime id/router, no centering, no teacher-future physical component",
          "source gain precedes and then propagates to deployment",
        ],
        "STOP_if":["any source-AUC cell regresses vs B","gain is only pass-rate/threshold movement","harmful pass remains in C/D/E range","state/provenance fails"],
        "forbidden_next_sweeps":["threshold/grid search","proposal expansion","option-specific free bias","generic AFE MLP width/class-weight sweeps","regime routing","broad root/margin encoder tuning","privileged teacher-physical distillation"],
        "safe_role":"shared-policy non-interference evaluation only after source GO",
      },
    }
    x.output.parent.mkdir(parents=True,exist_ok=True); x.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({"event":"v48_61_erwf_comparison","decision":doc["preregistered_decision"]["status"],"output":str(x.output)}))
if __name__=='__main__': main()
