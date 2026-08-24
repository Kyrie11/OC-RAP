#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
KINDS=('dev_near','dev_contact','certificate_near','certificate_contact'); VARIANTS=('balanced','precision')

def load(p): return json.loads(Path(p).read_text())
def cell(audit,arm,v,k,key): return (((audit.get('arms') or {}).get(arm) or {}).get(v) or {}).get(k,{}).get(key)
def sem(audit,arm,v,k,subset,key): return (((((audit.get('arms') or {}).get(arm) or {}).get(v) or {}).get(k,{}) .get('semantic_coverage_diagnostics') or {}).get(subset) or {}).get(key)
def delta(a,b): return None if a is None or b is None else float(a)-float(b)

def main():
 ap=argparse.ArgumentParser(description='v48.66 OC-ACRW route x persistent-reentry factorial attribution')
 for n in ('b','f','i','m65','n','o','p'): ap.add_argument('--'+n,type=Path,required=True)
 ap.add_argument('--feasibility-audit',type=Path,required=True); ap.add_argument('--v65-complete',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); x=ap.parse_args()
 audit=load(x.feasibility_audit); v65=load(x.v65_complete)
 deltas={}; selectivity=[]; safe=[]; trust=[]; false_cert=[]; auc_all=True; auc_ge=0
 for v in VARIANTS:
  deltas[v]={}
  for k in KINDS:
   B=cell(audit,'B_native',v,k,'absolute_feasibility_auc'); I=cell(audit,'I_ACTIVESET',v,k,'absolute_feasibility_auc')
   M65=cell(audit,'M65_OCCLRW',v,k,'absolute_feasibility_auc'); N=cell(audit,'N_ROUTE',v,k,'absolute_feasibility_auc'); O=cell(audit,'O_REENTRY',v,k,'absolute_feasibility_auc'); P=cell(audit,'P_Main_OCACRW',v,k,'absolute_feasibility_auc')
   pi=delta(P,I); pb=delta(P,B); ni=delta(N,I); oi=delta(O,I); po=delta(P,O); pn=delta(P,N)
   inter=None if None in (ni,oi,pi) else pi-ni-oi
   deltas[v][k]={'P_minus_B_absolute_feasibility_auc':pb,'P_minus_I_total_effect':pi,'P_minus_M65_recovery_from_classlocal':delta(P,M65),'N_minus_I_route_main_effect':ni,'O_minus_I_reentry_main_effect':oi,'P_minus_O_route_conditional_effect':po,'P_minus_N_reentry_conditional_effect':pn,'route_x_reentry_interaction':inter}
   if pb is None or pb<=0: auc_all=False
   if pb is not None and pb>=.01: auc_ge+=1
   pti=cell(audit,'P_Main_OCACRW',v,k,'teacher_infeasible_pass_fraction'); ph=cell(audit,'P_Main_OCACRW',v,k,'harmful_pass_fraction')
   fti=cell(audit,'F_ERWF',v,k,'teacher_infeasible_pass_fraction'); fh=cell(audit,'F_ERWF',v,k,'harmful_pass_fraction')
   ti_cap=min(.25,float(fti)-.10 if fti is not None else .25); h_cap=min(.25,float(fh)-.10 if fh is not None else .25)
   selectivity.append({'variant':v,'split':k,'P_teacher_infeasible_pass_fraction':pti,'P_harmful_pass_fraction':ph,'teacher_infeasible_cap':ti_cap,'harmful_cap':h_cap,'valid':pti is not None and ph is not None and float(pti)<=ti_cap and float(ph)<=h_cap})
   ps=cell(audit,'P_Main_OCACRW',v,k,'safe_positive_pass_fraction'); is_=cell(audit,'I_ACTIVESET',v,k,'safe_positive_pass_fraction'); sg=delta(ps,is_)
   safe.append({'variant':v,'split':k,'P_safe_positive_pass_fraction':ps,'I_safe_positive_pass_fraction':is_,'P_minus_I_safe_positive_pass_fraction':sg,'valid':sg is not None and sg>0})
   pp=cell(audit,'P_Main_OCACRW',v,k,'positive_certificate_teacher_feasible_precision'); ip=cell(audit,'I_ACTIVESET',v,k,'positive_certificate_teacher_feasible_precision'); pg=delta(pp,ip)
   trust.append({'variant':v,'split':k,'P_positive_certificate_precision':pp,'I_positive_certificate_precision':ip,'P_minus_I_precision':pg,'valid':pg is not None and pg>0})
   pfi=sem(audit,'P_Main_OCACRW',v,k,'teacher_infeasible','any_positive_common_option_fraction'); ifi=sem(audit,'I_ACTIVESET',v,k,'teacher_infeasible','any_positive_common_option_fraction'); fg=None if pfi is None or ifi is None else float(ifi)-float(pfi)
   false_cert.append({'variant':v,'split':k,'P_teacher_infeasible_positive_certificate_fraction':pfi,'I_teacher_infeasible_positive_certificate_fraction':ifi,'I_minus_P_false_certificate_reduction':fg,'valid':fg is not None and fg>0})
 auc_go=auc_all and auc_ge>=6
 sel_go=all(z['valid'] for z in selectivity)
 safe_go=all(z['valid'] for z in safe) and sum((z['P_minus_I_safe_positive_pass_fraction'] or -9)>=.15 for z in safe)>=6
 trust_go=all(z['valid'] for z in trust) and sum((z['P_minus_I_precision'] or -9)>=.10 for z in trust)>=6
 false_go=all(z['valid'] for z in false_cert) and sum((z['I_minus_P_false_certificate_reduction'] or -9)>=.10 for z in false_cert)>=6
 prereq=bool(v65.get('valid')) and bool(v65.get('attribution_ready')) and v65.get('engineering_version')=='v48.65.0-OC-CLRW' and v65.get('test_roots_read') is False
 go=bool(prereq and auc_go and sel_go and safe_go and trust_go and false_go)
 doc={'schema':'ocrap-v48.66-ocacrw-comparison-v1','source_deltas':deltas,'feasibility_role_audit':audit,
  'preregistered_decision':{'status':'GO' if go else 'STOP','v48_65_prerequisite_attribution_ready':prereq,'all_8_P_minus_B_auc_positive_and_at_least_6_ge_0p01':auc_go,'controlled_selectivity_without_F_permissive_relapse':sel_go,'safe_positive_pass_restored_over_I_ACTIVESET':safe_go,'positive_certificate_precision_improved_over_I':trust_go,'teacher_infeasible_positive_certificate_reduced_over_I':false_go,'selectivity_checks':selectivity,'safe_positive_pass_checks':safe,'witness_trust_checks':trust,'false_certificate_checks':false_cert},
  'attribution_order':['P-B is the primary absolute-source gate','N-I isolates observable route-constraint coverage','O-I isolates post-contact persistent-reentry coverage','P-O and P-N give conditional main effects; P-I-(N-I)-(O-I) is the interaction','P-M65 is diagnostic evidence that reverting the falsified class-local correction restores rather than masks source quality','deployment propagation and paired Safe nominal utility are interpreted only after source GO'],
  'scientific_contract':{'primary_hypothesis':'v48.65 falsified observation-class-local correction transport as the dominant bottleneck. The remaining failure is witness trust: the current observable positive certificate omits active physical constraints. Candidate-global common support plus observation-certifiable route consistency and persistent post-contact re-entry should reject false witnesses while preserving true recovery witnesses, allowing the same two shared gains to restore absolute-feasibility admission.',
   'frozen':['v48.56-A Stage-I tensors','candidate-global v48.64 common-option support / exists-forall correction locus','active-set alignment ON','legacy stopping (path-stop remains OFF)','OC-MERO compatibility/top_m=8','RIFA top-K=5 and threshold=0.5','relative filter/ranker','recovery option library','teacher labels/datasets'],
   'interventions':['N adds only observable executable-route barrier','O adds only persistent re-entry/no-secondary-deterioration barrier','P adds both; all use current-observation CV occupancy and deterministic recovery rollout'],
   'GO_requires':['P-B AUC >0 in 8/8 and >=+0.01 in >=6/8','harmful and teacher-infeasible pass <=0.25 and >=0.10 below v48.61-F in every cell','safe-positive pass > I in 8/8 and +0.15 in >=6/8','positive-certificate teacher-feasible precision > I in 8/8 and +0.10 in >=6/8','teacher-infeasible positive-certificate coverage < I in 8/8 and by >=0.10 in >=6/8'],
   'forbidden':['threshold/LR/horizon/feature-weight/class-weight grids','proposal expansion/top-K search','generic AFE/MLP','candidate-only CPHR','compensatory ERWF','per-option negative veto','quantifier-gain sweep','regime router/policy/threshold/budget','broad root/margin/encoder retraining','privileged future/component-margin distillation','relative-ranker changes before source GO','option-library expansion without teacher-option-support evidence','class-local correction transport as Main','path-stop as Main']}}
 x.output.parent.mkdir(parents=True,exist_ok=True); x.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n'); print(json.dumps({'event':'v48_66_ocacrw_comparison','decision':doc['preregistered_decision']['status'],'output':str(x.output)}))
if __name__=='__main__': main()
