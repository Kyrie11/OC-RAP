#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
KINDS=('dev_near','dev_contact','certificate_near','certificate_contact');VARIANTS=('balanced','precision')
def load(p):return json.loads(Path(p).read_text())
def dep(d):phase='fit' if bool(d.get('development_fit_only')) else 'verify';return phase,d.get(phase) or {}
def metric(run,v,k):
 fn={'dev_near':'dev_diagnostic_near_v48.json','dev_contact':'dev_diagnostic_contact_v48.json','certificate_near':'direct_value_risk_near_v48.json','certificate_contact':'direct_value_risk_contact_v48.json'}[k];p=run/'candidates'/v/'calibration'/fn
 if not p.is_file():return {'missing':str(p)}
 d=load(p);phase,z=dep(d);dk=('num_groups','num_selected','selection_rate','num_positive_selected','positive_recall','precision','precision_wilson_lcb90','num_harmful_selected','harmful_selected_rate','harmful_selected_ucb90','harmful_group_exposure','harmful_group_exposure_ucb90','num_opportunities');qk=('candidate_safe_positive_auc','candidate_harm_auc','candidate_pred_teacher_correlation','candidate_rank_teacher_correlation','proposal_evidence_top1_correlation','proposal_evidence_top1_safe_positive_auc','proposal_evidence_top1_harm_auc','proposal_deployed_rule_abstention_rate','proposal_deployed_rule_top1_safe_positive_auc','proposal_top_k','proposal_positive_group_count','proposal_oracle_best_hit_rate_positive_groups','proposal_any_positive_hit_rate_positive_groups')
 return {'path':str(p),'phase':phase,'valid_for_deployment':d.get('valid_for_deployment'),'rejection_kind':d.get('rejection_kind'),'absolute_feasibility_mode':d.get('absolute_feasibility_mode'),'absolute_feasibility_threshold':d.get('absolute_feasibility_threshold'),'deployment':{x:z.get(x) for x in dk if x in z},'ranking_and_selector_diagnostics':{x:d.get(x) for x in qk if x in d}}
def am(audit,arm,v,k,key):return (((audit.get('arms') or {}).get(arm) or {}).get(v) or {}).get(k,{}).get(key)
def sem(audit,arm,v,k,subset,key):
 d=(((audit.get('arms') or {}).get(arm) or {}).get(v) or {}).get(k,{})
 z=((d.get('semantic_coverage_diagnostics') or {}).get(subset) or {})
 if key.startswith('classlocal.'):
  return (z.get('classlocal_transport') or {}).get(key.split('.',1)[1])
 return z.get(key)
def delta(a,b):return None if a is None or b is None else float(a)-float(b)

def main():
 ap=argparse.ArgumentParser(description='v48.65 OC-CLRW observation-class-local transport factorial attribution')
 for n in ('a','b','f','h','i','l','m'):ap.add_argument('--'+n,type=Path,required=True)
 ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--teacher-semantics-audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);x=ap.parse_args();audit=load(x.feasibility_audit);ta=load(x.teacher_semantics_audit)
 arms={'A':x.a,'B_native':x.b,'F_ERWF':x.f,'H_OCQARW':x.h,'I_ACTIVESET':x.i,'L_CLASSLOCAL':x.l,'M_Main_OCCLRW':x.m}
 cells={};auc_all=True;auc_ge=0;selectivity=[];safe=[];factor=[];classdiag=[]
 for v in VARIANTS:
  cells[v]={}
  for k in KINDS:
   vals={n:am(audit,n,v,k,'absolute_feasibility_auc') for n in ('B_native','F_ERWF','H_OCQARW','I_ACTIVESET','L_CLASSLOCAL','M_OCCLRW')}
   B,F,H,I,L,M=[vals[n] for n in ('B_native','F_ERWF','H_OCQARW','I_ACTIVESET','L_CLASSLOCAL','M_OCCLRW')]
   mb=delta(M,B); lh=delta(L,H); mi=delta(M,I); ml=delta(M,L); ih=delta(I,H)
   interaction=None if None in (lh,mi) else mi-lh
   cells[v][k]={'M_minus_B_absolute_feasibility_auc':mb,'M_minus_I_classlocal_conditional_effect':mi,'L_minus_H_classlocal_main_effect':lh,'I_minus_H_active_set_main_effect':ih,'M_minus_L_active_set_conditional_effect':ml,'classlocal_x_active_interaction':interaction,'M_minus_F_absolute_feasibility_auc':delta(M,F)}
   if mb is None or mb<=0:auc_all=False
   if mb is not None and mb>=.01:auc_ge+=1
   mti=am(audit,'M_OCCLRW',v,k,'teacher_infeasible_pass_fraction'); mh=am(audit,'M_OCCLRW',v,k,'harmful_pass_fraction')
   fti=am(audit,'F_ERWF',v,k,'teacher_infeasible_pass_fraction'); fh=am(audit,'F_ERWF',v,k,'harmful_pass_fraction')
   ti_cap=min(.25,(float(fti)-.10) if fti is not None else .25)
   harmful_cap=min(.25,(float(fh)-.10) if fh is not None else .25)
   selectivity.append({'variant':v,'split':k,'M_teacher_infeasible_pass_fraction':mti,'M_harmful_pass_fraction':mh,
      'teacher_infeasible_cap':ti_cap,'harmful_cap':harmful_cap,
      'valid':mti is not None and mh is not None and float(mti)<=ti_cap and float(mh)<=harmful_cap})
   ms=am(audit,'M_OCCLRW',v,k,'safe_positive_pass_fraction'); is_=am(audit,'I_ACTIVESET',v,k,'safe_positive_pass_fraction'); sg=delta(ms,is_)
   safe.append({'variant':v,'split':k,'M_safe_positive_pass_fraction':ms,'I_safe_positive_pass_fraction':is_,'M_minus_I_safe_positive_pass_fraction':sg,'valid':sg is not None and sg>0})
   classdiag.append({'variant':v,'split':k,
      'safe_positive_classlocal_positive_certificate_fraction':sem(audit,'M_OCCLRW',v,k,'safe_positive','classlocal.positive_certificate_fraction'),
      'harmful_classlocal_positive_certificate_fraction':sem(audit,'M_OCCLRW',v,k,'harmful','classlocal.positive_certificate_fraction'),
      'teacher_infeasible_classlocal_positive_certificate_fraction':sem(audit,'M_OCCLRW',v,k,'teacher_infeasible','classlocal.positive_certificate_fraction'),
      'safe_positive_classlocal_viable_root_mass_mean':sem(audit,'M_OCCLRW',v,k,'safe_positive','classlocal.viable_root_mass_mean'),
      'safe_positive_classlocal_selected_support_mean':sem(audit,'M_OCCLRW',v,k,'safe_positive','classlocal.selected_support_mean')})
 auc_go=auc_all and auc_ge>=6;sel_go=all(z['valid'] for z in selectivity);safe_go=all(z['valid'] for z in safe) and sum(z['M_minus_I_safe_positive_pass_fraction'] is not None and z['M_minus_I_safe_positive_pass_fraction']>=.15 for z in safe)>=6
 engineering_truth_ok=bool(ta.get('valid')) and ta.get('test_roots_read') is False and ta.get('read_only_existing_dataset') is True and ta.get('dataset_reconstruction') is False
 go=bool(auc_go and sel_go and safe_go and engineering_truth_ok)
 doc={'schema':'ocrap-v48.65-clrw-comparison-v1','arms':{n:{v:{k:metric(r,v,k) for k in KINDS} for v in VARIANTS} for n,r in arms.items()},'feasibility_role_audit':audit,'teacher_certificate_semantics_audit':ta,'source_deltas':cells,
   'preregistered_decision':{'status':'GO' if go else 'STOP','all_8_M_minus_B_auc_positive_and_at_least_6_ge_0p01':bool(auc_go),'controlled_selectivity_without_F_permissive_relapse':bool(sel_go),'safe_positive_pass_restored_over_I_ACTIVESET':bool(safe_go),'teacher_truth_contract_audit_valid':engineering_truth_ok,'selectivity_checks':selectivity,'safe_positive_pass_checks':safe},
   'diagnostic_not_gate':{'classlocal_certificate_diagnostics':classdiag,'teacher_classlocal_transport_gap':(ta.get('overall') or {}).get('teacher_feasible_classlocal_minus_global_shared_score_mean'),'teacher_classlocal_required_for_sign_fraction':(ta.get('overall') or {}).get('teacher_feasible_classlocal_required_for_sign_fraction')},
   'attribution_order':['M-B is the primary absolute-source test','L-H isolates class-local transport without active-set repair','M-I isolates class-local transport after the v48.64 active-set repair','M-L isolates active-set repair under class-local transport','(M-I)-(L-H) is the interaction','M-A deployment propagation and paired Safe nominal-utility are interpreted only after source GO'],
   'scientific_contract':{'primary_hypothesis':'v48.64 repaired part of physical witness sign availability, but its candidate-global option correction is over-coupled across distinguishable post-prefix observation classes; applying the same two-gain correction at q[i,l], after compatible-root aggregation but before per-class option maximization, should convert recovered witness availability into selective absolute-feasibility ordering',
     'frozen':['v48.56-A Stage-I tensors','root/margin/observation heads','OC-MERO compatibility and top_m=8','RIFA top-K=5 and threshold=0.5','relative rank/filter/reranker','recovery option library','teacher labels/datasets','legacy stopping coordinate for this factor experiment'],
     'GO_requires':['M-B source AUC improves in all eight Near/Contact cells and >=0.01 in at least six','teacher-infeasible and harmful pass stay <=0.25 and >=0.10 below v48.61-F in every cell','safe-positive pass improves over v48.64 I_ACTIVESET in all eight cells and by >=0.15 in at least six','teacher truth-contract audit recomputes stored OC-MERO labels exactly without test-root access or dataset reconstruction','only the two shared bounded gains train; no regime/router/teacher-future/relative-ranking intervention'],
     'forbidden_next_sweeps':['threshold/LR/horizon/feature-weight/class-weight grids','proposal expansion/densification/top-K search','option-specific free bias','generic AFE/MLP capacity','candidate-only CPHR','compensatory ERWF sum','per-option negative veto','quantifier gain sweep','Safe/Near/Contact router/policy/threshold/budget','broad root/margin/encoder retraining','privileged teacher-future/component-margin distillation','relative centering/ranker changes before source GO','recovery-option library expansion before teacher-option-support evidence'],
     'path_stop_status':'v48.64 J_PATHSTOP is not carried into Main: it failed to restore Near witness coverage and regressed certificate/Contact AUC; legacy stopping is frozen to isolate class-local transport.'}}
 x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_65_clrw_comparison','decision':doc['preregistered_decision']['status'],'output':str(x.output)}))
if __name__=='__main__':main()
