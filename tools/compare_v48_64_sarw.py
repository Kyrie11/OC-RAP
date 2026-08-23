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
 return {'path':str(p),'phase':phase,'valid_for_deployment':d.get('valid_for_deployment'),'rejection_kind':d.get('rejection_kind'),'absolute_feasibility_mode':d.get('absolute_feasibility_mode'),'absolute_feasibility_threshold':d.get('absolute_feasibility_threshold'),'deployment':{x:z.get(x) for x in dk if x in z},'ranking_and_selector_diagnostics':{x:d.get(x) for x in qk if x in d},'proposal_constrained_oracle_gate':d.get('proposal_constrained_oracle_gate'),'proposal_support_curve':d.get('proposal_support_curve')}
def am(audit,arm,v,k,key):return (((audit.get('arms') or {}).get(arm) or {}).get(v) or {}).get(k,{}).get(key)
def cov(audit,arm,v,k,subset='safe_positive',semantic=True):
 d=(((audit.get('arms') or {}).get(arm) or {}).get(v) or {}).get(k,{})
 x=d.get('semantic_coverage_diagnostics' if semantic else 'quantifier_coverage_diagnostics') or {};return (x.get(subset) or {}).get('any_positive_common_option_fraction')
def main():
 ap=argparse.ArgumentParser(description='v48.64 OC-SARW constraint-semantics factorial attribution')
 for n in 'abcdefghijk':ap.add_argument('--'+n,type=Path,required=True)
 ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);x=ap.parse_args();audit=load(x.feasibility_audit)
 arms={'A':x.a,'B_native':x.b,'C_AFE':x.c,'D_ORFC':x.d,'E_CPHR':x.e,'F_ERWF':x.f,'G_OCCWRF':x.g,'H_OCQARW':x.h,'I_ACTIVESET':x.i,'J_PATHSTOP':x.j,'K_Main_OCSARW':x.k}
 deltas={};allpos=True;meaningful=0;selectivity=[];recall=[];coverage=[];factor={}
 for v in VARIANTS:
  deltas[v]={};factor[v]={}
  for k in KINDS:
   vals={n:am(audit,n,v,k,'absolute_feasibility_auc') for n in ('B_native','F_ERWF','H_OCQARW','I_ACTIVESET','J_PATHSTOP','K_OCSARW')}
   B,F,H,I,J,K=[vals[n] for n in ('B_native','F_ERWF','H_OCQARW','I_ACTIVESET','J_PATHSTOP','K_OCSARW')]
   def d(a,b):return None if a is None or b is None else float(a)-float(b)
   kb=d(K,B);deltas[v][k]={'K_minus_B_absolute_feasibility_auc':kb,'K_minus_H_absolute_feasibility_auc':d(K,H),'I_minus_H_active_set_main_effect':d(I,H),'J_minus_H_path_stop_main_effect':d(J,H),'K_minus_I_path_stop_conditional_effect':d(K,I),'K_minus_J_active_set_conditional_effect':d(K,J),'K_minus_F_absolute_feasibility_auc':d(K,F)}
   if kb is None or kb<=0:allpos=False
   if kb is not None and kb>=.01:meaningful+=1
   ki=am(audit,'K_OCSARW',v,k,'teacher_infeasible_pass_fraction');kh=am(audit,'K_OCSARW',v,k,'harmful_pass_fraction');fi=am(audit,'F_ERWF',v,k,'teacher_infeasible_pass_fraction');cap=min(.25,(float(fi)-.10) if fi is not None else .25);selectivity.append({'variant':v,'split':k,'K_teacher_infeasible_pass_fraction':ki,'K_harmful_pass_fraction':kh,'fixed_selectivity_cap':cap,'valid':ki is not None and kh is not None and float(ki)<=cap and float(kh)<=cap})
   ks=am(audit,'K_OCSARW',v,k,'safe_positive_pass_fraction');hs=am(audit,'H_OCQARW',v,k,'safe_positive_pass_fraction');gain=d(ks,hs);recall.append({'variant':v,'split':k,'K_safe_positive_pass_fraction':ks,'H_safe_positive_pass_fraction':hs,'K_minus_H_safe_positive_pass_fraction':gain,'valid':gain is not None and gain>0})
   kc=cov(audit,'K_OCSARW',v,k,semantic=True);hc=cov(audit,'H_OCQARW',v,k,semantic=False);cg=d(kc,hc);coverage.append({'variant':v,'split':k,'K_safe_positive_any_positive_common_option_fraction':kc,'H_safe_positive_any_positive_common_option_fraction':hc,'K_minus_H_witness_coverage':cg,'valid':cg is not None and cg>0})
 auc_go=allpos and meaningful>=6;sel_go=all(z['valid'] for z in selectivity);rec_go=all(z['valid'] for z in recall) and sum(z['K_minus_H_safe_positive_pass_fraction'] is not None and z['K_minus_H_safe_positive_pass_fraction']>=.15 for z in recall)>=6;cov_go=all(z['valid'] for z in coverage) and sum(z['K_minus_H_witness_coverage'] is not None and z['K_minus_H_witness_coverage']>=.15 for z in coverage)>=6;go=bool(auc_go and sel_go and rec_go and cov_go)
 doc={'schema':'ocrap-v48.64-sarw-comparison-v1','arms':{n:{v:{k:metric(r,v,k) for k in KINDS} for v in VARIANTS} for n,r in arms.items()},'feasibility_role_audit':audit,'source_deltas':deltas,
  'preregistered_decision':{'status':'GO' if go else 'STOP','all_8_K_minus_B_auc_positive_and_at_least_6_ge_0p01':bool(auc_go),'controlled_selectivity_without_F_permissive_relapse':bool(sel_go),'safe_positive_pass_restored_over_H':bool(rec_go),'safe_positive_witness_coverage_restored_over_H':bool(cov_go),'selectivity_checks':selectivity,'safe_positive_pass_checks':recall,'safe_positive_witness_coverage_checks':coverage},
  'attribution_order':['K-B semantics-aligned absolute source (primary)','I-H isolates observable stability active-set semantics','J-H isolates executable path-capacity stopping semantics','K-I and K-J expose interaction/conditional effects','K-H tests whether constraint semantics repair the v48.63 zero-positive-witness failure','limiting-constraint diagnostics decide the next representation layer if STOP','K-A deployment only after source GO; paired Safe non-interference only afterward'],
  'scientific_contract':{'primary_hypothesis':'the v48.63 witness-coverage collapse is caused by observation-only constraint-semantics mismatch, not recovery-option identity/common-support or option-set quantification','GO_requires':['K-B source AUC improves in all eight Near/Contact cells and >=0.01 in at least six','teacher-infeasible/harmful pass stays <=0.25 and >=0.10 below F in every cell','safe-positive pass improves over H in all eight cells and by >=0.15 in at least six','safe-positive positive-common-option coverage improves over H in all eight cells and by >=0.15 in at least six','I/J factor arms identify whether active-set, stopping-reach, or their interaction supplies the improvement','only two shared bounded gains train in each arm; Stage-I/common-support/quantifier logic remain frozen','fixed top-5 and threshold 0.5; no regime id/router, teacher future or relative-ranking change'],
   'STOP_if':['any K-B source AUC cell regresses','witness coverage remains collapsed despite semantics repair','coverage rises but harmful/infeasible pass returns toward F','state/provenance/factor isolation fails'],
   'forbidden_next_sweeps':['threshold/LR/horizon/grid search','proposal expansion/densification','option-specific free bias','generic AFE classifier sweeps','candidate-only CPHR','compensatory ERWF sum','per-option negative veto','quantifier-scalar tuning','regime routing/policy/threshold/budget','broad root/margin/encoder retraining','privileged teacher future/component distillation','relative ranking changes before source GO','recovery-option library expansion before a teacher-option-coverage audit demonstrates missing support'],
   'if_STOP_next_question':'use K limiting-constraint distributions plus teacher-option support audit to distinguish clearance/dynamic-occupancy reachability mismatch from genuine recovery-option taxonomy coverage failure','safe_role':'same shared mechanism; Safe is paired nominal-utility non-interference only after source GO'}}
 x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_64_sarw_comparison','decision':doc['preregistered_decision']['status'],'output':str(x.output)}))
if __name__=='__main__':main()
