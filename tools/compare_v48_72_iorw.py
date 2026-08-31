#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
KINDS=('dev_near','dev_contact','certificate_near','certificate_contact');VARIANTS=('balanced','precision')
def load(p):return json.loads(Path(p).read_text())
def cell(a,arm,v,k,key):return (((a.get('arms') or {}).get(arm) or {}).get(v) or {}).get(k,{}).get(key)
def delta(a,b):return None if a is None or b is None else float(a)-float(b)
def main():
 ap=argparse.ArgumentParser(description='v48.72 OC-IORW interaction-oriented observation-only reachability attribution');ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--reachability-audit',type=Path,required=True);ap.add_argument('--truth-strata',type=Path,required=True);ap.add_argument('--v71-complete',type=Path,required=True);ap.add_argument('--v71-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);x=ap.parse_args()
 a=load(x.feasibility_audit);tr=load(x.reachability_audit);truth=load(x.truth_strata);v71=load(x.v71_complete);c71=load(x.v71_comparison)
 deltas={};cert=[];main=[];nonfloor=[];source=[];select=[];safe=[]
 for v in VARIANTS:
  deltas[v]={}
  for k in KINDS:
   vals={arm:cell(a,arm,v,k,'absolute_feasibility_auc') for arm in ('B_native','F_ERWF','P66_OCACRW','E70_OCCSOFT','J71_HISTORY_TUBE','L72_BOX_SUPPORT','M72_OCIORW')}
   lj=delta(vals['L72_BOX_SUPPORT'],vals['J71_HISTORY_TUBE']);ml=delta(vals['M72_OCIORW'],vals['L72_BOX_SUPPORT']);mj=delta(vals['M72_OCIORW'],vals['J71_HISTORY_TUBE']);me=delta(vals['M72_OCIORW'],vals['E70_OCCSOFT']);mb=delta(vals['M72_OCIORW'],vals['B_native'])
   deltas[v][k]={'L_minus_J71_auc':lj,'M_minus_L72_auc':ml,'M_minus_J71_auc':mj,'M_minus_E70_auc':me,'M_minus_B_auc':mb};source.append(mb is not None and mb>0)
   q=(((tr.get('comparisons') or {}).get(v) or {}).get(k) or {});ljq=q.get('L72_minus_J71') or {};mjq=q.get('M72_minus_J71') or {};mlq=q.get('M72_minus_L72') or {}
   cert.append({'variant':v,'split':k,'L_equals_J':bool(ljq.get('positive_certificate_set_equal')),'M_equals_J':bool(mjq.get('positive_certificate_set_equal')),'M_equals_L':bool(mlq.get('positive_certificate_set_equal'))})
   mca=cell(a,'M72_OCIORW',v,k,'positive_certificate_probability_auc_for_teacher_feasibility');jca=cell(a,'J71_HISTORY_TUBE',v,k,'positive_certificate_probability_auc_for_teacher_feasibility');mh=cell(a,'M72_OCIORW',v,k,'harmful_pass_fraction');mti=cell(a,'M72_OCIORW',v,k,'teacher_infeasible_pass_fraction');jh=cell(a,'J71_HISTORY_TUBE',v,k,'harmful_pass_fraction');jti=cell(a,'J71_HISTORY_TUBE',v,k,'teacher_infeasible_pass_fraction');ret=(mjq.get('selective_retention') or {}).get('teacher_feasible_ratio_gt_teacher_infeasible') is True
   main.append({'variant':v,'split':k,'source_auc_delta_M_minus_J':mj,'positive_cert_probability_auc_delta_M_minus_J':delta(mca,jca),'harm_TI_nonincrease':float(mh)<=float(jh)+.02+1e-12 and float(mti)<=float(jti)+.02+1e-12,'teacher_feasible_retained_over_infeasible':ret})
   mnf=cell(truth,'M72_OCIORW',v,k,'absolute_feasibility_auc_excluding_exact_0p5_feasible');jnf=cell(truth,'J71_HISTORY_TUBE',v,k,'absolute_feasibility_auc_excluding_exact_0p5_feasible');nonfloor.append({'variant':v,'split':k,'M_minus_J_nonfloor_auc':delta(mnf,jnf),'M_nonfloor_auc':mnf,'J_nonfloor_auc':jnf})
   fh=cell(a,'F_ERWF',v,k,'harmful_pass_fraction');fti=cell(a,'F_ERWF',v,k,'teacher_infeasible_pass_fraction');caph=min(.25,float(fh)-.10);capt=min(.25,float(fti)-.10);select.append({'variant':v,'split':k,'M_harmful':mh,'M_teacher_infeasible':mti,'cap_harmful':caph,'cap_TI':capt,'valid':float(mh)<=caph and float(mti)<=capt})
   ms=cell(a,'M72_OCIORW',v,k,'safe_positive_pass_fraction');ps=cell(a,'P66_OCACRW',v,k,'safe_positive_pass_fraction');safe.append({'variant':v,'split':k,'M_minus_P66':delta(ms,ps),'valid':ms is not None and ps is not None and float(ms)>float(ps)})
 pre71=bool(v71.get('valid')) and bool(v71.get('attribution_ready')) and v71.get('engineering_version')=='v48.71.0-OC-BORW' and not v71.get('test_roots_read');pr71=c71.get('preregistered_decision') or {};branch=pr71.get('status')=='STOP' and pr71.get('occupancy_reachability_trust_go') is False and pr71.get('next_branch')=='interaction_aware_observation_only_occupancy_reachability'
 cert_go=all(z['L_equals_J'] and z['M_equals_J'] and z['M_equals_L'] for z in cert)
 trust_order=sum((z['source_auc_delta_M_minus_J'] or -9)>0 for z in main)>=6 and sum((z['positive_cert_probability_auc_delta_M_minus_J'] or -9)>0 for z in main)>=6 and all(z['harm_TI_nonincrease'] for z in main) and sum(z['teacher_feasible_retained_over_infeasible'] for z in main)>=6
 physical_go=sum((z['M_minus_J_nonfloor_auc'] or -9)>0 for z in nonfloor)>=6
 mechanism_go=bool(pre71 and branch and cert_go and trust_order and physical_go)
 label_only=bool(pre71 and branch and cert_go and trust_order and not physical_go)
 source_go=all(source) and sum((deltas[v][k]['M_minus_B_auc'] or -9)>=.01 for v in VARIANTS for k in KINDS)>=6
 select_go=all(z['valid'] for z in select);safe_go=all(z['valid'] for z in safe) and sum((z['M_minus_P66'] or -9)>=.05 for z in safe)>=6
 full=bool(mechanism_go and source_go and select_go and safe_go)
 box_positive=sum((deltas[v][k]['L_minus_J71_auc'] or -9)>0 for v in VARIANTS for k in KINDS)
 hull_positive=sum((deltas[v][k]['M_minus_L72_auc'] or -9)>0 for v in VARIANTS for k in KINDS)
 if full:status='GO';next_branch='deployment_propagation_and_safe_noninterference'
 elif mechanism_go:status='MECHANISM_GO_SOURCE_STOP';next_branch='trust_conditioned_boundary_transport'
 elif label_only:status='LABEL_GO_NONFLOOR_STOP';next_branch='teacher_truth_contract_adjudication'
 else:
  status='STOP';next_branch='interaction_response_aware_observation_only_reachability' if box_positive<6 and hull_positive<6 else 'retain_supported_directional_set_then_interaction_response_dynamics'
 doc={'schema':'ocrap-v48.72-iorw-comparison-v1','algorithm':'v48.72-DCP-DRFC-BCDE-RIFA-OC-IORW','deltas':deltas,'preregistered_decision':{'v48_71_prerequisite_attribution_ready':pre71,'v48_71_branch_contract_valid':branch,'certificate_identity_checks':cert,'certificate_set_identity_gate':cert_go,'main_M_minus_J_checks':main,'trust_ordering_gate':trust_order,'nonfloor_checks':nonfloor,'nonfloor_physical_consistency_gate':physical_go,'interaction_oriented_reachability_go':mechanism_go,'label_ordering_without_nonfloor_physical_go':label_only,'box_direction_main_effect_positive_cells':box_positive,'hull_tightening_main_effect_positive_cells':hull_positive,'selectivity_checks':select,'selectivity_gate':select_go,'safe_positive_checks':safe,'safe_positive_vs_P66_gate':safe_go,'source_gate':source_go,'full_source_go':full,'status':status,'next_branch':next_branch},'test_roots_read':False,'dataset_reconstruction':False}
 x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_72_iorw_comparison','status':status,'output':str(x.output)}))
 return 0
if __name__=='__main__':raise SystemExit(main())
