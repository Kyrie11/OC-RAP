#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
KINDS=('dev_near','dev_contact','certificate_near','certificate_contact');VARIANTS=('balanced','precision')
def load(p):return json.loads(Path(p).read_text())
def cell(a,arm,v,k,key):return (((a.get('arms') or {}).get(arm) or {}).get(v) or {}).get(k,{}).get(key)
def delta(a,b):return None if a is None or b is None else float(a)-float(b)
def main():
 ap=argparse.ArgumentParser(description='v48.73 OC-IRRW current-state-anchored interaction-response reachability attribution')
 ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--response-audit',type=Path,required=True);ap.add_argument('--truth-strata',type=Path,required=True);ap.add_argument('--v72-complete',type=Path,required=True);ap.add_argument('--v72-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);x=ap.parse_args()
 a=load(x.feasibility_audit);ra=load(x.response_audit);truth=load(x.truth_strata);v72=load(x.v72_complete);c72=load(x.v72_comparison)
 deltas={};cert=[];main=[];nonfloor=[];source=[];select=[];safe=[]
 for v in VARIANTS:
  deltas[v]={}
  for k in KINDS:
   vals={arm:cell(a,arm,v,k,'absolute_feasibility_auc') for arm in ('B_native','F_ERWF','P66_OCACRW','M72_OCIORW','N73_ANCHORED_HULL','O73_OCIRRW')}
   nm=delta(vals['N73_ANCHORED_HULL'],vals['M72_OCIORW']);on=delta(vals['O73_OCIRRW'],vals['N73_ANCHORED_HULL']);om=delta(vals['O73_OCIRRW'],vals['M72_OCIORW']);ob=delta(vals['O73_OCIRRW'],vals['B_native'])
   deltas[v][k]={'N_minus_M72_auc':nm,'O_minus_N73_auc':on,'O_minus_M72_auc':om,'O_minus_B_auc':ob};source.append(ob is not None and ob>0)
   q=(((ra.get('comparisons') or {}).get(v) or {}).get(k) or {});nmq=q.get('N73_minus_M72') or {};omq=q.get('O73_minus_M72') or {};onq=q.get('O73_minus_N73') or {}
   cert.append({'variant':v,'split':k,'N_equals_M':bool(nmq.get('positive_certificate_set_equal')),'O_equals_M':bool(omq.get('positive_certificate_set_equal')),'O_equals_N':bool(onq.get('positive_certificate_set_equal'))})
   oca=cell(a,'O73_OCIRRW',v,k,'positive_certificate_probability_auc_for_teacher_feasibility');mca=cell(a,'M72_OCIORW',v,k,'positive_certificate_probability_auc_for_teacher_feasibility')
   oh=cell(a,'O73_OCIRRW',v,k,'harmful_pass_fraction');oti=cell(a,'O73_OCIRRW',v,k,'teacher_infeasible_pass_fraction');mh=cell(a,'M72_OCIORW',v,k,'harmful_pass_fraction');mti=cell(a,'M72_OCIORW',v,k,'teacher_infeasible_pass_fraction')
   sel=omq.get('selective_retention') or {}
   main.append({'variant':v,'split':k,'source_auc_delta_O_minus_M':om,'positive_cert_probability_auc_delta_O_minus_M':delta(oca,mca),'harm_TI_nonincrease':float(oh)<=float(mh)+.02+1e-12 and float(oti)<=float(mti)+.02+1e-12,'teacher_feasible_retained_over_infeasible':sel.get('teacher_feasible_ratio_gt_teacher_infeasible') is True,'safe_positive_retained_over_harmful':sel.get('safe_positive_ratio_gt_harmful') is True})
   onf=cell(truth,'O73_OCIRRW',v,k,'absolute_feasibility_auc_excluding_exact_0p5_feasible');mnf=cell(truth,'M72_OCIORW',v,k,'absolute_feasibility_auc_excluding_exact_0p5_feasible');nonfloor.append({'variant':v,'split':k,'O_minus_M_nonfloor_auc':delta(onf,mnf),'O_nonfloor_auc':onf,'M_nonfloor_auc':mnf})
   fh=cell(a,'F_ERWF',v,k,'harmful_pass_fraction');fti=cell(a,'F_ERWF',v,k,'teacher_infeasible_pass_fraction');caph=min(.25,float(fh)-.10);capt=min(.25,float(fti)-.10);select.append({'variant':v,'split':k,'O_harmful':oh,'O_teacher_infeasible':oti,'cap_harmful':caph,'cap_TI':capt,'valid':float(oh)<=caph and float(oti)<=capt})
   os=cell(a,'O73_OCIRRW',v,k,'safe_positive_pass_fraction');ps=cell(a,'P66_OCACRW',v,k,'safe_positive_pass_fraction');safe.append({'variant':v,'split':k,'O_minus_P66':delta(os,ps),'valid':os is not None and ps is not None and float(os)>float(ps)})
 pre72=bool(v72.get('valid')) and bool(v72.get('attribution_ready')) and v72.get('engineering_version')=='v48.72.0-OC-IORW' and not v72.get('test_roots_read')
 pr72=c72.get('preregistered_decision') or {};branch=pr72.get('status')=='STOP' and pr72.get('interaction_oriented_reachability_go') is False and pr72.get('next_branch')=='retain_supported_directional_set_then_interaction_response_dynamics'
 cert_go=all(z['N_equals_M'] and z['O_equals_M'] and z['O_equals_N'] for z in cert)
 ordering_go=sum((z['source_auc_delta_O_minus_M'] if z['source_auc_delta_O_minus_M'] is not None else -9)>0 for z in main)>=6 and sum((z['positive_cert_probability_auc_delta_O_minus_M'] if z['positive_cert_probability_auc_delta_O_minus_M'] is not None else -9)>0 for z in main)>=6
 selective_go=sum(z['teacher_feasible_retained_over_infeasible'] for z in main)>=6 and sum(z['safe_positive_retained_over_harmful'] for z in main)>=6 and all(z['harm_TI_nonincrease'] for z in main)
 physical_go=sum((z['O_minus_M_nonfloor_auc'] if z['O_minus_M_nonfloor_auc'] is not None else -9)>0 for z in nonfloor)>=6
 mechanism_go=bool(pre72 and branch and cert_go and ordering_go and selective_go and physical_go)
 label_only=bool(pre72 and branch and cert_go and ordering_go and selective_go and not physical_go)
 source_go=all(source) and sum((deltas[v][k]['O_minus_B_auc'] if deltas[v][k]['O_minus_B_auc'] is not None else -9)>=.01 for v in VARIANTS for k in KINDS)>=6
 select_go=all(z['valid'] for z in select);safe_go=all(z['valid'] for z in safe) and sum((z['O_minus_P66'] if z['O_minus_P66'] is not None else -9)>=.05 for z in safe)>=6
 full=bool(mechanism_go and source_go and select_go and safe_go)
 anchor_positive=sum((deltas[v][k]['N_minus_M72_auc'] if deltas[v][k]['N_minus_M72_auc'] is not None else -9)>0 for v in VARIANTS for k in KINDS)
 response_positive=sum((deltas[v][k]['O_minus_N73_auc'] if deltas[v][k]['O_minus_N73_auc'] is not None else -9)>0 for v in VARIANTS for k in KINDS)
 if full:status='GO';next_branch='deployment_propagation_and_safe_noninterference'
 elif mechanism_go:status='MECHANISM_GO_SOURCE_STOP';next_branch='trust_conditioned_boundary_transport'
 elif label_only:status='LABEL_GO_NONFLOOR_STOP';next_branch='teacher_truth_contract_adjudication'
 else:
  status='STOP'
  if anchor_positive>=6 and response_positive<6:next_branch='retain_current_state_anchor_and_rethink_response_rate_set'
  elif response_positive>=6:next_branch='interaction_response_semantics_or_teacher_truth_contract_diagnosis'
  else:next_branch='interaction_response_reachability_stop_no_parameter_sweep'
 doc={'schema':'ocrap-v48.73-irrw-comparison-v1','algorithm':'v48.73-DCP-DRFC-BCDE-RIFA-OC-IRRW','deltas':deltas,'preregistered_decision':{'v48_72_prerequisite_attribution_ready':pre72,'v48_72_branch_contract_valid':branch,'certificate_identity_checks':cert,'certificate_set_identity_gate':cert_go,'main_O_minus_M_checks':main,'ordering_gate':ordering_go,'selective_retention_gate':selective_go,'nonfloor_checks':nonfloor,'nonfloor_physical_consistency_gate':physical_go,'interaction_response_reachability_go':mechanism_go,'label_ordering_without_nonfloor_physical_go':label_only,'anchor_main_effect_positive_cells':anchor_positive,'response_rate_main_effect_positive_cells':response_positive,'selectivity_checks':select,'selectivity_gate':select_go,'safe_positive_checks':safe,'safe_positive_vs_P66_gate':safe_go,'source_gate':source_go,'full_source_go':full,'status':status,'next_branch':next_branch},'test_roots_read':False,'dataset_reconstruction':False}
 x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_73_irrw_comparison','status':status,'output':str(x.output)}));return 0
if __name__=='__main__':raise SystemExit(main())
