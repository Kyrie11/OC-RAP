#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
KINDS=('dev_near','dev_contact','certificate_near','certificate_contact');VARIANTS=('balanced','precision')
def load(p):return json.loads(Path(p).read_text())
def cell(a,arm,v,k,key):return (((a.get('arms') or {}).get(arm) or {}).get(v) or {}).get(k,{}).get(key)
def delta(a,b):return None if a is None or b is None else float(a)-float(b)
def positive_count(rows,key):return sum((z.get(key) if z.get(key) is not None else -9)>0 for z in rows)
def material_count(rows,key,thr=.005):return sum((z.get(key) if z.get(key) is not None else -9)>=thr for z in rows)
def main():
 ap=argparse.ArgumentParser(description='v48.74 OC-SVBW preregistered attribution')
 ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--response-audit',type=Path,required=True);ap.add_argument('--truth-strata',type=Path,required=True);ap.add_argument('--v73-complete',type=Path,required=True);ap.add_argument('--v73-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);x=ap.parse_args()
 a=load(x.feasibility_audit);ra=load(x.response_audit);truth=load(x.truth_strata);v73=load(x.v73_complete);c73=load(x.v73_comparison)
 deltas={};cert=[];p_checks=[];qp_checks=[];q_checks=[];p_nonfloor=[];q_nonfloor=[];source=[];select=[];safe=[]
 for v in VARIANTS:
  deltas[v]={}
  for k in KINDS:
   vals={arm:cell(a,arm,v,k,'absolute_feasibility_auc') for arm in ('B_native','F_ERWF','P66_OCACRW','T68_FIDELITY','P74_FIRST_ORDER_SVBW','Q74_OCSVBW')}
   pr=delta(vals['P74_FIRST_ORDER_SVBW'],vals['T68_FIDELITY']);qp=delta(vals['Q74_OCSVBW'],vals['P74_FIRST_ORDER_SVBW']);qr=delta(vals['Q74_OCSVBW'],vals['T68_FIDELITY']);qb=delta(vals['Q74_OCSVBW'],vals['B_native'])
   deltas[v][k]={'P_minus_reference_auc':pr,'Q_minus_P_auc':qp,'Q_minus_reference_auc':qr,'Q_minus_B_auc':qb};source.append(qb is not None and qb>0)
   rr=(((ra.get('comparisons') or {}).get(v) or {}).get(k) or {});prr=rr.get('P74_minus_reference') or {};qpr=rr.get('Q74_minus_P74') or {};qrr=rr.get('Q74_minus_reference') or {}
   cert.append({'variant':v,'split':k,'P_equals_reference':bool(prr.get('positive_certificate_set_equal')),'Q_equals_P':bool(qpr.get('positive_certificate_set_equal')),'Q_equals_reference':bool(qrr.get('positive_certificate_set_equal'))})
   pca=cell(a,'P74_FIRST_ORDER_SVBW',v,k,'positive_certificate_probability_auc_for_teacher_feasibility');qca=cell(a,'Q74_OCSVBW',v,k,'positive_certificate_probability_auc_for_teacher_feasibility');rca=cell(a,'T68_FIDELITY',v,k,'positive_certificate_probability_auc_for_teacher_feasibility')
   ph=cell(a,'P74_FIRST_ORDER_SVBW',v,k,'harmful_pass_fraction');pti=cell(a,'P74_FIRST_ORDER_SVBW',v,k,'teacher_infeasible_pass_fraction');qh=cell(a,'Q74_OCSVBW',v,k,'harmful_pass_fraction');qti=cell(a,'Q74_OCSVBW',v,k,'teacher_infeasible_pass_fraction');rh=cell(a,'T68_FIDELITY',v,k,'harmful_pass_fraction');rti=cell(a,'T68_FIDELITY',v,k,'teacher_infeasible_pass_fraction')
   ps=(prr.get('selective_retention') or {});qps=(qpr.get('selective_retention') or {});qs=(qrr.get('selective_retention') or {})
   p_checks.append({'variant':v,'split':k,'source_auc_delta':pr,'positive_cert_probability_auc_delta':delta(pca,rca),'teacher_feasible_retained_over_infeasible':ps.get('teacher_feasible_ratio_gt_teacher_infeasible') is True,'safe_positive_retained_over_harmful':ps.get('safe_positive_ratio_gt_harmful') is True,'harm_TI_nonrelapse':float(ph)<=float(rh)+.02+1e-12 and float(pti)<=float(rti)+.02+1e-12})
   qp_checks.append({'variant':v,'split':k,'source_auc_delta':qp,'positive_cert_probability_auc_delta':delta(qca,pca),'teacher_feasible_retained_over_infeasible':qps.get('teacher_feasible_ratio_gt_teacher_infeasible') is True,'safe_positive_retained_over_harmful':qps.get('safe_positive_ratio_gt_harmful') is True,'harm_TI_nonrelapse':float(qh)<=float(ph)+.02+1e-12 and float(qti)<=float(pti)+.02+1e-12})
   q_checks.append({'variant':v,'split':k,'source_auc_delta':qr,'positive_cert_probability_auc_delta':delta(qca,rca),'teacher_feasible_retained_over_infeasible':qs.get('teacher_feasible_ratio_gt_teacher_infeasible') is True,'safe_positive_retained_over_harmful':qs.get('safe_positive_ratio_gt_harmful') is True,'harm_TI_nonrelapse':float(qh)<=float(rh)+.02+1e-12 and float(qti)<=float(rti)+.02+1e-12})
   pnf=cell(truth,'P74_FIRST_ORDER_SVBW',v,k,'absolute_feasibility_auc_excluding_exact_0p5_feasible');qnf=cell(truth,'Q74_OCSVBW',v,k,'absolute_feasibility_auc_excluding_exact_0p5_feasible');rnf=cell(truth,'T68_FIDELITY',v,k,'absolute_feasibility_auc_excluding_exact_0p5_feasible')
   p_nonfloor.append({'variant':v,'split':k,'P_minus_reference_nonfloor_auc':delta(pnf,rnf)});q_nonfloor.append({'variant':v,'split':k,'Q_minus_reference_nonfloor_auc':delta(qnf,rnf)})
   fh=cell(a,'F_ERWF',v,k,'harmful_pass_fraction');fti=cell(a,'F_ERWF',v,k,'teacher_infeasible_pass_fraction');caph=min(.25,float(fh)-.10);capt=min(.25,float(fti)-.10);select.append({'variant':v,'split':k,'Q_harmful':qh,'Q_teacher_infeasible':qti,'cap_harmful':caph,'cap_TI':capt,'valid':float(qh)<=caph and float(qti)<=capt})
   qsafe=cell(a,'Q74_OCSVBW',v,k,'safe_positive_pass_fraction');psafe=cell(a,'P66_OCACRW',v,k,'safe_positive_pass_fraction');safe.append({'variant':v,'split':k,'Q_minus_P66':delta(qsafe,psafe),'valid':qsafe is not None and psafe is not None and float(qsafe)>float(psafe)})
 pre73=bool(v73.get('valid')) and bool(v73.get('attribution_ready')) and v73.get('engineering_version')=='v48.73.0-OC-IRRW' and not v73.get('test_roots_read')
 pr73=c73.get('preregistered_decision') or {};branch=pr73.get('status')=='STOP' and pr73.get('interaction_response_reachability_go') is False and pr73.get('next_branch')=='interaction_response_reachability_stop_no_parameter_sweep'
 cert_go=all(z['P_equals_reference'] and z['Q_equals_P'] and z['Q_equals_reference'] for z in cert)
 def mechanism(rows,nonfloor,key):
  ordering=positive_count(rows,'source_auc_delta')>=6 and positive_count(rows,'positive_cert_probability_auc_delta')>=6
  dual=sum(z['teacher_feasible_retained_over_infeasible'] and z['safe_positive_retained_over_harmful'] for z in rows)>=6
  nonrelapse=all(z['harm_TI_nonrelapse'] for z in rows)
  physical=positive_count(nonfloor,key)>=6
  material=material_count(rows,'source_auc_delta')>0 or material_count(rows,'positive_cert_probability_auc_delta')>0
  return {'ordering_gate':ordering,'dual_selectivity_gate':dual,'harm_TI_nonrelapse_gate':nonrelapse,'nonfloor_gate':physical,'material_effect_gate':material,'go':bool(ordering and dual and nonrelapse and physical and material)}
 p_mech=mechanism(p_checks,p_nonfloor,'P_minus_reference_nonfloor_auc')
 q_main_mech=mechanism(q_checks,q_nonfloor,'Q_minus_reference_nonfloor_auc')
 qp_order=positive_count(qp_checks,'source_auc_delta')>=6 and positive_count(qp_checks,'positive_cert_probability_auc_delta')>=6
 qp_dual=sum(z['teacher_feasible_retained_over_infeasible'] and z['safe_positive_retained_over_harmful'] for z in qp_checks)>=6
 qp_nonrelapse=all(z['harm_TI_nonrelapse'] for z in qp_checks);qp_material=material_count(qp_checks,'source_auc_delta')>0 or material_count(qp_checks,'positive_cert_probability_auc_delta')>0
 q_increment_go=bool(qp_order and qp_dual and qp_nonrelapse and qp_material)
 source_go=all(source) and sum((deltas[v][k]['Q_minus_B_auc'] if deltas[v][k]['Q_minus_B_auc'] is not None else -9)>=.01 for v in VARIANTS for k in KINDS)>=6
 select_go=all(z['valid'] for z in select);safe_go=all(z['valid'] for z in safe) and sum((z['Q_minus_P66'] if z['Q_minus_P66'] is not None else -9)>=.05 for z in safe)>=6
 full=bool(pre73 and branch and cert_go and q_main_mech['go'] and source_go and select_go and safe_go)
 label_only=bool(pre73 and branch and cert_go and q_main_mech['ordering_gate'] and q_main_mech['dual_selectivity_gate'] and q_main_mech['harm_TI_nonrelapse_gate'] and not q_main_mech['nonfloor_gate'])
 if full:status='GO';next_branch='deployment_propagation_and_safe_noninterference'
 elif label_only:status='LABEL_GO_NONFLOOR_STOP';next_branch='teacher_truth_contract_adjudication'
 elif pre73 and branch and cert_go and q_main_mech['go']:status='MECHANISM_GO_SOURCE_STOP';next_branch='trust_conditioned_boundary_transport'
 elif pre73 and branch and cert_go and p_mech['go'] and not q_increment_go:status='FIRST_ORDER_GO_HIGH_ORDER_STOP';next_branch='promote_first_order_signed_viability_only'
 else:status='STOP';next_branch='signed_viability_stop_then_supervision_truth_contract_no_parameter_sweep'
 doc={'schema':'ocrap-v48.74-svbw-comparison-v2','algorithm':'v48.74-DCP-DRFC-BCDE-RIFA-OC-SVBW','deltas':deltas,'preregistered_decision':{'v48_73_prerequisite_attribution_ready':pre73,'v48_73_stop_branch_valid':branch,'certificate_identity_checks':cert,'certificate_set_identity_gate':cert_go,'P_minus_reference_checks':p_checks,'P_first_order_mechanism':p_mech,'Q_minus_P_checks':qp_checks,'Q_high_order_increment_gate':q_increment_go,'Q_minus_reference_checks':q_checks,'Q_main_mechanism':q_main_mech,'nonfloor_P_checks':p_nonfloor,'nonfloor_Q_checks':q_nonfloor,'selectivity_checks':select,'selectivity_gate':select_go,'safe_positive_checks':safe,'safe_positive_vs_P66_gate':safe_go,'source_gate':source_go,'full_source_go':full,'status':status,'next_branch':next_branch},'attribution_order':['engineering/runtime/reference/state isolation','positive-certificate sign/set identity P/reference, Q/reference, Q/P','P74-reference first-order finite-time viability','Q74-P74 high-order acceleration contribution','Q74-reference full trust ordering and dual selectivity','non-floor physical consistency','Q74-native B source + historical selectivity + safe-positive admission'],'scientific_contract':{'accepted_reference':'T68 projection-fidelity reference with active-set, route/reentry and actuator projection retained','coordinate_20':'raw first-order finite-time signed-viability debt','coordinate_21':'raw non-compensatory high-order signed-viability debt','trust':'w=1/(1+debt), strictly positive; certificate sign/set must remain fixed','minimum_material_effect':'~5e-3 ordering delta or positive bootstrap lower bound; this checker uses the registered ~5e-3 deterministic fallback because no bootstrap artifact is required by the launcher','boundary_transport':'OFF until trust + non-floor GO'},'test_roots_read':False,'dataset_reconstruction':False}
 x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_74_svbw_comparison','status':status,'output':str(x.output)}));return 0
if __name__=='__main__':raise SystemExit(main())
