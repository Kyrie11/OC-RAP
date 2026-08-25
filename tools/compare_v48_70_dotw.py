#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
KINDS=('dev_near','dev_contact','certificate_near','certificate_contact');VARIANTS=('balanced','precision')
def load(p):return json.loads(Path(p).read_text())
def cell(a,arm,v,k,key):return (((a.get('arms') or {}).get(arm) or {}).get(v) or {}).get(k,{}).get(key)
def delta(a,b):return None if a is None or b is None else float(a)-float(b)
def main():
 ap=argparse.ArgumentParser(description='v48.70 OC-DOTW soft occupancy-disagreement attribution')
 ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--trust-audit',type=Path,required=True);ap.add_argument('--truth-strata',type=Path,required=True);ap.add_argument('--v69-complete',type=Path,required=True);ap.add_argument('--v69-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);x=ap.parse_args()
 a=load(x.feasibility_audit);tr=load(x.trust_audit);truth=load(x.truth_strata);v69=load(x.v69_complete);c69=load(x.v69_comparison)
 deltas={};source=[];select=[];safe=[];certid=[];e_mech=[];g_mech=[];interaction=[]
 for v in VARIANTS:
  deltas[v]={}
  for k in KINDS:
   arms=('B_native','F_ERWF','P66_OCACRW','T68_FIDELITY','D69_DTRW','E70_OCCSOFT','G70_OCDOTW')
   vals={arm:cell(a,arm,v,k,'absolute_feasibility_auc') for arm in arms}
   gb=delta(vals['G70_OCDOTW'],vals['B_native']);et=delta(vals['E70_OCCSOFT'],vals['T68_FIDELITY']);gd=delta(vals['G70_OCDOTW'],vals['D69_DTRW']);ge=delta(vals['G70_OCDOTW'],vals['E70_OCCSOFT']);dt=delta(vals['D69_DTRW'],vals['T68_FIDELITY'])
   deltas[v][k]={'G_minus_B_auc':gb,'E_minus_T68_auc':et,'G_minus_D69_auc':gd,'G_minus_E70_auc':ge,'D69_minus_T68_auc':dt,'interaction_auc':None if any(z is None for z in (gd,et)) else gd-et}
   source.append(gb is not None and gb>0)
   gh=cell(a,'G70_OCDOTW',v,k,'harmful_pass_fraction');gti=cell(a,'G70_OCDOTW',v,k,'teacher_infeasible_pass_fraction');fh=cell(a,'F_ERWF',v,k,'harmful_pass_fraction');fti=cell(a,'F_ERWF',v,k,'teacher_infeasible_pass_fraction')
   caph=min(.25,float(fh)-.10);capt=min(.25,float(fti)-.10)
   select.append({'variant':v,'split':k,'G_harmful':gh,'G_teacher_infeasible':gti,'cap_harmful':caph,'cap_TI':capt,'valid':float(gh)<=caph and float(gti)<=capt})
   gs=cell(a,'G70_OCDOTW',v,k,'safe_positive_pass_fraction');ps=cell(a,'P66_OCACRW',v,k,'safe_positive_pass_fraction')
   safe.append({'variant':v,'split':k,'G_minus_P66':delta(gs,ps),'valid':gs is not None and ps is not None and float(gs)>float(ps)})
   q=(((tr.get('comparisons') or {}).get(v) or {}).get(k) or {});etq=q.get('E70_minus_T68') or {};gdq=q.get('G70_minus_D69') or {}
   certid.append({'variant':v,'split':k,'E_equals_T':bool(etq.get('positive_certificate_set_equal')),'G_equals_D':bool(gdq.get('positive_certificate_set_equal'))})
   eca=cell(a,'E70_OCCSOFT',v,k,'positive_certificate_probability_auc_for_teacher_feasibility');tca=cell(a,'T68_FIDELITY',v,k,'positive_certificate_probability_auc_for_teacher_feasibility')
   gca=cell(a,'G70_OCDOTW',v,k,'positive_certificate_probability_auc_for_teacher_feasibility');dca=cell(a,'D69_DTRW',v,k,'positive_certificate_probability_auc_for_teacher_feasibility')
   e_h=cell(a,'E70_OCCSOFT',v,k,'harmful_pass_fraction');e_ti=cell(a,'E70_OCCSOFT',v,k,'teacher_infeasible_pass_fraction');t_h=cell(a,'T68_FIDELITY',v,k,'harmful_pass_fraction');t_ti=cell(a,'T68_FIDELITY',v,k,'teacher_infeasible_pass_fraction');d_h=cell(a,'D69_DTRW',v,k,'harmful_pass_fraction');d_ti=cell(a,'D69_DTRW',v,k,'teacher_infeasible_pass_fraction')
   esc=(etq.get('selective_contraction') or {}).get('teacher_feasible_ratio_gt_teacher_infeasible') is True;gsc=(gdq.get('selective_contraction') or {}).get('teacher_feasible_ratio_gt_teacher_infeasible') is True
   e_mech.append({'variant':v,'split':k,'auc_delta':et,'positive_cert_probability_auc_delta':delta(eca,tca),'harm_TI_nonincrease':float(e_h)<=float(t_h)+.02+1e-12 and float(e_ti)<=float(t_ti)+.02+1e-12,'selective_support_contraction':esc})
   g_mech.append({'variant':v,'split':k,'auc_delta':gd,'positive_cert_probability_auc_delta':delta(gca,dca),'harm_TI_nonincrease':float(gh)<=float(d_h)+.02+1e-12 and float(gti)<=float(d_ti)+.02+1e-12,'selective_support_contraction':gsc})
   interaction.append({'variant':v,'split':k,'D_minus_T':dt,'G_minus_E':ge,'demand_sign_rescued':ge is not None and ge>0})
 pre69=bool(v69.get('valid')) and bool(v69.get('attribution_ready')) and v69.get('engineering_version')=='v48.69.1-OC-DTRW-ENGFIX' and not v69.get('test_roots_read')
 pr69=c69.get('preregistered_decision') or {};branch=pr69.get('status')=='STOP' and pr69.get('demand_tempering_mechanism_gate') is False
 source_go=all(source) and sum((deltas[v][k]['G_minus_B_auc'] or -9)>=.01 for v in VARIANTS for k in KINDS)>=6
 select_go=all(z['valid'] for z in select)
 safe_go=all(z['valid'] for z in safe) and sum((z['G_minus_P66'] or -9)>=.05 for z in safe)>=6
 cert_go=all(z['E_equals_T'] and z['G_equals_D'] for z in certid)
 # Mechanism gate is intentionally trust-focused, not threshold-focused.  E/G
 # must improve within-set probability ordering and source AUC without hidden
 # certificate deletion.  Support contraction must preferentially spare teacher
 # feasible witnesses in at least 6/8 cells for each contrast.
 e_go=(sum((z['auc_delta'] or -9)>0 for z in e_mech)>=6 and sum((z['positive_cert_probability_auc_delta'] or -9)>0 for z in e_mech)>=6 and all(z['harm_TI_nonincrease'] for z in e_mech) and sum(z['selective_support_contraction'] for z in e_mech)>=6)
 g_go=(sum((z['auc_delta'] or -9)>0 for z in g_mech)>=6 and sum((z['positive_cert_probability_auc_delta'] or -9)>0 for z in g_mech)>=6 and all(z['harm_TI_nonincrease'] for z in g_mech) and sum(z['selective_support_contraction'] for z in g_mech)>=6)
 interaction_go=sum(z['demand_sign_rescued'] for z in interaction)>=6
 trust_go=cert_go and (e_go or g_go)
 full_source_go=bool(pre69 and branch and source_go and select_go and safe_go and trust_go)
 mechanism_go=bool(pre69 and branch and trust_go)
 status='GO' if full_source_go else ('MECHANISM_GO_SOURCE_STOP' if mechanism_go else 'STOP')
 next_branch=('deployment_and_safe_noninterference' if full_source_go else ('trust_conditioned_boundary_transport_factorial' if mechanism_go else 'richer_observation_only_occupancy_reachability'))
 doc={'schema':'ocrap-v48.70-dotw-comparison-v2','source_deltas':deltas,'preregistered_decision':{'status':status,'next_branch':next_branch,'full_source_go':full_source_go,'trust_identification_go':mechanism_go,'v48_69_prerequisite_attribution_ready':pre69,'v48_69_stop_branch_valid':branch,'source_gate':source_go,'selectivity_gate':select_go,'safe_positive_vs_P66_gate':safe_go,'certificate_set_identity_gate':cert_go,'soft_occupancy_trust_mechanism_gate':trust_go,'E_minus_T_mechanism_gate':e_go,'G_minus_D_mechanism_gate':g_go,'demand_under_occupancy_interaction_gate':interaction_go,'selectivity_checks':select,'safe_positive_checks':safe,'certificate_identity_checks':certid,'E_minus_T_checks':e_mech,'G_minus_D_checks':g_mech,'interaction_checks':interaction},'attribution_order':['certificate-set identity E==T68 and G==D69 first proves the new factor is soft trust/support rather than hidden certificate deletion','E-T68 isolates soft occupancy disagreement without demand normalization','G-D69 isolates the same occupancy factor under v48.69 demand normalization','within-positive-certificate probability AUC and selective support contraction decide the preregistered trust-identification mechanism','G-E asks whether demand becomes useful only after occupancy uncertainty is represented','G-B/source, selectivity and safe-positive admission are then reported as the higher-level full-source gate; pure attenuation is not expected to create new positive passes','truth-floor strata is read-only diagnostic; boundary transport remains frozen until trust-identification GO','deployment/Safe nominal utility remain deferred until full absolute source GO'],'scientific_contract':{'primary_hypothesis':'v48.69 falsified observation demand as a sufficient explanation for large actuator projection; v48.67/69 diagnostics still show deterministic CV clearance can rank false witnesses as safer. v48.70 is deliberately a trust-identification round: it tests whether disagreement between the historical CV forecast and the already-defined bounded observed-acceleration counterfactual is useful as soft epistemic trust, without repeating the falsified v48.68 hard min. Because w_occ<=1 and boundary transport is frozen, the mechanism is not allowed to claim that attenuation alone solves positive admission.','factorial_arms':{'T68_FIDELITY':'historical: demand OFF, occupancy-soft OFF','D69_DTRW':'historical: demand ON, occupancy-soft OFF','E70_OCCSOFT':'new: demand OFF, occupancy-soft ON','G70_OCDOTW':'new Main: demand ON, occupancy-soft ON'},'intervention':'For each projected recovery option, delta_occ=max_{t,agent} ReLU(clear_CV-clear_CA)/distance_scale; w_occ=1/(1+delta_occ). CV remains the signed physical certificate. E/G multiply positive common support by w_occ; no hard min, new threshold, new horizon, or new trainable parameter.','frozen':['v48.56-A Stage-I tensors','v48.67 actuator projection ON','v48.66 route+reentry ON','v48.64 active-set alignment ON','v48.68 projection-fidelity ON','v48.68 hard robust-occupancy min OFF','v48.67 boundary transport OFF','class-local/path-stop OFF','OC-MERO/RIFA/top-K=5/threshold=0.5','relative filter/ranker','option library','teacher labels/datasets'],'forbidden':['threshold/LR/horizon/feature/class-weight grids','proposal expansion/top-K search','generic AFE/MLP','candidate-only CPHR','compensatory ERWF','per-option negative veto','quantifier gain sweep','regime router/policy/threshold/budget','broad root/margin/encoder retraining','privileged teacher future/component-margin distillation','relative ranker changes before source GO','option-library expansion without teacher-option-support evidence','class-local correction as Main','path-stop as Main','reintroducing hard control veto','v48.67 boundary transport before trust mechanism GO','unconditional projected-control confidence','v48.68 hard CV/current-acceleration occupancy min','acceleration multiplier/horizon/threshold sweep','demand-only projection forgiveness as Main']},'soft_occupancy_trust_audit':tr,'truth_floor_strata_audit':truth,'feasibility_role_audit':a}
 x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_70_dotw_comparison','decision':doc['preregistered_decision']['status'],'output':str(x.output)}))
if __name__=='__main__':main()
