#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
KINDS=('dev_near','dev_contact','certificate_near','certificate_contact'); VARIANTS=('balanced','precision')
def load(p):return json.loads(Path(p).read_text())
def cell(a,arm,v,k,key):return (((a.get('arms') or {}).get(arm) or {}).get(v) or {}).get(k,{}).get(key)
def delta(a,b):return None if a is None or b is None else float(a)-float(b)
def main():
 ap=argparse.ArgumentParser(description='v48.69 OC-DTRW demand-normalized projection-fidelity attribution')
 ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--demand-audit',type=Path,required=True);ap.add_argument('--truth-audit',type=Path,required=True);ap.add_argument('--v68-complete',type=Path,required=True);ap.add_argument('--v68-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);x=ap.parse_args()
 a=load(x.feasibility_audit);dem=load(x.demand_audit);truth=load(x.truth_audit);v68=load(x.v68_complete);c68=load(x.v68_comparison)
 deltas={};source=[];sel=[];safe_p66=[];safe_t=[];mechanism=[]
 for v in VARIANTS:
  deltas[v]={}
  for k in KINDS:
   vals={arm:cell(a,arm,v,k,'absolute_feasibility_auc') for arm in ('B_native','F_ERWF','P66_OCACRW','Q67_CTRLPROJ','T68_FIDELITY','D69_DTRW')}
   db=delta(vals['D69_DTRW'],vals['B_native']);dt=delta(vals['D69_DTRW'],vals['T68_FIDELITY']);dp=delta(vals['D69_DTRW'],vals['P66_OCACRW'])
   deltas[v][k]={'D_minus_B_auc':db,'D_minus_T68_auc':dt,'D_minus_P66_auc':dp,'T68_minus_Q67_auc':delta(vals['T68_FIDELITY'],vals['Q67_CTRLPROJ'])}
   source.append(db is not None and db>0)
   dh=cell(a,'D69_DTRW',v,k,'harmful_pass_fraction');dti=cell(a,'D69_DTRW',v,k,'teacher_infeasible_pass_fraction');fh=cell(a,'F_ERWF',v,k,'harmful_pass_fraction');fti=cell(a,'F_ERWF',v,k,'teacher_infeasible_pass_fraction');th=cell(a,'T68_FIDELITY',v,k,'harmful_pass_fraction');tti=cell(a,'T68_FIDELITY',v,k,'teacher_infeasible_pass_fraction');ph=cell(a,'P66_OCACRW',v,k,'harmful_pass_fraction');pti=cell(a,'P66_OCACRW',v,k,'teacher_infeasible_pass_fraction')
   caph=min(.25,float(fh)-.10); capt=min(.25,float(fti)-.10)
   sel.append({'variant':v,'split':k,'D_harmful':dh,'D_teacher_infeasible':dti,'T_harmful':th,'T_teacher_infeasible':tti,'P66_harmful':ph,'P66_teacher_infeasible':pti,'preregistered_cap_harmful':caph,'preregistered_cap_TI':capt,'valid':float(dh)<=caph and float(dti)<=capt})
   ds=cell(a,'D69_DTRW',v,k,'safe_positive_pass_fraction');ps=cell(a,'P66_OCACRW',v,k,'safe_positive_pass_fraction');ts=cell(a,'T68_FIDELITY',v,k,'safe_positive_pass_fraction')
   safe_p66.append({'variant':v,'split':k,'D_minus_P66':delta(ds,ps),'valid':ds is not None and ps is not None and float(ds)>float(ps)})
   safe_t.append({'variant':v,'split':k,'D_minus_T68':delta(ds,ts),'valid':ds is not None and ts is not None and float(ds)>float(ts)})
   dc=(((dem.get('comparisons') or {}).get(v) or {}).get(k) or {})
   cert_equal=bool(dc.get('positive_certificate_set_equal'))
   harm_bound=float(dh)<=float(th)+.02+1e-12 and float(dti)<=float(tti)+.02+1e-12
   mechanism.append({'variant':v,'split':k,'certificate_set_equal':cert_equal,'D_minus_T68_auc':dt,'D_minus_T68_safe_pass':delta(ds,ts),'harm_TI_within_T_plus_0p02':harm_bound,'newly_passed':dc.get('newly_passed'),'valid':cert_equal and harm_bound})
 source_go=all(source) and sum((deltas[v][k]['D_minus_B_auc'] or -9)>=.01 for v in VARIANTS for k in KINDS)>=6
 sel_go=all(z['valid'] for z in sel)
 safe_p66_go=all(z['valid'] for z in safe_p66) and sum((z['D_minus_P66'] or -9)>=.05 for z in safe_p66)>=6
 safe_t_go=sum(z['valid'] for z in safe_t)>=6 and sum((z['D_minus_T68'] or -9)>=.05 for z in safe_t)>=4
 mech_go=all(z['certificate_set_equal'] for z in mechanism) and sum((z['D_minus_T68_auc'] or -9)>0 for z in mechanism)>=6 and all(z['harm_TI_within_T_plus_0p02'] for z in mechanism) and safe_t_go
 pre68=bool(v68.get('valid')) and bool(v68.get('attribution_ready')) and v68.get('engineering_version')=='v48.68.0-OC-RTRW' and not v68.get('test_roots_read')
 pr68=c68.get('preregistered_decision') or {}; branch=pr68.get('status')=='STOP' and pr68.get('projection_fidelity_mechanism_gate') is True and pr68.get('robust_occupancy_mechanism_gate') is False
 go=bool(pre68 and branch and source_go and sel_go and safe_p66_go and mech_go)
 doc={'schema':'ocrap-v48.69-dtrw-comparison-v1','source_deltas':deltas,'preregistered_decision':{'status':'GO' if go else 'STOP','v48_68_prerequisite_attribution_ready':pre68,'v48_68_branch_contract_valid':branch,'source_gate':source_go,'selectivity_gate':sel_go,'safe_positive_vs_P66_gate':safe_p66_go,'demand_tempering_mechanism_gate':mech_go,'safe_positive_vs_T68_gate':safe_t_go,'selectivity_checks':sel,'safe_positive_vs_P66_checks':safe_p66,'safe_positive_vs_T68_checks':safe_t,'demand_tempering_checks':mechanism},'attribution_order':['D-B is the primary absolute-source gate','D-T68 isolates demand normalization of the validated projection-fidelity trust signal','certificate-set identity proves the intervention is trust transport rather than a hidden physical veto','newly-passed rows test whether urgency tempering rescues true safe positives without permissive relapse','truth-debt audit is diagnostic only; boundary transport remains OFF until trust/admission GO','deployment and Safe nominal utility are interpreted only after source GO'],'scientific_contract':{'primary_hypothesis':'v48.68 validated projection severity as soft trust but showed absolute severity disproportionately suppresses urgent safe-positive recoveries; v48.69 normalizes the projection penalty by observation-derived active recovery demand while keeping physical witness sign, Stage-I, route/re-entry, top-K, threshold and boundary transport frozen.','frozen':['v48.56-A Stage-I tensors','v48.67 actuator projection ON','v48.66 route+reentry constraints ON','v48.64 active-set alignment ON','v48.68 projection-fidelity signal ON','robust occupancy hard-min OFF','boundary transport OFF','class-local/path-stop OFF','OC-MERO/RIFA/top-K=5/threshold=0.5','relative filter/ranker','option library','teacher labels/datasets'],'intervention':'D69: demand-normalized projection fidelity f=(1+demand)/(1+demand+raw_projection_violation), with demand=max(clearance deficit, active stability deficit) reconstructed only from observation-derived signed witness coordinates. Zero demand is exact v48.68 T.','forbidden':['threshold/LR/horizon/feature/class-weight grids','proposal expansion/top-K search','generic AFE/MLP','candidate-only CPHR','compensatory ERWF','per-option negative veto','quantifier gain sweep','regime router/policy/threshold/budget','broad root/margin/encoder retraining','privileged teacher future/component-margin distillation','relative ranker changes before source GO','option-library expansion without teacher-option-support evidence','class-local correction as Main','path-stop as Main','v48.67 boundary transport before trust/admission GO','unconditional projected-control confidence','v48.68 hard CV/current-acceleration occupancy min','absolute projection-fidelity penalty that ignores observed recovery demand']},'demand_trust_audit':dem,'truth_debt_audit':truth,'feasibility_role_audit':a}
 x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_69_dtrw_comparison','decision':doc['preregistered_decision']['status'],'output':str(x.output)}))
if __name__=='__main__':main()
