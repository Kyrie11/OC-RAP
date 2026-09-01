#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

VARIANTS=('balanced','precision'); SPLITS=('dev_near','dev_contact','certificate_near','certificate_contact')
def load(p):return json.loads(Path(p).read_text())
def cells(audit,name,field):return [audit['comparisons'][name][v][s].get(field) for v in VARIANTS for s in SPLITS]
def positives(vals):return sum(x is not None and x>0 for x in vals)
def material(vals,thr=.005):return sum(x is not None and x>=thr for x in vals)
def alltrue(vals):return all(bool(x) for x in vals)
def arm_cells(audit,arm,field):return [audit['arms'][arm][v][s].get(field) for v in VARIANTS for s in SPLITS]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--truth-audit',type=Path,required=True);ap.add_argument('--v74-complete',type=Path,required=True);ap.add_argument('--v74-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();au=load(a.truth_audit);v74c=load(a.v74_complete);v74=load(a.v74_comparison);errors=[]
 pre=(v74.get('preregistered_decision') or {})
 prereq=bool(v74c.get('valid') and v74c.get('attribution_ready') and v74c.get('engineering_version')=='v48.74.2-OC-SVBW-ENGFIX' and not v74c.get('test_roots_read') and pre.get('status')=='STOP' and pre.get('next_branch')=='signed_viability_stop_then_supervision_truth_contract_no_parameter_sweep')
 if not prereq:errors.append('V48.74 prerequisite branch mismatch')
 c_q=cells(au,'C75_minus_Q67','nonfloor_auc_delta'); d_t=cells(au,'D75_minus_T68','nonfloor_auc_delta'); d_c=cells(au,'D75_minus_C75','nonfloor_auc_delta'); t_q=cells(au,'T68_minus_Q67','nonfloor_auc_delta')
 cert_cq=cells(au,'C75_minus_Q67','positive_certificate_set_equal');cert_dt=cells(au,'D75_minus_T68','positive_certificate_set_equal');labels=cells(au,'D75_minus_T68','teacher_labels_equal')+cells(au,'C75_minus_Q67','teacher_labels_equal')
 c_main=(positives(c_q)>=6 and material(c_q)>=4); d_main=(positives(d_t)>=6 and material(d_t)>=4)
 truth_go=c_main or d_main
 fidelity_go=positives(d_c)>=6
 # The interaction is descriptive: censored-fidelity effect relative to historical-fidelity effect.
 interaction=[(dc or 0.0)-(tq or 0.0) if dc is not None and tq is not None else None for dc,tq in zip(d_c,t_q)]
 harmful=arm_cells(au,'D75_FIDELITY_CENSORED','harmful_pass_fraction');ti=arm_cells(au,'D75_FIDELITY_CENSORED','teacher_infeasible_pass_fraction');selectivity=all(x is not None and x<=.25 for x in harmful+ti)
 safe_non=arm_cells(au,'D75_FIDELITY_CENSORED','safe_positive_nonfloor_rows');support_adequate=sum((x or 0)>=5 for x in safe_non)>=6
 cert_identity=alltrue(cert_cq+cert_dt); label_identity=alltrue(labels)
 full_dt=cells(au,'D75_minus_T68','full_auc_delta');full_cq=cells(au,'C75_minus_Q67','full_auc_delta')
 if truth_go and fidelity_go and support_adequate: status='TRUTH_CONTRACT_GO_FIDELITY_RETAIN'
 elif truth_go and fidelity_go: status='TRUTH_CONTRACT_GO_POLICY_UNDERPOWERED'
 elif truth_go: status='TRUTH_CONTRACT_GO_FIDELITY_STOP'
 else: status='STOP'
 if truth_go:
  next_branch='teacher_truth_contract_causal_confound_confirmed_reconcile_paper_and_teacher_before_planner_promotion'
 else:
  next_branch='truth_floor_debt_not_dominant_training_cause_audit_absolute_supervision_representation_no_geometry_sweep'
 doc={'schema':'ocrap-v48.75-stca-comparison-v1','algorithm':'v48.75-DCP-DRFC-BCDE-RIFA-OC-STCA','engineering_version':'v48.75.0-OC-STCA','errors':errors,'preregistered_decision':{
  'v48_74_prerequisite_valid':prereq,'certificate_identity_gate':cert_identity,'teacher_label_identity_gate':label_identity,
  'C75_minus_Q67_nonfloor_positive_cells':positives(c_q),'C75_minus_Q67_nonfloor_material_cells':material(c_q),'C75_supervision_main_effect_go':c_main,
  'D75_minus_T68_nonfloor_positive_cells':positives(d_t),'D75_minus_T68_nonfloor_material_cells':material(d_t),'D75_supervision_main_effect_go':d_main,
  'truth_contract_causal_confound_go':truth_go,'D75_minus_C75_fidelity_positive_cells':positives(d_c),'projection_fidelity_under_censored_truth_go':fidelity_go,
  'historical_fidelity_nonfloor_positive_cells':positives(t_q),'interaction_nonfloor_deltas':interaction,'absolute_selectivity_gate':selectivity,
  'nonfloor_safe_positive_support_adequate':support_adequate,'status':status,'next_branch':next_branch},
  'deltas':{'C75_minus_Q67_nonfloor':c_q,'D75_minus_T68_nonfloor':d_t,'D75_minus_C75_nonfloor':d_c,'T68_minus_Q67_nonfloor':t_q,'D75_minus_T68_full':full_dt,'C75_minus_Q67_full':full_cq},
  'scientific_contract':{'primary_endpoint':'nonfloor absolute-feasibility ordering; full-label AUC is diagnostic because the intervention deliberately censors exact-0.5 supervision','material_effect':'>= +0.005 in at least 4/8 cells, with >=6/8 positive','floor_value':.5,'floor_tolerance':1e-8,'floor_rows_relabelled':False,'boundary_transport':'OFF','relative_ranker':'frozen','regime_router':'forbidden','dataset_reconstruction':False},'dataset_reconstruction':False,'test_roots_read':False}
 valid=not errors and prereq and cert_identity and label_identity
 doc['valid']=valid;doc['attribution_ready']=valid
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_75_stca_comparison','valid':valid,'status':status,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
