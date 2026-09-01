#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
VARIANTS=('balanced','precision');SPLITS=('dev_near','dev_contact','certificate_near','certificate_contact')
def load(p):return json.loads(Path(p).read_text())
def vals(a,name,field):return [a['comparisons'][name][v][s].get(field) for v in VARIANTS for s in SPLITS]
def pos(x):return sum(v is not None and v>0 for v in x)
def mat(x,t=.005):return sum(v is not None and v>=t for v in x)
def dec(x):return sum(v is not None and v<0 for v in x)
def decmat(x,t=.01):return sum(v is not None and v<=-t for v in x)
def alltrue(x):return all(bool(v) for v in x)
def arm(a,name,field):return [a['arms'][name][v]['splits'][s].get(field) for v in VARIANTS for s in SPLITS]
def nonrelapse(a,new,base,tol=.02):
 z=[]
 for field in ('harmful_pass_fraction','teacher_infeasible_pass_fraction'):
  n=arm(a,new,field);b=arm(a,base,field);z.extend([(x is not None and y is not None and x<=.25 and x<=y+tol) for x,y in zip(n,b)])
 return all(z)
def source_go(a,name):
 nf=vals(a,name,'nonfloor_auc_delta');hb=vals(a,name,'nonfloor_huber_delta')
 return pos(nf)>=6 and mat(nf)>=4 and dec(hb)>=6 and decmat(hb)>=4, nf, hb
def full_go(a,name,new_arm,base_arm):
 full=vals(a,name,'full_auc_delta');nf=vals(a,name,'nonfloor_auc_delta')
 source=(pos(full)==8 and mat(full,.01)>=6 and pos(nf)>=6 and mat(nf,.005)>=4)
 n=arm(a,new_arm,'safe_positive_nonfloor_pass_fraction');b=arm(a,base_arm,'safe_positive_nonfloor_pass_fraction');cnt=arm(a,base_arm,'safe_positive_nonfloor_rows')
 adequate=[(x,y,c) for x,y,c in zip(n,b,cnt) if c is not None and c>=5 and x is not None and y is not None]
 admission=(len(adequate)>=4 and all(x>=y for x,y,c in adequate) and sum((x-y)>=.05 for x,y,c in adequate)>=3)
 return source and admission,{'full_auc_delta':full,'nonfloor_auc_delta':nf,'adequate_safe_positive_cells':len(adequate),'adequate_safe_positive_deltas':[x-y for x,y,c in adequate],'source_ordering_gate':source,'safe_positive_admission_gate':admission}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--v76-complete',type=Path,required=True);ap.add_argument('--v76-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();au=load(a.audit);c76=load(a.v76_complete);p76=load(a.v76_comparison);errors=[]
 pre=p76.get('preregistered_decision') or {};prereq=bool(c76.get('valid') and c76.get('attribution_ready') and c76.get('engineering_version')=='v48.76.0-OC-ICSM' and pre.get('status')=='STOP' and pre.get('signed_margin_supervision_go') is False and pre.get('next_branch')=='signed_margin_supervision_stop_two_gain_transport_representation_bottleneck_then_structured_absolute_source_interface')
 if not prereq:errors.append('V48.76 prerequisite branch mismatch')
 identity=alltrue(vals(au,'G77_minus_E76','teacher_labels_equal')+vals(au,'H77_minus_F76','teacher_labels_equal')+vals(au,'G77_minus_E76','positive_certificate_set_equal')+vals(au,'H77_minus_F76','positive_certificate_set_equal')+vals(au,'H77_minus_G77','positive_certificate_set_equal'))
 g_go,g_auc,g_hub=source_go(au,'G77_minus_E76');h_go,h_auc,h_hub=source_go(au,'H77_minus_F76');typed_go=g_go or h_go
 hg_auc=vals(au,'H77_minus_G77','nonfloor_auc_delta');hg_hub=vals(au,'H77_minus_G77','nonfloor_huber_delta');fidelity_go=pos(hg_auc)>=6 and dec(hg_hub)>=6 and nonrelapse(au,'H77_MAIN_ACTSI','G77_TYPED_PROJ')
 selectivity=nonrelapse(au,'G77_TYPED_PROJ','E76_MARGIN_PROJ') and nonrelapse(au,'H77_MAIN_ACTSI','F76_MARGIN_FIDELITY')
 g_full,g_full_diag=full_go(au,'G77_minus_C75','G77_TYPED_PROJ','C75_SIGN_PROJ');h_full,h_full_diag=full_go(au,'H77_minus_D75','H77_MAIN_ACTSI','D75_SIGN_FIDELITY');full_source_go=g_full or h_full
 if typed_go and full_source_go:
  status='ACTSI_SOURCE_GO'
  next_branch='freeze_absolute_source_then_truth_contract_reconciliation_and_frozen_rifa_safe_external_evaluation'
 elif typed_go:
  status='ACTSI_MECHANISM_GO_SOURCE_STOP'
  next_branch='active_typed_signed_source_identified_then_teacher_physical_structural_truth_contract_adjudication_before_any_boundary_transport'
 else:
  status='STOP'
  next_branch='active_typed_transport_stop_close_gain_transport_family_then_structured_ocmero_tail_source_interface_no_gain_sweep'
 doc={'schema':'ocrap-v48.77-actsi-comparison-v1','algorithm':'v48.77-DCP-DRFC-BCDE-RIFA-OC-ACTSI','engineering_version':'v48.77.0-OC-ACTSI','errors':errors,'preregistered_decision':{'v48_76_prerequisite_valid':prereq,'identity_gate':identity,'G77_typed_source_main_effect_go':g_go,'H77_typed_source_main_effect_go':h_go,'active_constraint_typed_source_go':typed_go,'projection_fidelity_under_typed_source_go':fidelity_go,'absolute_selectivity_gate':selectivity,'G77_full_source_go':g_full,'H77_full_source_go':h_full,'full_source_go':full_source_go,'status':status,'next_branch':next_branch},'deltas':{'G77_minus_E76_nonfloor_auc':g_auc,'G77_minus_E76_nonfloor_huber':g_hub,'H77_minus_F76_nonfloor_auc':h_auc,'H77_minus_F76_nonfloor_huber':h_hub,'H77_minus_G77_nonfloor_auc':hg_auc,'H77_minus_G77_nonfloor_huber':hg_hub,'G77_full_source_diagnostic':g_full_diag,'H77_full_source_diagnostic':h_full_diag},'scientific_contract':{'structured_source_go':'either G-E or H-F: non-floor AUC >0 in >=6/8 with >=4/8 >=+0.005 AND non-floor Huber lower in >=6/8 with >=4/8 <=-0.01','fidelity_repromotion':'H-G non-floor AUC positive >=6/8 and Huber lower >=6/8 with no harmful/TI relapse','full_source_go':'vs native C/D: full AUC positive 8/8, >=6/8 >=+0.01; non-floor >=6/8 positive and >=4/8 >=+0.005; every adequately powered non-floor safe-positive cell non-decreases and >=3 cells improve by >=0.05','adequate_safe_positive_definition':'base non-floor safe-positive rows >=5','absolute_selectivity':'new harmful/TI <=0.25 and <= matched global-source reference +0.02','active_constraint_rows':['clearance','stopping','control','stability','route','persistent_reentry'],'option_id_input':False,'regime_id_input':False,'boundary_transport':'OFF','relative_ranker':'frozen','geometry_sweep':'forbidden','gain_lr_threshold_sweep':'forbidden','dataset_reconstruction':False},'dataset_reconstruction':False,'test_roots_read':False}
 valid=not errors and prereq and identity and selectivity;doc['valid']=valid;doc['attribution_ready']=valid
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_77_actsi_comparison','valid':valid,'status':status,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
