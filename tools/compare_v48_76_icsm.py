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
def gains(a,name):
 z=[]
 for v in VARIANTS:
  st=(a['arms'][name][v].get('state') or {});g=st.get('effective_gain') or [0,0];z.append(g)
 return z

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--v75-complete',type=Path,required=True);ap.add_argument('--v75-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();au=load(a.audit);c75=load(a.v75_complete);p75=load(a.v75_comparison);errors=[]
 pre=p75.get('preregistered_decision') or {};prereq=bool(c75.get('valid') and c75.get('attribution_ready') and c75.get('engineering_version')=='v48.75.0-OC-STCA' and pre.get('status')=='STOP' and pre.get('next_branch')=='truth_floor_debt_not_dominant_training_cause_audit_absolute_supervision_representation_no_geometry_sweep')
 if not prereq:errors.append('V48.75 prerequisite branch mismatch')
 ec=vals(au,'E76_minus_C75','nonfloor_auc_delta');fd=vals(au,'F76_minus_D75','nonfloor_auc_delta');fe=vals(au,'F76_minus_E76','nonfloor_auc_delta');ec_h=vals(au,'E76_minus_C75','nonfloor_huber_delta');fd_h=vals(au,'F76_minus_D75','nonfloor_huber_delta');fe_h=vals(au,'F76_minus_E76','nonfloor_huber_delta')
 identity=alltrue(vals(au,'E76_minus_C75','teacher_labels_equal')+vals(au,'F76_minus_D75','teacher_labels_equal')+vals(au,'E76_minus_C75','positive_certificate_set_equal')+vals(au,'F76_minus_D75','positive_certificate_set_equal'))
 e_go=pos(ec)>=6 and mat(ec)>=4 and dec(ec_h)>=6 and decmat(ec_h)>=4; f_go=pos(fd)>=6 and mat(fd)>=4 and dec(fd_h)>=6 and decmat(fd_h)>=4; objective_go=e_go or f_go
 fidelity_go=pos(fe)>=6 and dec(fe_h)>=6
 harmful=arm(au,'F76_MARGIN_FIDELITY','harmful_pass_fraction');ti=arm(au,'F76_MARGIN_FIDELITY','teacher_infeasible_pass_fraction');selectivity=all(x is not None and x<=.25 for x in harmful+ti)
 safe=arm(au,'F76_MARGIN_FIDELITY','safe_positive_nonfloor_rows');support=sum((x or 0)>=5 for x in safe)>=6
 gain_e=gains(au,'E76_MARGIN_PROJ');gain_f=gains(au,'F76_MARGIN_FIDELITY');gain_active=all(any(float(x)>1e-6 for x in g) for g in gain_e+gain_f)
 if objective_go and fidelity_go:status='SIGNED_MARGIN_GO_FIDELITY_RETAIN'
 elif objective_go:status='SIGNED_MARGIN_GO_FIDELITY_STOP'
 else:status='STOP'
 if objective_go:next_branch='signed_margin_supervision_identified_then_formalize_dual_physical_structural_contract_before_boundary_transport'
 else:next_branch='signed_margin_supervision_stop_two_gain_transport_representation_bottleneck_then_structured_absolute_source_interface'
 doc={'schema':'ocrap-v48.76-icsm-comparison-v1','algorithm':'v48.76-DCP-DRFC-BCDE-RIFA-OC-ICSM','engineering_version':'v48.76.0-OC-ICSM','errors':errors,'preregistered_decision':{'v48_75_prerequisite_valid':prereq,'identity_gate':identity,'E76_supervision_geometry_go':e_go,'F76_supervision_geometry_go':f_go,'signed_margin_supervision_go':objective_go,'projection_fidelity_under_signed_margin_go':fidelity_go,'absolute_selectivity_gate':selectivity,'nonfloor_safe_positive_support_adequate':support,'semantic_gain_activation_diagnostic':gain_active,'status':status,'next_branch':next_branch},'deltas':{'E76_minus_C75_nonfloor_auc':ec,'F76_minus_D75_nonfloor_auc':fd,'F76_minus_E76_nonfloor_auc':fe,'E76_minus_C75_nonfloor_huber':ec_h,'F76_minus_D75_nonfloor_huber':fd_h,'F76_minus_E76_nonfloor_huber':fe_h},'scientific_contract':{'truth_contract':'censor_exact_0p5','new_objective':'signed_margin_huber','huber_beta':1.0,'supervision_go':'nonfloor AUC >0 in >=6/8 with >=4/8 >=+0.005 AND nonfloor Huber decreases in >=6/8 with >=4/8 <=-0.01','fidelity_go':'F-E nonfloor AUC positive >=6/8 and Huber decreases >=6/8','boundary_transport':'OFF','relative_ranker':'frozen','regime_router':'forbidden','geometry_sweep':'forbidden','dataset_reconstruction':False},'dataset_reconstruction':False,'test_roots_read':False}
 valid=not errors and prereq and identity and selectivity;doc['valid']=valid;doc['attribution_ready']=valid
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_76_icsm_comparison','valid':valid,'status':status,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
