#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
V=('balanced','precision');S=('dev_near','dev_contact','certificate_near','certificate_contact')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--v80-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();au=json.loads(a.audit.read_text());p80=json.loads(a.v80_comparison.read_text());errs=[]
 pre=p80.get('preregistered_decision') or {};prereq=bool(p80.get('valid') and p80.get('attribution_ready') and pre.get('status')=='PARTIAL_IDENTIFICATION_TRUTH_STOP')
 if not prereq:errs.append('V48.80 STOP prerequisite missing')
 cs=[au['comparisons']['M81_minus_L80'][v][s] for v in V for s in S];identity=all(x['teacher_labels_equal'] for x in cs)
 if not identity:errs.append('teacher labels changed')
 informative=sum(int(x['informative_rows'])>=100 for x in cs);loss_imp=sum(x['interval_huber_new']<x['interval_huber_base'] for x in cs);loss_mat=sum(x['interval_huber_new']-x['interval_huber_base']<=-.01 for x in cs);sat_imp=sum(x['interval_satisfaction_new']>x['interval_satisfaction_base'] for x in cs)
 selectivity=all((x['harmful_pass_new'] or 0)<=.25 and (x['harmful_pass_new'] or 0)<=(x['harmful_pass_base'] or 0)+.02 and (x['ti_pass_new'] or 0)<=.25 and (x['ti_pass_new'] or 0)<=(x['ti_pass_base'] or 0)+.02 for x in cs)
 old=au['v48_80_truth_index_summary']['roles'];new=au['truth_index_summary']['roles'];exact_inc=sum(new[s]['exact_physical_rows']>old[s]['exact_physical_rows'] for s in S if s in new and s in old)
 go=informative>=6 and exact_inc>=3 and loss_imp>=6 and loss_mat>=4 and sat_imp>=6 and selectivity
 status='SWITCH_INVERSE_TRUTH_GO' if go else 'SWITCH_INVERSE_TRUTH_STOP'
 nxt='formalize_partially_identified_physical_reserve_debt_plus_structural_admissibility_no_boundary_transport' if go else 'close_truth_contract_refinement_family_then_structured_nested_source_representation_no_boundary_transport'
 valid=prereq and identity and selectivity and not errs
 doc={'schema':'ocrap-v48.81-sitc-comparison-v1','engineering_version':'v48.81.1-OC-SITC-ENGFIX','valid':valid,'attribution_ready':valid,'errors':errs,'preregistered_decision':{'v48_80_stop_prerequisite':prereq,'identity_gate':identity,'informative_powered_cells':informative,'exact_physical_increased_splits':exact_inc,'interval_huber_improved_cells':loss_imp,'interval_huber_material_cells':loss_mat,'interval_satisfaction_improved_cells':sat_imp,'absolute_selectivity_gate':selectivity,'switch_inverse_truth_go':go,'status':status,'next_branch':nxt},'scientific_contract':{'source_capacity':'execution-identical J78/L80 one-scalar nested tail source','truth_intervention':'exact inverse image of the ordered monotone structural teacher operators before monotone OC-MERO propagation','hidden_branch':'unidentifiable -> no finite cell bound','teacher_metadata_input_to_model':False,'dataset_reconstruction':False,'boundary_transport':'OFF','new_representation':'none'},'dataset_reconstruction':False,'teacher_labels_changed':False,'test_roots_read':False}
 a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':valid,'status':status}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
