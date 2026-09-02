#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
V=('balanced','precision');S=('dev_near','dev_contact','certificate_near','certificate_contact')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--v79-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();au=json.loads(a.audit.read_text());p79=json.loads(a.v79_comparison.read_text());errs=[]
 pre=p79.get('preregistered_decision') or {}; prereq=bool(p79.get('valid') and p79.get('attribution_ready') and pre.get('status')=='UNDERPOWERED_TRUTH_ADJUDICATION')
 if not prereq:errs.append('V48.79 underpowered prerequisite missing')
 cs=[au['comparisons']['L80_minus_J78'][v][s] for v in V for s in S]
 identity=all(x['teacher_labels_equal'] for x in cs)
 if not identity:errs.append('teacher labels changed')
 informative=sum(int(x['informative_rows'])>=100 for x in cs)
 loss_imp=sum(x['interval_huber_base'] is not None and x['interval_huber_new'] is not None and x['interval_huber_new']<x['interval_huber_base'] for x in cs)
 loss_mat=sum(x['interval_huber_base'] is not None and x['interval_huber_new'] is not None and x['interval_huber_new']-x['interval_huber_base']<=-.01 for x in cs)
 sat_imp=sum(x['interval_satisfaction_new'] is not None and x['interval_satisfaction_base'] is not None and x['interval_satisfaction_new']>x['interval_satisfaction_base'] for x in cs)
 selectivity=all((x['harmful_pass_new'] or 0)<=.25 and (x['harmful_pass_new'] or 0)<=(x['harmful_pass_base'] or 0)+.02 and (x['ti_pass_new'] or 0)<=.25 and (x['ti_pass_new'] or 0)<=(x['ti_pass_base'] or 0)+.02 for x in cs)
 go=informative>=6 and loss_imp>=6 and loss_mat>=4 and sat_imp>=6 and selectivity
 status='PARTIAL_IDENTIFICATION_TRUTH_GO' if go else 'PARTIAL_IDENTIFICATION_TRUTH_STOP'
 nxt='formalize_dual_physical_interval_and_structural_deployability_contract_no_boundary_transport' if go else 'truth_contract_partial_identification_stop_close_low_capacity_truth_probe_then_structured_source_only_after_semantic_adjudication'
 valid=prereq and identity and selectivity and not errs
 doc={'schema':'ocrap-v48.80-pistc-comparison-v1','engineering_version':'v48.80.0-OC-PISTC','valid':valid,'attribution_ready':valid,'errors':errs,'preregistered_decision':{'v48_79_underpowered_prerequisite':prereq,'identity_gate':identity,'informative_powered_cells':informative,'interval_huber_improved_cells':loss_imp,'interval_huber_material_cells':loss_mat,'interval_satisfaction_improved_cells':sat_imp,'absolute_selectivity_gate':selectivity,'partial_identification_truth_go':go,'status':status,'next_branch':nxt},'scientific_contract':{'source_capacity':'execution-identical J78 one-scalar nested tail source','truth_intervention':'candidate-level conservative physical interval propagated through monotone OC-MERO from one-sided structural floor/override bounds','hidden_branch':'unidentifiable -> no finite bound/no loss','teacher_metadata_input_to_model':False,'dataset_reconstruction':False,'boundary_transport':'OFF','new_representation':'none'},'dataset_reconstruction':False,'teacher_labels_changed':False,'test_roots_read':False}
 a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':valid,'status':status}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
