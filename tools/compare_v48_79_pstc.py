#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

VARIANTS=('balanced','precision'); SPLITS=('dev_near','dev_contact','certificate_near','certificate_contact')

def load(p): return json.loads(Path(p).read_text())
def cells(a, field): return [a['comparisons']['K79_minus_J78'][v][s]['physical_identifiable'].get(field) for v in VARIANTS for s in SPLITS]
def counts(a, field): return [a['comparisons']['K79_minus_J78'][v][s]['physical_identifiable'].get(field) for v in VARIANTS for s in SPLITS]
def arm_full(a, arm, field): return [a['arms'][arm][v]['splits'][s]['full'].get(field) for v in VARIANTS for s in SPLITS]
def identity(a, field): return [a['comparisons']['K79_minus_J78'][v][s].get(field) for v in VARIANTS for s in SPLITS]


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--audit',type=Path,required=True); ap.add_argument('--v78-comparison',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    au=load(a.audit); p78=load(a.v78_comparison); errors=[]; pre=p78.get('preregistered_decision') or {}
    prereq=bool(p78.get('valid') and p78.get('attribution_ready') and pre.get('status')=='STOP' and pre.get('I78_root_shape_go') is False and pre.get('J78_root_tail_source_go') is False and pre.get('next_branch')=='root_tail_source_stop_close_low_capacity_absolute_source_adapter_family_then_teacher_truth_contract_adjudication_before_any_new_source_capacity')
    if not prereq: errors.append('V48.78 prerequisite branch mismatch')
    id_gate=all(bool(x) for x in identity(au,'teacher_labels_equal')+identity(au,'positive_certificate_set_equal'))
    if not id_gate: errors.append('K79/J78 label or certificate identity failed')
    pos=counts(au,'teacher_feasible_rows'); neg=counts(au,'teacher_infeasible_rows')
    powered=[(p is not None and n is not None and int(p)>=20 and int(n)>=20) for p,n in zip(pos,neg)]
    auc_delta=cells(au,'auc_delta'); hub_delta=cells(au,'huber_delta')
    powered_n=sum(powered)
    auc_pos=sum(powered[i] and auc_delta[i] is not None and auc_delta[i]>0 for i in range(8))
    auc_mat=sum(powered[i] and auc_delta[i] is not None and auc_delta[i]>=.005 for i in range(8))
    hub_lower=sum(powered[i] and hub_delta[i] is not None and hub_delta[i]<0 for i in range(8))
    hub_mat=sum(powered[i] and hub_delta[i] is not None and hub_delta[i]<=-.01 for i in range(8))
    truth_go=powered_n>=6 and auc_pos>=6 and auc_mat>=4 and hub_lower>=6 and hub_mat>=4
    harmful_k=arm_full(au,'K79_PHYSICAL_TAIL_PROBE','harmful_pass_fraction'); harmful_j=arm_full(au,'J78_RTSI','harmful_pass_fraction')
    ti_k=arm_full(au,'K79_PHYSICAL_TAIL_PROBE','teacher_infeasible_pass_fraction'); ti_j=arm_full(au,'J78_RTSI','teacher_infeasible_pass_fraction')
    selectivity=all(x is not None and y is not None and x<=.25 and x<=y+.02 for x,y in zip(harmful_k,harmful_j)) and all(x is not None and y is not None and x<=.25 and x<=y+.02 for x,y in zip(ti_k,ti_j))
    if not selectivity: errors.append('absolute selectivity relapse')
    if powered_n < 6:
        status='UNDERPOWERED_TRUTH_ADJUDICATION'; next_branch='report_physical_identifiability_power_limit_no_algorithm_promotion_and_no_dataset_reconstruction'
    elif truth_go:
        status='PHYSICAL_STRUCTURAL_TRUTH_CONFOUND_GO'; next_branch='formalize_two_object_physical_recovery_margin_and_structural_deployability_contract_before_any_new_source_capacity_or_boundary_transport'
    else:
        status='TRUTH_ADJUDICATION_STOP'; next_branch='structural_mixing_not_dominant_for_root_tail_source_close_low_capacity_probe_then_consider_higher_capacity_structured_ocmero_source_no_generic_mlp'
    valid=not errors and prereq and id_gate and selectivity
    doc={'schema':'ocrap-v48.79-pstc-comparison-v1','algorithm':'v48.79-DCP-DRFC-BCDE-RIFA-OC-PSTC','engineering_version':'v48.79.0-OC-PSTC','valid':valid,'attribution_ready':valid,'errors':errors,
         'preregistered_decision':{'v48_78_prerequisite_valid':prereq,'identity_gate':id_gate,'powered_cells':powered_n,'powered_mask':powered,
             'physical_auc_positive_cells':auc_pos,'physical_auc_material_cells':auc_mat,'physical_huber_lower_cells':hub_lower,'physical_huber_material_cells':hub_mat,
             'physical_structural_truth_confound_go':truth_go,'absolute_selectivity_gate':selectivity,'status':status,'next_branch':next_branch},
         'deltas':{'K79_minus_J78_physical_auc':auc_delta,'K79_minus_J78_physical_huber':hub_delta,'physical_teacher_feasible_rows':pos,'physical_teacher_infeasible_rows':neg},
         'scientific_contract':{
             'entry_condition':'V48.78 I/J STOP closes the low-capacity absolute-source adapter family and preregisters teacher truth-contract adjudication before any new source capacity.',
             'physical_identifiability':'candidate is retained only if the exact nested teacher OC-MERO active tail has zero conservative exposure to structural floor/override/hidden-branch semantics; no label is rewritten.',
             'power':'a cell is powered for physical AUC iff it contains >=20 physical-identifiable teacher-feasible and >=20 teacher-infeasible proposal rows; primary adjudication requires >=6/8 powered cells.',
             'truth_confound_go':'on powered cells: K-J physical AUC >0 in >=6/8 and >=4/8 >=+0.005; signed-margin Huber lower in >=6/8 and >=4/8 <=-0.01.',
             'absolute_selectivity':'K harmful/TI <=0.25 and <=J78+0.02 in all 8 full-population cells.',
             'v48_75_relationship':'uses the V48.75 material physical-ordering standard, but replaces crude exact-0.5 censoring with nested-tail structural exposure and retains V48.76 signed-margin error as a second required axis.',
             'source_capacity':'execution-identical J78 nested zero-translation one-scalar source; no new source capacity.',
             'boundary_transport':'OFF','relative_ranker':'frozen','teacher_rewrite':'forbidden','dataset_reconstruction':'forbidden','generic_mlp':'forbidden','gain_lr_threshold_sweep':'forbidden'},
         'dataset_reconstruction':False,'teacher_labels_changed':False,'test_roots_read':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_79_pstc_comparison','valid':valid,'status':status,'output':str(a.output)})); return 0 if valid else 30

if __name__=='__main__': raise SystemExit(main())
