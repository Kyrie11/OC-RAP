#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

V=('balanced','precision')
S=('dev_near','dev_contact','certificate_near','certificate_contact')


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--v82-comparison',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); au=json.loads(a.audit.read_text()); p82=json.loads(a.v82_comparison.read_text()); d82=p82.get('preregistered_decision') or {}; errors=[]
    prereq=bool(p82.get('valid') and p82.get('attribution_ready') and d82.get('status')=='SIGNED_NESTED_TAIL_FIELD_STOP' and d82.get('signed_channel_increment_go'))
    if not prereq: errors.append('V48.82 signed-channel-increment / STOP prerequisite missing')

    def cells(name): return [au['comparisons'][name][v][s] for v in V for s in S]
    def stats(name, *, base_slack:float):
        cs=cells(name)
        auc_pos=sum(x['source_auc_new'] is not None and x['source_auc_base'] is not None and x['source_auc_new']>x['source_auc_base'] for x in cs)
        auc_mat=sum(x['source_auc_new'] is not None and x['source_auc_base'] is not None and x['source_auc_new']-x['source_auc_base']>=.003 for x in cs)
        hub=sum(x['interval_huber_new'] is not None and x['interval_huber_base'] is not None and x['interval_huber_new']<x['interval_huber_base'] for x in cs)
        hub_mat=sum(x['interval_huber_new'] is not None and x['interval_huber_base'] is not None and x['interval_huber_new']-x['interval_huber_base']<=-.003 for x in cs)
        absolute_cap=all((x['harmful_pass_new'] or 0)<=.25 and (x['ti_pass_new'] or 0)<=.25 for x in cs)
        matched_cap=all((x['harmful_pass_new'] or 0)<=(x['harmful_pass_base'] or 0)+base_slack and (x['ti_pass_new'] or 0)<=(x['ti_pass_base'] or 0)+base_slack for x in cs)
        false_nonincrease=sum((x['harmful_pass_new'] or 0)<=(x['harmful_pass_base'] or 0)+1e-12 and (x['ti_pass_new'] or 0)<=(x['ti_pass_base'] or 0)+1e-12 for x in cs)
        powered=[(v,s,au['comparisons'][name][v][s]) for v in V for s in S if int(au['comparisons'][name][v][s].get('safe_positive_rows',0))>=5]
        safe_no_decline=all((x.get('safe_positive_pass_new') or 0)>=(x.get('safe_positive_pass_base') or 0)-1e-12 for _,_,x in powered)
        safe_material=[(v,s,(x.get('safe_positive_pass_new') or 0)-(x.get('safe_positive_pass_base') or 0)) for v,s,x in powered if (x.get('safe_positive_pass_new') or 0)-(x.get('safe_positive_pass_base') or 0)>=.05]
        safe_near=any('near' in s and d>=.05 for v,s,d in safe_material)
        safe_contact=any('contact' in s and d>=.05 for v,s,d in safe_material)
        return {'auc_positive':auc_pos,'auc_material':auc_mat,'huber_improved':hub,'huber_material':hub_mat,'absolute_selectivity':absolute_cap,'matched_selectivity':matched_cap,'false_nonincrease_cells':false_nonincrease,'powered_safe_positive_cells':len(powered),'safe_positive_no_decline':safe_no_decline,'safe_positive_material_cells':len(safe_material),'safe_positive_near_material':safe_near,'safe_positive_contact_material':safe_contact}

    mech=stats('P83_minus_O82',base_slack=.005)
    full=stats('P83_minus_L80',base_slack=.02)
    mechanism_go=(mech['auc_positive']>=6 and mech['auc_material']>=4 and mech['huber_improved']>=6 and mech['huber_material']>=4 and mech['absolute_selectivity'] and mech['matched_selectivity'] and mech['false_nonincrease_cells']>=6 and mech['safe_positive_no_decline'] and mech['safe_positive_material_cells']>=2 and mech['safe_positive_near_material'] and mech['safe_positive_contact_material'])
    full_go=(full['auc_positive']==8 and full['auc_material']>=6 and full['huber_improved']>=6 and full['absolute_selectivity'] and full['matched_selectivity'] and full['safe_positive_no_decline'] and full['safe_positive_material_cells']>=2 and full['safe_positive_near_material'] and full['safe_positive_contact_material'])
    if full_go:
        status='COUNTERFACTUAL_RECOVERY_TAIL_FIELD_GO'; nxt='freeze_absolute_source_then_safe_near_contact_closed_loop_and_external_baselines'
    elif mechanism_go:
        status='MECHANISM_GO_SOURCE_STOP'; nxt='counterfactual_action_response_validated_then_adjudicate_absolute_boundary_debt_without_reopening_source_capacity'
    else:
        status='COUNTERFACTUAL_RECOVERY_TAIL_FIELD_STOP'; nxt='close_frozen_structured_tail_adapter_family_then_test_stage_i_root_action_observability_no_field_capacity_sweep'
    doc={'schema':'ocrap-v48.83-crtf-comparison-v1','engineering_version':'v48.83.0-OC-CRTF','valid':prereq and not errors,'attribution_ready':prereq and not errors,'errors':errors,
         'preregistered_decision':{'v48_82_stop_and_signed_increment_prerequisite':prereq,'counterfactual_action_response_go':mechanism_go,'full_source_go':full_go,'P83_minus_O82':mech,'P83_minus_L80':full,'status':status,'next_branch':nxt},
         'scientific_contract':{'truth_contract':'V48.80 structural_interval_bounds frozen scaffold','boundary_transport':'OFF','dataset_reconstruction':False,'generic_mlp':False,'regime_id_input':False,'relative_ranker_modified':False,'new_source_capacity_vs_O82':'none (same 2x192 signed field)','intervention':'candidate-minus-nominal frozen root-option interaction before signed nested-tail field'}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n'); print(json.dumps({'valid':doc['valid'],'status':status})); return 0 if doc['valid'] else 30

if __name__=='__main__': raise SystemExit(main())
