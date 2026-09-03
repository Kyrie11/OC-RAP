#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
V=('balanced','precision');S=('dev_near','dev_contact','certificate_near','certificate_contact')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--v84-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();au=json.loads(a.audit.read_text());p84=json.loads(a.v84_comparison.read_text());d84=p84.get('preregistered_decision') or {};errors=[]
 prereq=bool(p84.get('valid') and d84.get('status')=='STAGE_I_ACTION_OBSERVABILITY_STOP')
 if not prereq:errors.append('V48.84 Stage-I observability STOP prerequisite missing')
 def cells(name):return [au['comparisons'][name][v][s] for v in V for s in S]
 def stats(name,slack):
  cs=cells(name)
  auc_pos=sum(x['source_auc_new'] is not None and x['source_auc_base'] is not None and x['source_auc_new']>x['source_auc_base'] for x in cs)
  auc_mat=sum(x['source_auc_new'] is not None and x['source_auc_base'] is not None and x['source_auc_new']-x['source_auc_base']>=.003 for x in cs)
  auc_strong=sum(x['source_auc_new'] is not None and x['source_auc_base'] is not None and x['source_auc_new']-x['source_auc_base']>=.01 for x in cs)
  hub=sum(x['interval_huber_new'] is not None and x['interval_huber_base'] is not None and x['interval_huber_new']<x['interval_huber_base'] for x in cs)
  hub_mat=sum(x['interval_huber_new'] is not None and x['interval_huber_base'] is not None and x['interval_huber_new']-x['interval_huber_base']<=-.003 for x in cs)
  abs_sel=all((x['harmful_pass_new'] or 0)<=.25 and (x['ti_pass_new'] or 0)<=.25 for x in cs)
  matched=all((x['harmful_pass_new'] or 0)<=(x['harmful_pass_base'] or 0)+slack and (x['ti_pass_new'] or 0)<=(x['ti_pass_base'] or 0)+slack for x in cs)
  false_nonincrease=sum((x['harmful_pass_new'] or 0)<=(x['harmful_pass_base'] or 0)+1e-12 and (x['ti_pass_new'] or 0)<=(x['ti_pass_base'] or 0)+1e-12 for x in cs)
  powered=[(v,s,au['comparisons'][name][v][s]) for v in V for s in S if int(au['comparisons'][name][v][s].get('safe_positive_rows',0))>=5]
  no_decline=all((x.get('safe_positive_pass_new') or 0)>=(x.get('safe_positive_pass_base') or 0)-1e-12 for _,_,x in powered)
  mats=[(v,s,(x.get('safe_positive_pass_new') or 0)-(x.get('safe_positive_pass_base') or 0)) for v,s,x in powered if (x.get('safe_positive_pass_new') or 0)-(x.get('safe_positive_pass_base') or 0)>=.05]
  return {'auc_positive':auc_pos,'auc_material_003':auc_mat,'auc_material_010':auc_strong,'huber_improved':hub,'huber_material':hub_mat,'absolute_selectivity':abs_sel,'matched_selectivity':matched,'false_nonincrease_cells':false_nonincrease,'powered_safe_positive_cells':len(powered),'safe_positive_no_decline':no_decline,'safe_positive_material_cells':len(mats),'safe_positive_near_material':any('near' in s for _,s,_ in mats),'safe_positive_contact_material':any('contact' in s for _,s,_ in mats)}
 q=stats('Q85_minus_L80',.02);inc=stats('R85_minus_Q85',.005);full=stats('R85_minus_L80',.02)
 q_go=q['auc_positive']>=6 and q['auc_material_003']>=4 and q['huber_improved']>=6 and q['huber_material']>=4 and q['absolute_selectivity'] and q['matched_selectivity'] and q['safe_positive_no_decline'] and q['safe_positive_material_cells']>=2 and q['safe_positive_near_material'] and q['safe_positive_contact_material']
 inc_go=inc['auc_positive']>=6 and inc['auc_material_003']>=4 and inc['huber_improved']>=6 and inc['absolute_selectivity'] and inc['matched_selectivity'] and inc['false_nonincrease_cells']>=6 and inc['safe_positive_no_decline'] and inc['safe_positive_contact_material']
 full_go=full['auc_positive']==8 and full['auc_material_010']>=6 and full['huber_improved']>=6 and full['absolute_selectivity'] and full['matched_selectivity'] and full['safe_positive_no_decline'] and full['safe_positive_material_cells']>=2 and full['safe_positive_near_material'] and full['safe_positive_contact_material']
 if full_go:status='STATE_ACTION_RECOVERY_REPRESENTATION_GO';nxt='freeze_absolute_source_then_safe_near_contact_closed_loop_and_external_baselines'
 elif q_go and inc_go:status='REPRESENTATION_GO_SOURCE_STOP';nxt='state_conditioned_raw_action_response_validated_then_adjudicate_remaining_absolute_boundary_debt_no_capacity_sweep'
 elif q_go:status='ACTION_RESPONSE_GO_STATE_GATE_STOP';nxt='promote_action_response_without_state_gate_then_retest_full_source_no_capacity_sweep'
 else:status='STATE_ACTION_RECOVERY_REPRESENTATION_STOP';nxt='narrow_action_response_representation_stop_then_adjudicate_action_response_supervision_or_raw_action_sufficiency_no_broad_encoder_retrain'
 doc={'schema':'ocrap-v48.85-sarr-comparison-v1','engineering_version':'v48.85.1-OC-SARR-ENGFIX','valid':prereq and not errors,'attribution_ready':prereq and not errors,'errors':errors,'preregistered_decision':{'v48_84_stop_prerequisite':prereq,'action_response_representation_go':q_go,'state_conditioning_increment_go':inc_go,'full_source_go':full_go,'Q85_minus_L80':q,'R85_minus_Q85':inc,'R85_minus_L80':full,'status':status,'next_branch':nxt},'scientific_contract':{'truth_contract':'V48.80 structural_interval_bounds frozen scaffold','boundary_transport':'OFF','dataset_reconstruction':False,'generic_mlp':False,'broad_encoder_root_retraining':False,'signed_reserve_debt_channels':True,'intervention':'narrow absolute-only raw candidate action response injected into frozen root-margin representation; R arm adds parameter-free nominal-root state gating at identical trainable capacity'}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':doc['valid'],'status':status}));return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
