#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
V=('balanced','precision');S=('dev_near','dev_contact','certificate_near','certificate_contact')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--v85-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();au=json.loads(a.audit.read_text());p85=json.loads(a.v85_comparison.read_text());d85=p85.get('preregistered_decision') or {};errors=[]
 prereq=bool(p85.get('valid') and d85.get('status')=='STATE_ACTION_RECOVERY_REPRESENTATION_STOP' and not d85.get('action_response_representation_go'))
 if not prereq:errors.append('V48.85 SARR STOP prerequisite missing')
 def cells(name):return [au['comparisons'][name][v][s] for v in V for s in S]
 def stats(name,base_slack):
  cs=cells(name); auc_pos=sum(x['source_auc_new']>x['source_auc_base'] for x in cs); auc_mat=sum(x['source_auc_new']-x['source_auc_base']>=.003 for x in cs); auc_strong=sum(x['source_auc_new']-x['source_auc_base']>=.01 for x in cs)
  abs_sel=all((x['harmful_pass_new'] or 0)<=.25 and (x['ti_pass_new'] or 0)<=.25 for x in cs)
  matched=all((x['harmful_pass_new'] or 0)<=(x['harmful_pass_base'] or 0)+base_slack and (x['ti_pass_new'] or 0)<=(x['ti_pass_base'] or 0)+base_slack for x in cs)
  noninc=sum((x['harmful_pass_new'] or 0)<=(x['harmful_pass_base'] or 0)+1e-12 and (x['ti_pass_new'] or 0)<=(x['ti_pass_base'] or 0)+1e-12 for x in cs)
  false_drop=sum(((x['harmful_pass_base'] or 0)-(x['harmful_pass_new'] or 0)>=.03) and ((x['ti_pass_base'] or 0)-(x['ti_pass_new'] or 0)>=.03) for x in cs)
  powered=[(v,s,au['comparisons'][name][v][s]) for v in V for s in S if int(au['comparisons'][name][v][s].get('safe_positive_rows',0))>=5]
  no_decline=all((x.get('safe_positive_pass_new') or 0)>=(x.get('safe_positive_pass_base') or 0)-1e-12 for _,_,x in powered)
  mats=[(v,s,(x.get('safe_positive_pass_new') or 0)-(x.get('safe_positive_pass_base') or 0)) for v,s,x in powered if (x.get('safe_positive_pass_new') or 0)-(x.get('safe_positive_pass_base') or 0)>=.05]
  hub=sum(x['interval_huber_new']<=x['interval_huber_base']+1e-12 for x in cs)
  return {'auc_positive':auc_pos,'auc_material_003':auc_mat,'auc_material_010':auc_strong,'absolute_selectivity':abs_sel,'matched_selectivity':matched,'false_nonincrease_cells':noninc,'false_drop_003_cells':false_drop,'interval_huber_nondegrade_cells':hub,'safe_positive_no_decline':no_decline,'safe_positive_material_cells':len(mats),'safe_positive_near_material':any('near' in s for _,s,_ in mats),'safe_positive_contact_material':any('contact' in s for _,s,_ in mats)}
 resp=stats('S86_minus_Q85',.005); sel=stats('T86_minus_S86',.005); full=stats('T86_minus_L80',.02)
 resp_go=resp['auc_positive']>=6 and resp['auc_material_003']>=4 and resp['absolute_selectivity'] and resp['false_nonincrease_cells']>=6 and resp['safe_positive_no_decline'] and resp['safe_positive_near_material'] and resp['safe_positive_contact_material']
 sel_go=sel['auc_positive']>=6 and sel['auc_material_003']>=4 and sel['absolute_selectivity'] and sel['false_nonincrease_cells']>=6 and sel['false_drop_003_cells']>=4 and sel['safe_positive_no_decline'] and sel['safe_positive_near_material'] and sel['safe_positive_contact_material']
 full_go=full['auc_positive']==8 and full['auc_material_010']>=6 and full['absolute_selectivity'] and full['matched_selectivity'] and full['interval_huber_nondegrade_cells']>=6 and full['safe_positive_no_decline'] and full['safe_positive_near_material'] and full['safe_positive_contact_material']
 if full_go:status='COUNTERFACTUAL_RECOVERY_SUPERVISION_GO';nxt='freeze_absolute_source_then_safe_near_contact_closed_loop_and_external_baselines'
 elif resp_go and sel_go:status='SUPERVISION_MECHANISM_GO_SOURCE_STOP';nxt='counterfactual_response_and_structural_admissibility_validated_then_adjudicate_remaining_absolute_boundary_debt_no_capacity_increase'
 elif resp_go:status='RESPONSE_SUPERVISION_GO_SELECTIVE_STOP';nxt='retain_counterfactual_response_supervision_close_structural_response_constraint_then_retest_full_source_no_capacity_sweep'
 else:status='COUNTERFACTUAL_RECOVERY_SUPERVISION_STOP';nxt='response_aligned_supervision_stop_close_raw_action_response_branch_then_open_narrow_observation_action_interaction_representation_no_broad_encoder_retrain'
 doc={'schema':'ocrap-v48.86-crsc-comparison-v1','engineering_version':'v48.86.0-OC-CRSC','valid':prereq and not errors,'attribution_ready':prereq and not errors,'errors':errors,'preregistered_decision':{'v48_85_stop_prerequisite':prereq,'counterfactual_response_supervision_go':resp_go,'structural_selective_increment_go':sel_go,'full_source_go':full_go,'S86_minus_Q85':resp,'T86_minus_S86':sel,'T86_minus_L80':full,'status':status,'next_branch':nxt},'scientific_contract':{'representation':'V48.85 Q85 raw action-response adapter frozen in form/capacity','state_conditioning':False,'physical_response_supervision':'candidate-minus-nominal partially identified R_dep interval','structural_admissibility_supervision':'safe-benefit prefers positive signed response; harmful prefers nonpositive signed response via pairwise logistic ordering','truth_contract':'V48.80 structural intervals + exact PCD training-only sidecar','boundary_transport':'OFF','dataset_reconstruction':False,'generic_mlp':False,'relative_ranker_modified':False}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':doc['valid'],'status':status}));return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
