#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
V=('balanced','precision');S=('dev_near','dev_contact','certificate_near','certificate_contact')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--v86-comparison',type=Path,required=True);ap.add_argument('--u-run',type=Path,required=True);ap.add_argument('--v-run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();au=json.loads(a.audit.read_text());p86=json.loads(a.v86_comparison.read_text());d86=p86.get('preregistered_decision') or {};errors=[]
 prereq=bool(p86.get('valid') and d86.get('status')=='COUNTERFACTUAL_RECOVERY_SUPERVISION_STOP' and not d86.get('counterfactual_response_supervision_go') and 'observation_action_interaction' in str(d86.get('next_branch','')))
 if not prereq:errors.append('V48.86 CRSC STOP / narrow interaction prerequisite missing')
 def cells(name):return [au['comparisons'][name][v][s] for v in V for s in S]
 def stats(name,base_slack):
  cs=cells(name); auc_pos=sum(x['source_auc_new']>x['source_auc_base'] for x in cs); auc_mat=sum(x['source_auc_new']-x['source_auc_base']>=.003 for x in cs); auc_strong=sum(x['source_auc_new']-x['source_auc_base']>=.01 for x in cs)
  abs_sel=all((x['harmful_pass_new'] or 0)<=.25 and (x['ti_pass_new'] or 0)<=.25 for x in cs)
  matched=all((x['harmful_pass_new'] or 0)<=(x['harmful_pass_base'] or 0)+base_slack and (x['ti_pass_new'] or 0)<=(x['ti_pass_base'] or 0)+base_slack for x in cs)
  powered=[(v,s,au['comparisons'][name][v][s]) for v in V for s in S if int(au['comparisons'][name][v][s].get('safe_positive_rows',0))>=5]
  mats=[(v,s,(x.get('safe_positive_pass_new') or 0)-(x.get('safe_positive_pass_base') or 0)) for v,s,x in powered if (x.get('safe_positive_pass_new') or 0)-(x.get('safe_positive_pass_base') or 0)>=.05]
  hub=sum(x['interval_huber_new']<=x['interval_huber_base']+1e-12 for x in cs)
  return {'auc_positive':auc_pos,'auc_material_003':auc_mat,'auc_material_010':auc_strong,'absolute_selectivity':abs_sel,'matched_selectivity':matched,'interval_huber_nondegrade_cells':hub,'safe_positive_material_cells':len(mats),'safe_positive_near_material':any('near' in s for _,s,_ in mats),'safe_positive_contact_material':any('contact' in s for _,s,_ in mats),'newly_admitted_safe_positive':sum(int(x.get('newly_admitted_safe_positive',0)) for x in cs),'newly_admitted_harmful':sum(int(x.get('newly_admitted_harmful',0)) for x in cs)}
 def train_meta(run):
  out={}
  for v in V:
   p=run/'candidates'/v/'TRAINING_COMPLETE.json'
   if not p.is_file(): out[v]={'exists':False,'best_epoch':None}; continue
   d=json.loads(p.read_text()); out[v]={'exists':True,'best_epoch':int(d.get('best_epoch',-1)),'epochs_completed':int(d.get('epochs_completed',-1)),'best_metric':d.get('best_metric')}
  return out
 u=stats('U87_minus_S86',.02); vt=stats('V87_minus_T86',.02); vu=stats('V87_minus_U87',.02); full=stats('V87_minus_L80',.02)
 um=train_meta(a.u_run);vm=train_meta(a.v_run);u_learns=all(um[v]['exists'] and um[v]['best_epoch']>0 for v in V);v_learns=all(vm[v]['exists'] and vm[v]['best_epoch']>0 for v in V)
 # V-T is the primary representation test because T86 already contains the structural sign objective;
 # this prevents the ~95% zero-compatible physical response intervals from making U-S the sole gate.
 interaction_go=v_learns and vt['auc_positive']>=6 and vt['auc_material_003']>=4 and vt['absolute_selectivity'] and vt['safe_positive_near_material'] and vt['safe_positive_contact_material'] and vt['newly_admitted_harmful']<=8
 interval_interaction_support=u_learns and u['auc_positive']>=5 and u['interval_huber_nondegrade_cells']>=4
 selective_increment_go=vu['auc_positive']>=6 and vu['auc_material_003']>=4 and vu['absolute_selectivity'] and vu['safe_positive_near_material'] and vu['safe_positive_contact_material']
 full_go=full['auc_positive']==8 and full['auc_material_010']>=6 and full['absolute_selectivity'] and full['matched_selectivity'] and full['interval_huber_nondegrade_cells']>=6 and full['safe_positive_near_material'] and full['safe_positive_contact_material']
 if full_go and interaction_go: status='BILINEAR_ACTION_ROOT_RESPONSE_FULL_GO'; nxt='freeze_absolute_source_then_frozen_RIFA_safe_noninterference_near_closed_loop_contact_postcollision_external_SOTA'
 elif interaction_go: status='BILINEAR_ACTION_ROOT_INTERACTION_GO_SOURCE_STOP'; nxt='freeze_interaction_representation_then_adjudicate_remaining_target_or_absolute_boundary_debt_no_capacity_increase'
 elif interval_interaction_support and not selective_increment_go: status='INTERACTION_INTERVAL_SUPPORT_SELECTIVE_STOP'; nxt='retain_bilinear_interaction_with_response_interval_close_current_pairwise_selective_constraint_then_retest_without_capacity_sweep'
 else: status='BILINEAR_ACTION_ROOT_INTERACTION_STOP'; nxt='close_low_rank_root_action_bilinear_branch_then_adjudicate_root_local_response_target_identifiability_no_broad_encoder_or_regime_router'
 doc={'schema':'ocrap-v48.87-barr-comparison-v1','engineering_version':'v48.87.0-OC-BARR','valid':prereq and not errors,'attribution_ready':prereq and not errors,'errors':errors,'preregistered_decision':{'v48_86_stop_prerequisite':prereq,'U87_minus_S86':u,'V87_minus_T86':vt,'V87_minus_U87':vu,'V87_minus_L80':full,'training_meta_U87':um,'training_meta_V87':vm,'U87_heldout_learning':u_learns,'V87_heldout_learning':v_learns,'interval_interaction_support':interval_interaction_support,'bilinear_action_root_interaction_go':interaction_go,'structural_selective_increment_go':selective_increment_go,'full_source_go':full_go,'status':status,'next_branch':nxt},'scientific_contract':{'representation':'shared low-rank bilinear frozen-root x executable-action response, rank=51','new_trainable_parameters':53550,'q85_trainable_parameters':54144,'capacity_not_increased_vs_Q85':True,'regime_conditioning':False,'state_gate':False,'generic_mlp':False,'broad_encoder_retraining':False,'physical_response_supervision':'V48.86 candidate-minus-nominal partially identified response interval','structural_admissibility_supervision':'V48.86 pairwise safe-positive / harmful ordering','boundary_transport':'OFF','dataset_reconstruction':False,'relative_ranker_modified':False}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':doc['valid'],'status':status}));return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
