#!/usr/bin/env python3
import argparse,json
from pathlib import Path
V=('balanced','precision');S=('dev_near','dev_contact','certificate_near','certificate_contact')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--v81-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();au=json.loads(a.audit.read_text());p81=json.loads(a.v81_comparison.read_text());pre=p81.get('preregistered_decision') or {};errs=[]
 prereq=bool(p81.get('valid') and p81.get('attribution_ready') and pre.get('status')=='SWITCH_INVERSE_TRUTH_STOP');
 if not prereq:errs.append('V48.81 STOP prerequisite missing')
 def cells(name):return [au['comparisons'][name][v][s] for v in V for s in S]
 def stats(name):
  cs=cells(name);auc_pos=sum(x['source_auc_new'] is not None and x['source_auc_base'] is not None and x['source_auc_new']>x['source_auc_base'] for x in cs);auc_mat=sum(x['source_auc_new'] is not None and x['source_auc_base'] is not None and x['source_auc_new']-x['source_auc_base']>=.005 for x in cs);hub=sum(x['interval_huber_new']<x['interval_huber_base'] for x in cs);hubm=sum(x['interval_huber_new']-x['interval_huber_base']<=-.01 for x in cs);sel=all((x['harmful_pass_new'] or 0)<=.25 and (x['ti_pass_new'] or 0)<=.25 and (x['harmful_pass_new'] or 0)<=(x['harmful_pass_base'] or 0)+.02 and (x['ti_pass_new'] or 0)<=(x['ti_pass_base'] or 0)+.02 for x in cs);return {'auc_positive':auc_pos,'auc_material':auc_mat,'huber_improved':hub,'huber_material':hubm,'selectivity':sel}
 n=stats('N82_minus_L80');o=stats('O82_minus_N82');full=stats('O82_minus_L80')
 n_go=n['auc_positive']>=6 and n['auc_material']>=4 and n['huber_improved']>=6 and n['selectivity'];o_go=o['auc_positive']>=6 and o['huber_improved']>=6 and o['selectivity'];full_go=full['auc_positive']==8 and full['auc_material']>=6 and full['huber_improved']>=6 and full['selectivity']
 status='SIGNED_NESTED_TAIL_FIELD_GO' if full_go else 'SIGNED_NESTED_TAIL_FIELD_STOP';nxt='freeze_structured_source_then_truth_semantics_and_closed_loop' if full_go else 'structured_tail_field_stop_adjudicate_root_observability_or_supervision_no_capacity_sweep'
 doc={'schema':'ocrap-v48.82-sntf-comparison-v1','engineering_version':'v48.82.1-OC-SNTF-ENGFIX','valid':prereq and not errs,'attribution_ready':prereq and not errs,'errors':errs,'preregistered_decision':{'v48_81_stop_prerequisite':prereq,'single_field_go':n_go,'signed_channel_increment_go':o_go,'full_source_go':full_go,'N82_minus_L80':n,'O82_minus_N82':o,'O82_minus_L80':full,'status':status,'next_branch':nxt},'scientific_contract':{'truth_contract':'V48.80 structural_interval_bounds frozen scaffold','boundary_transport':'OFF','dataset_reconstruction':False,'generic_mlp':False,'regime_id_input':False}}
 a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':doc['valid'],'status':status}))
 return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
