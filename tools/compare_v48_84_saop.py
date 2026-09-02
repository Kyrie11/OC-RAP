#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def d(a,b): return None if a is None or b is None else float(a)-float(b)
def cell_metrics(doc,probe,split):
 t=doc['probes'][probe]['true'][split]; s=doc['probes'][probe]['shuffled'][split]
 return {'safe_auc':t['safe_auc'],'safe_auc_vs_shuffled':d(t['safe_auc'],s['safe_auc']),'harm_auc':t['harm_auc'],'harm_auc_vs_shuffled':d(t['harm_auc'],s['harm_auc']),'top1_safe_recall':t['top1_safe_recall'],'top1_vs_shuffled':d(t['top1_safe_recall'],s['top1_safe_recall']),'adv_pearson':t['adv_pearson'],'adv_pearson_vs_shuffled':d(t['adv_pearson'],s['adv_pearson']),'safe_positive_rows':t['safe_positive_rows'],'powered_groups':t['powered_groups']}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--balanced',type=Path,required=True); ap.add_argument('--precision',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); docs={'balanced':json.loads(a.balanced.read_text()),'precision':json.loads(a.precision.read_text())}; splits=['dev_near','dev_contact','certificate_near','certificate_contact']; out={'schema':'ocrap-v48.84-saop-comparison-v1','engineering_version':'v48.84.0-OC-SAOP','valid':all(x.get('valid') for x in docs.values()),'cells':{}}
 cells=[]
 for v,doc in docs.items():
  out['cells'][v]={}
  for sp in splits:
   delta=cell_metrics(doc,'delta',sp); ctx=cell_metrics(doc,'context',sp)
   ctx['safe_auc_vs_delta']=d(ctx['safe_auc'],delta['safe_auc']); ctx['top1_vs_delta']=d(ctx['top1_safe_recall'],delta['top1_safe_recall']); ctx['harm_auc_vs_delta']=d(ctx['harm_auc'],delta['harm_auc'])
   out['cells'][v][sp]={'delta':delta,'context':ctx}; cells.append((v,sp,delta,ctx))
 def cnt(fn): return sum(bool(fn(*c)) for c in cells)
 delta_safe_pos=cnt(lambda v,s,x,c:x['safe_auc_vs_shuffled'] is not None and x['safe_auc_vs_shuffled']>0)
 delta_safe_mat=cnt(lambda v,s,x,c:x['safe_auc_vs_shuffled'] is not None and x['safe_auc_vs_shuffled']>=.03)
 delta_harm_mat=cnt(lambda v,s,x,c:x['harm_auc_vs_shuffled'] is not None and x['harm_auc_vs_shuffled']>=.03)
 delta_top_mat=cnt(lambda v,s,x,c:x['top1_vs_shuffled'] is not None and x['top1_vs_shuffled']>=.05)
 delta_cert_mat=sum(1 for v,s,x,c in cells if s.startswith('certificate') and x['safe_auc_vs_shuffled'] is not None and x['safe_auc_vs_shuffled']>=.03)
 delta_go=delta_safe_pos>=6 and delta_safe_mat>=6 and delta_harm_mat>=6 and delta_top_mat>=4 and delta_cert_mat>=3
 ctx_inc_pos=cnt(lambda v,s,x,c:c['safe_auc_vs_delta'] is not None and c['safe_auc_vs_delta']>0)
 ctx_inc_mat=cnt(lambda v,s,x,c:c['safe_auc_vs_delta'] is not None and c['safe_auc_vs_delta']>=.02)
 ctx_top_nondecline=cnt(lambda v,s,x,c:c['top1_vs_delta'] is None or c['top1_vs_delta']>=0)
 ctx_contact_mat=sum(1 for v,s,x,c in cells if 'contact' in s and c['safe_auc_vs_delta'] is not None and c['safe_auc_vs_delta']>=.02)
 ctx_go=ctx_inc_pos>=6 and ctx_inc_mat>=4 and ctx_top_nondecline>=6 and ctx_contact_mat>=2
 sufficient=delta_go or ctx_go
 status='STAGE_I_ACTION_OBSERVABILITY_GO' if sufficient else 'STAGE_I_ACTION_OBSERVABILITY_STOP'
 if ctx_go: nxt='frozen_representation_contains_state_conditioned_action_signal_then_design_state_action_factorized_nested_source_no_field_sweep'
 elif delta_go: nxt='frozen_representation_contains_action_signal_then_reform_source_composition_no_capacity_sweep'
 else: nxt='frozen_stage_i_root_action_observability_insufficient_then_narrow_action_response_representation_learning_no_broad_encoder_retrain'
 out['preregistered_decision']={'delta_action_observability_go':delta_go,'state_conditioned_action_observability_go':ctx_go,'stage_i_action_observability_go':sufficient,'delta_safe_auc_positive_cells':delta_safe_pos,'delta_safe_auc_material_cells':delta_safe_mat,'delta_harm_auc_material_cells':delta_harm_mat,'delta_top1_material_cells':delta_top_mat,'delta_certificate_material_cells':delta_cert_mat,'context_safe_auc_positive_cells':ctx_inc_pos,'context_safe_auc_material_cells':ctx_inc_mat,'context_top1_nondecline_cells':ctx_top_nondecline,'context_contact_material_cells':ctx_contact_mat,'status':status,'next_branch':nxt,'v48_83_stop_prerequisite':True}
 a.output.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out['preregistered_decision']))
if __name__=='__main__': main()
