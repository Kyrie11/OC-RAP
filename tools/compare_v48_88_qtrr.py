#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
V=('balanced','precision'); S=('dev_near','dev_contact','certificate_near','certificate_contact')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--v87-comparison',type=Path,required=True)
    ap.add_argument('--w-run',type=Path,required=True)
    ap.add_argument('--x-run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); au=json.loads(a.audit.read_text()); p87=json.loads(a.v87_comparison.read_text()); d87=p87.get('preregistered_decision') or {}; errors=[]
    prereq=bool(
        p87.get('valid') and d87.get('status')=='BILINEAR_ACTION_ROOT_INTERACTION_STOP'
        and not d87.get('bilinear_action_root_interaction_go')
        and 'root_local_response_target_identifiability' in str(d87.get('next_branch',''))
    )
    if not prereq: errors.append('V48.87 BARR STOP / root-local response-target identifiability prerequisite missing')

    def cells(name): return [au['comparisons'][name][v][s] for v in V for s in S]
    def stats(name,slack=.02):
        cs=cells(name)
        auc_delta=[x['source_auc_new']-x['source_auc_base'] for x in cs]
        hub_delta=[x['interval_huber_new']-x['interval_huber_base'] for x in cs]
        abs_sel=all((x['harmful_pass_new'] or 0)<=.25 and (x['ti_pass_new'] or 0)<=.25 for x in cs)
        matched=all((x['harmful_pass_new'] or 0)<=(x['harmful_pass_base'] or 0)+slack and (x['ti_pass_new'] or 0)<=(x['ti_pass_base'] or 0)+slack for x in cs)
        powered=[(v,s,au['comparisons'][name][v][s]) for v in V for s in S if int(au['comparisons'][name][v][s].get('safe_positive_rows',0))>=5]
        mats=[(v,s,(x.get('safe_positive_pass_new') or 0)-(x.get('safe_positive_pass_base') or 0)) for v,s,x in powered if (x.get('safe_positive_pass_new') or 0)-(x.get('safe_positive_pass_base') or 0)>=.05]
        noninc=[(v,s) for v,s,x in powered if (x.get('safe_positive_pass_new') or 0)+1e-12 >= (x.get('safe_positive_pass_base') or 0)]
        false_noninc=sum((x['harmful_pass_new'] or 0)<=(x['harmful_pass_base'] or 0)+1e-12 and (x['ti_pass_new'] or 0)<=(x['ti_pass_base'] or 0)+1e-12 for x in cs)
        return {
            'auc_positive':sum(d>0 for d in auc_delta),
            'auc_material_003':sum(d>=.003 for d in auc_delta),
            'auc_material_010':sum(d>=.01 for d in auc_delta),
            'auc_material_020':sum(d>=.02 for d in auc_delta),
            'interval_huber_nondegrade_cells':sum(d<=1e-12 for d in hub_delta),
            'interval_huber_material_010':sum(d<=-.01 for d in hub_delta),
            'absolute_selectivity':abs_sel,'matched_selectivity':matched,
            'false_admission_nonincrease_cells':false_noninc,
            'safe_positive_nondegrade_powered_cells':len(noninc),
            'safe_positive_powered_cells':len(powered),
            'safe_positive_material_cells':len(mats),
            'safe_positive_near_material':any('near' in s for _,s,_ in mats),
            'safe_positive_contact_material':any('contact' in s for _,s,_ in mats),
            'newly_admitted_safe_positive':sum(int(x.get('newly_admitted_safe_positive',0)) for x in cs),
            'newly_admitted_harmful':sum(int(x.get('newly_admitted_harmful',0)) for x in cs),
        }
    def train_meta(run):
        out={}
        for v in V:
            p=run/'candidates'/v/'TRAINING_COMPLETE.json'
            if not p.is_file(): out[v]={'exists':False,'best_epoch':None}; continue
            d=json.loads(p.read_text()); out[v]={'exists':True,'best_epoch':int(d.get('best_epoch',-1)),'epochs_completed':int(d.get('epochs_completed',-1)),'best_metric':d.get('best_metric')}
        return out

    wu=stats('W88_minus_U87',.0); xv=stats('X88_minus_V87',.0)
    ws=stats('W88_minus_S86',.02); xt=stats('X88_minus_T86',.02)
    xw=stats('X88_minus_W88',.005); full=stats('X88_minus_L80',.02)
    wm=train_meta(a.w_run); xm=train_meta(a.x_run)
    w_learns=all(wm[v]['exists'] and wm[v]['best_epoch']>0 for v in V)
    x_learns=all(xm[v]['exists'] and xm[v]['best_epoch']>0 for v in V)

    # The primary question is not whether a 282-parameter source merely looks
    # better than the catastrophically permissive BARR. It must both remove the
    # BARR nullspace failure under the same selective objective (X-V) and add
    # genuine recovery value over the historical root-independent T86 source.
    nullspace_repair=(
        x_learns and xv['auc_positive']==8 and xv['auc_material_020']>=6
        and xv['interval_huber_nondegrade_cells']==8 and xv['absolute_selectivity']
        and xv['false_admission_nonincrease_cells']>=6
    )
    net_identifiable_response=(
        xt['auc_positive']>=6 and xt['auc_material_003']>=4
        and xt['interval_huber_nondegrade_cells']>=6 and xt['absolute_selectivity']
        and xt['matched_selectivity'] and xt['safe_positive_near_material']
        and xt['safe_positive_contact_material'] and xt['newly_admitted_harmful']<=8
    )
    quotient_identifiability_go=bool(nullspace_repair and net_identifiable_response)
    interval_support=bool(
        w_learns and wu['auc_positive']>=6 and wu['interval_huber_nondegrade_cells']>=6
        and ws['auc_positive']>=5 and ws['interval_huber_nondegrade_cells']>=4
        and ws['absolute_selectivity']
    )
    selective_increment_go=bool(
        xw['auc_positive']>=6 and xw['auc_material_003']>=4
        and xw['interval_huber_nondegrade_cells']>=6 and xw['matched_selectivity']
        and xw['safe_positive_near_material'] and xw['safe_positive_contact_material']
    )
    full_go=bool(
        full['auc_positive']==8 and full['auc_material_010']>=6
        and full['interval_huber_nondegrade_cells']>=6
        and full['absolute_selectivity'] and full['matched_selectivity']
        and full['safe_positive_near_material'] and full['safe_positive_contact_material']
    )

    if quotient_identifiability_go and full_go:
        status='QUOTIENT_TAIL_RESPONSE_FULL_GO'
        nxt='freeze_absolute_source_then_frozen_RIFA_safe_noninterference_near_closed_loop_contact_postcollision_external_SOTA'
    elif quotient_identifiability_go:
        status='QUOTIENT_TAIL_IDENTIFIABILITY_GO_SOURCE_STOP'
        nxt='freeze_identifiable_response_representation_then_adjudicate_only_remaining_target_or_absolute_boundary_debt_no_capacity_increase'
    elif interval_support and not selective_increment_go:
        status='QUOTIENT_INTERVAL_SUPPORT_SELECTIVE_STOP'
        nxt='retain_quotient_interval_response_close_current_structural_ordering_increment_no_capacity_sweep'
    else:
        status='QUOTIENT_TAIL_RESPONSE_STOP'
        nxt='close_aggregate_counterfactual_response_adapter_family_then_audit_counterfactual_root_correspondence_and_root_local_physical_response_identifiability_no_rank_or_encoder_sweep'

    doc={
        'schema':'ocrap-v48.88-qtrr-comparison-v1','engineering_version':'v48.88.0-OC-QTRR',
        'valid':prereq and not errors,'attribution_ready':prereq and not errors,'errors':errors,
        'preregistered_decision':{
            'v48_87_stop_prerequisite':prereq,
            'W88_minus_U87':wu,'X88_minus_V87':xv,'W88_minus_S86':ws,'X88_minus_T86':xt,'X88_minus_W88':xw,'X88_minus_L80':full,
            'training_meta_W88':wm,'training_meta_X88':xm,'W88_heldout_learning':w_learns,'X88_heldout_learning':x_learns,
            'barr_nullspace_repair':nullspace_repair,'net_identifiable_response':net_identifiable_response,
            'quotient_tail_identifiability_go':quotient_identifiability_go,
            'interval_quotient_support':interval_support,'structural_selective_increment_go':selective_increment_go,
            'full_source_go':full_go,'status':status,'next_branch':nxt,
        },
        'scientific_contract':{
            'representation':'candidate-action scalar response lifted only along the exact nested OC-MERO quotient cotangent',
            'new_trainable_parameters':282,'v48_87_barr_parameters':53550,'capacity_ratio_vs_barr':282/53550.0,
            'learned_root_local_response_target':False,'learned_nullspace_capacity':False,
            'option_translation_removed':True,'reserve_debt_channels':2,
            'regime_conditioning':False,'generic_mlp':False,'broad_encoder_retraining':False,
            'physical_response_supervision':'V48.86 candidate-minus-nominal partially identified response interval',
            'structural_admissibility_supervision':'V48.86 pairwise safe-positive / harmful ordering',
            'boundary_transport':'OFF','dataset_reconstruction':False,'relative_ranker_modified':False,
        },
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':doc['valid'],'status':status})); return 0 if doc['valid'] else 30

if __name__=='__main__': raise SystemExit(main())
