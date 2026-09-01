#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
VARIANTS=('balanced','precision');SPLITS=('dev_near','dev_contact','certificate_near','certificate_contact')
def load(p):return json.loads(Path(p).read_text())
def vals(a,name,field):return [a['comparisons'][name][v][s].get(field) for v in VARIANTS for s in SPLITS]
def pos(x):return sum(v is not None and v>0 for v in x)
def mat(x,t):return sum(v is not None and v>=t for v in x)
def dec(x):return sum(v is not None and v<0 for v in x)
def decmat(x,t):return sum(v is not None and v<=-t for v in x)
def alltrue(x):return all(bool(v) for v in x)
def arm(a,name,field):return [a['arms'][name][v]['splits'][s].get(field) for v in VARIANTS for s in SPLITS]
def mechanism(a,name,auc_mat=.005,hub_mat=.01):
    au=vals(a,name,'nonfloor_auc_delta');hb=vals(a,name,'nonfloor_huber_delta')
    return pos(au)>=6 and mat(au,auc_mat)>=4 and dec(hb)>=6 and decmat(hb,hub_mat)>=4,au,hb
def selectivity(a,new,base,tol=.02):
    out=[]
    for f in ('harmful_pass_fraction','teacher_infeasible_pass_fraction'):
        n=arm(a,new,f);b=arm(a,base,f);out += [x is not None and y is not None and x<=.25 and x<=y+tol for x,y in zip(n,b)]
    return all(out)
def full_go(a,name,new,base):
    full=vals(a,name,'full_auc_delta');nf=vals(a,name,'nonfloor_auc_delta')
    order=pos(full)==8 and mat(full,.01)>=6 and pos(nf)>=6 and mat(nf,.005)>=4
    n=arm(a,new,'safe_positive_nonfloor_pass_fraction');b=arm(a,base,'safe_positive_nonfloor_pass_fraction');cnt=arm(a,base,'safe_positive_nonfloor_rows')
    adequate=[(x,y,c) for x,y,c in zip(n,b,cnt) if c is not None and c>=5 and x is not None and y is not None]
    admission=len(adequate)>=4 and all(x>=y for x,y,c in adequate) and sum((x-y)>=.05 for x,y,c in adequate)>=3
    return order and admission,{'full_auc_delta':full,'nonfloor_auc_delta':nf,'adequate_safe_positive_cells':len(adequate),'adequate_safe_positive_deltas':[x-y for x,y,c in adequate],'source_ordering_gate':order,'safe_positive_admission_gate':admission}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--v77-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();au=load(a.audit);p77=load(a.v77_comparison);errors=[]
    pre=p77.get('preregistered_decision') or {};prereq=bool(p77.get('valid') and p77.get('attribution_ready') and pre.get('status')=='STOP' and pre.get('active_constraint_typed_source_go') is False and pre.get('next_branch')=='active_typed_transport_stop_close_gain_transport_family_then_structured_ocmero_tail_source_interface_no_gain_sweep')
    if not prereq:errors.append('V48.77 prerequisite branch mismatch')
    identity=alltrue(vals(au,'I78_minus_E76','teacher_labels_equal')+vals(au,'J78_minus_I78','teacher_labels_equal')+vals(au,'I78_minus_E76','positive_certificate_set_equal')+vals(au,'J78_minus_I78','positive_certificate_set_equal'))
    i_go,i_auc,i_hub=mechanism(au,'I78_minus_E76',.005,.01)
    # Incremental tail localization has a lower material threshold because it is
    # a deterministic factor on top of an already non-translational source, but
    # still requires cross-cell consistency and signed-error improvement.
    j_i_go,j_i_auc,j_i_hub=mechanism(au,'J78_minus_I78',.003,.005)
    j_go,j_auc,j_hub=mechanism(au,'J78_minus_E76',.005,.01)
    sel=selectivity(au,'I78_ROOT_SHAPE','E76_GLOBAL_SIGNED_SOURCE') and selectivity(au,'J78_MAIN_RTSI','E76_GLOBAL_SIGNED_SOURCE')
    full,diag=full_go(au,'J78_minus_C75','J78_MAIN_RTSI','C75_NATIVE_SIGN_PROJ')
    if j_go and full:
        status='RTSI_SOURCE_GO';next_branch='freeze_absolute_source_then_teacher_truth_contract_reconciliation_frozen_rifa_and_three_regime_closed_loop'
    elif j_go:
        status='RTSI_MECHANISM_GO_SOURCE_STOP';next_branch='root_tail_source_identified_then_teacher_physical_structural_truth_contract_adjudication_before_boundary_transport'
    elif i_go:
        status='ROOT_SHAPE_GO_TAIL_STOP';next_branch='promote_zero_mean_root_shape_only_then_truth_contract_adjudication_no_tail_or_gain_sweep'
    else:
        status='STOP';next_branch='root_tail_source_stop_close_low_capacity_absolute_source_adapter_family_then_teacher_truth_contract_adjudication_before_any_new_source_capacity'
    doc={'schema':'ocrap-v48.78-rtsi-comparison-v1','algorithm':'v48.78-DCP-DRFC-BCDE-RIFA-OC-RTSI','engineering_version':'v48.78.0-OC-RTSI','errors':errors,'preregistered_decision':{'v48_77_prerequisite_valid':prereq,'identity_gate':identity,'I78_root_shape_go':i_go,'J78_tail_localization_increment_go':j_i_go,'J78_root_tail_source_go':j_go,'absolute_selectivity_gate':sel,'full_source_go':full,'status':status,'next_branch':next_branch},'deltas':{'I78_minus_E76_nonfloor_auc':i_auc,'I78_minus_E76_nonfloor_huber':i_hub,'J78_minus_I78_nonfloor_auc':j_i_auc,'J78_minus_I78_nonfloor_huber':j_i_hub,'J78_minus_E76_nonfloor_auc':j_auc,'J78_minus_E76_nonfloor_huber':j_hub,'J78_full_source_diagnostic':diag},'scientific_contract':{'root_shape_go':'I-E: non-floor AUC >0 >=6/8 and >=4/8 >=+0.005; Huber lower >=6/8 and >=4/8 <=-0.01','tail_increment_go':'J-I: non-floor AUC >0 >=6/8 and >=4/8 >=+0.003; Huber lower >=6/8 and >=4/8 <=-0.005','main_root_tail_go':'J-E uses the strong root-shape GO thresholds','full_source_go':'J-C: full AUC 8/8 positive, >=6/8 >=+0.01; non-floor >=6/8 positive and >=4/8 >=+0.005; adequately powered safe-positive cells non-decrease and >=3 improve >=0.05','absolute_selectivity':'I/J harmful and TI <=0.25 and <= E76 +0.02','translation_contract':'p-weighted root correction is exactly zero for every option','projection_fidelity':'OFF','typed_gain_transport':'OFF','boundary_transport':'OFF','relative_ranker':'frozen','gain_lr_threshold_sweep':'forbidden','geometry_sweep':'forbidden','regime_router':'forbidden'},'dataset_reconstruction':False,'test_roots_read':False}
    valid=not errors and prereq and identity and sel;doc['valid']=valid;doc['attribution_ready']=valid
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_78_rtsi_comparison','valid':valid,'status':status,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
