from __future__ import annotations
import hashlib, json, os, subprocess
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def _w(p:Path,d): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d)+'\n')
def _sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()

def test_v4854_reference_reuse_is_semantic_not_transient_seal_sha(tmp_path:Path):
    src=tmp_path/'source'; ref=tmp_path/'ref'
    checks={}
    for v in ('balanced','precision'):
        p=src/'candidates'/v/'model_v48_trac_sr'/'best.pt'; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(('ckpt-'+v).encode()); checks[v]={'sha256':_sha(p)}
    roots={}
    for role in ('safe_calibration','near_certificate','contact_certificate','near_threshold_fit_dev','contact_threshold_fit_dev'):
        r=tmp_path/role; r.mkdir(); (r/'manifest.csv').write_text(role+'\n'); roots[role]=r
    protocol={'version':'v48.36-OCAF','confidence':{'level':.9,'bound_type':'one_sided'},'benefit':{'positive_gain':.015,'opportunity_label_mode':'raw_benefit','gate_positive_mode':'safe_benefit'},'policy':{'proposal_top_k':5,'selection_semantics':'rank_topk_then_filter_then_evidence_rerank','option_execution_semantics':'observation_class'},'harm':{},'near':{},'contact':{},'datasets':[{'role':k,'manifest_sha256':_sha(v/'manifest.csv')} for k,v in roots.items()],'threshold_source':'pooled_evidence_adapt_dev_shared_rule','shared_deployment_rule':True,'strategy_regime_conditioning':False,'certificate_mode':'external_rule_full_verification','certificate_labels_used_for_threshold_fit':False,'fit_verify_scene_disjoint':True,'test_roots_read':False}
    canonical=json.dumps(protocol,sort_keys=True,separators=(',',':')).encode(); psha=hashlib.sha256(canonical).hexdigest()
    _w(ref/'AUTHORITATIVE_RUN_STATUS.json',{'authoritative_exit_code':20,'pipeline_valid':True})
    _w(ref/'SOURCE_CHECKPOINT_CONTRACT.json',{'checks':checks})
    _w(ref/'GATE_SPEC.json',{'protocol':protocol,'protocol_sha256':psha,'created_unix':1})
    _w(ref/'V48_53_FACTOR_CONTRACT.json',{'version':'v48.53-DCP-DRFC-BCDE-CSE','arm':'A','boundary_complete_frontier':True,'physical_teacher_sign_alignment':False,'physical_student_sign_alignment':False,'native_physical_student_drs':False,'native_certificate_preservation':True,'native_advantage_preservation':True,'native_exact_advantage_preservation':False,'native_boundary_complete_advantage_preservation':False,'student_sign_coordinate':'hard_qbest_ge_zero_root_mass_exact_pcd','teacher_sign_coordinate':'q_hard_proxy_drs_exact_pcd','frontier_order_coordinate':'smooth_boundary_drs_smooth_pcd','strategy_regime_conditioning':False,'test_roots_read':False})
    out=tmp_path/'reuse.json'
    cmd=['python',str(ROOT/'tools/check_v48_54_reference_reuse.py'),'--reference',str(ref),'--source-run',str(src),'--safe',str(roots['safe_calibration']),'--near-cert',str(roots['near_certificate']),'--contact-cert',str(roots['contact_certificate']),'--near-dev',str(roots['near_threshold_fit_dev']),'--contact-dev',str(roots['contact_threshold_fit_dev']),'--output',str(out)]
    r=subprocess.run(cmd,cwd=ROOT); assert r.returncode==0
    d=json.loads(out.read_text()); assert d['valid'] and d['ignored_transient_identity']

def test_v4854_ipbd_checkpoint_contract(tmp_path:Path):
    run=tmp_path/'candidates'/'precision'; stage=run/'v48_47_recovery_frontier'; ck=stage/'model_v48_47_witness'/'best.pt'; ck.parent.mkdir(parents=True)
    st={'stage':'frontier','option_execution_semantics':'observation_class','boundary_complete_frontier':True,'decision_equivalent_frontier':False,'physical_teacher_sign_alignment':False,'physical_student_sign_alignment':False,'teacher_sign_coordinate':'q_hard_proxy_drs_exact_pcd','student_sign_coordinate':'hard_qbest_ge_zero_root_mass_exact_pcd','frontier_order_coordinate':'smooth_boundary_drs_smooth_pcd','invariant_physical_boundary_distillation':True,'physical_boundary_distillation_coordinate':'teacher_q_selected_mstar_zero_to_predicted_margin','physical_boundary_distillation_weight':.5,'frontier_sign_weight':.5}
    _w(stage/'V48_47_WITNESS_STAGE.json',st)
    torch.save({'cfg':{'training':{'recovery_frontier_boundary_complete':True,'recovery_frontier_physical_teacher_sign_alignment':False,'recovery_frontier_physical_student_sign_alignment':False,'invariant_physical_boundary_distillation':True,'option_execution_semantics':'observation_class'},'loss_weights':{'physical_boundary_distill':.5}}},ck)
    _w(stage/'V48_47_WITNESS_COMPLETE.json',{'checkpoint_sha256':_sha(ck)})
    out=tmp_path/'ipbd.json'; r=subprocess.run(['python',str(ROOT/'tools/check_v48_54_ipbd_contract.py'),'--run',str(run),'--expect-ipbd','true','--output',str(out)],cwd=ROOT)
    assert r.returncode==0 and json.loads(out.read_text())['valid']

def test_v4854_postgate_blocks_rc20_and_allows_rc0(tmp_path:Path):
    main=tmp_path/'main'; main.mkdir(); marker=tmp_path/'marker'; nextp=main/'NEXT_COMMANDS.txt'; nextp.write_text(f'echo ok > {marker}\n')
    factor={'version':'v48.54-DCP-DRFC-BCDE-IPBD','arm':'B','invariant_physical_boundary_distillation':True,'physical_teacher_sign_alignment':False,'physical_student_sign_alignment':False,'native_physical_student_drs':False,'student_sign_coordinate':'hard_qbest_ge_zero_root_mass_exact_pcd','teacher_sign_coordinate':'q_hard_proxy_drs_exact_pcd','frontier_order_coordinate':'smooth_boundary_drs_smooth_pcd','strategy_regime_conditioning':False,'new_tuned_thresholds':False,'test_roots_read':False}
    _w(main/'V48_54_FACTOR_CONTRACT.json',factor)
    env={**os.environ,'MAIN_RUN':str(main)}
    _w(main/'AUTHORITATIVE_RUN_STATUS.json',{'valid':True,'pipeline_valid':True,'authoritative_exit_code':20,'checks':{'certificate_executed':True,'gate_evaluated':True}})
    r=subprocess.run(['bash',str(ROOT/'scripts/run_v48_54_postgate_if_authorized.sh')],cwd=ROOT,env=env); assert r.returncode!=0 and not marker.exists()
    _w(main/'AUTHORITATIVE_RUN_STATUS.json',{'valid':True,'pipeline_valid':True,'authoritative_exit_code':0,'checks':{'certificate_executed':True,'gate_evaluated':True}})
    r=subprocess.run(['bash',str(ROOT/'scripts/run_v48_54_postgate_if_authorized.sh')],cwd=ROOT,env=env); assert r.returncode==0 and marker.exists()
