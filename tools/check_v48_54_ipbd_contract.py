#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import torch

def sha256(p: Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def btext(x:str)->bool:
    s=str(x).strip().lower()
    if s in {'1','true','yes','on'}: return True
    if s in {'0','false','no','off'}: return False
    raise argparse.ArgumentTypeError(x)

def main()->int:
    ap=argparse.ArgumentParser(description='Fail-closed v48.54 IPBD witness contract check.')
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--expect-ipbd',type=btext,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); err=[]; stage={}; complete={}; training={}; lw={}
    root=a.run/'v48_47_recovery_frontier'; sp=root/'V48_47_WITNESS_STAGE.json'; cp=root/'V48_47_WITNESS_COMPLETE.json'; ck=root/'model_v48_47_witness'/'best.pt'
    for p,n in ((sp,'stage_contract'),(cp,'stage_complete'),(ck,'witness_checkpoint')):
        if not p.is_file(): err.append(f'missing_{n}:{p}')
    if sp.is_file():
        try: stage=json.loads(sp.read_text())
        except Exception as e: err.append(f'stage_unreadable:{e!r}')
    if cp.is_file():
        try: complete=json.loads(cp.read_text())
        except Exception as e: err.append(f'complete_unreadable:{e!r}')
    if ck.is_file():
        try:
            z=torch.load(ck,map_location='cpu',weights_only=False); cfg=z.get('cfg',{}) if isinstance(z,dict) else {}
            training=cfg.get('training',{}) if isinstance(cfg.get('training'),dict) else {}
            lw=cfg.get('loss_weights',{}) if isinstance(cfg.get('loss_weights'),dict) else {}
        except Exception as e: err.append(f'checkpoint_unreadable:{e!r}')
    ipbd=bool(a.expect_ipbd); expected_w=float(stage.get('frontier_sign_weight',0.5)) if ipbd else 0.0
    checks={
      'stage_frontier':stage.get('stage')=='frontier',
      'observation_class':stage.get('option_execution_semantics')=='observation_class',
      'boundary_complete_frontier':stage.get('boundary_complete_frontier') is True,
      'old_de_frontier_off':stage.get('decision_equivalent_frontier') is False,
      'teacher_physical_off':stage.get('physical_teacher_sign_alignment') is False,
      'student_physical_off':stage.get('physical_student_sign_alignment') is False,
      'teacher_coordinate_qhard':stage.get('teacher_sign_coordinate')=='q_hard_proxy_drs_exact_pcd',
      'student_coordinate_qhard':stage.get('student_sign_coordinate')=='hard_qbest_ge_zero_root_mass_exact_pcd',
      'smooth_order':stage.get('frontier_order_coordinate')=='smooth_boundary_drs_smooth_pcd',
      'ipbd_stage':bool(stage.get('invariant_physical_boundary_distillation',False)) is ipbd,
      'ipbd_coordinate':stage.get('physical_boundary_distillation_coordinate')=='teacher_q_selected_mstar_zero_to_predicted_margin',
      'ipbd_weight':abs(float(stage.get('physical_boundary_distillation_weight',0.0))-expected_w)<1e-12,
      'checkpoint_bcfc':bool(training.get('recovery_frontier_boundary_complete',False)) is True,
      'checkpoint_teacher_physical_off':bool(training.get('recovery_frontier_physical_teacher_sign_alignment',False)) is False,
      'checkpoint_student_physical_off':bool(training.get('recovery_frontier_physical_student_sign_alignment',False)) is False,
      'checkpoint_ipbd':bool(training.get('invariant_physical_boundary_distillation',False)) is ipbd,
      'checkpoint_observation_class':training.get('option_execution_semantics')=='observation_class',
      'checkpoint_ipbd_loss_weight':abs(float(lw.get('physical_boundary_distill',0.0))-expected_w)<1e-12,
    }
    for n,ok in checks.items():
        if not ok: err.append('failed:'+n)
    actual=sha256(ck) if ck.is_file() else None
    if actual and complete and complete.get('checkpoint_sha256')!=actual: err.append('checkpoint_sha_mismatch')
    doc={'event':'v48_54_ipbd_contract','version':'v48.54-DCP-DRFC-BCDE-IPBD','run':str(a.run),'expect_ipbd':ipbd,'checks':checks,'stage_ipbd_weight':stage.get('physical_boundary_distillation_weight'),'checkpoint_ipbd_weight':lw.get('physical_boundary_distill'),'witness_checkpoint_sha256':actual,'valid':not err,'errors':err,'strategy_regime_conditioning':False,'test_roots_read':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return 0 if not err else 4
if __name__=='__main__': raise SystemExit(main())
