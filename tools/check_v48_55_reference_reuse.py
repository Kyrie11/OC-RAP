#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(p:Path)->str:
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def load(p): return json.loads(Path(p).read_text())
def main()->int:
 ap=argparse.ArgumentParser(description='Semantic fail-closed reuse check for v48.55 A reference.')
 ap.add_argument('--reference',type=Path,required=True); ap.add_argument('--source-run',type=Path,required=True)
 ap.add_argument('--safe',type=Path,required=True); ap.add_argument('--near-cert',type=Path,required=True); ap.add_argument('--contact-cert',type=Path,required=True); ap.add_argument('--near-dev',type=Path,required=True); ap.add_argument('--contact-dev',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args(); e=[]; r=a.reference
 for f in ['AUTHORITATIVE_RUN_STATUS.json','SOURCE_CHECKPOINT_CONTRACT.json','GATE_SPEC.json']:
  if not (r/f).is_file(): e.append('missing_reference:'+f)
 factor_path = r/'V48_53_FACTOR_CONTRACT.json'
 if not factor_path.is_file(): factor_path = r/'V48_54_FACTOR_CONTRACT.json'
 if not factor_path.is_file(): e.append('missing_reference:V48_53_OR_V48_54_FACTOR_CONTRACT.json')
 if e:
  doc={'event':'v48_55_reference_reuse_contract','valid':False,'errors':e}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2)+'\n'); return 4
 status=load(r/'AUTHORITATIVE_RUN_STATUS.json'); source=load(r/'SOURCE_CHECKPOINT_CONTRACT.json'); gate=load(r/'GATE_SPEC.json'); factor=load(factor_path)
 if int(status.get('authoritative_exit_code',99)) not in (0,20) or not status.get('pipeline_valid',status.get('valid',False)): e.append('reference_not_authoritative_algorithm_result')
 # v48.53-A is exactly the q-hard BC-FC + smooth-NAP reference required by v48.55.
 required={'arm':'A','boundary_complete_frontier':True,'physical_teacher_sign_alignment':False,'physical_student_sign_alignment':False,'native_physical_student_drs':False,'native_certificate_preservation':True,'native_advantage_preservation':True,'native_exact_advantage_preservation':False,'native_boundary_complete_advantage_preservation':False,'strategy_regime_conditioning':False,'test_roots_read':False}
 for k,v in required.items():
  if factor.get(k)!=v: e.append(f'factor_mismatch:{k}')
 if factor.get('version')=='v48.54-DCP-DRFC-BCDE-IPBD' and factor.get('invariant_physical_boundary_distillation') is not False:
  e.append('factor_mismatch:v48.54_A_must_have_ipbd_off')
 elif factor.get('version') not in {'v48.53-DCP-DRFC-BCDE-CSE','v48.54-DCP-DRFC-BCDE-IPBD'}:
  e.append('factor_mismatch:unsupported_reference_version')
 if factor.get('student_sign_coordinate')!='hard_qbest_ge_zero_root_mass_exact_pcd': e.append('student_sign_not_qhard')
 if factor.get('teacher_sign_coordinate')!='q_hard_proxy_drs_exact_pcd': e.append('teacher_sign_not_qproxy')
 if factor.get('frontier_order_coordinate')!='smooth_boundary_drs_smooth_pcd': e.append('order_not_smooth')
 # Source checkpoint byte identity.
 checks=source.get('checks') or {}
 cur_src={}
 for variant in ('balanced','precision'):
  p=a.source_run/'candidates'/variant/'model_v48_trac_sr'/'best.pt'
  if not p.is_file(): e.append(f'missing_current_source:{variant}'); continue
  cur_src[variant]=sha(p)
  if (checks.get(variant) or {}).get('sha256')!=cur_src[variant]: e.append(f'source_sha_mismatch:{variant}')
 # Dataset semantic identity. Ignore V48_45_PROTOCOL_SEAL byte SHA because it contains transient creation metadata.
 current_roots={'safe_calibration':a.safe,'near_certificate':a.near_cert,'contact_certificate':a.contact_cert,'near_threshold_fit_dev':a.near_dev,'contact_threshold_fit_dev':a.contact_dev}
 ref_ds={str(x.get('role')):x for x in ((gate.get('protocol') or {}).get('datasets') or []) if isinstance(x,dict)}
 cur_ds={}
 for role,root in current_roots.items():
  m=root/'manifest.csv'
  if not m.is_file(): e.append(f'missing_manifest:{role}'); continue
  cur_ds[role]=sha(m)
  if (ref_ds.get(role) or {}).get('manifest_sha256')!=cur_ds[role]: e.append(f'manifest_sha_mismatch:{role}')
 # Frozen gate semantics must match the preregistered v48.36 protocol expected by v48.55.
 p=gate.get('protocol') or {}
 expected_checks={
  'version':p.get('version')=='v48.36-OCAF','shared_rule':p.get('shared_deployment_rule') is True,
  'no_regime_conditioning':p.get('strategy_regime_conditioning') is False,'no_test':p.get('test_roots_read') is False,
  'topk':((p.get('policy') or {}).get('proposal_top_k')==5),'obs_class':((p.get('policy') or {}).get('option_execution_semantics')=='observation_class'),
  'positive_gain':abs(float((p.get('benefit') or {}).get('positive_gain',-1))-0.015)<1e-12,
  'safe_benefit':((p.get('benefit') or {}).get('gate_positive_mode')=='safe_benefit'),
  'scene_disjoint':p.get('fit_verify_scene_disjoint') is True,
 }
 for k,ok in expected_checks.items():
  if not ok: e.append('gate_semantic_mismatch:'+k)
 doc={'event':'v48_55_reference_reuse_contract','version':'v48.55-DCP-DRFC-BCDE-TCBC','reference':str(r),'reference_authoritative_exit_code':status.get('authoritative_exit_code'),'source_checkpoint_sha256':cur_src,'dataset_manifest_sha256':cur_ds,'ignored_transient_identity':['V48_45_PROTOCOL_SEAL.json byte sha / created_unix'],'gate_semantic_checks':expected_checks,'valid':not e,'errors':e,'strategy_regime_conditioning':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n'); return 0 if not e else 4
if __name__=='__main__': raise SystemExit(main())
