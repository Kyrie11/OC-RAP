#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path

def btext(x:str)->bool:
 s=str(x).strip().lower()
 if s in {'1','true','yes','on'}: return True
 if s in {'0','false','no','off'}: return False
 raise argparse.ArgumentTypeError(x)

def csv(x):
 s=str(x or '').strip()
 if not s or s.lower() in {'none','null','~'}: return []
 return [float(v.strip()) for v in s.split(',') if v.strip()]

def same(a,b,tol=1e-9):
 return len(a)==len(b) and all(math.isfinite(x) and abs(x-y)<=tol*max(1.,abs(x),abs(y)) for x,y in zip(a,b))

def bval(x): return x in {True,'true','True','1',1}

def main()->int:
 ap=argparse.ArgumentParser(description='Fail-closed v48.56 Decision-Role Aligned Certificate contract.')
 ap.add_argument('--run',type=Path,required=True)
 ap.add_argument('--expect-dep-boundary-aligned',type=btext,required=True)
 ap.add_argument('--expect-gap-ordinal-only',type=btext,required=True)
 ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args(); err=[]
 factor=a.run/'factor_stage'; arch_p=factor/'STAGE_ARCHITECTURE.json'; cache_p=factor/'FACTOR_CACHE_CONTRACT.json'
 for p,n in ((arch_p,'stage_architecture'),(cache_p,'factor_cache_contract')):
  if not p.is_file(): err.append(f'missing_{n}:{p}')
 try: arch=json.loads(arch_p.read_text()) if arch_p.is_file() else {}
 except Exception as e: arch={}; err.append(f'architecture_unreadable:{e!r}')
 try: cache=json.loads(cache_p.read_text()) if cache_p.is_file() else {}
 except Exception as e: cache={}; err.append(f'cache_unreadable:{e!r}')
 settings=cache.get('settings',{}) if isinstance(cache.get('settings'),dict) else {}
 expected_rel=[1.,1.,0.,0.,0.] if a.expect_gap_ordinal_only else [1.,1.,1.,0.,0.]
 checks={
  'algorithm_v48_56':str(arch.get('algorithm_variant','')).startswith('v48.56-'),
  'no_regime_exposure':arch.get('regime_id_exposed_to_evidence_model') is False,
  'test_roots_not_read':arch.get('test_roots_read') is False,
  'raw_component_target':arch.get('component_margin_target_mode')=='raw' and settings.get('component_margin_target_mode')=='raw',
  'component_reliability':same(csv(arch.get('component_reliability')),expected_rel),
  'component_reliability_cache':same(csv(settings.get('component_reliability')),expected_rel),
  'regression_reliability':same(csv(arch.get('component_margin_regression_reliability')),expected_rel),
  'regression_reliability_cache':same(csv(settings.get('component_margin_regression_reliability')),expected_rel),
  'dep_role_setting':bval(settings.get('v4856_dep_boundary_aligned',False))==bool(a.expect_dep_boundary_aligned),
  'native_dep_role_setting':bval(settings.get('native_dep_boundary_aligned',False))==bool(a.expect_dep_boundary_aligned),
  'gap_role_setting':bval(settings.get('v4856_gap_ordinal_only',False))==bool(a.expect_gap_ordinal_only),
  'native_certificate_preserved':bval(settings.get('native_certificate_preservation',False)),
  'native_advantage_preserved':bval(settings.get('native_advantage_preservation',False)),
  'native_margin_complete_off':not bval(settings.get('native_margin_complete_preservation',False)),
  'physical_teacher_off':not bval(settings.get('v4852_physical_teacher_sign_alignment',False)),
  'physical_student_off':not bval(settings.get('v4853_physical_student_sign_alignment',False)),
  'ipbd_off':not bval(settings.get('v4854_invariant_physical_boundary_distillation',False)),
  'target_scale_unchanged':abs(float(arch.get('component_margin_target_scale',-1))-0.10)<1e-12,
  'component_weight_unchanged':abs(float(arch.get('component_margin_regression_weight',-1))-1.0)<1e-12,
 }
 for k,v in checks.items():
  if not v: err.append('failed:'+k)
 doc={'event':'v48_56_drac_contract','version':'v48.56-DCP-DRFC-BCDE-DRAC','run':str(a.run),
      'expect_dep_boundary_aligned':bool(a.expect_dep_boundary_aligned),'expect_gap_ordinal_only':bool(a.expect_gap_ordinal_only),
      'expected_component_reliability':expected_rel,'checks':checks,'valid':not err,'errors':err,
      'strategy_regime_conditioning':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
 return 0 if not err else 4
if __name__=='__main__': raise SystemExit(main())
