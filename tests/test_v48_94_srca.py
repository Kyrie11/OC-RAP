from __future__ import annotations
from pathlib import Path
import json
import subprocess
import sys

from ocrap.v48_94_support_reserve_admission import support_reserve_admission

ROOT=Path(__file__).resolve().parents[1]

def test_support_establishment_uses_exact_zero_state():
    x=support_reserve_admission([0.25,0.10,0.30,0.9],[0.0,0.8,0.1,0.9])
    assert x.state=='support_establishment'
    assert x.passed
    assert abs(x.score-0.25)<1e-12

def test_reserve_debt_requires_support_and_native_zero_boundary():
    ok=support_reserve_admission([0.8,0.6,0.9,0.8],[0.5,0.7,0.9,0.9])
    assert ok.state=='reserve_debt' and ok.passed
    assert abs(ok.score-0.48)<1e-12
    bad_dep=support_reserve_admission([0.8,0.49,0.9,0.8],[0.5,0.7,0.9,0.9])
    assert not bad_dep.passed
    lost=support_reserve_admission([0.0,0.99,0.9,0.8],[0.5,0.7,0.9,0.9])
    assert not lost.passed

def test_gap_is_diagnostic_not_positive_admission():
    a=support_reserve_admission([1.0,0.7,0.9,1.0],[1.0,0.7,0.9,1.0])
    b=support_reserve_admission([1.0,0.7,0.9,0.01],[1.0,0.7,0.9,1.0])
    assert a.passed==b.passed and a.score==b.score

def test_only_historical_boundaries_are_allowed():
    try: support_reserve_admission([1,.6,.8,1],[1,.6,.8,1],deployability_threshold=.6)
    except ValueError: pass
    else: raise AssertionError('threshold sweep must be rejected')

def test_calibrator_exposes_fixed_mode_and_diagnostics():
    text=(ROOT/'tools/calibrate_policy_risk_v48.py').read_text()
    assert '"support_reserve"' in text
    assert 'support_reserve_state' in text
    assert 'native_candidate_certificate' in text
    assert 'native_nominal_certificate' in text

def test_shared_controller_accepts_fixed_support_reserve_mode_without_sweep():
    text=(ROOT/'scripts/calibrate_v48_36_shared_certificate_pool.sh').read_text()
    assert 'off|native|learned|support_reserve' in text
    assert 'requires ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' in text

def test_runner_is_fixed_capacity_and_regime_agnostic():
    text=(ROOT/'scripts/run_v48_94_dcp_drfc_bcde_rifa_srca_two_gpu.sh').read_text()
    assert 'V93_COMPLETE' in text and 'PCD_FACTOR_COMPLEMENTARITY_GO' in text
    assert '--absolute-feasibility-mode=support_reserve' in text
    assert 'train.py' not in text
    assert 'V4891_WOMD_SOURCE' not in text

def test_runtime_contract_passes_on_repo(tmp_path):
    out=tmp_path/'runtime.json'
    subprocess.run([sys.executable,str(ROOT/'tools/check_v48_94_runtime_code_contract.py'),'--repo',str(ROOT),'--output',str(out)],check=True,cwd=ROOT)
    d=json.loads(out.read_text())
    assert d['valid'] and d['attribution_ready']
    assert d['scientific_contract']['new_planner_parameters']==0
    assert not d['scientific_contract']['regime_conditioning']
