from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import torch

from ocrap.cli.train import (
    _absolute_feasibility_bce,
    _absolute_feasibility_signed_margin_huber,
    _absolute_feasibility_supervision_loss,
)

ROOT=Path(__file__).resolve().parents[1]

def _batch():
    return {
        'r_dep_star': torch.tensor([0.5,0.2,-0.7,-2.0,0.5],dtype=torch.float32),
        'is_nominal': torch.zeros(5,dtype=torch.float32),
        'bucket_id': torch.tensor([1,1,2,2,3]),
        'time_index': torch.zeros(5,dtype=torch.long),
    }

def _out():
    return {'direct_recovery_absolute_feasibility_logit':torch.tensor([0.0,0.1,-0.2,-1.0,0.0],dtype=torch.float32)}

def test_v4876_binary_sign_is_execution_exact_legacy_objective():
    b=_batch(); o=_out(); cfg={'direct_value_absolute_feasibility_truth_contract':'censor_exact_0p5','direct_value_absolute_feasibility_supervision_objective':'binary_sign'}
    assert torch.equal(_absolute_feasibility_supervision_loss(o,b,cfg),_absolute_feasibility_bce(o,b,cfg))

def test_v4876_signed_margin_huber_uses_nonfloor_raw_margin():
    b=_batch(); o=_out(); cfg={'direct_value_absolute_feasibility_truth_contract':'censor_exact_0p5','direct_value_absolute_feasibility_supervision_objective':'signed_margin_huber'}
    # rows 1,2,3 are supervised; exact 0.5 row 0 is censored and bucket-3 row 4 is excluded.
    expected=torch.nn.functional.smooth_l1_loss(torch.tensor([0.1,-0.2,-1.0]),torch.tensor([0.2,-0.7,-2.0]),beta=1.0)
    got=_absolute_feasibility_signed_margin_huber(o,b,cfg)
    assert torch.allclose(got,expected,atol=0,rtol=0)
    assert torch.equal(_absolute_feasibility_supervision_loss(o,b,cfg),got)

def test_v4876_signed_margin_has_same_zero_boundary_not_relabel():
    b=_batch(); o=_out(); cfg={'direct_value_absolute_feasibility_truth_contract':'censor_exact_0p5','direct_value_absolute_feasibility_supervision_objective':'signed_margin_huber'}
    # changing a censored floor target's prediction must not change the loss.
    a=_absolute_feasibility_supervision_loss(o,b,cfg)
    o2={'direct_recovery_absolute_feasibility_logit':o['direct_recovery_absolute_feasibility_logit'].clone()};o2['direct_recovery_absolute_feasibility_logit'][0]=100.0
    assert torch.equal(a,_absolute_feasibility_supervision_loss(o2,b,cfg))

def test_v4876_unknown_objective_fails_closed():
    try:_absolute_feasibility_supervision_loss(_out(),_batch(),{'direct_value_absolute_feasibility_supervision_objective':'mystery'})
    except ValueError:return
    raise AssertionError('unknown objective did not fail closed')

def test_v4876_shell_wires_objective_and_truth_contract():
    train=(ROOT/'scripts/train_ocrap_v48_trac_sr.sh').read_text(); adapt=(ROOT/'scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh').read_text(); run=(ROOT/'scripts/run_v48_76_dcp_drfc_bcde_rifa_icsm_two_gpu.sh').read_text()
    assert 'direct_value_absolute_feasibility_supervision_objective' in train
    assert 'ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE' in adapt
    assert 'ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE=signed_margin_huber' in run
    assert 'ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=censor_exact_0p5' in run

def test_v4876_runtime_contract_synthetic(tmp_path:Path):
    out=tmp_path/'runtime.json'
    r=subprocess.run([sys.executable,str(ROOT/'tools/check_v48_76_runtime_code_contract.py'),'--repo',str(ROOT),'--output',str(out)],capture_output=True,text=True)
    assert r.returncode==0,(r.stdout,r.stderr)
    d=json.loads(out.read_text());assert d['valid'] and d['attribution_ready'];assert d['supervision_contract']['synthetic_check']['valid']

def test_v4876_changelog_records_v75_stop_and_no_geometry_sweep():
    t=(ROOT/'ALGORITHM_CHANGELOG.md').read_text().lower()
    assert 'v48.76' in t and 'signed-margin' in t
    assert 'no geometry' in t or 'geometry/kinematic' in t
