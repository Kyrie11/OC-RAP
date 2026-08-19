from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_index(path: Path, buckets=(1, 2)) -> None:
    rows=[]
    # Two groups, each with one nominal and two candidates. Values are chosen so
    # DEP/GAP RMS is nonzero while the pooled calculation is insensitive to the
    # diagnostic bucket label.
    for gi,b in enumerate(buckets):
        scene=f"s{gi}"; time=0
        rows.append({"bucket":b,"scene":scene,"time":time,"candidate":0,"nominal":True,
                     "teacher_drs":0.8,"teacher_r_dep":0.5,"teacher_gap":0.2,
                     "teacher_hard_violation":0.0,"teacher_harm_proxy":0.0})
        rows.append({"bucket":b,"scene":scene,"time":time,"candidate":1,"nominal":False,
                     "teacher_drs":0.4,"teacher_r_dep":0.1,"teacher_gap":0.6,
                     "teacher_hard_violation":0.0,"teacher_harm_proxy":0.0})
        rows.append({"bucket":b,"scene":scene,"time":time,"candidate":2,"nominal":False,
                     "teacher_drs":0.9,"teacher_r_dep":0.8,"teacher_gap":0.05,
                     "teacher_hard_violation":0.0,"teacher_harm_proxy":0.0})
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))


def _scales(tmp_path: Path, buckets=(1,2)) -> dict:
    idx=tmp_path/("idx_"+"_".join(map(str,buckets))+".jsonl")
    out=tmp_path/("scale_"+"_".join(map(str,buckets))+".json")
    _write_index(idx,buckets)
    subprocess.run([sys.executable,str(ROOT/'tools/compute_v48_55_component_boundary_scales.py'),
                    '--index',str(idx),'--output',str(out)],check=True,cwd=ROOT)
    return json.loads(out.read_text())


def test_v4855_pooled_rms_scaling_is_linear_regime_free_and_keeps_drs_raw(tmp_path: Path):
    a=_scales(tmp_path,(1,2)); b=_scales(tmp_path,(9,7))
    assert a['canonical_scales']==b['canonical_scales']
    assert a['canonical_scales'][0]==0.10
    assert a['canonical_scales'][1] > 0 and a['canonical_scales'][2] > 0
    assert a['strategy_regime_conditioning'] is False
    assert a['test_roots_read'] is False
    assert a['saturating_transform'] is False
    assert a['zero_crossing_preserved'] is True
    assert a['within_component_order_preserved'] is True
    assert a['continuous_components_canonicalized']==['deployability','gap_quality']


def test_v4855_loss_separates_component_sign_support_from_magnitude_regression():
    text=(ROOT/'src/ocrap/models/losses.py').read_text()
    helper=text[text.index('def component_margin_regression_targets'):text.index('def direct_uncertainty_recovery_value_loss')]
    assert 'if target_mode == "raw":' in helper
    assert 'if target_mode == "frontier_tanh":' in helper
    assert 'if target_mode != "pooled_rms_linear":' in helper
    assert 'return float(target_scale) * raw_margin / scales.unsqueeze(0)' in helper
    assert 'preserves' in helper and 'zero crossing' in helper and 'does not' in helper
    block=text[text.index('target_component_margins_regression = component_margin_regression_targets'):]
    assert 'ordinal_evidence_component_margin_regression_reliability' in text
    assert 'effective_regression_reliability = component_reliability * regression_reliability' in block[:6200]


def test_v4855_factor_map_is_exact_2x2_and_physical_family_is_off():
    text=(ROOT/'scripts/run_v48_55_dcp_drfc_bcde_tcbc_arm.sh').read_text()
    assert 'A: X=0 Y=0' in text and 'B: X=1 Y=0' in text and 'C: X=0 Y=1' in text and 'D: X=1 Y=1' in text
    assert 'FACTOR_COMPONENT_MARGIN_REGRESSION_RELIABILITY="0,1,1,0,0"' in text
    assert 'FACTOR_COMPONENT_MARGIN_TARGET_MODE=pooled_rms_linear' in text
    assert 'V4854_INVARIANT_PHYSICAL_BOUNDARY_DISTILLATION=false' in text
    assert 'V4852_PHYSICAL_TEACHER_SIGN_ALIGNMENT=false' in text
    assert 'V4853_PHYSICAL_STUDENT_SIGN_ALIGNMENT=false' in text
    assert "'strategy_regime_conditioning':False" in text


def test_v4855_launcher_semantically_reuses_A_and_runs_BC_before_D():
    text=(ROOT/'scripts/run_v48_55_dcp_drfc_bcde_tcbc_two_gpu.sh').read_text()
    assert 'check_v48_55_reference_reuse.py' in text
    assert 'run_arm B "$B_RUN" "$GPU0" "$GPU0" 1 & pb=$!' in text
    assert 'run_arm C "$C_RUN" "$GPU1" "$GPU1" 1 & pc=$!' in text
    assert 'run_arm D "$D_RUN" "$GPU0" "$GPU1" 0' in text
    assert 'compare_v48_55_dcp_drfc_bcde_tcbc_2x2.py' in text


def test_v4855_dedicated_contract_runs_before_certificate_controller():
    text=(ROOT/'scripts/run_v48_36_ocaf_dedicated.sh').read_text()
    assert text.index('check_v48_55_tcbc_contract.py') < text.index('calibrate_v48_36_shared_certificate_pool.sh')
    assert 'compute_v48_55_component_boundary_scales.py' in text
