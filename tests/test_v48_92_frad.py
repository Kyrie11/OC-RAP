from __future__ import annotations
from pathlib import Path
import math

from ocrap.v48_92_factorized_recovery_advantage import (
    FACTOR_NAMES,
    factorize_recovery_advantage,
    pcd_factors,
    pcd_from_factors,
    shapley_product_delta,
)

ROOT=Path(__file__).resolve().parents[1]


def test_factorization_exactly_reconstructs_pcd_advantage():
    f=factorize_recovery_advantage(candidate_drs=.75,nominal_drs=.4,candidate_r_dep=.6,nominal_r_dep=-.2,candidate_gap=.15,nominal_gap=.55)
    assert f.shapley_sum_error < 1e-12
    assert abs((f.shapley_drs+f.shapley_deployability_gate+f.shapley_gap_discount)-f.teacher_adv_reconstructed)<1e-12


def test_shapley_is_zero_when_factor_does_not_change():
    nom=pcd_factors(drs=.5,r_dep=.2,gap=.3)
    cand=dict(nom);cand['drs']=.8
    phi=shapley_product_delta(nom,cand)
    assert phi['drs']>0
    assert abs(phi['deployability_gate'])<1e-15
    assert abs(phi['gap_discount'])<1e-15


def test_candidate_gap_reduction_has_positive_gap_contribution():
    f=factorize_recovery_advantage(candidate_drs=.6,nominal_drs=.6,candidate_r_dep=.1,nominal_r_dep=.1,candidate_gap=.1,nominal_gap=.8)
    assert f.shapley_gap_discount>0
    assert abs(f.shapley_drs)<1e-15
    assert abs(f.shapley_deployability_gate)<1e-15


def test_runner_is_audit_only_and_reuses_v4891_sidecar():
    text=(ROOT/'scripts/run_v48_92_dcp_drfc_bcde_rifa_frad.sh').read_text()
    assert 'V91_SIDECAR' in text
    assert 'build_v48_92_factorized_recovery_advantage_audit.py' in text
    assert 'build_v48_91_common_exogenous_physical_sidecar.py' not in text
    assert 'V4891_WOMD_SOURCE' not in text
    assert 'train.py' not in text


def test_runtime_contract_freezes_forbidden_families():
    text=(ROOT/'tools/check_v48_92_runtime_code_contract.py').read_text()
    for token in ('boundary_transport_off','regime_conditioning_off','capacity_sweep_off','raw_womd_replay_disabled','audit_only_zero_planner_parameters'):
        assert token in text


def test_l80_component_label_union_is_variant_identity_checked(tmp_path):
    import json
    from tools.build_v48_92_factorized_recovery_advantage_audit import _component_labels
    from tools.build_v48_89_root_correspondence_audit import ROLE_FILES
    row={
        'scene':'scene','time':1,'candidate':2,'macro':5,'teacher_harmful':False,
        'teacher_adv':0.1,'teacher_candidate_drs':0.8,'teacher_nominal_drs':0.6,
        'teacher_candidate_r_dep':0.4,'teacher_nominal_r_dep':0.2,
        'teacher_candidate_gap':0.1,'teacher_nominal_gap':0.3,
        'teacher_candidate_hard':0.0,'teacher_nominal_hard':0.0,
        'teacher_candidate_harm_proxy':0.0,'teacher_nominal_harm_proxy':0.0,
    }
    for variant in ('balanced','precision'):
        d=tmp_path/'candidates'/variant/'calibration';d.mkdir(parents=True,exist_ok=True)
        for filename in ROLE_FILES.values():
            (d/filename).write_text(json.dumps(row)+'\n')
    labels,identity=_component_labels(tmp_path)
    assert all(len(v)==1 for v in labels.values())
    assert all(v['component_identity_on_overlap'] for v in identity.values())
