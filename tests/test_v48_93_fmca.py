from __future__ import annotations
from pathlib import Path

from ocrap.v48_93_factor_mediation import adjudicate_factor_mediation, exact_factor_counterfactuals

ROOT = Path(__file__).resolve().parents[1]


def test_drs_activation_mode_is_exact():
    x = adjudicate_factor_mediation(
        nominal_drs=0.0, candidate_drs=1.0,
        nominal_deployability_gate=0.3, candidate_deployability_gate=0.6,
        nominal_gap_discount=1.0, candidate_gap_discount=1.0,
    )
    assert x.mediation_mode == "drs_activation"
    assert x.necessary_drs and x.sufficient_drs
    assert not x.necessary_deployability_gate


def test_deployability_gain_mode_is_exact():
    x = adjudicate_factor_mediation(
        nominal_drs=1.0, candidate_drs=1.0,
        nominal_deployability_gate=0.3, candidate_deployability_gate=0.6,
        nominal_gap_discount=1.0, candidate_gap_discount=1.0,
    )
    assert x.mediation_mode == "deployability_gain"
    assert x.necessary_deployability_gate and x.sufficient_deployability_gate
    assert not x.necessary_drs


def test_gap_knockout_is_interpretable():
    x = adjudicate_factor_mediation(
        nominal_drs=1.0, candidate_drs=1.0,
        nominal_deployability_gate=0.5, candidate_deployability_gate=0.5,
        nominal_gap_discount=0.4, candidate_gap_discount=1.0,
    )
    assert x.mediation_mode == "gap_gain"
    assert x.necessary_gap_discount and x.sufficient_gap_discount


def test_counterfactual_identity_uses_exact_pcd_product():
    n = {"drs": 0.4, "deployability_gate": 0.3, "gap_discount": 0.8}
    c = {"drs": 0.9, "deployability_gate": 0.6, "gap_discount": 1.0}
    x = exact_factor_counterfactuals(n, c)
    assert abs(float(x["full_advantage"]) - (0.9 * 0.6 * 1.0 - 0.4 * 0.3 * 0.8)) < 1e-12


def test_runner_is_audit_only_and_reuses_v4892():
    text = (ROOT / "scripts/run_v48_93_dcp_drfc_bcde_rifa_fmca.sh").read_text()
    assert "V92_AUDIT" in text
    assert "build_v48_93_factor_mediation_audit.py" in text
    assert "V4891_WOMD_SOURCE" not in text
    assert "train.py" not in text


def test_runtime_contract_freezes_forbidden_families():
    text = (ROOT / "tools/check_v48_93_runtime_code_contract.py").read_text()
    for token in (
        "boundary_transport_off", "regime_conditioning_off", "capacity_sweep_off",
        "raw_womd_replay_disabled", "audit_only_zero_planner_parameters",
    ):
        assert token in text
