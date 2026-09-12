from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from ocrap.audits.constraint_native_orientation import (
    GEOMETRY_DIM,
    MATCHED_DIM as LEGACY_MATCHED_DIM,
    RAW_CANDIDATE_DIM,
    contract_checks as legacy_contract_checks,
    derive_candidate_semantics,
    fit_closed_form_ridge,
    teacher_factor_tuple,
)
from ocrap.audits.heterogeneous_constraint_normal_cone import (
    CONE_GEOMETRY_DIM,
    MATCHED_DIM as HCNC_MATCHED_DIM,
    contract_checks as hcnc_contract_checks,
)
from ocrap.audits.executable_constraint_jacobian import (
    ENGINEERING_VERSION as ECJ_ENGINEERING_VERSION,
    JACOBIAN_GEOMETRY_DIM,
    MATCHED_DIM as ECJ_MATCHED_DIM,
    SCIENTIFIC_VERSION as ECJ_SCIENTIFIC_VERSION,
    contract_checks as ecj_contract_checks,
)
from ocrap.audits.common_option_constraint_work import (
    ENGINEERING_VERSION as CCW_ENGINEERING_VERSION,
    MATCHED_DIM as CCW_MATCHED_DIM,
    SCIENTIFIC_VERSION as CCW_SCIENTIFIC_VERSION,
    WORK_GEOMETRY_DIM,
    contract_checks as ccw_contract_checks,
)
from ocrap.audits.recovery_set_constraint_flow import (
    ENGINEERING_VERSION as RSCF_ENGINEERING_VERSION,
    MATCHED_DIM as RSCF_MATCHED_DIM,
    SCIENTIFIC_VERSION as RSCF_SCIENTIFIC_VERSION,
    SET_GEOMETRY_DIM,
    contract_checks as rscf_contract_checks,
)
from ocrap.audits.weak_root_recovery_set_flow import (
    ENGINEERING_VERSION as WRCF_ENGINEERING_VERSION,
    MATCHED_DIM as WRCF_MATCHED_DIM,
    SCIENTIFIC_VERSION as WRCF_SCIENTIFIC_VERSION,
    TAIL_GEOMETRY_DIM,
    contract_checks as wrcf_contract_checks,
)
from ocrap.audits.tail_boundary_crossing_flow import (
    BOUNDARY_GEOMETRY_DIM,
    ENGINEERING_VERSION as TBCF_ENGINEERING_VERSION,
    MATCHED_DIM as TBCF_MATCHED_DIM,
    SCIENTIFIC_VERSION as TBCF_SCIENTIFIC_VERSION,
    contract_checks as tbcf_contract_checks,
)
from ocrap.audits.viability_survival_envelope import (
    ENVELOPE_GEOMETRY_DIM,
    ENGINEERING_VERSION as VSE_ENGINEERING_VERSION,
    MATCHED_DIM as VSE_MATCHED_DIM,
    SCIENTIFIC_VERSION as VSE_SCIENTIFIC_VERSION,
    contract_checks as vse_contract_checks,
)
from ocrap.audits.viability_order_profile import (
    PROFILE_GEOMETRY_DIM,
    ENGINEERING_VERSION as VOP_ENGINEERING_VERSION,
    MATCHED_DIM as VOP_MATCHED_DIM,
    SCIENTIFIC_VERSION as VOP_SCIENTIFIC_VERSION,
    ORDER_MASSES,
    contract_checks as vop_contract_checks,
)
from ocrap.audits.viability_rank_transport import (
    ENGINEERING_VERSION as VRT_ENGINEERING_VERSION,
    MATCHED_DIM as VRT_MATCHED_DIM,
    SCIENTIFIC_VERSION as VRT_SCIENTIFIC_VERSION,
    TRANSPORT_GEOMETRY_DIM,
    TRANSPORT_MODE_DEGREES,
    contract_checks as vrt_contract_checks,
)
from ocrap.audits.viability_rank_persistence import (
    ENGINEERING_VERSION as VRPC_ENGINEERING_VERSION,
    MATCHED_DIM as VRPC_MATCHED_DIM,
    SCIENTIFIC_VERSION as VRPC_SCIENTIFIC_VERSION,
    COUPLING_GEOMETRY_DIM,
    COUPLING_MODE_NAMES,
    contract_checks as vrpc_contract_checks,
)
from ocrap.audits.signed_viability_rank_state import (
    ENGINEERING_VERSION, MATCHED_DIM, SCIENTIFIC_VERSION,
    STATE_GEOMETRY_DIM, STATE_MODE_NAMES, contract_checks,
)


def test_legacy_cnro_primitive_contract_remains_available():
    checks = legacy_contract_checks()
    assert checks and all(checks.values()), checks
    assert RAW_CANDIDATE_DIM == 156
    assert GEOMETRY_DIM == 32
    assert LEGACY_MATCHED_DIM == 188


def test_hcnc_prerequisite_primitive_remains_available():
    checks = hcnc_contract_checks()
    assert checks and all(checks.values()), checks
    assert CONE_GEOMETRY_DIM == 64
    assert HCNC_MATCHED_DIM == 220


def test_ecj_prerequisite_contract_checks():
    checks = ecj_contract_checks()
    assert checks and all(checks.values()), checks
    assert JACOBIAN_GEOMETRY_DIM == 64
    assert ECJ_MATCHED_DIM == 220
    assert ECJ_ENGINEERING_VERSION == "v48.113.0-OC-ECJ"
    assert ECJ_SCIENTIFIC_VERSION == "v48.113-OC-ECJ"


def test_historical_ccw_contract_checks():
    checks = ccw_contract_checks()
    assert checks and all(checks.values()), checks
    assert WORK_GEOMETRY_DIM == 64
    assert CCW_MATCHED_DIM == 220
    assert CCW_ENGINEERING_VERSION == "v48.114.0-OC-CCW"
    assert CCW_SCIENTIFIC_VERSION == "v48.114-OC-CCW"


def test_historical_rscf_contract_checks():
    checks = rscf_contract_checks()
    assert checks and all(checks.values()), checks
    assert SET_GEOMETRY_DIM == 64
    assert RSCF_MATCHED_DIM == 220
    assert RSCF_ENGINEERING_VERSION == "v48.115.0-OC-RSCF"
    assert RSCF_SCIENTIFIC_VERSION == "v48.115-OC-RSCF"


def test_historical_wrcf_contract_checks():
    checks = wrcf_contract_checks()
    assert checks and all(checks.values()), checks
    assert TAIL_GEOMETRY_DIM == 64
    assert WRCF_MATCHED_DIM == 220
    assert WRCF_ENGINEERING_VERSION == "v48.116.1-OC-WRCF"
    assert WRCF_SCIENTIFIC_VERSION == "v48.116-OC-WRCF"


def test_historical_tbcf_contract_checks():
    checks = tbcf_contract_checks()
    assert checks and all(checks.values()), checks
    assert BOUNDARY_GEOMETRY_DIM == 64
    assert TBCF_MATCHED_DIM == 220
    assert TBCF_ENGINEERING_VERSION == "v48.117.0-OC-TBCF"
    assert TBCF_SCIENTIFIC_VERSION == "v48.117-OC-TBCF"


def test_historical_vse_contract_checks():
    checks = vse_contract_checks()
    assert checks and all(checks.values()), checks
    assert ENVELOPE_GEOMETRY_DIM == 64
    assert VSE_MATCHED_DIM == 220
    assert VSE_ENGINEERING_VERSION == "v48.118.0-OC-VSE"
    assert VSE_SCIENTIFIC_VERSION == "v48.118-OC-VSE"

def test_historical_vop_contract_checks():
    checks = vop_contract_checks()
    assert checks and all(checks.values()), checks
    assert PROFILE_GEOMETRY_DIM == 64
    assert VOP_MATCHED_DIM == 220
    assert VOP_ENGINEERING_VERSION == "v48.119.0-OC-VOP"
    assert VOP_SCIENTIFIC_VERSION == "v48.119-OC-VOP"
    assert np.array_equal(ORDER_MASSES, np.asarray([0.25, 0.5, 0.75, 1.0]))


def test_historical_vrt_contract_checks():
    checks = vrt_contract_checks()
    assert checks and all(checks.values()), checks
    assert TRANSPORT_GEOMETRY_DIM == 64
    assert VRT_MATCHED_DIM == 220
    assert VRT_ENGINEERING_VERSION == "v48.120.0-OC-VRT"
    assert VRT_SCIENTIFIC_VERSION == "v48.120-OC-VRT"
    assert TRANSPORT_MODE_DEGREES == (0, 1, 2, 3)

def test_historical_vrpc_contract_checks():
    checks = vrpc_contract_checks()
    assert checks and all(checks.values()), checks
    assert COUPLING_GEOMETRY_DIM == 64
    assert VRPC_MATCHED_DIM == 220
    assert VRPC_ENGINEERING_VERSION == "v48.121.0-OC-VRPC"
    assert VRPC_SCIENTIFIC_VERSION == "v48.121-OC-VRPC"
    assert COUPLING_MODE_NAMES == ("global_shift", "nominal_rank_tilt", "rank_persistence_defect", "rank_persistence_interaction")


def test_current_svrt_contract_checks():
    checks = contract_checks()
    assert checks and all(checks.values()), checks
    assert STATE_GEOMETRY_DIM == 64
    assert MATCHED_DIM == 220
    assert ENGINEERING_VERSION == "v48.122.0-OC-SVRT"
    assert SCIENTIFIC_VERSION == "v48.122-OC-SVRT"
    assert STATE_MODE_NAMES == (
        "global_shift", "nominal_rank_tilt", "signed_nominal_state_coupling", "rank_signed_nominal_state_interaction"
    )


def test_ridge_owner_is_still_unique_and_closed_form():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 12))
    y = np.asarray([0, 1] * 32)
    model = fit_closed_form_ridge(x, y)
    assert model.coef.shape == (12,)
    assert model.ridge_lambda > 0
    assert model.normal_equation_residual <= 1e-7


def test_current_launcher_reuses_authoritative_v121_and_keeps_command_name():
    repo = Path(__file__).resolve().parents[1]
    launcher = (repo / "scripts/run_constraint_native_orientation_audit.sh").read_text()
    assert "OC-RAP-v48.121-PIPELINE_COMPLETE.json" in launcher
    assert "OC-RAP-v48.121-DCP-DRFC-BCDE-RIFA-OC-VRPC-comparison.json" in launcher
    assert "OC-RAP-v48.121-VRPC-balanced.json" in launcher
    assert "OC-RAP-v48.121-VRPC-precision.json" in launcher
    assert "OC-RAP-v48.93-factor-mediation-audit.jsonl" in launcher
    assert "OC-RAP-v48.122-SVRT-balanced.json" in launcher
    assert "OC-RAP-v48.122-OC-SVRT-results.zip" in launcher
    assert "signed nominal viability" in launcher.lower()
    assert "close_nominal_rank_persistence_coupling" in launcher
    assert "--run-id" in launcher

def test_unversioned_runner_keeps_v93_role_filter_semantics(tmp_path):
    tool = Path(__file__).resolve().parents[1] / "tools" / "run_constraint_native_recovery_orientation_audit.py"
    spec = importlib.util.spec_from_file_location("orientation_audit_runner_test", tool)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    build_v93_map, label_groups = mod.build_v93_map, mod.label_groups

    def pcd_row(*, bucket, scene, candidate, nominal, drs, rdep, gap, harmful=False):
        row = {
            "bucket": bucket,
            "scene": scene,
            "time": 10,
            "candidate": candidate,
            "macro": 2,
            "nominal": nominal,
            "path": str(tmp_path / f"{scene}_{candidate}.npz"),
            "teacher_drs": drs,
            "teacher_r_dep": rdep,
            "teacher_gap": gap,
            "component_harmful": harmful,
        }
        factors = teacher_factor_tuple(row)
        row["teacher_pcd"] = factors["drs"] * factors["deployability_gate"] * factors["gap_discount"]
        return row

    nominal = pcd_row(bucket=1, scene="s", candidate=0, nominal=True, drs=.2, rdep=-.5, gap=.2)
    keep = pcd_row(bucket=1, scene="s", candidate=1, nominal=False, drs=.7, rdep=.5, gap=.1)
    omit = pcd_row(bucket=1, scene="s", candidate=2, nominal=False, drs=.1, rdep=-1., gap=.7, harmful=True)
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in (nominal, keep, omit)))
    sem = derive_candidate_semantics(nominal, keep)
    v93 = tmp_path / "v93.jsonl"
    v93.write_text(json.dumps({
        "dataset_role": "dev_near",
        "scene_id": "s",
        "time_index": 10,
        "candidate_index": 1,
        "safe_positive": sem["safe_positive"],
        "teacher_harmful": sem["teacher_harmful"],
        "mediation_mode": sem["mediation_mode"],
    }) + "\n")
    groups = label_groups(index, role_filter="dev_near", v93_map=build_v93_map(v93))
    assert len(groups) == 1
    assert [c["candidate"] for c in groups[0]["candidates"]] == [1]


def test_current_result_packager_uses_only_canonical_v122_artifacts():
    tool = Path(__file__).resolve().parents[1] / "tools" / "package_constraint_native_orientation_results.py"
    spec = importlib.util.spec_from_file_location("orientation_packager_test", tool)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.EXPECTED["balanced"] == "OC-RAP-v48.122-SVRT-balanced.json"
    assert mod.EXPECTED["precision"] == "OC-RAP-v48.122-SVRT-precision.json"
    assert mod.ENGINEERING_VERSION == "v48.122.0-OC-SVRT"
    assert mod.SCIENTIFIC_VERSION == "v48.122-OC-SVRT"


def test_v122_comparison_preregisters_signed_rank_state_transport():
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "tools/compare_constraint_native_recovery_orientation.py").read_text()
    assert '"reentry_contact_coverage_go"' in text
    assert "full_signed_state_minus_v121_full_persistence" in text
    assert "exposed_signed_state_minus_v121_exposed_persistence" in text
    assert "full_set_signed_state_effect" in text
    assert "SIGNED_VIABILITY_RANK_STATE_TRANSPORT_GO" in text
    assert "SIGNED_VIABILITY_RANK_STATE_TRANSPORT_STOP" in text
    assert "close_signed_viability_rank_state_transport" in text


def _vse_field(full_values: np.ndarray):
    from ocrap.audits.executable_constraint_jacobian import ExecutableConstraintField
    arr = np.asarray(full_values, dtype=np.float64)
    L, T, C = arr.shape
    masks = np.ones_like(arr, dtype=bool)
    return ExecutableConstraintField(
        values=arr.copy(), masks=masks.copy(), full_values=arr.copy(), full_masks=masks.copy(),
        option_valid=np.ones(L, dtype=bool), option_scores=np.zeros(L, dtype=np.float64),
        option_modes=tuple(f"m{i}" for i in range(L)), diagnostics={},
    )


def test_vse_joint_envelope_rejects_cross_option_frankenstein_feasibility():
    from ocrap.audits.viability_survival_envelope import _field_envelope
    # At every time option 0 fails constraint 1 and option 1 fails constraint 0.
    # A per-constraint max would look safe, but no same recovery option is jointly safe.
    x = np.ones((2, 4, 4), dtype=np.float64)
    x[0, :, 1] = -0.25
    x[1, :, 0] = -0.40
    f = _vse_field(x)
    contrib, diag = _field_envelope(f, f.full_masks, np.ones(2, dtype=bool), reverse=False)
    scalar = np.asarray(diag["scalar_envelope"])
    assert np.all(scalar < 0.0)
    assert np.allclose(contrib.sum(axis=1), scalar, atol=1e-12, rtol=0.0)


def test_vse_prefix_envelope_encodes_first_joint_viability_loss():
    from ocrap.audits.viability_survival_envelope import _field_envelope
    x = np.ones((2, 5, 4), dtype=np.float64)
    x[0, 2:, 0] = -0.2
    x[1, 4:, 1] = -0.3
    f = _vse_field(x)
    _, diag = _field_envelope(f, f.full_masks, np.ones(2, dtype=bool), reverse=False)
    scalar = np.asarray(diag["scalar_envelope"])
    assert np.all(scalar[:4] > 0.0)  # option 1 keeps one full prefix viable
    assert scalar[4] < 0.0          # by t=4 every option has violated some constraint


def test_vse_suffix_envelope_encodes_persistent_safe_reentry():
    from ocrap.audits.viability_survival_envelope import _field_envelope
    x = np.ones((2, 5, 4), dtype=np.float64)
    x[0, :3, 0] = -0.5
    x[1, :2, 1] = -0.4
    f = _vse_field(x)
    _, diag = _field_envelope(f, f.full_masks, np.ones(2, dtype=bool), reverse=True)
    scalar = np.asarray(diag["scalar_envelope"])
    assert scalar[0] < 0.0
    assert scalar[1] < 0.0
    assert np.all(scalar[2:] > 0.0)  # option 1 has a persistent-safe suffix from t=2
