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
    ENGINEERING_VERSION,
    MATCHED_DIM,
    SCIENTIFIC_VERSION,
    TAIL_GEOMETRY_DIM,
    contract_checks,
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


def test_current_wrcf_contract_checks():
    checks = contract_checks()
    assert checks and all(checks.values()), checks
    assert TAIL_GEOMETRY_DIM == 64
    assert MATCHED_DIM == 220
    assert ENGINEERING_VERSION == "v48.116.0-OC-WRCF"
    assert SCIENTIFIC_VERSION == "v48.116-OC-WRCF"


def test_ridge_owner_is_still_unique_and_closed_form():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 12))
    y = np.asarray([0, 1] * 32)
    model = fit_closed_form_ridge(x, y)
    assert model.coef.shape == (12,)
    assert model.ridge_lambda > 0
    assert model.normal_equation_residual <= 1e-7


def test_current_launcher_reuses_authoritative_v115_and_keeps_command_name():
    repo = Path(__file__).resolve().parents[1]
    launcher = (repo / "scripts/run_constraint_native_orientation_audit.sh").read_text()
    assert "OC-RAP-v48.115-PIPELINE_COMPLETE.json" in launcher
    assert "OC-RAP-v48.115-DCP-DRFC-BCDE-RIFA-OC-RSCF-comparison.json" in launcher
    assert "OC-RAP-v48.115-RSCF-balanced.json" in launcher
    assert "OC-RAP-v48.115-RSCF-precision.json" in launcher
    assert "OC-RAP-v48.93-factor-mediation-audit.jsonl" in launcher
    assert "OC-RAP-v48.116-WRCF-balanced.json" in launcher
    assert "OC-RAP-v48.116-OC-WRCF-results.zip" in launcher
    assert "weak-root-conditioned recovery-set flow branch" in launcher
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


def test_current_result_packager_uses_only_canonical_v116_artifacts():
    tool = Path(__file__).resolve().parents[1] / "tools" / "package_constraint_native_orientation_results.py"
    spec = importlib.util.spec_from_file_location("orientation_packager_test", tool)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.EXPECTED["balanced"] == "OC-RAP-v48.116-WRCF-balanced.json"
    assert mod.EXPECTED["precision"] == "OC-RAP-v48.116-WRCF-precision.json"
    assert mod.ENGINEERING_VERSION == "v48.116.0-OC-WRCF"
    assert mod.SCIENTIFIC_VERSION == "v48.116-OC-WRCF"


def test_v116_comparison_preregisters_weak_root_tail_flow():
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "tools/compare_constraint_native_recovery_orientation.py").read_text()
    assert '"reentry_contact_coverage_go"' in text
    assert "tail_integral_vs_v48_115_uniform_integral_support_gate" in text
    assert "tail_work_vs_v48_115_uniform_work_support_gate" in text
    assert "tail_work_vs_tail_integral_support_gate" in text
    assert "WEAK_ROOT_RECOVERY_SET_FLOW_GO" in text
    assert "nominal_ocmero_tail_weighted" in text
