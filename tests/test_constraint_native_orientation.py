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
    ENGINEERING_VERSION,
    MATCHED_DIM,
    SCIENTIFIC_VERSION,
    contract_checks,
)


def test_legacy_cnro_primitive_contract_remains_available():
    checks = legacy_contract_checks()
    assert checks and all(checks.values()), checks
    assert RAW_CANDIDATE_DIM == 156
    assert GEOMETRY_DIM == 32
    assert LEGACY_MATCHED_DIM == 188


def test_current_hcnc_contract_checks():
    checks = contract_checks()
    assert checks and all(checks.values()), checks
    assert CONE_GEOMETRY_DIM == 64
    assert MATCHED_DIM == 220
    assert ENGINEERING_VERSION == "v48.112.0-OC-HCNC"
    assert SCIENTIFIC_VERSION == "v48.112-OC-HCNC"


def test_ridge_owner_is_still_unique_and_closed_form():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 12))
    y = np.asarray([0, 1] * 32)
    model = fit_closed_form_ridge(x, y)
    assert model.coef.shape == (12,)
    assert model.ridge_lambda > 0
    assert model.normal_equation_residual <= 1e-7


def test_current_launcher_reuses_authoritative_v111_and_keeps_command_name():
    repo = Path(__file__).resolve().parents[1]
    launcher = (repo / "scripts/run_constraint_native_orientation_audit.sh").read_text()
    assert "OC-RAP-v48.111-PIPELINE_COMPLETE.json" in launcher
    assert "OC-RAP-v48.111-DCP-DRFC-BCDE-RIFA-OC-CNRO-comparison.json" in launcher
    assert "OC-RAP-v48.111-CNRO-balanced.json" in launcher
    assert "OC-RAP-v48.111-CNRO-precision.json" in launcher
    assert "OC-RAP-v48.93-factor-mediation-audit.jsonl" in launcher
    assert "OC-RAP-v48.112-HCNC-balanced.json" in launcher
    assert "OC-RAP-v48.112-OC-HCNC-results.zip" in launcher
    assert "CNGO" not in launcher
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


def test_current_result_packager_uses_only_canonical_v112_artifacts():
    tool = Path(__file__).resolve().parents[1] / "tools" / "package_constraint_native_orientation_results.py"
    spec = importlib.util.spec_from_file_location("orientation_packager_test", tool)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.EXPECTED["balanced"] == "OC-RAP-v48.112-HCNC-balanced.json"
    assert mod.EXPECTED["precision"] == "OC-RAP-v48.112-HCNC-precision.json"
    assert mod.ENGINEERING_VERSION == "v48.112.0-OC-HCNC"
    assert mod.SCIENTIFIC_VERSION == "v48.112-OC-HCNC"
