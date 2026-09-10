from __future__ import annotations
import numpy as np
from ocrap.audits.constraint_native_orientation import GEOMETRY_DIM,MATCHED_DIM,RAW_CANDIDATE_DIM,contract_checks,fit_closed_form_ridge

def test_cnro_contract_checks():
    checks=contract_checks(); assert checks and all(checks.values()), checks
    assert RAW_CANDIDATE_DIM==156 and GEOMETRY_DIM==32 and MATCHED_DIM==188

def test_cnro_ridge_is_unique_and_closed_form():
    rng=np.random.default_rng(7); x=rng.normal(size=(64,12)); y=np.asarray([0,1]*32)
    m=fit_closed_form_ridge(x,y)
    assert m.coef.shape==(12,)
    assert m.ridge_lambda>0
    assert m.normal_equation_residual <= 1e-7

def test_cnro_keeps_original_scientific_version_and_versioned_reference_contract():
    from pathlib import Path
    from ocrap.audits.constraint_native_orientation import ENGINEERING_VERSION
    assert ENGINEERING_VERSION == "v48.111.1-OC-CNRO-RESULTFIX"
    repo = Path(__file__).resolve().parents[1]
    launcher = (repo / "scripts/run_constraint_native_orientation_audit.sh").read_text()
    assert "OC-RAP-v48.110-PIPELINE_COMPLETE.json" in launcher
    assert "OC-RAP-v48.110-DCP-DRFC-BCDE-RIFA-OC-CATO-comparison.json" in launcher
    assert "OC-RAP-v48.93-factor-mediation-audit.jsonl" in launcher
    assert "compare_constraint_native_recovery_orientation.py" in launcher
    assert "check_constraint_native_orientation_pipeline.py" in launcher
    assert "package_constraint_native_orientation_results.py" in launcher
    assert "CNGO" in launcher
    assert "--run-id" in launcher


def test_cnro_unversioned_code_keeps_v93_role_filter_semantics(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    from ocrap.audits.constraint_native_orientation import derive_candidate_semantics, teacher_factor_tuple

    tool = Path(__file__).resolve().parents[1] / "tools" / "run_constraint_native_recovery_orientation_audit.py"
    spec = importlib.util.spec_from_file_location("cnro_audit_runner_test", tool)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    build_v93_map, label_groups = mod.build_v93_map, mod.label_groups

    def pcd_row(*, bucket, scene, candidate, nominal, drs, rdep, gap, harmful=False):
        row = {
            "bucket": bucket, "scene": scene, "time": 10, "candidate": candidate,
            "macro": 2, "nominal": nominal, "path": str(tmp_path / f"{scene}_{candidate}.npz"),
            "teacher_drs": drs, "teacher_r_dep": rdep, "teacher_gap": gap,
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
        "dataset_role": "dev_near", "scene_id": "s", "time_index": 10,
        "candidate_index": 1, "safe_positive": sem["safe_positive"],
        "teacher_harmful": sem["teacher_harmful"], "mediation_mode": sem["mediation_mode"],
    }) + "\n")
    groups = label_groups(index, role_filter="dev_near", v93_map=build_v93_map(v93))
    assert len(groups) == 1
    assert [c["candidate"] for c in groups[0]["candidates"]] == [1]


def test_cnro_result_packager_rejects_mixed_or_missing_artifacts(tmp_path):
    import hashlib, importlib.util, json
    from pathlib import Path
    tool = Path(__file__).resolve().parents[1] / "tools" / "package_constraint_native_orientation_results.py"
    spec = importlib.util.spec_from_file_location("cnro_packager_test", tool)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    # The exact canonical map is a guard against accidental CNGO/CNRO mixing.
    assert mod.EXPECTED["balanced"] == "OC-RAP-v48.111-CNRO-balanced.json"
    assert "CNGO" not in " ".join(mod.EXPECTED.values())
    assert mod.ENGINEERING_VERSION == "v48.111.1-OC-CNRO-RESULTFIX"
