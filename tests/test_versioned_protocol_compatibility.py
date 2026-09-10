from __future__ import annotations

from pathlib import Path


def test_code_filenames_are_unversioned_but_current_outputs_keep_versioned_protocols():
    repo = Path(__file__).resolve().parents[1]
    for root_name in ("src", "tools", "tests", "scripts"):
        for p in (repo / root_name).rglob("*"):
            if not p.is_file():
                continue
            name = p.name.lower()
            assert not (name.endswith((".py", ".sh")) and ("v48_" in name or "v48." in name)), p
    eval_text = (repo / "scripts" / "run_ocrap_evaluation.sh").read_text(encoding="utf-8")
    assert "ocrap_v48_111_submission_three_regime" in eval_text
    assert "V48.111-DEPLOYABLE-STACK" in eval_text
    assert "V48.111-SUBMISSION-THREE-REGIME-SUMMARY" in eval_text
    assert "SAFE_WOMD=auto NEAR_WOMD=auto CONTACT_WOMD=auto" in eval_text
    baseline = (repo / "scripts" / "run_external_baselines.sh").read_text(encoding="utf-8")
    assert "export CL_WOMD=auto CALIB_WOMD=auto" in baseline
    checker = (repo / "tools" / "check_deployable_stack.py").read_text(encoding="utf-8")
    assert '"schema": "ocrap-v48.111-deployable-stack-contract-v1"' in checker
    assert '"engineering_version": "v48.111-deployed-evidence-frozen-1"' in checker


def test_signed_viability_semantic_module_preserves_historical_schema_env_contract(monkeypatch):
    import ocrap.signed_viability as sv

    monkeypatch.delenv("OCRAP_V48_74_SIGNED_VIABILITY", raising=False)
    assert sv.enabled() is False
    monkeypatch.setenv("OCRAP_V48_74_SIGNED_VIABILITY", "1")
    assert sv.enabled() is True
    frag = sv.runtime_contract_fragment()
    assert frag["engineering_version"] == "v48.74.2-OC-SVBW-ENGFIX"
    assert frag["schema"] == 10
    assert frag["feature_dim"] == 22
    assert frag["source"] == "signed_finite_time_viability_projected_recovery_witness"
