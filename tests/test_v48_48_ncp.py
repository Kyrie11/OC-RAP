from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v4848_factor_matrix_is_clean_2x2_and_no_dwok() -> None:
    text = (ROOT / "scripts/run_v48_48_ncp_ablation_arm.sh").read_text(encoding="utf-8")
    assert "V4847_DECISION_OBS=0" in text
    assert "TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class" in text
    assert "EVAL_OPTION_EXECUTION_SEMANTICS=observation_class" in text
    assert 'B)\n    export OCRAP_ALGORITHM_VERSION="v48.48-NCP-DRFC-ablation-B"' in text
    assert "export V4848_NATIVE_CERTIFICATE=1" in text
    assert 'C)\n    export OCRAP_ALGORITHM_VERSION="v48.48-NCP-DRFC-ablation-C"' in text
    assert "export V4847_RECOVERY_FRONTIER=1" in text
    assert 'D)\n    export OCRAP_ALGORITHM_VERSION="v48.48-NCP-DRFC"' in text
    assert "regime_conditioning':False" in text or "strategy_regime_conditioning':False" in text


def test_v4848_native_flag_roundtrips_through_train_and_inference_plumbing() -> None:
    train_sh = (ROOT / "scripts/train_ocrap_v48_trac_sr.sh").read_text(encoding="utf-8")
    train_py = (ROOT / "src/ocrap/cli/train.py").read_text(encoding="utf-8")
    infer_py = (ROOT / "src/ocrap/models/inference.py").read_text(encoding="utf-8")
    contract = (ROOT / "tools/check_v48_36_ocaf_model_contract.py").read_text(encoding="utf-8")
    for key in (
        "direct_recovery_evidence_native_certificate_preservation",
        "direct_recovery_evidence_native_drs_tolerance",
        "direct_recovery_evidence_native_deployability_tolerance",
    ):
        assert key in train_py
        assert key in infer_py
    assert "EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION" in train_sh
    assert "expect-native-certificate-preservation" in contract


def test_v4848_witness_stage_cannot_inherit_native_downstream_path() -> None:
    text = (ROOT / "scripts/adapt_ocrap_v48_47_dsofr_witness_stage.sh").read_text(encoding="utf-8")
    assert "EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=false" in text


def test_v4848_launcher_uses_two_arm_parallelism_but_serial_variants_by_default() -> None:
    text = (ROOT / "scripts/run_v48_48_ncp_2x2_two_gpu.sh").read_text(encoding="utf-8")
    assert 'for pair in "A B" "C D"' in text
    assert 'V4848_VARIANT_MODE:-serial' in text
    assert 'default_min_free_mb=12000' in text and 'default_min_free_mb=20000' in text
    assert "wait_for_gpu_lease" in text
    assert "nvidia-smi --id=\"$gpu\" --query-gpu=memory.free" in text
    assert "V48_48_GPU_SCHEDULER_DECISION.json" in text
    assert "requires two distinct GPU ids" in text
    # The arm launcher receives the same physical GPU for Balanced/Precision;
    # serial mode means only one variant process occupies it at a time.
    assert 'GPU0="$gpu" GPU1="$gpu" SERIAL_VARIANTS_ON_ONE_GPU="$SERIAL_VARIANTS"' in text


def test_v4848_comparator_requires_all_four_pipeline_valid_arms() -> None:
    text = (ROOT / "tools/compare_v48_48_ncp_2x2.py").read_text(encoding="utf-8")
    assert '"B":(True,False)' in text
    assert '"C":(False,True)' in text
    assert '"D":(True,True)' in text
    assert "pipeline_valid" in text
    assert "native_certificate_preservation" in text
    assert "clean DRFC main effect" in text


def test_v4848_serial_workers_isolate_errexit_state() -> None:
    calibration = (ROOT / "scripts/calibrate_v48_36_shared_certificate_pool.sh").read_text(encoding="utf-8")
    controller = (ROOT / "scripts/run_v48_36_ocaf_dedicated.sh").read_text(encoding="utf-8")
    # Natural RC=20 from Balanced certificate must be captured rather than
    # letting a function-local `set -e` terminate the serial controller.
    assert '( calibrate_variant balanced "$GPU0" ); S0=$?' in calibration
    assert '( calibrate_variant precision "$GPU1" ); S1=$?' in calibration
    # The same shell-state leak exists for nonzero adaptation failures; isolate
    # those workers too so the second variant and failure provenance still run.
    assert '( run_variant balanced "$GPU0" ); s0=$?' in controller
    assert '( run_variant precision "$GPU1" ); s1=$?' in controller


def test_v4848_serial_errexit_regression_executes_both_natural_failures(tmp_path: Path) -> None:
    import subprocess

    script = tmp_path / "errexit_regression.sh"
    script.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
calibrate_variant(){ local name="$1"; echo "$name"; set -e; return 20; }
S0=0; S1=0
set +e
( calibrate_variant balanced ); S0=$?
( calibrate_variant precision ); S1=$?
set -e
printf 'S0=%s S1=%s\n' "$S0" "$S1"
""",
        encoding="utf-8",
    )
    completed = subprocess.run(["bash", str(script)], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["balanced", "precision", "S0=20 S1=20"]


def test_v4848_launcher_attempts_second_wave_even_after_engineering_failure() -> None:
    text = (ROOT / "scripts/run_v48_48_ncp_2x2_two_gpu.sh").read_text(encoding="utf-8")
    pair_loop = text.split('for pair in "A B" "C D"; do', 1)[1].split('done', 1)[0]
    assert '[[ "$engineering_failed" == 0 ]] || break' not in pair_loop
    assert "all four waves were attempted but attribution is blocked" in text
