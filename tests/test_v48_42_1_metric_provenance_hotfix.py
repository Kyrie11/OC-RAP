from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v48421_metric_contract", ROOT / "tools" / "check_v48_36_metric_calibration_contract.py"
)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
_load_metric_row = MOD._load_metric_row


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path, *, recorded_prefix: str) -> tuple[Path, Path, Path]:
    candidate = root / "runs" / "run" / "candidates" / "balanced"
    factor_model = candidate / "factor_stage" / "model_v48_trac_sr"
    final_model = candidate / "model_v48_trac_sr"
    factor_model.mkdir(parents=True)
    final_model.mkdir(parents=True)
    ckpt = factor_model / "best.pt"
    ckpt.write_bytes(b"factor-checkpoint")
    source = factor_model / "train_summary.json"
    source.write_text(json.dumps({
        "best_epoch": 7,
        "history": [{"epoch": 7, "val": {"direct_group_count_near": 110}}],
    }), encoding="utf-8")
    stage = final_model / "train_summary.json"
    prefix = Path(recorded_prefix)
    source_raw = prefix / "factor_stage" / "model_v48_trac_sr" / "train_summary.json"
    ckpt_raw = prefix / "factor_stage" / "model_v48_trac_sr" / "best.pt"
    stage.write_text(json.dumps({
        "best_epoch": 0,
        "epochs_completed": 0,
        "total_train_steps": 0,
        "history": [],
        "materialized_without_training": True,
        "parameter_update_performed": False,
        "factor_checkpoint_reused_without_parameter_update": True,
        "source_factor_checkpoint": str(ckpt_raw),
        "source_factor_checkpoint_sha256": _sha(ckpt),
        "metric_source_checkpoint_sha256": _sha(ckpt),
        "metric_source_train_summary": str(source_raw),
        "metric_source_train_summary_sha256": _sha(source),
        "source_factor_best_epoch": 7,
    }), encoding="utf-8")
    return stage, source, ckpt


def test_repo_relative_metric_provenance_resolves_from_controller_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage, source, _ = _fixture(tmp_path, recorded_prefix="runs/run/candidates/balanced")
    monkeypatch.chdir(tmp_path)
    _, best, epoch, provenance = _load_metric_row(stage)
    assert epoch == 7 and best["epoch"] == 7
    assert Path(provenance["metric_source_train_summary"]) == source.resolve()
    assert provenance["metric_source_train_summary_resolution"] == "cwd_relative"
    assert provenance["metric_source_checkpoint_resolution"] == "cwd_relative"


def test_relocated_run_uses_sha_pinned_factor_stage_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a packaged/moved run: the recorded historical repo prefix no
    # longer exists, but the factor-stage suffix under the current candidate is
    # byte-identical and therefore auditable.
    stage, source, _ = _fixture(tmp_path / "moved", recorded_prefix="runs/original/candidates/balanced")
    empty_cwd = tmp_path / "empty-cwd"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    _, best, epoch, provenance = _load_metric_row(stage)
    assert epoch == 7 and best["epoch"] == 7
    assert Path(provenance["metric_source_train_summary"]) == source.resolve()
    assert provenance["metric_source_train_summary_resolution"] == "factor_stage_suffix"
    assert provenance["metric_source_checkpoint_resolution"] == "factor_stage_suffix"


def test_relative_provenance_still_fails_closed_on_byte_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage, source, _ = _fixture(tmp_path, recorded_prefix="runs/run/candidates/balanced")
    monkeypatch.chdir(tmp_path)
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA256 mismatch"):
        _load_metric_row(stage)


def test_v48421_changes_implementation_not_algorithm_identity() -> None:
    main = (ROOT / "scripts" / "run_v48_42_hpfr_dedicated.sh").read_text(encoding="utf-8")
    assert 'OCRAP_ALGORITHM_VERSION="v48.42-HPFR"' in main
    assert 'OCRAP_IMPLEMENTATION_VERSION="v48.42.1-HPFR-METRIC-PROVENANCE-HOTFIX"' in main
    # No algorithm flags are changed by the engineering hotfix.
    assert 'EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=true' in main
    assert 'EVIDENCE_RANK_BENEFIT_SKIP=true' in main
    assert 'FACTOR_COMPONENT_MARGIN_TARGET_MODE=raw' in main


def test_certificate_pool_access_occurs_after_metric_provenance_contract() -> None:
    text = (ROOT / "scripts" / "calibrate_v48_36_shared_certificate_pool.sh").read_text(encoding="utf-8")
    metric_pos = text.index("check_v48_36_metric_calibration_contract.py")
    certificate_calibration_pos = text.index('local datasets=("$CERT_NEAR,$CERT_CONTACT"')
    verification_pos = text.index("--verification-only")
    assert metric_pos < certificate_calibration_pos < verification_pos


def test_fast_rerun_is_new_output_and_exact_cache_only() -> None:
    text = (ROOT / "scripts" / "run_v48_42_1_hpfr_from_exact_factor_cache.sh").read_text(encoding="utf-8")
    assert "OUTPUTDIR must differ from OLD_OUTPUTDIR" in text
    assert "V4842_ALLOW_EXACT_FACTOR_CACHE=1" in text
    assert 'V4836_FACTOR_CACHE_BALANCED="$OLD_OUTPUTDIR/candidates/balanced/factor_stage"' in text
    assert 'V4836_FACTOR_CACHE_PRECISION="$OLD_OUTPUTDIR/candidates/precision/factor_stage"' in text
    assert "run_v48_42_hpfr_dedicated.sh" in text
