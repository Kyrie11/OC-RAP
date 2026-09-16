from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_artifact(tmp: Path, keys: list[str]) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / "closed_loop_x.json"
    out.write_text(json.dumps({"method":"x","num_scenes":len(keys),"bucket_target_count":len(keys),"run_fingerprint":"abc","timing":{"execution_contract":"isolated_single_process_single_gpu"}}))
    out.with_suffix(out.suffix + ".progress.json").write_text(json.dumps({"status":"complete","run_fingerprint":"abc"}))
    with out.with_suffix(out.suffix + ".scenes.jsonl").open("w") as f:
        for k in keys:
            f.write(json.dumps({"run_fingerprint":"abc","scene":{"target_key":k}})+"\n")
    return out


def test_closed_loop_artifact_check_is_target_lock_aware(tmp_path: Path) -> None:
    out = _write_artifact(tmp_path, ["a", "b"])
    lock = tmp_path / "lock.json"; lock.write_text(json.dumps({"target_keys":["a","b","c"]}))
    cmd=[sys.executable,str(ROOT/"tools/check_closed_loop_artifact.py"),"--output",str(out),"--target-keys-file",str(lock),"--quiet"]
    assert subprocess.run(cmd).returncode != 0
    lock.write_text(json.dumps({"target_keys":["a","b"]}))
    assert subprocess.run(cmd).returncode == 0


def test_unified_external_launcher_propagates_zero_as_full_cohort() -> None:
    text=(ROOT/"scripts/run_external_baselines.sh").read_text()
    assert 'export CL_MAX_SCENARIOS="$MAX_SCENARIOS"' in text


def test_final_latency_contract_uses_isolated_artifacts() -> None:
    ext=(ROOT/"scripts/profile_external_baselines_latency.sh").read_text()
    ocrap=(ROOT/"scripts/profile_ocrap_latency.sh").read_text()
    tables=(ROOT/"scripts/build_final_regime_comparison_tables.sh").read_text()
    assert "isolated_single_process_single_gpu" in ext
    assert "isolated_single_process_single_gpu" in ocrap
    assert "--ocrap-latency-run" in tables
    assert "--safe-latency-run" in tables


def test_generative_checkpoint_contract_bumped_after_validation_fix() -> None:
    safe=(ROOT/"scripts/run_external_baselines_safe.sh").read_text()
    near=(ROOT/"scripts/run_external_baselines_near.sh").read_text()
    assert "diffusion_planner_womd_lattice_port_v63" in safe
    assert "flow_planner_womd_lattice_port_v63" in near


def test_comparison_table_rejects_latency_target_mismatch(tmp_path: Path) -> None:
    acc = _write_artifact(tmp_path / "acc", ["a", "b"])
    lat = _write_artifact(tmp_path / "lat", ["a", "c"])
    out = tmp_path / "tables"
    cmd = [
        sys.executable,
        str(ROOT / "tools/build_regime_comparison_tables.py"),
        "--regime", "safe",
        "--input", f"x={acc}",
        "--latency-input", f"x={lat}",
        "--output-dir", str(out),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env=env)
    assert proc.returncode != 0
    assert "latency target set does not match accuracy target set" in (proc.stdout + proc.stderr)
