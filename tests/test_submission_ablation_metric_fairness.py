from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools/check_publication_closed_loop_artifact.py"


def _contract(regime: str = "contact") -> dict:
    return {
        "metric_semantics_version": "publication_v55_signed_clearance_unclipped_v1",
        "max_steps": 40,
        "replan_interval_steps": 1,
        "metric_dt_s": 0.1,
        "num_candidate_prefixes": 24,
        "num_recovery_options": 12,
        "use_sdc_paths": True,
        "require_observation_legal_route": True,
        "allow_future_route_proxy": False,
        "dataloader_include_sdc_paths": True,
        "allow_logged_sdc_route_fallback": False,
        "compute_future_metrics": False,
        "publication_geometry_metric": "exact_oriented_box_signed_clearance+penetration+swept_sat_constant_velocity_ttc_v55",
        "clearance_is_signed": True,
        "duration_auc_support": "left_endpoint_t0_to_tN_minus_1_no_fictitious_terminal_interval",
        "womd_source_role": "validation",
    }


def _write_contact_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    key = "scene-a:t10"
    lock = tmp_path / "contact.json"
    lock.write_text(json.dumps({"target_keys": [key]}), encoding="utf-8")
    manifest = tmp_path / "contact_anchor_manifest.json"
    manifest_doc = {
        "schema": "ocrap-contact-anchor-manifest-v1",
        "valid": True,
        "pre_treatment_policy": "exact_a0",
        "min_post_steps": 40,
        "num_selected_anchors": 1,
        "num_selected_scenes": 1,
        "target_keys": [key],
        "anchors": [{
            "target_key": key,
            "contact_anchor_fingerprint": "fp-a",
            "contact_anchor_time_index": 15,
            "contact_anchor_prelude_env_steps": 5,
        }],
    }
    manifest.write_text(json.dumps(manifest_doc), encoding="utf-8")
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = tmp_path / "closed_loop_ocrap.json"
    output.write_text(json.dumps({
        "num_scenes": 1,
        "metrics_valid": True,
        "route_ineligible_target_count": 0,
        "evaluation_contract": _contract(),
        "runtime_contract": {
            "publication_geometry_metric": "exact_oriented_box_signed_clearance+penetration+swept_sat_constant_velocity_ttc_v55",
            "publication_metrics_include_target_state_t0": True,
        },
        "clearance_metric_full_coverage_scene_rate": 1.0,
        "ttc_metric_full_coverage_scene_rate": 1.0,
        "overlap_metric_full_coverage_scene_rate": 1.0,
        "offroad_metric_full_coverage_scene_rate": 1.0,
        "timing": {
            "execution_contract": "isolated_single_process_single_gpu",
            "steady_state_deployed_planner_s": {"mean": 0.01},
        },
        "contact_anchor_protocol": "exact_a0_pretreatment_prelude_v1",
        "contact_anchor_manifest_sha256": sha,
        "contact_anchor_state_fingerprint_required": True,
        "observed_contact_scene_rate": 1.0,
        "post_contact_metric_eligible_scene_rate": 1.0,
    }), encoding="utf-8")
    Path(str(output) + ".scenes.jsonl").write_text(json.dumps({"scene": {
        "target_key": key,
        "contact_anchor_protocol": "exact_a0_pretreatment_prelude_v1",
        "contact_anchor_fingerprint": "fp-a",
        "contact_anchor_time_index": 15,
        "contact_anchor_prelude_env_steps": 5,
        "contact_anchor_found": True,
    }}) + "\n", encoding="utf-8")
    return output, lock, manifest


def _check(output: Path, lock: Path, manifest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        sys.executable, str(CHECKER),
        "--output", str(output), "--regime", "contact",
        "--target-keys-file", str(lock),
        "--contact-anchor-manifest", str(manifest),
        "--require-latency-contract", "isolated_single_process_single_gpu",
        "--require-finite-timing",
    ], cwd=ROOT, text=True, capture_output=True)


def test_publication_checker_accepts_shared_contact_anchor(tmp_path: Path) -> None:
    output, lock, manifest = _write_contact_fixture(tmp_path)
    proc = _check(output, lock, manifest)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["valid"] is True


def test_publication_checker_rejects_contact_fingerprint_mismatch(tmp_path: Path) -> None:
    output, lock, manifest = _write_contact_fixture(tmp_path)
    journal = Path(str(output) + ".scenes.jsonl")
    row = json.loads(journal.read_text())
    row["scene"]["contact_anchor_fingerprint"] = "wrong"
    journal.write_text(json.dumps(row) + "\n")
    proc = _check(output, lock, manifest)
    assert proc.returncode != 0
    report = json.loads(proc.stdout)
    assert any(e["check"] == "per_scene_contact_anchor_reproduction" for e in report["errors"])


def test_submission_ablation_launcher_uses_final_metric_and_contact_contracts() -> None:
    text = (ROOT / "scripts/run_submission_ablations.sh").read_text()
    assert "publication_v55_signed_clearance_unclipped_v1" in text
    assert "CONTACT_ANCHOR_PRELUDE_ENABLED=true" in text
    assert "CONTACT_ANCHOR_MANIFEST_FILE" in text
    assert "check_publication_closed_loop_artifact.py" in text
    assert "build_final_observation_legal_target_locks.sh" in text
    assert "isolated_single_process_single_gpu" in text
    assert 'table_args+=(--latency-root "$LATENCY_ROOT")' in text
    assert "archive_incompatible_partial_contact_if_needed" in text


def test_ablation_table_builder_never_silently_uses_accuracy_latency() -> None:
    text = (ROOT / "tools/build_submission_ablation_tables.py").read_text()
    assert "--latency-root" in text
    assert 'cmd.append("--omit-latency")' in text
    assert '"--latency-input"' in text
