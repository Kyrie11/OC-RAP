from __future__ import annotations

import re
from pathlib import Path

from ocrap.external_baselines.provenance import MAIN_TABLE_BY_REGIME

ROOT = Path(__file__).resolve().parents[1]


def _block(text: str, var: str) -> str:
    m = re.search(rf"{re.escape(var)}=\(\n(?P<body>.*?)\n\)", text, flags=re.S)
    assert m, f"missing bash array {var}"
    return m.group("body")


def _method_lines(block: str) -> tuple[str, ...]:
    out: list[str] = []
    for raw in block.splitlines():
        line = raw.strip().strip('"').strip("'")
        if not line or line.startswith("#"):
            continue
        # Safe SPECS entries use method|config|kind|checkpoint.
        out.append(line.split("|", 1)[0])
    return tuple(out)


def test_regime_launchers_match_main_table_registry() -> None:
    safe = (ROOT / "scripts/run_external_baselines_safe.sh").read_text()
    near = (ROOT / "scripts/run_external_baselines_near.sh").read_text()
    contact = (ROOT / "scripts/run_external_baselines_contact.sh").read_text()

    assert _method_lines(_block(safe, "SPECS")) == MAIN_TABLE_BY_REGIME["safe"]
    assert _method_lines(_block(near, "METHODS")) == MAIN_TABLE_BY_REGIME["near"]
    assert _method_lines(_block(contact, "METHODS")) == MAIN_TABLE_BY_REGIME["contact"]


def test_launchers_are_two_gpu_bounded_and_wire_train_calibration() -> None:
    for rel in (
        "scripts/run_external_baselines_safe.sh",
        "scripts/run_external_baselines_near.sh",
        "scripts/run_external_baselines_contact.sh",
    ):
        text = (ROOT / rel).read_text()
        assert ': "${CUDA_DEVICES:=0,1}"' in text
        assert ': "${JOBS_PER_GPU:=3}"' in text
        assert ': "${MAX_PARALLEL:=6}"' in text
        assert 'GPU_SLOTS=()' in text
        assert '((MAX_PARALLEL <= ${#GPU_SLOTS[@]})) || MAX_PARALLEL="${#GPU_SLOTS[@]}"' in text

    near = (ROOT / "scripts/run_external_baselines_near.sh").read_text()
    assert 'tools/calibrate_external_baselines.py' in near
    assert 'DO_CALIBRATE' in near
    assert 'CONFORMAL_INTERVALS' in near
    assert 'conformal_prediction_intervals_m' in near
    assert 'runtime_resolve_bucket_womd_spec' in near
    assert 'CALIB_WOMD_ROLE' in near

    contact = (ROOT / "scripts/run_external_baselines_contact.sh").read_text()
    assert 'tools/register_external_nonlearning_baselines.py' in contact
    assert 'DO_TRAIN' in contact

    master = (ROOT / "scripts/run_external_baselines_all.sh").read_text()
    assert 'DO_TRAIN="$DO_TRAIN_CONTACT"' in master
    assert 'DO_CALIBRATE="$DO_CALIBRATE_NEAR"' in master


def test_clean_wrapper_defaults_publication_role_and_propagates_force_reregister() -> None:
    text = (ROOT / "scripts/run_external_baselines.sh").read_text()
    assert 'WOMD_ROLE="${PRIMARY_WOMD_ROLE:-validation}"' in text
    assert 'CL_WOMD_ROLE="$WOMD_ROLE"' in text
    assert 'CALIB_WOMD_ROLE="$WOMD_ROLE"' in text
    assert 'FORCE_REREGISTER="$FORCE_RETRAIN"' in text


def test_nonlearning_launchers_support_registration_reuse() -> None:
    for rel in (
        "scripts/run_external_baselines_safe.sh",
        "scripts/run_external_baselines_near.sh",
        "scripts/run_external_baselines_contact.sh",
    ):
        text = (ROOT / rel).read_text()
        assert "check_external_nonlearning_registration.py" in text
        assert "FORCE_REREGISTER" in text


def test_safe_and_near_use_per_baseline_train_test_pipeline_with_dynamic_refill() -> None:
    for rel in ("scripts/run_external_baselines_safe.sh", "scripts/run_external_baselines_near.sh"):
        text = (ROOT / rel).read_text()
        assert ': "${USE_DYNAMIC_SCHEDULER:=true}"' in text
        assert "run_baseline_pipeline()" in text
        assert 'prepare_or_offline_method "$spec" "$gpu"' in text
        assert 'run_closed_loop_method "$spec" "$gpu"' in text
        assert 'run_queue run_baseline_pipeline "${PIPELINE_SPECS[@]}"' in text
        # There must not be separate all-method prepare and closed-loop queues,
        # otherwise the train/test phase barrier returns.
        assert 'run_queue prepare_or_offline_method "${SPECS[@]}"' not in text


def test_near_offline_evaluator_treats_flow_and_planr1_as_pure_learned() -> None:
    text = (ROOT / "src/ocrap/external_baselines/evaluate.py").read_text()
    for name in ('"flow_planner"', '"flowplanner"', '"plan_r1"', '"planr1"'):
        assert text.count(name) >= 2


def test_all_regime_launcher_inherits_six_slot_dynamic_defaults() -> None:
    text = (ROOT / "scripts/run_external_baselines_all.sh").read_text()
    assert ': "${USE_DYNAMIC_SCHEDULER:=true}"' in text
    assert ': "${JOBS_PER_GPU:=3}"' in text
    assert ': "${MAX_PARALLEL:=6}"' in text
    assert 'USE_DYNAMIC_SCHEDULER="$USE_DYNAMIC_SCHEDULER"' in text


def test_near_publication_closed_loop_disables_teacher_audit_and_uses_fast_history() -> None:
    text = (ROOT / "scripts/run_external_baselines_near.sh").read_text()
    assert ': "${CL_LABEL_MODE:=fast}"' in text
    assert ': "${CL_AUDIT_EVERY_N_STEPS:=0}"' in text
    assert ': "${CL_FAST_WAYMAX_HISTORY:=true}"' in text
    assert '--set "closed_loop.fast_waymax_history=$CL_FAST_WAYMAX_HISTORY"' in text


def test_zero_closed_loop_audit_cadence_is_a_real_disable_switch() -> None:
    text = (ROOT / "src/ocrap/simulation/closed_loop_runner.py").read_text()
    assert 'audit_enabled = audit_every_n_steps > 0' in text
    assert 'if audit_enabled and (selected_label_audit or coverage_label_audit)' in text
