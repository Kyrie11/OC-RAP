from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_selected_trace_rerun_uses_portable_scheduler_and_skips_reusable_regimes() -> None:
    text = (ROOT / 'scripts/generate_selected_regime_traces.sh').read_text()
    assert 'trace_pending_count()' in text
    assert 'OCRAP_PENDING="$(trace_pending_count \'*\' ocrap)"' in text
    assert 'SAFE_EXTERNAL_PENDING="$(trace_pending_count safe external)"' in text
    assert 'NEAR_EXTERNAL_PENDING="$(trace_pending_count near external)"' in text
    assert 'CONTACT_EXTERNAL_PENDING="$(trace_pending_count contact external)"' in text
    assert ': "${USE_DYNAMIC_SCHEDULER:=auto}"' in text
    assert text.count('USE_DYNAMIC_SCHEDULER="$USE_DYNAMIC_SCHEDULER"') >= 3
    assert 'CHECKPOINT_ROOT="$NEAR_EXTERNAL_ROOT/checkpoints"' in text
    assert 'if (( OCRAP_PENDING > 0 )); then' in text
    assert 'if (( SAFE_EXTERNAL_PENDING > 0 )); then' in text
    assert 'if (( NEAR_EXTERNAL_PENDING > 0 )); then' in text
    assert 'if (( CONTACT_EXTERNAL_PENDING > 0 )); then' in text
    assert '[VIS-TRACE][REUSE] ocrap_selected' in text
    assert '[VIS-TRACE][REUSE] external_safe' in text
    assert '[VIS-TRACE][REUSE] external_near' in text
    assert '[VIS-TRACE][REUSE] external_contact' in text
