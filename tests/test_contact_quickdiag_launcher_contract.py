from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ocrap_three_regime_resolves_only_enabled_buckets():
    text = (ROOT / "scripts/run_ocrap_three_regime_evaluation.sh").read_text(encoding="utf-8")
    assert text.index(': "${RUN_SAFE:=1}"') < text.index('resolve_regime_womd()')
    assert 'if [[ "$enabled" != 1 ]]; then' in text
    assert 'SAFE_WOMD="$(resolve_regime_womd "$RUN_SAFE" "$SAFE_WOMD" "$SAFE_BUCKET")"' in text
    assert 'NEAR_WOMD="$(resolve_regime_womd "$RUN_NEAR" "$NEAR_WOMD" "$NEAR_BUCKET")"' in text
    assert 'CONTACT_WOMD="$(resolve_regime_womd "$RUN_CONTACT" "$CONTACT_WOMD" "$CONTACT_BUCKET")"' in text


def test_nominal_three_regime_resolves_only_enabled_buckets():
    text = (ROOT / "scripts/run_nominal_three_regime_control.sh").read_text(encoding="utf-8")
    assert 'if [[ "$enabled" != 1 ]]; then' in text
    assert 'SAFE_WOMD="$(resolve_spec "$RUN_SAFE" "$SAFE_WOMD" "$SAFE_BUCKET")"' in text
    assert 'CONTACT_WOMD="$(resolve_spec "$RUN_CONTACT" "$CONTACT_WOMD" "$CONTACT_BUCKET")"' in text


def test_quickdiag_reuse_is_bound_to_requested_subset_and_anchor_hash():
    text = (ROOT / "scripts/run_contact_recovery_quick_diagnostic.sh").read_text(encoding="utf-8")
    assert "requested_num_targets" in text
    assert "max_targets_per_scene" in text
    assert "source_target_keys_sha256" in text
    assert "DIAG_MIN_POST_STEPS" in text
    assert 'SAFE_BUCKET="$OCRAP_ROOT/val_safe"' in text
    assert 'NEAR_BUCKET="$OCRAP_ROOT/val_near_contact"' in text
    assert 'CONTACT_BUCKET="$OCRAP_ROOT/val_contact"' in text
