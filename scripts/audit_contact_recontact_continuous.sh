#!/usr/bin/env bash
# Zero-GPU continuous re-contact audit on already-generated full Contact traces.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
: "${BASE_OUT:=runs}"
: "${AUDIT_MODE:=supplement26}"  # official8 | supplement26
: "${MAIN_VIS_ROOT:=$BASE_OUT/regime_visualization_v48_124_final_optimized}"
: "${MAIN_CHARACTERIZATION_ROOT:=$BASE_OUT/ocrap_v48_124_final_characterization}"
: "${SOURCE_SUPPLEMENT_NAME:=supplement_01}"
: "${OUT:=$BASE_OUT/contact_recontact_continuous_audit_${AUDIT_MODE}}"
METHODS=(postimpact_mpc_lite post_crash_braking postimpact_motion_tvlqr post_collision_restoration compensatory_postimpact_mpc robust_postimpact_control)
case "$AUDIT_MODE" in
  official8)
    TRACE_ROOT="$MAIN_VIS_ROOT/selective_traces"
    OCR_TRACE="$TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
    EXT_ROOT="$TRACE_ROOT/external/contact"
    KEYS="$MAIN_CHARACTERIZATION_ROOT/target_keys/contact.json"
    ;;
  supplement26)
    SRC="$BASE_OUT/contact_qualitative_supplements/$SOURCE_SUPPLEMENT_NAME"
    OCR_TRACE="$SRC/traces/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
    EXT_ROOT="$SRC/traces/external/contact"
    KEYS="$SRC/target_keys/contact.json"
    ;;
  *) echo "AUDIT_MODE must be official8 or supplement26" >&2; exit 2 ;;
esac
[[ -s "$OCR_TRACE" && -s "$KEYS" ]] || { echo "missing OC-RAP trace or target keys for mode=$AUDIT_MODE" >&2; exit 30; }
ARGS=(--trace "ocrap=$OCR_TRACE")
for m in "${METHODS[@]}"; do
  p="$EXT_ROOT/closed_loop_${m}.json.scenes.jsonl"
  [[ -s "$p" ]] || { echo "missing baseline full trace: $p" >&2; exit 30; }
  ARGS+=(--trace "$m=$p")
done
mkdir -p "$OUT"
python tools/audit_contact_recontact_continuous.py \
  "${ARGS[@]}" --target-keys-file "$KEYS" \
  --output-json "$OUT/CONTACT_RECONTACT_CONTINUOUS_AUDIT.json" \
  --output-csv "$OUT/CONTACT_RECONTACT_CONTINUOUS_AUDIT.csv" \
  --per-scene-csv "$OUT/CONTACT_RECONTACT_PER_SCENE.csv"
echo "[CONTACT-RECONTACT-AUDIT][DONE] $OUT/CONTACT_RECONTACT_CONTINUOUS_AUDIT.csv"
