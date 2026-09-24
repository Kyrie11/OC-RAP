#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
: "${BASE_OUT:=runs}"
: "${OCRAP_RECONTACT_SOURCE:=official8}" # official8 | supplement26 | frozen_confirmation
: "${MAIN_VIS_ROOT:=$BASE_OUT/regime_visualization_v48_124_final_optimized}"
: "${MAIN_CHARACTERIZATION_ROOT:=$BASE_OUT/ocrap_v48_124_final_characterization}"
: "${SOURCE_SUPPLEMENT_NAME:=supplement_01}"
: "${FROZEN_OUT:=$BASE_OUT/contact_recovery_frozen8_v137}"
case "$OCRAP_RECONTACT_SOURCE" in
  official8)
    TRACE="$MAIN_VIS_ROOT/selective_traces/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
    KEYS="$MAIN_CHARACTERIZATION_ROOT/target_keys/contact.json"
    OUT="$BASE_OUT/contact_recontact_ocrap_official8"
    ;;
  supplement26)
    SRC="$BASE_OUT/contact_qualitative_supplements/$SOURCE_SUPPLEMENT_NAME"
    TRACE="$SRC/traces/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
    KEYS="$SRC/target_keys/contact.json"
    OUT="$BASE_OUT/contact_recontact_ocrap_${SOURCE_SUPPLEMENT_NAME}"
    ;;
  frozen_confirmation)
    TRACE="$FROZEN_OUT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
    KEYS="$MAIN_CHARACTERIZATION_ROOT/target_keys/contact.json"
    OUT="$FROZEN_OUT/ocrap/contact/recontact_continuous"
    ;;
  *) echo "OCRAP_RECONTACT_SOURCE must be official8, supplement26, or frozen_confirmation" >&2; exit 2 ;;
esac
[[ -s "$TRACE" && -s "$KEYS" ]] || { echo "missing full OC-RAP trace or target lock: trace=$TRACE keys=$KEYS" >&2; exit 30; }
mkdir -p "$OUT"
python tools/audit_contact_recontact_continuous.py \
  --trace "ocrap=$TRACE" --target-keys-file "$KEYS" \
  --output-json "$OUT/OCRAP_CONTACT_RECONTACT_CONTINUOUS.json" \
  --output-csv "$OUT/OCRAP_CONTACT_RECONTACT_CONTINUOUS.csv" \
  --per-scene-csv "$OUT/OCRAP_CONTACT_RECONTACT_PER_SCENE.csv"
echo "[OCRAP-CONTACT-RECONTACT][DONE] $OUT/OCRAP_CONTACT_RECONTACT_CONTINUOUS.csv"
