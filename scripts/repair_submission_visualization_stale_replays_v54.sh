#!/usr/bin/env bash
# Repair only the population closed-loop artifacts whose recorded raw WOMD
# source role is stale relative to the canonical OC-RAP test-bucket provenance.
# This script preserves the public result roots and filenames.  Existing stale
# artifacts are archived before rerun so closed-loop resume cannot silently
# reuse validation_interactive scenes under the new validation fingerprint.
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

: "${BASE_OUT:=/home/senzeyu2/code/OC-RAP/runs}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${OCRAP_MODEL_RUN:=$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
: "${MODEL_VARIANT:=balanced}"
: "${OCRAP_SUBMISSION_ROOT:=$BASE_OUT/ocrap_v48_111_submission_three_regime}"
: "${OCRAP_RESULTS_ROOT:=$OCRAP_SUBMISSION_ROOT/$MODEL_VARIANT}"
: "${CONTACT_EXTERNAL_ROOT:=$BASE_OUT/contact_external}"
: "${CUDA_DEVICES:=0,1}"
: "${MAX_STEPS:=40}"
: "${NUM_CANDIDATES:=24}"
: "${NUM_RECOVERY_OPTIONS:=12}"

stamp="$(date +%Y%m%d-%H%M%S)"
backup="$BASE_OUT/submission_visualization_stale_replay_backup_$stamp"
mkdir -p "$backup"

resolve_role() {
  local dataset="$1"
  python tools/resolve_womd_replay_source.py \
    --dataset "$dataset" --split test --womd-root "$WOMD_ROOT" --shards 150 --json \
    | python -c 'import json,sys; print(json.load(sys.stdin)["resolved_role"])'
}

safe_role="$(resolve_role "$OCRAP_ROOT/test_safe")"
near_role="$(resolve_role "$OCRAP_ROOT/test_near_contact")"
contact_role="$(resolve_role "$OCRAP_ROOT/test_contact")"
printf '[CANONICAL] Safe=%s Near=%s Contact=%s\n' "$safe_role" "$near_role" "$contact_role"

# This repair is intentionally scoped to the stale artifacts reported by the
# v53 input contract in the user's current run.  We do not touch Safe or Near
# external population results because they already match bucket provenance.
archive_ocrap_regime() {
  local regime="$1" src="$OCRAP_RESULTS_ROOT/$regime" dst="$backup/ocrap_$regime"
  mkdir -p "$dst"
  shopt -s nullglob
  local files=("$src"/closed_loop_ocrap.json* "$src"/closed_loop_ocrap.log "$src"/closed_loop_dataset_support.json)
  local f
  for f in "${files[@]}"; do
    [[ -e "$f" ]] || continue
    mv "$f" "$dst/"
    echo "[ARCHIVE] $f -> $dst/"
  done
  shopt -u nullglob
  # Phase files live one directory above the per-regime output.
  if [[ -f "$OCRAP_RESULTS_ROOT/$regime.phase.json" ]]; then
    mv "$OCRAP_RESULTS_ROOT/$regime.phase.json" "$dst/"
  fi
}

archive_contact_external() {
  local src="$CONTACT_EXTERNAL_ROOT" dst="$backup/external_contact"
  mkdir -p "$dst"
  shopt -s nullglob
  local files=("$src"/closed_loop_*.json* "$src"/closed_loop_*.log "$src"/jax_waymax_runtime_preflight.json)
  local f
  for f in "${files[@]}"; do
    [[ -e "$f" ]] || continue
    mv "$f" "$dst/"
    echo "[ARCHIVE] $f -> $dst/"
  done
  shopt -u nullglob
}

archive_ocrap_regime near
archive_ocrap_regime contact
archive_contact_external

echo '[RERUN] OC-RAP Near + Contact only; Safe remains untouched.'
python tools/check_v48_111_deployable_stack.py \
  --model-run "$OCRAP_MODEL_RUN" --variant "$MODEL_VARIANT" \
  --output "$backup/V48.111-DEPLOYABLE-STACK-${MODEL_VARIANT}.json"
MODEL_RUN="$OCRAP_MODEL_RUN" \
MODEL_VARIANT="$MODEL_VARIANT" \
CUDA_DEVICES="$CUDA_DEVICES" \
OCRAP_ROOT="$OCRAP_ROOT" \
WOMD_ROOT="$WOMD_ROOT" \
OUT="$OCRAP_RESULTS_ROOT" \
MAX_SCENARIOS=0 \
MAX_STEPS="$MAX_STEPS" \
NUM_CANDIDATES="$NUM_CANDIDATES" \
NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
ALLOW_DIAGNOSTIC_RC20=1 \
RUN_SAFE=0 RUN_NEAR=1 RUN_CONTACT=1 \
SKIP_COMPLETE_REGIMES=false FINALIZE_COMPLETE_JOURNALS=true \
bash scripts/run_ocrap_three_regime_closed_loop.sh

echo '[RERUN] Contact external population only; no retraining/offline evaluation.'
OCRAP_ROOT="$OCRAP_ROOT" \
WOMD_ROOT="$WOMD_ROOT" \
CUDA_DEVICES="$CUDA_DEVICES" \
RUN="$CONTACT_EXTERNAL_ROOT" \
CL_WOMD=auto \
CL_MAX_SCENARIOS=0 \
DO_TRAIN=false \
DO_OFFLINE=false \
DO_CLOSED_LOOP=true \
SKIP_COMPLETE_METHODS=false \
CL_RESUME_FORCE=false \
bash scripts/run_contact_external_baselines.sh

echo "[DONE] stale artifacts archived under: $backup"
echo '[NEXT] run scripts/build_submission_visualization_v54.sh'
