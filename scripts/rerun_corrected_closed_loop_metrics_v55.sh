#!/usr/bin/env bash
# Recompute publication closed-loop metrics after the v55 geometry/aggregation fix.
# Preserves the user's public run roots and trained/calibrated artifacts.
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

: "${BASE_OUT:=/home/senzeyu2/code/OC-RAP/runs}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${CUDA_DEVICES:=0,1}"
: "${MODEL_RUN:=$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
: "${VARIANTS:=balanced,precision}"
: "${MAX_SCENARIOS:=0}"
: "${MAX_STEPS:=40}"
: "${NUM_CANDIDATES:=24}"
: "${NUM_RECOVERY_OPTIONS:=12}"

# Existing public roots used by build_submission_visualization_v54.sh.
: "${OCRAP_SUBMISSION_ROOT:=$BASE_OUT/ocrap_v48_111_submission_three_regime}"
: "${SAFE_EXTERNAL_ROOT:=$BASE_OUT/safe_external}"
: "${NEAR_EXTERNAL_ROOT:=$BASE_OUT/near_external}"
: "${CONTACT_EXTERNAL_ROOT:=$BASE_OUT/contact_external}"
: "${TABLE_OUT:=$BASE_OUT/submission_external_baseline_tables_v55}"

# Fast metric-only baseline replay: six methods = three processes per GPU.
# If this concurrency is too aggressive for the local card memory, each regime
# is retried automatically at one process/GPU and already-complete methods are reused.
: "${JOBS_PER_GPU:=3}"
: "${MAX_PARALLEL:=6}"
: "${FALLBACK_JOBS_PER_GPU:=1}"
: "${FALLBACK_MAX_PARALLEL:=2}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"

stamp="$(date +%Y%m%d-%H%M%S)"
backup="$BASE_OUT/metric_repair_backup_v55/$stamp"
mkdir -p "$backup"

resolve_role() {
  local dataset="$1"
  python tools/resolve_womd_replay_source.py \
    --dataset "$dataset" --split test --womd-root "$WOMD_ROOT" --shards 150 --json \
    | python -c 'import json,sys; d=json.load(sys.stdin); print(d["resolved_role"])'
}

echo '[PREFLIGHT] resolving each bucket from stored dataset provenance (never hard-code validation vs validation_interactive)'
safe_role="$(resolve_role "$OCRAP_ROOT/test_safe")"
near_role="$(resolve_role "$OCRAP_ROOT/test_near_contact")"
contact_role="$(resolve_role "$OCRAP_ROOT/test_contact")"
printf '[SOURCE] test_safe=%s test_near_contact=%s test_contact=%s\n' "$safe_role" "$near_role" "$contact_role"

archive_matches() {
  local dst="$1"; shift
  mkdir -p "$dst"
  shopt -s nullglob
  local pattern f
  for pattern in "$@"; do
    for f in $pattern; do
      [[ -e "$f" ]] || continue
      mv "$f" "$dst/"
      echo "[ARCHIVE] $f -> $dst/"
    done
  done
  shopt -u nullglob
}

# Remove only evaluation artifacts. Keep model checkpoints, calibration files,
# train summaries and JAX caches so there is no retraining/recalibration cost.
IFS=',' read -r -a _variants <<< "$VARIANTS"
for variant in "${_variants[@]}"; do
  variant="$(echo "$variant" | xargs)"; [[ -n "$variant" ]] || continue
  for regime in safe near contact; do
    archive_matches "$backup/ocrap/$variant/$regime" \
      "$OCRAP_SUBMISSION_ROOT/$variant/$regime/closed_loop_ocrap.json*" \
      "$OCRAP_SUBMISSION_ROOT/$variant/$regime/closed_loop_ocrap.log" \
      "$OCRAP_SUBMISSION_ROOT/$variant/$regime/closed_loop_dataset_support.json"
  done
  archive_matches "$backup/ocrap/$variant" \
    "$OCRAP_SUBMISSION_ROOT/$variant/safe.phase.json" \
    "$OCRAP_SUBMISSION_ROOT/$variant/near.phase.json" \
    "$OCRAP_SUBMISSION_ROOT/$variant/contact.phase.json" \
    "$OCRAP_SUBMISSION_ROOT/$variant/OCRAP_THREE_REGIME_RUN_INDEX.json"
done
archive_matches "$backup/ocrap/summary" \
  "$OCRAP_SUBMISSION_ROOT/V48.111-SUBMISSION-THREE-REGIME-SUMMARY.json" \
  "$OCRAP_SUBMISSION_ROOT/V48.111-SUBMISSION-THREE-REGIME-SUMMARY.csv" \
  "$OCRAP_SUBMISSION_ROOT/V48.111-SUBMISSION-THREE-REGIME-SUMMARY.md"

for spec in \
  "safe:$SAFE_EXTERNAL_ROOT" \
  "near:$NEAR_EXTERNAL_ROOT" \
  "contact:$CONTACT_EXTERNAL_ROOT"; do
  regime="${spec%%:*}"; root="${spec#*:}"
  archive_matches "$backup/external/$regime" \
    "$root/closed_loop_*.json*" \
    "$root/closed_loop_*.log" \
    "$root/closed_loop_summary.json" \
    "$root/jax_waymax_runtime_preflight.json" \
    "$root/${regime}_external_baselines_summary.json"
done
if [[ -e "$TABLE_OUT" ]]; then
  mv "$TABLE_OUT" "$backup/$(basename "$TABLE_OUT")"
  echo "[ARCHIVE] $TABLE_OUT -> $backup/"
fi

echo '[1/5] OC-RAP: rerun all three regimes for both frozen variants; no training/recalibration.'
GPU0=0 GPU1=1 \
CUDA_DEVICES="$CUDA_DEVICES" \
BASE_OUT="$BASE_OUT" \
OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" \
VARIANTS="$VARIANTS" \
MODEL_RUN="$MODEL_RUN" \
MAX_SCENARIOS="$MAX_SCENARIOS" \
MAX_STEPS="$MAX_STEPS" \
NUM_CANDIDATES="$NUM_CANDIDATES" \
NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
OUT_ROOT="$OCRAP_SUBMISSION_ROOT" \
bash scripts/run_v48_111_submission_three_regime.sh

run_safe() {
  local jpg="$1" mp="$2"
  OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
  RUN="$SAFE_EXTERNAL_ROOT" CL_WOMD=auto CL_MAX_SCENARIOS="$MAX_SCENARIOS" CL_MAX_STEPS="$MAX_STEPS" \
  CL_NUM_CANDIDATES="$NUM_CANDIDATES" CL_LABEL_MODE=fast \
  DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_NOMINAL_CONTROL=false RUN_LEGACY_SAFE=false \
  SKIP_COMPLETE_METHODS=true CL_RESUME_FORCE=false JOBS_PER_GPU="$jpg" MAX_PARALLEL="$mp" \
  XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \
  bash scripts/run_safe_regime_external_baselines.sh
}
run_near() {
  local jpg="$1" mp="$2"
  OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
  RUN="$NEAR_EXTERNAL_ROOT" CL_WOMD=auto CL_MAX_SCENARIOS="$MAX_SCENARIOS" CL_MAX_STEPS="$MAX_STEPS" \
  CL_NUM_CANDIDATES="$NUM_CANDIDATES" CL_NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" CL_LABEL_MODE=fast \
  DO_TRAIN=false DO_CALIBRATE=false DO_OFFLINE=false DO_CLOSED_LOOP=true \
  RUN_ORACLE_CLOSED_LOOP=false RUN_LEGACY_NEAR=false \
  SKIP_COMPLETE_METHODS=true CL_RESUME_FORCE=false JOBS_PER_GPU="$jpg" MAX_PARALLEL="$mp" \
  XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \
  bash scripts/run_near_contact_external_baselines_2gpu_optimized.sh
}
run_contact() {
  local jpg="$1" mp="$2"
  OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
  RUN="$CONTACT_EXTERNAL_ROOT" CL_WOMD=auto CL_MAX_SCENARIOS="$MAX_SCENARIOS" CL_MAX_STEPS="$MAX_STEPS" \
  CL_NUM_CANDIDATES="$NUM_CANDIDATES" CL_NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" CL_LABEL_MODE=fast \
  DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_LEGACY_CONTACT=false \
  SKIP_COMPLETE_METHODS=true CL_RESUME_FORCE=false JOBS_PER_GPU="$jpg" MAX_PARALLEL="$mp" \
  XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \
  bash scripts/run_contact_external_baselines.sh
}

run_with_memory_fallback() {
  local label="$1" fn="$2"
  echo "[$label] fast replay: JOBS_PER_GPU=$JOBS_PER_GPU MAX_PARALLEL=$MAX_PARALLEL"
  if "$fn" "$JOBS_PER_GPU" "$MAX_PARALLEL"; then
    return 0
  fi
  echo "[WARN] $label failed at high concurrency; retrying incomplete methods with JOBS_PER_GPU=$FALLBACK_JOBS_PER_GPU MAX_PARALLEL=$FALLBACK_MAX_PARALLEL" >&2
  "$fn" "$FALLBACK_JOBS_PER_GPU" "$FALLBACK_MAX_PARALLEL"
}

echo '[2/5] Safe external baselines: reuse checkpoints, metric-only replay.'
run_with_memory_fallback SAFE run_safe

echo '[3/5] Near-contact external baselines: reuse checkpoints + existing conformal calibration; skip audit-only selected teacher labels.'
run_with_memory_fallback NEAR run_near

echo '[4/5] Contact external baselines: controllers are already registered; metric-only replay.'
run_with_memory_fallback CONTACT run_contact

echo '[5/5] Rebuild paired comparison tables with corrected publication metrics.'
python tools/build_submission_external_baseline_tables.py \
  --ocrap-run "$OCRAP_SUBMISSION_ROOT" \
  --safe-run "$SAFE_EXTERNAL_ROOT" \
  --near-run "$NEAR_EXTERNAL_ROOT" \
  --contact-run "$CONTACT_EXTERNAL_ROOT" \
  --variants "$VARIANTS" \
  --output-dir "$TABLE_OUT"

python - "$OCRAP_SUBMISSION_ROOT" "$SAFE_EXTERNAL_ROOT" "$NEAR_EXTERNAL_ROOT" "$CONTACT_EXTERNAL_ROOT" <<'PY'
import json, pathlib, sys
expected = 'exact_oriented_box_signed_clearance+penetration+swept_sat_constant_velocity_ttc_v55'
roots = [pathlib.Path(x) for x in sys.argv[1:]]
files=[]
files += list((roots[0]).glob('*/*/closed_loop_ocrap.json'))
for root in roots[1:]: files += list(root.glob('closed_loop_*.json'))
bad=[]
checked=[]
for p in files:
    d=json.loads(p.read_text())
    got=((d.get('runtime_contract') or {}).get('publication_geometry_metric'))
    # closed_loop_summary.json and support documents share the filename prefix
    # but are not per-method aggregate artifacts.
    if got is None:
        continue
    checked.append(p)
    if got != expected: bad.append((str(p),got))
if bad:
    raise SystemExit('metric protocol verification failed: '+repr(bad[:5]))
if len(checked) < 24:
    raise SystemExit(f'expected at least 24 corrected main artifacts (6 OC-RAP regime/variant + 18 external), got {len(checked)}')
print(json.dumps({'event':'v55_metric_protocol_verified','artifacts':len(checked),'protocol':expected}))
PY

printf '\n[DONE] Corrected closed-loop results are in the original run roots.\n'
printf '  OC-RAP:   %s\n  Safe ext: %s\n  Near ext: %s\n  Contact:  %s\n  Tables:   %s\n' \
  "$OCRAP_SUBMISSION_ROOT" "$SAFE_EXTERNAL_ROOT" "$NEAR_EXTERNAL_ROOT" "$CONTACT_EXTERNAL_ROOT" "$TABLE_OUT"
printf 'Old metric artifacts were archived at: %s\n' "$backup"
printf '%s\n' '[IMPORTANT] Do NOT run repair_submission_visualization_stale_replays_v54.sh after this full rerun; all launchers already resolve CL_WOMD from dataset provenance with auto.'
