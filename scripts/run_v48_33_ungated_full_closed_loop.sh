#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

OUT="${OUT:?Set OUT to runs/ocrap_v48_33_eligible_set_policy_dedicated_4833}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
WOMD_VAL="${WOMD_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord}"
if [[ "$WOMD_VAL" == *@* ]]; then
  WOMD_SOURCE="$WOMD_VAL"
else
  WOMD_SOURCE="$WOMD_VAL@${WOMD_EXPECTED_SHARDS:-150}"
fi
SAFE_TEST="${SAFE_TEST:-$OCRAP_ROOT/test_safe}"
NEAR_TEST="${NEAR_TEST:-$OCRAP_ROOT/test_near_contact}"
CONTACT_TEST="${CONTACT_TEST:-$OCRAP_ROOT/test_contact}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
VARIANTS="${VARIANTS:-balanced,precision}"
FULL_ROOT="${FULL_ROOT:-$OUT/ungated_full_closed_loop}"
mkdir -p "$FULL_ROOT"

# The main v48.33 controller correctly blocks test roots when Natural gate fails.
# This script is a deliberately separate exploratory path.  It never edits the
# official V48_33_COMPLETE.json and requires an explicit opt-in.
python - "$OUT" "$FULL_ROOT" "${ALLOW_UNGATED_TEST:-0}" <<'PY_PREFLIGHT'
import json, pathlib, sys, time
out=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2])
allow=str(sys.argv[3]).strip().lower() in {'1','true','yes','on'}
status_path=out/'V48_33_COMPLETE.json'
try:
    status=json.load(open(status_path))
except Exception as exc:
    raise SystemExit(f'missing_or_invalid_main_status:{status_path}:{exc}')
if not bool(status.get('pipeline_valid')):
    raise SystemExit('main_pipeline_invalid; do not run closed-loop test')
if not bool(status.get('gate_evaluated')):
    raise SystemExit('deployment gate was not evaluated')
if not bool(status.get('gate_passed')) and not allow:
    raise SystemExit('Natural gate failed. Re-run only with explicit ALLOW_UNGATED_TEST=1 for exploratory diagnostics.')
doc={
  'event':'v48_33_ungated_full_closed_loop_preflight',
  'created_unix':time.time(),
  'pipeline_valid':bool(status.get('pipeline_valid')),
  'gate_evaluated':bool(status.get('gate_evaluated')),
  'gate_passed':bool(status.get('gate_passed')),
  'explicit_ungated_opt_in':allow,
  'reads_test_roots':True,
  'exploratory_only':not bool(status.get('gate_passed')),
  'paper_result':False if not bool(status.get('gate_passed')) else None,
  'valid_for_deployment':bool(status.get('gate_passed')),
  'main_status_path':str(status_path),
}
(root/'UNGATED_EXPLORATORY_ONLY.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
PY_PREFLIGHT

CANDIDATE_ROOT="${CANDIDATE_ROOT:-$OUT/candidates}"
if [[ ! -d "$CANDIDATE_ROOT" && -d "$OUT/dedicated_candidates" ]]; then
  CANDIDATE_ROOT="$OUT/dedicated_candidates"
fi

count_targets() {
  local dataset="$1" split="$2"
  python tools/count_closed_loop_targets.py --dataset "$dataset" --split "$split" --max-targets-per-scene 1 --count-only
}

SAFE_COUNT="${SAFE_COUNT:-$(count_targets "$SAFE_TEST" test)}"
NEAR_COUNT="${NEAR_COUNT:-$(count_targets "$NEAR_TEST" test)}"
CONTACT_COUNT="${CONTACT_COUNT:-$(count_targets "$CONTACT_TEST" test)}"
for pair in "safe:$SAFE_COUNT" "near_contact:$NEAR_COUNT" "contact:$CONTACT_COUNT"; do
  name="${pair%%:*}"; count="${pair##*:}"
  [[ "$count" =~ ^[0-9]+$ && "$count" -gt 0 ]] || { echo "invalid/empty $name target count: $count" >&2; exit 31; }
done
printf 'full closed-loop target counts: safe=%s near=%s contact=%s\n' "$SAFE_COUNT" "$NEAR_COUNT" "$CONTACT_COUNT"

python - "$FULL_ROOT" "$SAFE_COUNT" "$NEAR_COUNT" "$CONTACT_COUNT" "$SAFE_TEST" "$NEAR_TEST" "$CONTACT_TEST" <<'PY_COUNTS'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1])
doc={'created_unix':time.time(),'split':'test','max_targets_per_scene':1,
     'counts':{'safe':int(sys.argv[2]),'near_contact':int(sys.argv[3]),'contact':int(sys.argv[4])},
     'datasets':{'safe':sys.argv[5],'near_contact':sys.argv[6],'contact':sys.argv[7]}}
(root/'FULL_TARGET_COUNTS.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
PY_COUNTS

run_variant() {
  local variant="$1"
  local base="$CANDIDATE_ROOT/$variant"
  local run="$FULL_ROOT/$variant"
  mkdir -p "$run"
  [[ -f "$base/model_v48_trac_sr/best.pt" ]] || { echo "missing checkpoint: $base/model_v48_trac_sr/best.pt" >&2; return 32; }
  [[ -f "$base/calibration/gamma_rec_by_bucket_v48.json" ]] || { echo "missing gamma map: $base" >&2; return 32; }
  [[ -f "$base/calibration/direct_value_risk_near_v48.json" ]] || { echo "missing near selector certificate: $base" >&2; return 32; }
  [[ -f "$base/calibration/direct_value_risk_contact_v48.json" ]] || { echo "missing contact selector certificate: $base" >&2; return 32; }

  # Near and Contact are run in parallel by regime on GPU0/GPU1; control then
  # OC-RAP are sequential on each card to preserve exact paired target order.
  DEV_SHADOW_DIAGNOSTIC=1 \
  BASE_RUN="$base" RUN="$run" OCRAP_ROOT="$OCRAP_ROOT" \
  WOMD_VAL="$WOMD_VAL" DEV_SHADOW_WOMD_SOURCE="$WOMD_SOURCE" SAFE_WOMD_SOURCE="$WOMD_SOURCE" \
  SAFE_TEST="$SAFE_TEST" NEAR_TEST="$NEAR_TEST" CONTACT_TEST="$CONTACT_TEST" \
  BUCKET_SPLIT=test SAFE_BUCKET_SPLIT=test \
  RUN_OFFLINE_EVAL=0 RUN_AUDITS=1 RUN_SAFE_CLOSED_LOOP=1 \
  RUN_SCALAR_BASELINES=1 RUN_SAFE_PAIRED_SCALAR=1 PAIR_BUCKETS_ON_TWO_GPUS=1 \
  RUN_NEAR_AUDIT=1 RUN_CONTACT_AUDIT=1 \
  GPU_NEAR="$GPU0" GPU_CONTACT="$GPU1" GPU_SAFE_BASELINE="$GPU0" GPU_SAFE="$GPU1" \
  NEAR_AUDIT_TARGETS="$NEAR_COUNT" CONTACT_AUDIT_TARGETS="$CONTACT_COUNT" \
  NEAR_AUDIT_MAX_ROLLOUTS="$NEAR_COUNT" CONTACT_AUDIT_MAX_ROLLOUTS="$CONTACT_COUNT" \
  SAFE_MAX_TARGETS="$SAFE_COUNT" SAFE_MAX_ROLLOUTS="$SAFE_COUNT" \
  SAFE_MAX_STEPS="${SAFE_MAX_STEPS:-40}" \
  NEAR_AUDIT_MAX_STEPS="${NEAR_MAX_STEPS:-50}" \
  CONTACT_AUDIT_MAX_STEPS="${CONTACT_MAX_STEPS:-60}" \
  AUDIT_LABEL_MODE="${FULL_LABEL_MODE:-fast}" AUDIT_LABELS="${FULL_AUDIT_LABELS:-0}" \
  AUDIT_EVERY_N_STEPS="${FULL_AUDIT_EVERY_N_STEPS:-8}" \
  AUDIT_TOP_K="${FULL_AUDIT_TOP_K:-5}" AUDIT_MAX_EXTRA_CANDIDATES="${FULL_AUDIT_MAX_EXTRA_CANDIDATES:-2}" \
  AUDIT_NUM_CANDIDATES="${FULL_NUM_CANDIDATES:-12}" AUDIT_NUM_RECOVERY_OPTIONS="${FULL_NUM_RECOVERY_OPTIONS:-8}" \
  SAVE_CLOSED_LOOP_TRACES=true CL_RESUME="${CL_RESUME:-true}" \
  DEV_SHADOW_ALLOW_LEGACY_SOURCE_INDEX_TARGETS="${ALLOW_LEGACY_SOURCE_INDEX_TARGETS:-false}" \
  bash scripts/run_ocrap_v48_trac_sr.sh >"$run/full_controller.log" 2>&1

  local safe_control="$run/closed_loop_safe_fast_v48_scalar.json"
  local safe_method="$run/closed_loop_safe_fast_v48.json"
  local near_control="$run/audit_near_contact_selected_topk_v48_scalar.json"
  local near_method="$run/audit_near_contact_selected_topk_v48_v48.json"
  local contact_control="$run/audit_contact_selected_topk_v48_scalar.json"
  local contact_method="$run/audit_contact_selected_topk_v48_v48.json"
  for path in "$safe_control" "$safe_method" "$near_control" "$near_method" "$contact_control" "$contact_method"; do
    [[ -s "$path" ]] || { echo "missing closed-loop output: $path" >&2; return 33; }
  done

  python tools/compare_paired_closed_loop.py "$safe_control" "$safe_method" \
    --output "$run/safe_test_paired.json" --bootstrap "${FULL_BOOTSTRAP:-5000}"
  python tools/compare_paired_closed_loop.py "$near_control" "$near_method" \
    --output "$run/near_test_paired.json" --bootstrap "${FULL_BOOTSTRAP:-5000}"
  python tools/compare_paired_closed_loop.py "$contact_control" "$contact_method" \
    --output "$run/contact_test_paired.json" --bootstrap "${FULL_BOOTSTRAP:-5000}"

  python tools/select_critical_closed_loop_scenes.py "$near_control" "$near_method" \
    --regime near_contact --output "$run/critical_near_contact.json" \
    --top-k-each "${CRITICAL_TOP_K_EACH:-8}" --max-selected "${CRITICAL_MAX_SELECTED:-24}"
  python tools/select_critical_closed_loop_scenes.py "$contact_control" "$contact_method" \
    --regime contact --output "$run/critical_contact.json" \
    --top-k-each "${CRITICAL_TOP_K_EACH:-8}" --max-selected "${CRITICAL_MAX_SELECTED:-24}"

  python tools/visualize_closed_loop_critical.py "$near_control" "$near_method" "$run/critical_near_contact.json" \
    --output-dir "$run/visualizations/near_contact" --max-scenes "${CRITICAL_MAX_SELECTED:-24}" --dpi "${VIS_DPI:-160}"
  python tools/visualize_closed_loop_critical.py "$contact_control" "$contact_method" "$run/critical_contact.json" \
    --output-dir "$run/visualizations/contact" --max-scenes "${CRITICAL_MAX_SELECTED:-24}" --dpi "${VIS_DPI:-160}"

  python tools/summarize_ungated_closed_loop.py \
    --safe "$run/safe_test_paired.json" --near "$run/near_test_paired.json" --contact "$run/contact_test_paired.json" \
    --variant "$variant" --output "$run/UNGATED_FULL_CLOSED_LOOP_SUMMARY.json"

  # Optional: rerun only selected critical targets with sparse teacher/PCD labels.
  # This does not replace the fast physical results; it adds diagnostic audits.
  if [[ "${RERUN_CRITICAL_WITH_LABELS:-0}" == "1" ]]; then
    local audit_run="$run/critical_label_audit"
    local near_selected contact_selected
    near_selected="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["selected_keys"]))' "$run/critical_near_contact.json")"
    contact_selected="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["selected_keys"]))' "$run/critical_contact.json")"
    mkdir -p "$audit_run"
    DEV_SHADOW_DIAGNOSTIC=1 \
    BASE_RUN="$base" RUN="$audit_run" OCRAP_ROOT="$OCRAP_ROOT" \
    WOMD_VAL="$WOMD_VAL" DEV_SHADOW_WOMD_SOURCE="$WOMD_SOURCE" \
    NEAR_TEST="$NEAR_TEST" CONTACT_TEST="$CONTACT_TEST" BUCKET_SPLIT=test \
    RUN_OFFLINE_EVAL=0 RUN_AUDITS=1 RUN_SAFE_CLOSED_LOOP=0 RUN_SCALAR_BASELINES=0 \
    RUN_NEAR_AUDIT=1 RUN_CONTACT_AUDIT=1 PAIR_BUCKETS_ON_TWO_GPUS=1 \
    GPU_NEAR="$GPU0" GPU_CONTACT="$GPU1" \
    NEAR_AUDIT_TARGETS="$near_selected" CONTACT_AUDIT_TARGETS="$contact_selected" \
    NEAR_AUDIT_MAX_ROLLOUTS="$near_selected" CONTACT_AUDIT_MAX_ROLLOUTS="$contact_selected" \
    NEAR_AUDIT_TARGET_KEYS_FILE="$run/critical_near_contact.json" \
    CONTACT_AUDIT_TARGET_KEYS_FILE="$run/critical_contact.json" \
    NEAR_AUDIT_MAX_STEPS="${NEAR_MAX_STEPS:-50}" CONTACT_AUDIT_MAX_STEPS="${CONTACT_MAX_STEPS:-60}" \
    AUDIT_LABEL_MODE=selected_topk AUDIT_LABELS="${CRITICAL_AUDIT_LABELS:-1024}" \
    AUDIT_EVERY_N_STEPS="${CRITICAL_AUDIT_EVERY_N_STEPS:-2}" AUDIT_TOP_K="${CRITICAL_AUDIT_TOP_K:-5}" \
    AUDIT_MAX_EXTRA_CANDIDATES="${CRITICAL_AUDIT_MAX_EXTRA_CANDIDATES:-3}" \
    SAVE_CLOSED_LOOP_TRACES=true CL_RESUME="${CL_RESUME:-true}" \
    DEV_SHADOW_ALLOW_LEGACY_SOURCE_INDEX_TARGETS="${ALLOW_LEGACY_SOURCE_INDEX_TARGETS:-false}" \
    bash scripts/run_ocrap_v48_trac_sr.sh >"$audit_run/controller.log" 2>&1
  fi

  python - "$run" "$variant" <<'PY_DONE'
import json,pathlib,sys,time
run=pathlib.Path(sys.argv[1]); variant=sys.argv[2]
(run/'UNGATED_VARIANT_COMPLETE.json').write_text(json.dumps({
  'event':'v48_33_ungated_variant_complete','created_unix':time.time(),
  'variant':variant,'complete':True,'exploratory_only':True,
  'visualization_index_near':str(run/'visualizations/near_contact/index.html'),
  'visualization_index_contact':str(run/'visualizations/contact/index.html'),
},ensure_ascii=False,indent=2)+'\n')
PY_DONE
}

IFS=',' read -r -a variant_array <<< "$VARIANTS"
for variant in "${variant_array[@]}"; do
  variant="${variant//[[:space:]]/}"
  [[ -n "$variant" ]] || continue
  echo "===== full ungated closed-loop: $variant ====="
  run_variant "$variant"
done

python - "$FULL_ROOT" "$VARIANTS" <<'PY_COMPLETE'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); variants=[x.strip() for x in sys.argv[2].split(',') if x.strip()]
complete=all((root/v/'UNGATED_VARIANT_COMPLETE.json').is_file() for v in variants)
doc={'event':'v48_33_ungated_full_closed_loop_complete','created_unix':time.time(),
     'variants':variants,'complete':complete,'exploratory_only':True,'valid_for_deployment':False}
(root/'UNGATED_FULL_CLOSED_LOOP_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
if not complete: raise SystemExit(34)
PY_COMPLETE

echo "Completed exploratory full closed-loop outputs under: $FULL_ROOT"
