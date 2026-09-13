#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=scripts/lib/runtime.sh
source scripts/lib/runtime.sh

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${OUT:=runs/ocrap_nominal_three_regime}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${WOMD_NUM_SHARDS:=150}"
: "${SAFE_BUCKET:=$OCRAP_ROOT/test_safe}"
: "${NEAR_BUCKET:=$OCRAP_ROOT/test_near_contact}"
: "${CONTACT_BUCKET:=$OCRAP_ROOT/test_contact}"
: "${BUCKET_SPLIT:=test}"
: "${SAFE_WOMD:=auto}"
: "${NEAR_WOMD:=auto}"
: "${CONTACT_WOMD:=auto}"
: "${CUDA_DEVICES:=0,1}"
: "${MAX_SCENARIOS:=0}"
: "${MAX_STEPS:=40}"
: "${REPLAN_INTERVAL:=1}"
: "${LABEL_MODE:=fast}"
: "${INCLUDE_SCENES_IN_RESULT:=true}"
: "${RESULT_SCENE_DETAIL:=metrics}"
: "${RESUME_FORCE:=false}"

: "${RUN_SAFE:=1}"
: "${RUN_NEAR:=1}"
: "${RUN_CONTACT:=1}"
: "${CONTACT_TARGET_KEYS_FILE:=}"
: "${CONTACT_ANCHOR_PRELUDE_ENABLED:=false}"
: "${CONTACT_ANCHOR_PRELUDE_MAX_STEPS:=60}"
: "${CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL:=1}"
: "${CONTACT_ANCHOR_REQUIRE_FOUND:=true}"
: "${CONTACT_ANCHOR_MANIFEST_FILE:=}"

resolve_spec() {
  local value="$1" bucket="$2"
  if [[ "${value,,}" == auto ]]; then
    runtime_resolve_bucket_womd_spec "$bucket" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "${WOMD_ROLE:-auto}"
  else
    runtime_normalize_womd_spec "$value" "$WOMD_NUM_SHARDS"
  fi
}
SAFE_WOMD="$(resolve_spec "$SAFE_WOMD" "$SAFE_BUCKET")"
NEAR_WOMD="$(resolve_spec "$NEAR_WOMD" "$NEAR_BUCKET")"
CONTACT_WOMD="$(resolve_spec "$CONTACT_WOMD" "$CONTACT_BUCKET")"

nominal_exact_a0_ok() {
  local output="$1"
  python - "$output" <<'PY'
import json, math, sys
p=sys.argv[1]
try:
    d=json.load(open(p,encoding='utf-8'))
except Exception:
    raise SystemExit(1)
if str(d.get('method','')).lower() != 'nominal':
    raise SystemExit(1)
rate=d.get('intervention_rate')
try:
    if not math.isfinite(float(rate)) or abs(float(rate)) > 1e-12:
        raise SystemExit(1)
except Exception:
    raise SystemExit(1)
reasons=d.get('selection_reason_counts') or {}
if set(reasons) != {'nominal_prefix_exact_a0'}:
    raise SystemExit(1)
for scene in d.get('scenes') or []:
    try:
        if abs(float(scene.get('intervention_rate',0.0))) > 1e-12:
            raise SystemExit(1)
    except Exception:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"; ((${#GPUS[@]})) || GPUS=(0)
mkdir -p "$OUT/safe" "$OUT/near" "$OUT/contact"

run_one() {
  local regime="$1" womd="$2" bucket="$3" gpu="$4" target_keys="${5:-}"
  local run_dir="$OUT/$regime" output="$OUT/$regime/closed_loop_nominal.json"
  local extra=()
  if [[ -n "$target_keys" ]]; then
    extra+=(--set "closed_loop.target_keys_file=$target_keys" --set closed_loop.require_target_keys=true)
  fi
  if [[ "$regime" == contact && "${CONTACT_ANCHOR_PRELUDE_ENABLED,,}" == true ]]; then
    extra+=(
      --set closed_loop.contact_anchor_prelude_enabled=true
      --set "closed_loop.contact_anchor_prelude_max_steps=$CONTACT_ANCHOR_PRELUDE_MAX_STEPS"
      --set "closed_loop.contact_anchor_prelude_replan_interval_steps=$CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL"
      --set "closed_loop.contact_anchor_require_found=$CONTACT_ANCHOR_REQUIRE_FOUND"
    )
    [[ -n "$CONTACT_ANCHOR_MANIFEST_FILE" ]] && extra+=(--set "closed_loop.contact_anchor_manifest_file=$CONTACT_ANCHOR_MANIFEST_FILE")
  fi
  mkdir -p "$run_dir"
  if python tools/check_closed_loop_artifact.py --output "$output" --method nominal --bucket-dataset "$bucket" --quiet \
     && nominal_exact_a0_ok "$output"; then
    echo "[REUSE] exact-a0 nominal $regime complete: $output"
    return 0
  fi
  support_args=()
  if [[ -n "$target_keys" ]]; then support_args=(--target-keys-file "$target_keys" --require-target-keys); fi
  python tools/check_closed_loop_dataset_support.py --dataset "$bucket" --split "$BUCKET_SPLIT" \
    --womd-pattern "$womd" --expected-source-role "${WOMD_ROLE:-auto}" "${support_args[@]}" --output "$run_dir/closed_loop_dataset_support.json"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
  export JAX_COMPILATION_CACHE_DIR="$run_dir/.jax_compilation_cache"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
  python -u -m ocrap.cli closed-loop \
    --config configs/external_baselines/nominal_log_replay.yaml \
    --dataset "$womd" --output "$output" \
    --set closed_loop.method=nominal \
    --set "closed_loop.max_scenarios=$MAX_SCENARIOS" \
    --set "closed_loop.max_bucket_targets=$MAX_SCENARIOS" \
    --set "closed_loop.bucket_dataset=$bucket" \
    --set "closed_loop.bucket_split=$BUCKET_SPLIT" \
    --set closed_loop.require_bucket_targets=true \
    --set closed_loop.max_targets_per_scene=1 \
    --set "closed_loop.max_steps=$MAX_STEPS" \
    --set "closed_loop.replan_interval_steps=$REPLAN_INTERVAL" \
    --set "closed_loop.label_mode=$LABEL_MODE" \
    --set closed_loop.render_trace=false \
    --set closed_loop.save_partial=true \
    --set "closed_loop.resume_force=$RESUME_FORCE" \
    --set closed_loop.resume=true \
    --set closed_loop.result_scene_detail="$RESULT_SCENE_DETAIL" \
    --set closed_loop.scene_journal_detail="$RESULT_SCENE_DETAIL" \
    --set closed_loop.memory_scene_detail="$RESULT_SCENE_DETAIL" \
    --set "closed_loop.include_scenes_in_result=$INCLUDE_SCENES_IN_RESULT" \
    --set closed_loop.include_scenes_in_partial=false \
    --set waymax.dataloader_include_sdc_paths=false \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.use_jit_scan_rollouts=true \
    "${extra[@]}" \
    2>&1 | tee -a "$run_dir/closed_loop_nominal.log"
  python tools/check_closed_loop_artifact.py --output "$output" --method nominal --bucket-dataset "$bucket"
  nominal_exact_a0_ok "$output" || { echo "nominal control violated exact-a0 semantics: $output" >&2; return 30; }
}

failed=0
if ((${#GPUS[@]} >= 2)); then
  p0= p1=
  if [[ "$RUN_SAFE" == 1 ]]; then run_one safe "$SAFE_WOMD" "$SAFE_BUCKET" "${GPUS[0]}" & p0=$!; fi
  if [[ "$RUN_NEAR" == 1 ]]; then run_one near "$NEAR_WOMD" "$NEAR_BUCKET" "${GPUS[1]}" & p1=$!; fi
  [[ -z "$p0" ]] || wait "$p0" || failed=1
  [[ -z "$p1" ]] || wait "$p1" || failed=1
  if [[ "$RUN_CONTACT" == 1 ]]; then run_one contact "$CONTACT_WOMD" "$CONTACT_BUCKET" "${GPUS[0]}" "$CONTACT_TARGET_KEYS_FILE" || failed=1; fi
else
  [[ "$RUN_SAFE" != 1 ]] || run_one safe "$SAFE_WOMD" "$SAFE_BUCKET" "${GPUS[0]}" || failed=1
  [[ "$RUN_NEAR" != 1 ]] || run_one near "$NEAR_WOMD" "$NEAR_BUCKET" "${GPUS[0]}" || failed=1
  [[ "$RUN_CONTACT" != 1 ]] || run_one contact "$CONTACT_WOMD" "$CONTACT_BUCKET" "${GPUS[0]}" "$CONTACT_TARGET_KEYS_FILE" || failed=1
fi
((failed==0)) || exit 30
