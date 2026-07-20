#!/usr/bin/env bash
set -euo pipefail

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${TRAIN_CONTACT:=$OCRAP_ROOT/train_contact}"
: "${VAL_CONTACT:=$OCRAP_ROOT/val_contact}"
: "${TEST_CONTACT:=$OCRAP_ROOT/test_contact}"
: "${RUN:=runs/contact_external_baselines}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}"
: "${CL_WOMD:=${WOMD_VAL_INTERACTIVE}@150}"
: "${CL_MAX_SCENARIOS:=50}"
: "${CL_MAX_STEPS:=40}"
: "${CL_LABEL_MAX_CANDIDATES:=10}"
: "${CL_NUM_RECOVERY_OPTIONS:=12}"
: "${CL_TEACHER_TOP_K_OPTIONS:=4}"
: "${CL_EXHAUSTIVE_TEACHER_LABELS:=false}"
: "${CL_SAVE_PARTIAL:=false}"
: "${CL_PROFILE_TIMING:=true}"
: "${DO_TRAIN_NONLEARNED:=false}"
: "${DO_OFFLINE:=true}"
: "${DO_CLOSED_LOOP:=true}"

: "${CUDA_DEVICES:=0}"
IFS=',' read -r -a GPU_LIST <<< "$CUDA_DEVICES"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then GPU_LIST=(0); fi
: "${MAX_PARALLEL:=${#GPU_LIST[@]}}"
if [ "$MAX_PARALLEL" -lt 1 ]; then MAX_PARALLEL=1; fi
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)"
: "${THREADS_PER_JOB:=$(( CPU_COUNT / MAX_PARALLEL ))}"
if [ "$THREADS_PER_JOB" -lt 1 ]; then THREADS_PER_JOB=1; fi
: "${JAX_CACHE_DIR:=$RUN/.jax_compilation_cache}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
: "${WARM_JAX_CACHE:=true}"
: "${WARMUP_MAX_STEPS:=2}"

export RUN
mkdir -p "$RUN" "$JAX_CACHE_DIR"

CONTACT_METHODS=(
  postimpact_mpc_lite
  post_crash_braking
  post_collision_restoration
  severity_minimization
)
CONTACT_CSV="$(IFS=,; echo "${CONTACT_METHODS[*]}")"

run_with_runtime_env() {
  local gpu="$1"; shift
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="$THREADS_PER_JOB" \
    MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
    NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \
    JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR" \
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
    PYTHONUNBUFFERED=1 \
    "$@"
}

run_parallel_batches() {
  local runner="$1"; shift
  local items=("$@")
  local total="${#items[@]}"
  local base j idx failed pid
  for ((base=0; base<total; base+=MAX_PARALLEL)); do
    local pids=()
    local names=()
    failed=0
    for ((j=0; j<MAX_PARALLEL && base+j<total; j++)); do
      idx=$((base+j))
      "$runner" "${items[$idx]}" "$idx" &
      pids+=("$!")
      names+=("${items[$idx]}")
    done
    for j in "${!pids[@]}"; do
      pid="${pids[$j]}"
      if ! wait "$pid"; then
        echo "[ERROR] parallel job failed: ${names[$j]}" >&2
        failed=1
      fi
    done
    if [ "$failed" -ne 0 ]; then return 1; fi
  done
}

register_method() {
  local method="$1" idx="$2"
  local gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  run_with_runtime_env "$gpu" python -u -m ocrap.cli train-baseline \
    --config configs/external_baselines/contact_external_baselines.yaml \
    --dataset "$TRAIN_CONTACT" --val-dataset "$VAL_CONTACT" \
    --baseline "$method" --output "$RUN/$method" \
    --set external_baselines.training.validate_dataset=false \
    2>&1 | tee "$RUN/train_${method}.log"
}

warm_jax_cache() {
  local gpu="${GPU_LIST[0]}"
  local out="$RUN/.warmup_contact.json"
  rm -f "$out" "$out.partial.jsonl"
  echo "[WARMUP] compiling shared Waymax/teacher kernels on gpu=$gpu"
  run_with_runtime_env "$gpu" python -u -m ocrap.cli closed-loop \
    --config configs/external_baselines/contact_external_baselines.yaml \
    --dataset "$CL_WOMD" \
    --output "$out" \
    --set closed_loop.method=postimpact_mpc_lite \
    --set closed_loop.max_scenarios=1 \
    --set closed_loop.max_steps="$WARMUP_MAX_STEPS" \
    --set closed_loop.replan_interval_steps=1 \
    --set closed_loop.label_mode=all \
    --set closed_loop.external_sparse_labels=true \
    --set closed_loop.external_label_max_candidates="$CL_LABEL_MAX_CANDIDATES" \
    --set closed_loop.external_label_macro_diversity=true \
    --set closed_loop.exhaustive_teacher_labels="$CL_EXHAUSTIVE_TEACHER_LABELS" \
    --set closed_loop.num_candidate_prefixes=24 \
    --set closed_loop.num_recovery_options="$CL_NUM_RECOVERY_OPTIONS" \
    --set closed_loop.save_partial=false \
    --set closed_loop.profile_timing=false \
    --set waymax.dataloader_include_sdc_paths=false \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.teacher_rollout_top_k_options="$CL_TEACHER_TOP_K_OPTIONS" \
    --set waymax.use_jit_scan_rollouts=true \
    2>&1 | tee "$RUN/warmup_contact.log"
}

run_closed_loop_method() {
  local method="$1" idx="$2"
  local gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  echo "[START] method=$method gpu=$gpu" | tee "$RUN/closed_loop_${method}.launch.log"
  run_with_runtime_env "$gpu" python -u -m ocrap.cli closed-loop \
    --config configs/external_baselines/contact_external_baselines.yaml \
    --dataset "$CL_WOMD" \
    --output "$RUN/closed_loop_${method}.json" \
    --set closed_loop.method="$method" \
    --set closed_loop.max_scenarios="$CL_MAX_SCENARIOS" \
    --set closed_loop.max_steps="$CL_MAX_STEPS" \
    --set closed_loop.replan_interval_steps=1 \
    --set closed_loop.label_mode=all \
    --set closed_loop.external_sparse_labels=true \
    --set closed_loop.external_label_max_candidates="$CL_LABEL_MAX_CANDIDATES" \
    --set closed_loop.external_label_macro_diversity=true \
    --set closed_loop.exhaustive_teacher_labels="$CL_EXHAUSTIVE_TEACHER_LABELS" \
    --set closed_loop.num_candidate_prefixes=24 \
    --set closed_loop.num_recovery_options="$CL_NUM_RECOVERY_OPTIONS" \
    --set closed_loop.save_partial="$CL_SAVE_PARTIAL" \
    --set closed_loop.profile_timing="$CL_PROFILE_TIMING" \
    --set waymax.dataloader_include_sdc_paths=false \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.teacher_rollout_top_k_options="$CL_TEACHER_TOP_K_OPTIONS" \
    --set waymax.use_jit_scan_rollouts=true \
    2>&1 | tee "$RUN/closed_loop_${method}.log"
  echo "[DONE] method=$method gpu=$gpu"
}

if [ "$DO_TRAIN_NONLEARNED" = "true" ]; then
  run_parallel_batches register_method "${CONTACT_METHODS[@]}"
fi

if [ "$DO_OFFLINE" = "true" ]; then
  run_with_runtime_env "${GPU_LIST[0]}" python -u -m ocrap.cli evaluate-baseline \
    --config configs/external_baselines/contact_external_baselines.yaml \
    --dataset "$TEST_CONTACT" --split test \
    --output "$RUN/eval_contact_external_baselines.json" \
    --baselines "$CONTACT_CSV" \
    2>&1 | tee "$RUN/eval_contact_external_baselines.log"
fi

if [ "$DO_CLOSED_LOOP" = "true" ]; then
  if [ "$WARM_JAX_CACHE" = "true" ]; then warm_jax_cache; fi
  run_parallel_batches run_closed_loop_method "${CONTACT_METHODS[@]}"
fi

python - <<'PY'
import glob, json, os
run=os.environ['RUN']
rows=[]
for p in sorted(glob.glob(os.path.join(run,'closed_loop_*.json'))):
    try:
        d=json.load(open(p))
    except Exception:
        continue
    row={k:d.get(k) for k in [
        'method','num_scenes','num_decisions','closed_loop_FRA_exec',
        'closed_loop_FRA_cand','closed_loop_DRS','closed_loop_ODG',
        'closed_loop_post_contact_deployability','closed_loop_bounded_NUP',
        'intervention_rate']}
    row['timing_per_decision_s']=(d.get('timing') or {}).get('per_decision_s')
    rows.append(row)
out=os.path.join(run,'closed_loop_summary.json')
json.dump(rows, open(out,'w'), indent=2)
print({'event':'contact_closed_loop_summary','output':out,'num_methods':len(rows)})
PY
