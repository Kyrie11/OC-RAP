#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${TRAIN_NEAR:=$OCRAP_ROOT/train_near_contact}"
: "${VAL_NEAR:=$OCRAP_ROOT/val_near_contact}"
: "${TEST_NEAR:=$OCRAP_ROOT/test_near_contact}"
: "${RUN:=runs/near_contact_external_baselines_optimized}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}"
: "${CL_WOMD:=${WOMD_VAL_INTERACTIVE}@150}"
: "${CL_MAX_SCENARIOS:=50}"
: "${CL_BUCKET_DATASET:=${TEST_NEAR}}"
: "${CL_BUCKET_SPLIT:=test}"
: "${CL_MAX_TARGETS_PER_SCENE:=1}"
: "${CL_RENDER_TRACE:=false}"
: "${CL_RENDER_MAX_AGENTS:=64}"
: "${CL_PREFLIGHT:=true}"
: "${CL_ORACLE_MAX_SCENARIOS:=$CL_MAX_SCENARIOS}"
: "${CL_MAX_STEPS:=40}"
: "${CL_REPLAN_INTERVAL_STEPS:=1}"
: "${CL_NUM_CANDIDATES:=24}"
: "${CL_AUDIT_EVERY_N_STEPS:=1}"
: "${CL_NUM_RECOVERY_OPTIONS:=12}"
: "${CL_SAVE_PARTIAL:=true}"
: "${CL_PROFILE_TIMING:=true}"
: "${DO_OFFLINE:=true}"
: "${DO_CLOSED_LOOP:=true}"
: "${TRAIN_GAMEFORMER_IF_MISSING:=true}"
: "${FORCE_RETRAIN_GAMEFORMER:=false}"
: "${GAMEFORMER_CHECKPOINT:=$RUN/gameformer_lite/best.pt}"
: "${CUDA_DEVICES:=0,1}"
: "${GAMEFORMER_GLOBAL_BATCH_SIZE:=64}"
: "${GAMEFORMER_NUM_WORKERS_TOTAL:=8}"
: "${GAMEFORMER_TRAIN_GPUS:=2}"

IFS=',' read -r -a GPU_LIST <<< "$CUDA_DEVICES"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then GPU_LIST=(0 1); fi
: "${MAX_PARALLEL:=${#GPU_LIST[@]}}"
if [ "$MAX_PARALLEL" -gt "${#GPU_LIST[@]}" ]; then MAX_PARALLEL="${#GPU_LIST[@]}"; fi
if [ "$GAMEFORMER_TRAIN_GPUS" -gt "${#GPU_LIST[@]}" ]; then GAMEFORMER_TRAIN_GPUS="${#GPU_LIST[@]}"; fi

CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)"
: "${THREADS_PER_JOB:=$(( CPU_COUNT / (2 * MAX_PARALLEL) ))}"
if [ "$THREADS_PER_JOB" -lt 1 ]; then THREADS_PER_JOB=1; fi
if [ "$THREADS_PER_JOB" -gt 8 ]; then THREADS_PER_JOB=8; fi
: "${JAX_CACHE_DIR:=$RUN/.jax_compilation_cache}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
export RUN
mkdir -p "$RUN" "$JAX_CACHE_DIR"

if [ "$DO_CLOSED_LOOP" = true ] && [ "$CL_PREFLIGHT" = true ]; then
  python tools/check_closed_loop_dataset_support.py \
    --dataset "$CL_BUCKET_DATASET" --split "$CL_BUCKET_SPLIT" --womd-pattern "$CL_WOMD" \
    --expected-source-role auto --output "$RUN/closed_loop_dataset_support.json"
fi


# Longest/most variable jobs first; the dynamic scheduler immediately backfills
# whichever GPU finishes. This balances wall-clock load, not just job count.
CLOSED_LOOP_METHODS=(
  oracle_recovery_filter
  gameformer_lite
  marc_lite
  racp_lite
  predictive_safety_filter
  dro_cvar_filter
  cvar_risk_filter
  expected_risk_filter
)
NONLEARNED=(marc_lite racp_lite expected_risk_filter cvar_risk_filter dro_cvar_filter predictive_safety_filter oracle_recovery_filter)
NONLEARNED_CSV="$(IFS=,; echo "${NONLEARNED[*]}")"

common_env=(
  OMP_NUM_THREADS="$THREADS_PER_JOB"
  MKL_NUM_THREADS="$THREADS_PER_JOB"
  OPENBLAS_NUM_THREADS="$THREADS_PER_JOB"
  NUMEXPR_NUM_THREADS="$THREADS_PER_JOB"
  XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE"
  TF_FORCE_GPU_ALLOW_GROWTH=true
  JAX_ENABLE_X64=0
  JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR"
  JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  PYTHONUNBUFFERED=1
)

run_env_gpu() {
  local gpu="$1"; shift
  env CUDA_VISIBLE_DEVICES="$gpu" "${common_env[@]}" "$@"
}

run_env_cpu() {
  env CUDA_VISIBLE_DEVICES="" "${common_env[@]}" "$@"
}

join_first_gpus() {
  local count="$1" out="" i
  for ((i=0; i<count; i++)); do
    [ -n "$out" ] && out+=","
    out+="${GPU_LIST[$i]}"
  done
  printf '%s' "$out"
}

train_gameformer() {
  local visible
  visible="$(join_first_gpus "$GAMEFORMER_TRAIN_GPUS")"
  local command=(python -u -m ocrap.cli train-baseline)
  if [ "$GAMEFORMER_TRAIN_GPUS" -gt 1 ]; then
    command=(torchrun --standalone --nproc_per_node="$GAMEFORMER_TRAIN_GPUS" -m ocrap.cli train-baseline)
  fi
  env CUDA_VISIBLE_DEVICES="$visible" "${common_env[@]}" "${command[@]}" \
    --config configs/external_baselines/near_contact_gameformer_lite.yaml \
    --dataset "$TRAIN_NEAR" --val-dataset "$VAL_NEAR" \
    --baseline gameformer_lite --output "$RUN/gameformer_lite" \
    --set external_baselines.training.global_batch_size="$GAMEFORMER_GLOBAL_BATCH_SIZE" \
    --set external_baselines.training.num_workers_total="$GAMEFORMER_NUM_WORKERS_TOTAL" \
    2>&1 | tee "$RUN/train_gameformer_lite.log"
}

if [ "$FORCE_RETRAIN_GAMEFORMER" = true ] || { [ ! -f "$GAMEFORMER_CHECKPOINT" ] && [ "$TRAIN_GAMEFORMER_IF_MISSING" = true ]; }; then
  train_gameformer
fi
if { [ "$DO_OFFLINE" = true ] || [ "$DO_CLOSED_LOOP" = true ]; } && [ ! -f "$GAMEFORMER_CHECKPOINT" ]; then
  echo "Missing GameFormer checkpoint: $GAMEFORMER_CHECKPOINT" >&2
  exit 2
fi

python - "$GAMEFORMER_CHECKPOINT" <<'PY'
import sys, torch
p=sys.argv[1]
try: c=torch.load(p,map_location='cpu',weights_only=False)
except TypeError: c=torch.load(p,map_location='cpu')
contract=c.get('input_contract') or {}
if contract.get('version',0) < 2 or contract.get('deployable_feature_only') is not True:
    raise SystemExit('Checkpoint is legacy/teacher-conditioned. Retrain with near_contact_gameformer_lite.yaml.')
print({'event':'gameformer_input_contract_ok','checkpoint':p,'version':contract.get('version')})
PY

if [ "$DO_OFFLINE" = true ]; then
  # Rule-based baselines are NumPy/CPU work. Keeping them off CUDA avoids GPU
  # context contention while GameFormer evaluates concurrently on GPU 0.
  run_env_cpu python -u -m ocrap.cli evaluate-baseline \
    --config configs/external_baselines/near_contact_external_baselines.yaml \
    --dataset "$TEST_NEAR" --split test \
    --output "$RUN/eval_near_contact_nonlearned.json" --baselines "$NONLEARNED_CSV" \
    2>&1 | tee "$RUN/eval_near_contact_nonlearned.log" & p_cpu=$!

  run_env_gpu "${GPU_LIST[0]}" python -u -m ocrap.cli evaluate-baseline \
    --config configs/external_baselines/near_contact_gameformer_lite.yaml \
    --dataset "$TEST_NEAR" --checkpoint "$GAMEFORMER_CHECKPOINT" --split test \
    --output "$RUN/eval_near_contact_gameformer_lite.json" --baselines gameformer_lite \
    2>&1 | tee "$RUN/eval_near_contact_gameformer_lite.log" & p_gpu=$!

  failed=0
  wait "$p_cpu" || failed=1
  wait "$p_gpu" || failed=1
  [ "$failed" -eq 0 ] || exit 1
fi

run_closed_loop_method() {
  local method="$1" gpu="$2"
  local config=configs/external_baselines/near_contact_external_baselines.yaml
  local label_mode=selected max_scenes="$CL_MAX_SCENARIOS" exhaustive=false
  local ckpt=()
  if [ "$method" = gameformer_lite ]; then
    config=configs/external_baselines/near_contact_gameformer_lite.yaml
    ckpt=(--checkpoint "$GAMEFORMER_CHECKPOINT")
  elif [ "$method" = oracle_recovery_filter ]; then
    # Non-deployable upper bound: all candidates must be labelled before selection.
    label_mode=all
    exhaustive=true
    max_scenes="$CL_ORACLE_MAX_SCENARIOS"
  fi
  echo "[START] near method=$method gpu=$gpu label_mode=$label_mode"
  run_env_gpu "$gpu" python -u -m ocrap.cli closed-loop \
    --config "$config" --dataset "$CL_WOMD" "${ckpt[@]}" \
    --output "$RUN/closed_loop_${method}.json" \
    --set closed_loop.method="$method" \
    --set closed_loop.max_scenarios="$max_scenes" \
    --set closed_loop.bucket_dataset="$CL_BUCKET_DATASET" \
    --set closed_loop.bucket_split="$CL_BUCKET_SPLIT" \
    --set closed_loop.require_bucket_targets=true \
    --set closed_loop.max_bucket_targets="$CL_MAX_SCENARIOS" \
    --set closed_loop.max_targets_per_scene="$CL_MAX_TARGETS_PER_SCENE" \
    --set closed_loop.render_trace="$CL_RENDER_TRACE" \
    --set closed_loop.render_max_agents="$CL_RENDER_MAX_AGENTS" \
    --set closed_loop.max_steps="$CL_MAX_STEPS" \
    --set closed_loop.replan_interval_steps="$CL_REPLAN_INTERVAL_STEPS" \
    --set closed_loop.label_mode="$label_mode" \
    --set closed_loop.force_teacher_baselines=false \
    --set closed_loop.external_sparse_labels=false \
    --set closed_loop.exhaustive_teacher_labels="$exhaustive" \
    --set closed_loop.num_candidate_prefixes="$CL_NUM_CANDIDATES" \
    --set closed_loop.num_recovery_options="$CL_NUM_RECOVERY_OPTIONS" \
    --set closed_loop.save_partial="$CL_SAVE_PARTIAL" \
    --set closed_loop.profile_timing="$CL_PROFILE_TIMING" \
    --set closed_loop.audit_every_n_steps="$CL_AUDIT_EVERY_N_STEPS" \
    --set waymax.dataloader_include_sdc_paths=false \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.use_jit_scan_rollouts=true \
    2>&1 | tee "$RUN/closed_loop_${method}.log"
  echo "[DONE] near method=$method gpu=$gpu"
}

run_closed_loop_dynamic() {
  local -a methods=("$@")
  local next=0 active=0 failed=0 gpu method pid done_pid status
  declare -A PID_GPU=() PID_METHOD=()

  launch_one() {
    local m="$1" g="$2"
    run_closed_loop_method "$m" "$g" &
    local p=$!
    PID_GPU[$p]="$g"
    PID_METHOD[$p]="$m"
    active=$((active + 1))
  }

  for ((i=0; i<MAX_PARALLEL && next<${#methods[@]}; i++)); do
    launch_one "${methods[$next]}" "${GPU_LIST[$i]}"
    next=$((next + 1))
  done

  while [ "$active" -gt 0 ]; do
    done_pid=""
    if wait -n -p done_pid; then status=0; else status=$?; fi
    gpu="${PID_GPU[$done_pid]}"
    method="${PID_METHOD[$done_pid]}"
    unset 'PID_GPU[$done_pid]' 'PID_METHOD[$done_pid]'
    active=$((active - 1))
    if [ "$status" -ne 0 ]; then
      echo "[ERROR] $method failed on GPU $gpu (status=$status)" >&2
      failed=1
    fi
    if [ "$next" -lt "${#methods[@]}" ]; then
      launch_one "${methods[$next]}" "$gpu"
      next=$((next + 1))
    fi
  done
  return "$failed"
}

run_closed_loop_fallback() {
  local -a methods=("$@")
  local base j idx failed=0 pids names
  for ((base=0; base<${#methods[@]}; base+=MAX_PARALLEL)); do
    pids=(); names=()
    for ((j=0; j<MAX_PARALLEL && base+j<${#methods[@]}; j++)); do
      idx=$((base+j))
      run_closed_loop_method "${methods[$idx]}" "${GPU_LIST[$j]}" &
      pids+=("$!"); names+=("${methods[$idx]}")
    done
    for j in "${!pids[@]}"; do
      wait "${pids[$j]}" || { echo "[ERROR] ${names[$j]} failed" >&2; failed=1; }
    done
  done
  return "$failed"
}

if [ "$DO_CLOSED_LOOP" = true ]; then
  if [ "${BASH_VERSINFO[0]}" -ge 5 ]; then
    run_closed_loop_dynamic "${CLOSED_LOOP_METHODS[@]}"
  else
    echo "[WARN] Bash < 5: using static two-GPU batches instead of wait -n dynamic scheduling." >&2
    run_closed_loop_fallback "${CLOSED_LOOP_METHODS[@]}"
  fi
fi

python - <<'PY'
import glob,json,os
run=os.environ['RUN']; rows=[]
for p in sorted(glob.glob(os.path.join(run,'closed_loop_*.json'))):
    try:d=json.load(open(p))
    except Exception:continue
    rows.append({k:d.get(k) for k in ['method','source','label_mode','num_scenes','num_decisions','closed_loop_FRA_exec','closed_loop_FRA_cand','closed_loop_DRS','closed_loop_ODG','closed_loop_post_contact_deployability','closed_loop_bounded_NUP','intervention_rate','timing']})
out=os.path.join(run,'closed_loop_summary.json')
with open(out,'w') as f: json.dump(rows,f,indent=2)
print({'event':'near_contact_closed_loop_summary','output':out,'num_methods':len(rows)})
PY
