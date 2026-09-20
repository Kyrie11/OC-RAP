#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# shellcheck source=scripts/lib/runtime.sh
source scripts/lib/runtime.sh

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${TRAIN_NEAR:=$OCRAP_ROOT/train_near_contact}"
: "${VAL_NEAR:=$OCRAP_ROOT/val_near_contact}"
: "${CALIB_NEAR:=$OCRAP_ROOT/calibration_near_contact}"
: "${TEST_NEAR:=$OCRAP_ROOT/test_near_contact}"
: "${RUN:=runs/near_contact_external_baselines}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${WOMD_VAL:=$WOMD_ROOT/validation/validation_tfexample.tfrecord@150}"
: "${WOMD_VAL_INTERACTIVE:=$WOMD_ROOT/validation_interactive/validation_interactive_tfexample.tfrecord@150}"
: "${WOMD_NUM_SHARDS:=150}"
: "${CL_MAX_SCENARIOS:=50}"
: "${CL_BUCKET_DATASET:=$TEST_NEAR}"
: "${CL_BUCKET_SPLIT:=test}"
# Dataset provenance, not a launcher default, owns the replay collection.
# Explicit CL_WOMD still overrides auto mode, but the normal path is robust to
# either validation or validation_interactive buckets under WOMD_ROOT.
: "${CL_WOMD:=auto}"
: "${CL_WOMD_ROLE:=${PRIMARY_WOMD_ROLE:-validation}}"
if [[ "${CL_WOMD,,}" == auto ]]; then
  CL_WOMD="$(runtime_resolve_bucket_womd_spec "$CL_BUCKET_DATASET" "$CL_BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "$CL_WOMD_ROLE")"
else
  CL_WOMD="$(runtime_normalize_womd_spec "$CL_WOMD" "$WOMD_NUM_SHARDS")"
fi
: "${CALIB_WOMD:=auto}"
: "${CALIB_WOMD_ROLE:=${PRIMARY_WOMD_ROLE:-validation}}"
: "${CL_MAX_TARGETS_PER_SCENE:=1}"
: "${CL_TARGET_KEYS_FILE:=}"
: "${CL_RENDER_TRACE:=false}"
: "${CL_RENDER_MAX_AGENTS:=48}"
: "${CL_PREFLIGHT:=true}"
: "${JAX_RUNTIME_PREFLIGHT:=true}"
: "${CL_ORACLE_MAX_SCENARIOS:=20}"
: "${RUN_ORACLE_CLOSED_LOOP:=false}"
: "${RUN_LEGACY_NEAR:=false}"
: "${RUN_SUPPLEMENTARY_NEAR:=true}"
: "${FORCE_RETRAIN_NEAR:=false}"
: "${CHECKPOINT_ROOT:=$RUN/checkpoints}"
: "${TRAIN_NUM_WORKERS_PER_JOB:=0}"
: "${CL_MAX_STEPS:=40}"
: "${CL_REPLAN_INTERVAL_STEPS:=1}"
: "${CL_NUM_CANDIDATES:=24}"
: "${CL_NUM_RECOVERY_OPTIONS:=12}"
: "${CL_LABEL_MODE:=selected}"
: "${CL_AUDIT_EVERY_N_STEPS:=0}"
: "${CL_SAVE_PARTIAL:=true}"
: "${CL_PROFILE_TIMING:=true}"
: "${CL_LATENCY_EXECUTION_CONTRACT:=throughput_or_unspecified}"
: "${CL_LATENCY_WARMUP_DECISIONS:=3}"
: "${CL_RESUME:=true}"
: "${CL_RESUME_FORCE:=false}"
: "${CL_METRIC_SEMANTICS_VERSION:=publication_v55_signed_clearance_unclipped_v1}"
: "${CL_PARTIAL_WRITE_EVERY_SCENES:=32}"
: "${CL_PROGRESS_EVERY_STEPS:=10}"
: "${SKIP_COMPLETE_METHODS:=true}"
: "${USE_DYNAMIC_SCHEDULER:=true}"
: "${NEAR_SAFETY_FILTER_DEDICATED_GPUS:=true}"
: "${NEAR_SAFETY_FILTER_START_ALL_PARALLEL:=true}"
: "${DO_OFFLINE:=true}"
: "${DO_CLOSED_LOOP:=true}"
: "${DO_TRAIN:=true}"
: "${DO_CALIBRATE:=true}"
: "${FORCE_RECALIBRATE:=false}"
: "${FORCE_REREGISTER:=false}"
: "${CONFORMAL_DELTA:=${CONFORMAL_ALPHA:-0.10}}"
: "${CONFORMAL_PREDICTION_HORIZON:=7}"
: "${CONFORMAL_MISSION_HORIZON:=$CL_MAX_STEPS}"
: "${CONFORMAL_CALIBRATION_UNIT:=group}"
: "${CONFORMAL_CALIBRATION:=$RUN/conformal_calibration.json}"
: "${CONFORMAL_INTERVALS:=}"
# A raw calibration replay source is needed only when intervals are loaded from
# or fitted into a calibration artifact. Explicit interval injection does not
# require legacy calibration provenance.
if [[ -z "$CONFORMAL_INTERVALS" ]]; then
  if [[ "${CALIB_WOMD,,}" == auto ]]; then
    CALIB_WOMD="$(runtime_resolve_bucket_womd_spec "$CALIB_NEAR" calibration "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "$CALIB_WOMD_ROLE")"
  else
    CALIB_WOMD="$(runtime_normalize_womd_spec "$CALIB_WOMD" "$WOMD_NUM_SHARDS")"
  fi
elif [[ "${CALIB_WOMD,,}" == auto ]]; then
  CALIB_WOMD=""
else
  CALIB_WOMD="$(runtime_normalize_womd_spec "$CALIB_WOMD" "$WOMD_NUM_SHARDS")"
fi
: "${CUDA_DEVICES:=0,1}"
: "${JOBS_PER_GPU:=3}"                    # requested 3-way per-GPU concurrency
: "${MAX_PARALLEL:=6}"                     # empty => all GPU slots

IFS=',' read -r -a GPU_LIST <<< "$CUDA_DEVICES"
((${#GPU_LIST[@]})) || GPU_LIST=(0 1)
((JOBS_PER_GPU >= 1)) || JOBS_PER_GPU=1
GPU_SLOTS=()
for ((_slot=0; _slot<JOBS_PER_GPU; _slot++)); do
  for _gpu in "${GPU_LIST[@]}"; do GPU_SLOTS+=("$_gpu"); done
done
[[ -n "$MAX_PARALLEL" ]] || MAX_PARALLEL="${#GPU_SLOTS[@]}"
((MAX_PARALLEL >= 1)) || MAX_PARALLEL=1
((MAX_PARALLEL <= ${#GPU_SLOTS[@]})) || MAX_PARALLEL="${#GPU_SLOTS[@]}"
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)"
: "${THREADS_PER_JOB:=$(( CPU_COUNT / (2 * MAX_PARALLEL) ))}"
((THREADS_PER_JOB >= 1)) || THREADS_PER_JOB=1
((THREADS_PER_JOB <= 8)) || THREADS_PER_JOB=8
if ((TRAIN_NUM_WORKERS_PER_JOB <= 0)); then
  TRAIN_NUM_WORKERS_PER_JOB=$(( CPU_COUNT / (2 * MAX_PARALLEL) ))
  ((TRAIN_NUM_WORKERS_PER_JOB >= 2)) || TRAIN_NUM_WORKERS_PER_JOB=2
  ((TRAIN_NUM_WORKERS_PER_JOB <= 6)) || TRAIN_NUM_WORKERS_PER_JOB=6
fi
: "${JAX_CACHE_DIR:=$RUN/.jax_compilation_cache}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
export RUN CL_WOMD
mkdir -p "$RUN" "$JAX_CACHE_DIR" "$CHECKPOINT_ROOT"

if [[ "$CONFORMAL_CALIBRATION_UNIT" == "group" ]]; then
  echo "[CPSF] calibration_unit=group preserves the legacy launcher contract, but formal exchangeability is only group-level. For a stricter WOMD-scene certificate use CONFORMAL_CALIBRATION_UNIT=scene_max and ensure delta/T is supported by the number of independent calibration scenes." >&2
fi

CONFIG=configs/external_baselines/near_contact_external_baselines.yaml
# Six published Near-Contact main-table controls plus source-backed supplementary
# learned planners. The scheduler below caps concurrency at 6 (3 processes/GPU).
METHODS=(
  marc_lite
  racp_lite
  robust_scenario_mpc
  predictive_safety_filter
  dr_cvar_safety_filter
  conformal_predictive_safety_filter
)
if runtime_bool_true "$RUN_LEGACY_NEAR"; then METHODS+=(severity_minimization); fi
METHODS_CSV="$(IFS=,; echo "${METHODS[*]}")"
SPECS=()
for _m in "${METHODS[@]}"; do SPECS+=("${_m}|$CONFIG|nonlearning||"); done
if runtime_bool_true "$RUN_SUPPLEMENTARY_NEAR"; then
  SPECS+=(
    "flow_planner|configs/external_baselines/flow_planner.yaml|learned|$CHECKPOINT_ROOT/flow_planner/best.pt|flow_planner_womd_lattice_port_v63"
    "plan_r1|configs/external_baselines/plan_r1.yaml|learned|$CHECKPOINT_ROOT/plan_r1/best.pt|plan_r1_source_core_womd_adapter_v62"
    "betopnet|configs/external_baselines/betopnet.yaml|learned|$CHECKPOINT_ROOT/betopnet/best.pt|betop_source_core_topology_adapter_v62"
  )
fi
ALL_METHODS=()
for _spec in "${SPECS[@]}"; do IFS='|' read -r _m _rest <<< "$_spec"; ALL_METHODS+=("$_m"); done
ALL_METHODS_CSV="$(IFS=,; echo "${ALL_METHODS[*]}")"

common_env=(
  OMP_NUM_THREADS="$THREADS_PER_JOB"
  MKL_NUM_THREADS="$THREADS_PER_JOB"
  OPENBLAS_NUM_THREADS="$THREADS_PER_JOB"
  NUMEXPR_NUM_THREADS="$THREADS_PER_JOB"
  TF_NUM_INTRAOP_THREADS="$THREADS_PER_JOB"
  TF_NUM_INTEROP_THREADS=2
  MALLOC_ARENA_MAX=4
  XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE"
  TF_FORCE_GPU_ALLOW_GROWTH=true
  JAX_ENABLE_X64=0
  JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  PYTHONUNBUFFERED=1
)
run_env_gpu() {
  local gpu="$1"; shift
  local cache="$JAX_CACHE_DIR/gpu_${gpu//[^[:alnum:]_.-]/_}"
  mkdir -p "$cache"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpu" OCRAP_TENSORFLOW_CPU_ONLY=1 JAX_COMPILATION_CACHE_DIR="$cache" "${common_env[@]}" "$@"
}
run_env_cpu() {
  local cache="$JAX_CACHE_DIR/cpu"; mkdir -p "$cache"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OCRAP_TENSORFLOW_CPU_ONLY=1 JAX_COMPILATION_CACHE_DIR="$cache" "${common_env[@]}" "$@"
}
# JAX CUDA wheels currently still initialize the CUDA PJRT plugin during plugin
# discovery even when JAX_PLATFORMS=cpu.  Hiding every GPU with
# CUDA_VISIBLE_DEVICES="" can therefore emit/fail with CUDA_ERROR_NO_DEVICE
# before the requested CPU backend is selected.  CPSF calibration needs JAX/
# Waymax but must compute on CPU, so keep one allocated GPU *visible only for
# CUDA plugin discovery* while JAX_PLATFORMS=cpu forces all JAX computation to
# the CPU backend.  TensorFlow remains CPU-only via its own visibility API.
: "${CPU_JAX_VISIBLE_DEVICE:=${GPU_LIST[0]}}"
run_env_jax_cpu() {
  local cache="$JAX_CACHE_DIR/jax_cpu"; mkdir -p "$cache"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$CPU_JAX_VISIBLE_DEVICE" JAX_PLATFORMS=cpu \
    OCRAP_TENSORFLOW_CPU_ONLY=1 JAX_COMPILATION_CACHE_DIR="$cache" "${common_env[@]}" "$@"
}

if runtime_bool_true "$DO_CLOSED_LOOP" && runtime_bool_true "$JAX_RUNTIME_PREFLIGHT"; then
  echo "[PREFLIGHT] validating JAX/Waymax GPU runtime on CUDA device ${GPU_LIST[0]}"
  run_env_gpu "${GPU_LIST[0]}" python tools/check_jax_waymax_runtime.py \
    --require-gpu --output "$RUN/jax_waymax_runtime_preflight.json"
fi

if runtime_bool_true "$DO_CLOSED_LOOP" && runtime_bool_true "$CL_PREFLIGHT"; then
  preflight_target_args=()
  if [[ -n "$CL_TARGET_KEYS_FILE" ]]; then
    preflight_target_args=(--target-keys-file "$CL_TARGET_KEYS_FILE" --require-target-keys)
  fi
  python tools/check_closed_loop_dataset_support.py \
    --dataset "$CL_BUCKET_DATASET" --split "$CL_BUCKET_SPLIT" --womd-pattern "$CL_WOMD" \
    --expected-source-role auto "${preflight_target_args[@]}" \
    --output "$RUN/closed_loop_dataset_support.json"
fi

# These six baselines fit no neural weights. DO_TRAIN validates the regime data
# and writes one train_summary.json per method, but intentionally creates no .pt.
# The optimized registrar scans train/val only once for all six methods.
NEAR_NONLEARNING_SPECS="$(python - "$CONFIG" "$METHODS_CSV" <<'PY_SPECS'
import sys
cfg, methods = sys.argv[1:]
print(','.join(f"{m.strip()}={cfg}" for m in methods.split(',') if m.strip()))
PY_SPECS
)"
if runtime_bool_true "$DO_TRAIN"; then
  # Validate each non-learning registration independently.  If only one method
  # is missing/stale, do not rewrite all six train summaries; the registrar still
  # scans train/val once for the subset that actually needs registration.
  MISSING_NONLEARNING_METHODS=()
  for _m in "${METHODS[@]}"; do
    if runtime_bool_true "$FORCE_REREGISTER" || ! python tools/check_external_nonlearning_registration.py \
        --root "$RUN" --dataset "$TRAIN_NEAR" --val-dataset "$VAL_NEAR" \
        --specs "$_m=$CONFIG" >/dev/null 2>&1; then
      MISSING_NONLEARNING_METHODS+=("$_m")
      echo "[STATUS] method=$_m training/registration=incomplete"
    else
      echo "[STATUS] method=$_m training/registration=complete"
    fi
  done
  if ((${#MISSING_NONLEARNING_METHODS[@]})); then
    MISSING_NONLEARNING_CSV="$(IFS=,; echo "${MISSING_NONLEARNING_METHODS[*]}")"
    echo "[REGISTER] missing Near-Contact non-learning methods: $MISSING_NONLEARNING_CSV"
    run_env_cpu python -u tools/register_external_nonlearning_baselines.py \
      --config "$CONFIG" --dataset "$TRAIN_NEAR" --val-dataset "$VAL_NEAR" \
      --baselines "$MISSING_NONLEARNING_CSV" --output-root "$RUN" \
      2>&1 | tee "$RUN/register_nonlearning_near.log"
  else
    echo "[REUSE] all Near-Contact non-learning baseline registrations are compatible"
  fi
fi

calibration_valid() {
  local artifact="$1"
  [[ -f "$artifact" ]] || return 1
  python - "$artifact" "$CONFIG" "$CALIB_NEAR" "$CALIB_WOMD" "$CONFORMAL_DELTA" "$CONFORMAL_PREDICTION_HORIZON" "$CONFORMAL_MISSION_HORIZON" "$CONFORMAL_CALIBRATION_UNIT" <<'PY' >/dev/null
import hashlib, json, math, sys
from pathlib import Path
from ocrap.config import load_config
artifact, config_path, dataset, womd, delta, H, T, unit = sys.argv[1:]
delta=float(delta); H=int(H); T=int(T)
try:
    d=json.load(open(artifact))
    cfg=load_config(config_path)
    fp=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':'),ensure_ascii=False,default=str).encode()).hexdigest()
    vals=[float(x) for x in d['conformal_prediction_intervals_m']]
    ok=(d.get('requested_config_fingerprint')==fp and str(d.get('dataset'))==str(Path(dataset)) and
        str(d.get('split'))=='calibration' and str(d.get('womd_pattern'))==str(womd) and
        math.isclose(float(d.get('delta')),delta,rel_tol=0,abs_tol=1e-12) and
        int(d.get('prediction_horizon'))==H and int(d.get('mission_horizon'))==T and str(d.get('calibration_unit'))==str(unit) and
        len(vals)==H and all(math.isfinite(x) and x >= 0.0 for x in vals) and
        d.get('teacher_labels_used') is False and d.get('test_labels_used') is False)
except Exception:
    ok=False
raise SystemExit(0 if ok else 1)
PY
}

validate_intervals() {
  python - "$1" "$CONFORMAL_PREDICTION_HORIZON" <<'PY' >/dev/null
import json, math, sys
vals=json.loads(sys.argv[1]); H=int(sys.argv[2])
assert isinstance(vals,list) and len(vals)==H, (len(vals) if isinstance(vals,list) else type(vals), H)
assert all(math.isfinite(float(x)) and float(x)>=0.0 for x in vals), vals
PY
}

if [[ -n "$CONFORMAL_INTERVALS" ]]; then
  validate_intervals "$CONFORMAL_INTERVALS"
  echo "[CALIBRATION] using explicit CONFORMAL_INTERVALS=$CONFORMAL_INTERVALS"
else
  if runtime_bool_true "$DO_CALIBRATE"; then
    if runtime_bool_true "$FORCE_RECALIBRATE" || ! calibration_valid "$CONFORMAL_CALIBRATION"; then
      echo "[CALIBRATION] fitting CPSF horizon-wise conformal prediction intervals from $CALIB_NEAR against dataset-owned WOMD source: $CALIB_WOMD"
      echo "[CALIBRATION-RUNTIME] JAX compute backend=cpu; CUDA plugin discovery sees device $CPU_JAX_VISIBLE_DEVICE"
      run_env_jax_cpu python -u tools/calibrate_external_baselines.py \
        --config "$CONFIG" --dataset "$CALIB_NEAR" --split calibration \
        --womd-pattern "$CALIB_WOMD" --delta "$CONFORMAL_DELTA" \
        --prediction-horizon "$CONFORMAL_PREDICTION_HORIZON" \
        --mission-horizon "$CONFORMAL_MISSION_HORIZON" \
        --calibration-unit "$CONFORMAL_CALIBRATION_UNIT" \
        --output "$CONFORMAL_CALIBRATION" \
        2>&1 | tee "$RUN/calibrate_conformal.log"
    else
      echo "[REUSE] valid CPSF conformal calibration $CONFORMAL_CALIBRATION"
    fi
  elif ! calibration_valid "$CONFORMAL_CALIBRATION"; then
    echo "DO_CALIBRATE=false but no compatible CPSF calibration artifact exists. Set DO_CALIBRATE=true or CONFORMAL_INTERVALS='[...]' explicitly." >&2
    exit 2
  fi
  CONFORMAL_INTERVALS="$(python - "$CONFORMAL_CALIBRATION" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(json.dumps([float(x) for x in d['conformal_prediction_intervals_m']],separators=(',',':')))
PY
)"
  validate_intervals "$CONFORMAL_INTERVALS"
fi
export CONFORMAL_INTERVALS CONFORMAL_DELTA CONFORMAL_PREDICTION_HORIZON CONFORMAL_MISSION_HORIZON CONFORMAL_CALIBRATION_UNIT WOMD_VAL CALIB_WOMD


artifact_complete() {
  local output="$1"
  local args=(--output "$output" --quiet --require-metric-semantics-version "$CL_METRIC_SEMANTICS_VERSION")
  [[ -n "$CL_TARGET_KEYS_FILE" ]] && args+=(--target-keys-file "$CL_TARGET_KEYS_FILE")
  python tools/check_closed_loop_artifact.py "${args[@]}"
}

checkpoint_valid() {
  local ckpt="$1" expected_impl="$2" config="$3"
  [[ -n "$ckpt" && -f "$ckpt" ]] || return 1
  python tools/check_external_training_complete.py \
    --checkpoint "$ckpt" --summary "$(dirname "$ckpt")/train_summary.json" --config "$config" \
    --require-deployable-contract --require-implementation-version "$expected_impl" >/dev/null 2>&1
}

is_slow_near_filter() {
  case "$1" in
    predictive_safety_filter|dr_cvar_safety_filter|conformal_predictive_safety_filter) return 0 ;;
    *) return 1 ;;
  esac
}

nonlearning_registration_valid() {
  local method="$1" config="$2"
  python tools/check_external_nonlearning_registration.py \
    --root "$RUN" --dataset "$TRAIN_NEAR" --val-dataset "$VAL_NEAR" \
    --specs "$method=$config" >/dev/null 2>&1
}

print_baseline_status() {
  local spec method config kind ckpt expected_impl train_status cl_status
  echo "[STATUS] Near-Contact baseline completion check (training/registration + closed loop)"
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r method config kind ckpt expected_impl <<< "$spec"
    if [[ "$kind" == learned ]]; then
      if checkpoint_valid "$ckpt" "$expected_impl" "$config"; then train_status=complete; else train_status=incomplete; fi
    else
      if nonlearning_registration_valid "$method" "$config"; then train_status=complete; else train_status=incomplete; fi
    fi
    if artifact_complete "$RUN/closed_loop_${method}.json"; then cl_status=complete; else cl_status=incomplete; fi
    echo "[STATUS] method=$method training=$train_status closed_loop=$cl_status"
  done
}

prepare_or_offline_method() {
  local spec="$1" gpu="$2" method config kind ckpt expected_impl train_dir
  IFS='|' read -r method config kind ckpt expected_impl <<< "$spec"
  if ! runtime_bool_true "$FORCE_RETRAIN_NEAR" \
      && runtime_bool_true "$DO_CLOSED_LOOP" && runtime_bool_true "$SKIP_COMPLETE_METHODS" \
      && artifact_complete "$RUN/closed_loop_${method}.json"; then
    echo "[REUSE] near method=$method already has a complete closed-loop artifact; training/registration/offline preparation skipped"
    return 0
  fi
  if [[ "$kind" == learned ]]; then
    if runtime_bool_true "$FORCE_RETRAIN_NEAR" || ! checkpoint_valid "$ckpt" "$expected_impl" "$config"; then
      if ! runtime_bool_true "$DO_TRAIN"; then
        echo "Missing/invalid checkpoint and training disabled: $ckpt" >&2; return 2
      fi
      train_dir="$(dirname "$ckpt")"; mkdir -p "$train_dir"
      # Remove stale incompatible artifacts before retraining; otherwise an old
      # best.pt could remain after an interrupted training attempt.
      rm -f "$train_dir/best.pt" "$train_dir/latest.pt" "$train_dir/train_summary.json"
      echo "[TRAIN] near method=$method gpu=$gpu"
      run_env_gpu "$gpu" python -u -m ocrap.cli train-baseline \
        --config "$config" --dataset "$TRAIN_NEAR" --val-dataset "$VAL_NEAR" \
        --baseline "$method" --output "$train_dir" \
        --set external_baselines.training.distributed=false \
        --set "external_baselines.training.num_workers=$TRAIN_NUM_WORKERS_PER_JOB" \
        --set external_baselines.training.tqdm=false \
        2>&1 | tee "$RUN/train_${method}.log"
      checkpoint_valid "$ckpt" "$expected_impl" "$config" || { echo "Training produced invalid checkpoint: $ckpt" >&2; return 2; }
    else
      echo "[REUSE] validated checkpoint $ckpt"
    fi
  fi
  if runtime_bool_true "$DO_OFFLINE"; then
    local checkpoint_args=() conformal_args=()
    [[ "$kind" == learned ]] && checkpoint_args=(--checkpoint "$ckpt")
    [[ "$method" == conformal_predictive_safety_filter ]] && conformal_args=(--set "external_baselines.policy.conformal_prediction_intervals_m=$CONFORMAL_INTERVALS")
    echo "[OFFLINE] near method=$method gpu=$gpu"
    run_env_gpu "$gpu" python -u -m ocrap.cli evaluate-baseline \
      --config "$config" --dataset "$TEST_NEAR" --split test \
      --output "$RUN/eval_near_contact_${method}.json" --baselines "$method" \
      "${checkpoint_args[@]}" "${conformal_args[@]}" \
      2>&1 | tee "$RUN/eval_near_contact_${method}.log"
  fi
}

supports_wait_pid_capture() {
  help wait 2>/dev/null | grep -Eq -- '(^|[[:space:]])-p([[:space:]]|[[:punct:]])'
}
run_queue_dynamic() {
  local runner="$1"; shift; local -a items=("$@")
  local next=0 active=0 failed=0 done_pid status gpu item i
  declare -A PID_GPU=() PID_ITEM=()
  launch_one() { local x="$1" g="$2"; "$runner" "$x" "$g" & local p=$!; PID_GPU[$p]="$g"; PID_ITEM[$p]="$x"; active=$((active+1)); }
  for ((i=0;i<MAX_PARALLEL && next<${#items[@]};i++)); do launch_one "${items[$next]}" "${GPU_SLOTS[$i]}"; next=$((next+1)); done
  while ((active>0)); do
    done_pid=""; if wait -n -p done_pid; then status=0; else status=$?; fi
    gpu="${PID_GPU[$done_pid]}"; item="${PID_ITEM[$done_pid]}"; unset 'PID_GPU[$done_pid]' 'PID_ITEM[$done_pid]'; active=$((active-1))
    if ((status!=0)); then echo "[ERROR] $item failed on GPU $gpu (status=$status)" >&2; failed=1; fi
    if ((next<${#items[@]})); then launch_one "${items[$next]}" "$gpu"; next=$((next+1)); fi
  done
  return "$failed"
}
run_queue_fixed() {
  local runner="$1"; shift; local -a items=("$@") pids=() names=(); local base j idx failed=0
  for ((base=0;base<${#items[@]};base+=MAX_PARALLEL)); do
    pids=(); names=()
    for ((j=0;j<MAX_PARALLEL && base+j<${#items[@]};j++)); do idx=$((base+j)); "$runner" "${items[$idx]}" "${GPU_SLOTS[$j]}" & pids+=("$!"); names+=("${items[$idx]}"); done
    for j in "${!pids[@]}"; do wait "${pids[$j]}" || { echo "[ERROR] ${names[$j]} failed" >&2; failed=1; }; done
  done
  return "$failed"
}
run_queue() {
  local runner="$1"; shift; local use_dynamic=false
  case "${USE_DYNAMIC_SCHEDULER,,}" in
    1|true|yes|on) supports_wait_pid_capture || { echo "USE_DYNAMIC_SCHEDULER requested but Bash lacks wait -p" >&2; return 2; }; use_dynamic=true ;;
    auto|'') supports_wait_pid_capture && use_dynamic=true ;;
    0|false|no|off) use_dynamic=false ;;
    *) echo "Invalid USE_DYNAMIC_SCHEDULER=$USE_DYNAMIC_SCHEDULER" >&2; return 2 ;;
  esac
  if [[ "$use_dynamic" == true ]]; then run_queue_dynamic "$runner" "$@"; else run_queue_fixed "$runner" "$@"; fi
}

run_queue_all_parallel_round_robin() {
  # Start every unfinished safety-filter job immediately. GPU assignment is
  # round-robin over the visible physical devices, so with two GPUs the three
  # publication filters start as 0,1,0 instead of making the third filter wait.
  # This is execution-only: configs, target locks, candidate sets, teacher
  # labels and metric aggregation are unchanged.
  local runner="$1"; shift; local -a items=("$@")
  ((${#items[@]})) || return 0
  local -a pids=() names=() gpus=()
  local i gpu failed=0 status
  for ((i=0;i<${#items[@]};i++)); do
    gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
    echo "[PARALLEL-START] safety-filter item=${items[$i]%%|*} gpu=$gpu"
    "$runner" "${items[$i]}" "$gpu" &
    pids+=("$!"); names+=("${items[$i]}"); gpus+=("$gpu")
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then status=0; else status=$?; fi
    if ((status!=0)); then
      echo "[ERROR] ${names[$i]} failed on GPU ${gpus[$i]} (status=$status)" >&2
      failed=1
    fi
  done
  return "$failed"
}

run_queue_one_per_gpu() {
  # Compatibility scheduler retained for users who explicitly disable
  # NEAR_SAFETY_FILTER_START_ALL_PARALLEL. It permits at most one safety-filter
  # process per physical GPU.
  local runner="$1"; shift; local -a items=("$@")
  ((${#items[@]})) || return 0
  local next=0 active=0 failed=0 done_pid status gpu item i
  declare -A PID_GPU=() PID_ITEM=()
  launch_one() {
    local x="$1" g="$2"
    "$runner" "$x" "$g" &
    local p=$!
    PID_GPU[$p]="$g"; PID_ITEM[$p]="$x"; active=$((active+1))
  }
  if supports_wait_pid_capture; then
    for ((i=0;i<${#GPU_LIST[@]} && next<${#items[@]};i++)); do
      launch_one "${items[$next]}" "${GPU_LIST[$i]}"; next=$((next+1))
    done
    while ((active>0)); do
      done_pid=""; if wait -n -p done_pid; then status=0; else status=$?; fi
      gpu="${PID_GPU[$done_pid]}"; item="${PID_ITEM[$done_pid]}"
      unset 'PID_GPU[$done_pid]' 'PID_ITEM[$done_pid]'; active=$((active-1))
      if ((status!=0)); then echo "[ERROR] $item failed on GPU $gpu (status=$status)" >&2; failed=1; fi
      if ((next<${#items[@]})); then launch_one "${items[$next]}" "$gpu"; next=$((next+1)); fi
    done
    return "$failed"
  fi
  local base j idx pids=() names=()
  for ((base=0;base<${#items[@]};base+=${#GPU_LIST[@]})); do
    pids=(); names=()
    for ((j=0;j<${#GPU_LIST[@]} && base+j<${#items[@]};j++)); do
      idx=$((base+j)); "$runner" "${items[$idx]}" "${GPU_LIST[$j]}" &
      pids+=("$!"); names+=("${items[$idx]}")
    done
    for j in "${!pids[@]}"; do wait "${pids[$j]}" || { echo "[ERROR] ${names[$j]} failed" >&2; failed=1; }; done
  done
  return "$failed"
}

run_closed_loop_method() {
  local spec="$1" gpu="$2" method config kind ckpt expected_impl
  IFS='|' read -r method config kind ckpt expected_impl <<< "$spec"
  local output="$RUN/closed_loop_${method}.json"
  if runtime_bool_true "$SKIP_COMPLETE_METHODS" && ! runtime_bool_true "$FORCE_RETRAIN_NEAR" \
      && artifact_complete "$output"; then
    echo "[REUSE] near closed-loop method=$method is already complete: $output"
    return 0
  fi
  local label_mode="$CL_LABEL_MODE" max_scenes="$CL_MAX_SCENARIOS" exhaustive=false sparse=true
  local target_args=() checkpoint_args=()
  if [[ "$kind" == learned ]]; then
    checkpoint_valid "$ckpt" "$expected_impl" "$config" || { echo "Missing/invalid checkpoint: $ckpt" >&2; return 2; }
    checkpoint_args=(--checkpoint "$ckpt")
  fi
  if [[ "$method" == oracle_recovery_filter ]]; then
    label_mode=all; exhaustive=true; sparse=false; max_scenes="$CL_ORACLE_MAX_SCENARIOS"
  fi
  if [[ -n "$CL_TARGET_KEYS_FILE" ]]; then
    target_args=(--set "closed_loop.target_keys_file=$CL_TARGET_KEYS_FILE" --set closed_loop.require_target_keys=true)
  fi
  echo "[START] near method=$method gpu=$gpu label_mode=$label_mode max_scenes=$max_scenes"
  run_env_gpu "$gpu" python -u -m ocrap.cli closed-loop \
    --config "$config" --dataset "$CL_WOMD" "${checkpoint_args[@]}" --output "$output" \
    --set "external_baselines.policy.conformal_prediction_intervals_m=$CONFORMAL_INTERVALS" \
    --set "closed_loop.method=$method" \
    --set "closed_loop.max_scenarios=$max_scenes" \
    --set "closed_loop.max_bucket_targets=$max_scenes" \
    --set "closed_loop.bucket_dataset=$CL_BUCKET_DATASET" \
    --set "closed_loop.bucket_split=$CL_BUCKET_SPLIT" \
    --set closed_loop.require_bucket_targets=true \
    --set "closed_loop.max_targets_per_scene=$CL_MAX_TARGETS_PER_SCENE" \
    --set "closed_loop.render_trace=$CL_RENDER_TRACE" \
    --set "closed_loop.render_max_agents=$CL_RENDER_MAX_AGENTS" \
    --set "closed_loop.max_steps=$CL_MAX_STEPS" \
    --set "closed_loop.replan_interval_steps=$CL_REPLAN_INTERVAL_STEPS" \
    --set "closed_loop.label_mode=$label_mode" \
    --set closed_loop.force_teacher_baselines=false \
    --set "closed_loop.external_sparse_labels=$sparse" \
    --set "closed_loop.exhaustive_teacher_labels=$exhaustive" \
    --set "closed_loop.num_candidate_prefixes=$CL_NUM_CANDIDATES" \
    --set "closed_loop.num_recovery_options=$CL_NUM_RECOVERY_OPTIONS" \
    --set "closed_loop.save_partial=$CL_SAVE_PARTIAL" \
    --set "closed_loop.resume=$CL_RESUME" \
    --set "closed_loop.resume_force=$CL_RESUME_FORCE" \
    --set "closed_loop.metric_semantics_version=$CL_METRIC_SEMANTICS_VERSION" \
    --set "closed_loop.partial_write_every_scenes=$CL_PARTIAL_WRITE_EVERY_SCENES" \
    --set "closed_loop.progress_every_steps=$CL_PROGRESS_EVERY_STEPS" \
    --set closed_loop.result_scene_detail=metrics \
    --set closed_loop.scene_journal_detail=metrics \
    --set closed_loop.memory_scene_detail=metrics \
    --set closed_loop.include_scenes_in_result=false \
    --set closed_loop.include_scenes_in_partial=false \
    --set "closed_loop.profile_timing=$CL_PROFILE_TIMING" \
    --set "closed_loop.latency_execution_contract=$CL_LATENCY_EXECUTION_CONTRACT" \
    --set "closed_loop.latency_warmup_decisions=$CL_LATENCY_WARMUP_DECISIONS" \
    --set "closed_loop.audit_every_n_steps=$CL_AUDIT_EVERY_N_STEPS" \
    --set closed_loop.use_sdc_paths=true \
    --set closed_loop.require_observation_legal_route=true \
    --set closed_loop.allow_future_route_proxy=false \
    --set waymax.dataloader_include_sdc_paths=true \
    --set waymax.allow_logged_sdc_route_fallback=false \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.use_jit_scan_rollouts=true \
    "${target_args[@]}" \
    2>&1 | tee "$RUN/closed_loop_${method}.log"
  echo "[DONE] near method=$method gpu=$gpu"
}

run_baseline_pipeline() {
  local spec="$1" gpu="$2"
  prepare_or_offline_method "$spec" "$gpu"
  if runtime_bool_true "$DO_CLOSED_LOOP"; then
    run_closed_loop_method "$spec" "$gpu"
  fi
}

# Check all methods before launching per-baseline work. The validators are
# target-lock/config aware, so a complete method is not scheduled again.
print_baseline_status

PIPELINE_SPECS=()
for spec in "${SPECS[@]}"; do IFS='|' read -r _m _c _kind _rest <<< "$spec"; [[ "$_kind" == learned ]] && PIPELINE_SPECS+=("$spec"); done
for spec in "${SPECS[@]}"; do IFS='|' read -r _m _c _kind _rest <<< "$spec"; [[ "$_kind" == learned ]] || PIPELINE_SPECS+=("$spec"); done

if runtime_bool_true "$NEAR_SAFETY_FILTER_DEDICATED_GPUS"; then
  # Preparation is separated from closed loop so the three expensive filters can
  # use one Waymax/JAX process per physical GPU instead of competing 3-per-GPU.
  # Learned planners are still prepared first; with PRETRAINED_BASELINE_ROOT this
  # is normally a fast checkpoint validation rather than retraining.
  if runtime_bool_true "$DO_TRAIN" || runtime_bool_true "$DO_OFFLINE"; then
    run_queue prepare_or_offline_method "${PIPELINE_SPECS[@]}"
  fi

  if runtime_bool_true "$DO_CLOSED_LOOP"; then
    SLOW_FILTER_SPECS=()
    OTHER_CLOSED_LOOP_SPECS=()
    for spec in "${PIPELINE_SPECS[@]}"; do
      IFS='|' read -r _m _c _kind _ckpt _impl <<< "$spec"
      if runtime_bool_true "$SKIP_COMPLETE_METHODS" && ! runtime_bool_true "$FORCE_RETRAIN_NEAR" \
          && artifact_complete "$RUN/closed_loop_${_m}.json"; then
        echo "[REUSE] near method=$_m excluded from scheduler: complete closed-loop artifact"
        continue
      fi
      if is_slow_near_filter "$_m"; then SLOW_FILTER_SPECS+=("$spec"); else OTHER_CLOSED_LOOP_SPECS+=("$spec"); fi
    done

    # Run the slow filters first so interrupted publication runs make progress on
    # the exact methods that previously stalled at 1/250 scenes. By default all
    # unfinished safety filters are forked immediately; set
    # NEAR_SAFETY_FILTER_START_ALL_PARALLEL=false to restore one-per-GPU gating.
    if runtime_bool_true "$NEAR_SAFETY_FILTER_START_ALL_PARALLEL"; then
      run_queue_all_parallel_round_robin run_closed_loop_method "${SLOW_FILTER_SPECS[@]}"
    else
      run_queue_one_per_gpu run_closed_loop_method "${SLOW_FILTER_SPECS[@]}"
    fi
    run_queue run_closed_loop_method "${OTHER_CLOSED_LOOP_SPECS[@]}"
  fi
else
  # Compatibility path: retain the historical per-baseline pipeline and dynamic
  # refill behavior when dedicated safety-filter GPU scheduling is disabled.
  if runtime_bool_true "$DO_TRAIN" || runtime_bool_true "$DO_OFFLINE" || runtime_bool_true "$DO_CLOSED_LOOP"; then
    run_queue run_baseline_pipeline "${PIPELINE_SPECS[@]}"
  fi
fi

# Oracle recovery is a teacher-only audit upper bound, not an external baseline.
if runtime_bool_true "$DO_CLOSED_LOOP" && runtime_bool_true "$RUN_ORACLE_CLOSED_LOOP"; then
  run_queue run_closed_loop_method "oracle_recovery_filter|$CONFIG|nonlearning||"
fi

python tools/summarize_external_closed_loop.py \
  --run "$RUN" --regime near --output "$RUN/closed_loop_summary.json" \
  --methods "$ALL_METHODS_CSV" --womd-spec "$CL_WOMD"
python - "$RUN/closed_loop_summary.json" <<'PY'
import json, os, sys
p=sys.argv[1]
d=json.load(open(p))
d['conformal_calibration']={
    'delta': float(os.environ['CONFORMAL_DELTA']),
    'prediction_horizon': int(os.environ['CONFORMAL_PREDICTION_HORIZON']),
    'mission_horizon': int(os.environ['CONFORMAL_MISSION_HORIZON']),
    'calibration_unit': os.environ['CONFORMAL_CALIBRATION_UNIT'],
    'prediction_intervals_m': json.loads(os.environ['CONFORMAL_INTERVALS']),
    'raw_calibration_womd_spec': os.environ.get('CALIB_WOMD') or None,
}
with open(p,'w') as f: json.dump(d,f,indent=2)
print({'event':'near_contact_closed_loop_summary_augmented','output':p,'num_methods':len(d.get('methods',[]))})
PY
