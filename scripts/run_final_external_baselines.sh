#!/usr/bin/env bash
# Paired external-baseline characterization on the immutable target sets emitted
# by run_final_locked_three_regime_characterization.sh.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
REGIME="${1:-all}"
OCRAP_OUT="${OCRAP_FINAL_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"
OUT="${FINAL_EXTERNAL_BASELINE_OUT:-$BASE_OUT/external_baselines_v48_124_final}"
GPU_LIST="${GPU_LIST:-0,1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
PROFILE_LATENCY="${PROFILE_LATENCY:-true}"
LATENCY_GPU="${LATENCY_GPU:-0}"
WOMD_ROLE="${WOMD_ROLE:-validation}"

run_one() {
  local r="$1" keyfile="$OCRAP_OUT/target_keys/$1.json"
  [[ -s "$keyfile" ]] || { echo "missing paired target-key lock: $keyfile" >&2; exit 30; }
  echo "[FINAL BASELINE] regime=$r target_keys=$keyfile"
  env CL_TARGET_KEYS_FILE="$keyfile" USE_DYNAMIC_SCHEDULER=auto \
    RUN_SUPPLEMENTARY_SAFE=true RUN_SUPPLEMENTARY_NEAR=true \
    bash scripts/run_external_baselines.sh \
      --regime "$r" --out "$OUT" --gpus "$GPU_LIST" \
      --jobs-per-gpu "$JOBS_PER_GPU" --max-parallel "$MAX_PARALLEL" \
      --max-scenarios 0 --womd-role "$WOMD_ROLE"

  if [[ "${PROFILE_LATENCY,,}" == true || "${PROFILE_LATENCY,,}" == 1 || "${PROFILE_LATENCY,,}" == yes ]]; then
    env CL_TARGET_KEYS_FILE="$keyfile" RUN_SUPPLEMENTARY_SAFE=true RUN_SUPPLEMENTARY_NEAR=true \
      bash scripts/profile_external_baselines_latency.sh \
        --regime "$r" --source-run "$OUT" --out "${OUT}_latency_isolated" \
        --gpu "$LATENCY_GPU" --max-scenarios 0 --womd-role "$WOMD_ROLE"
  fi
}

case "$REGIME" in
  safe|near|contact) run_one "$REGIME" ;;
  all) run_one safe; run_one near; run_one contact ;;
  *) echo "usage: $0 [safe|near|contact|all]" >&2; exit 2 ;;
esac
