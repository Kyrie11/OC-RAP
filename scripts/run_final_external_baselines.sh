#!/usr/bin/env bash
# Paired external-baseline characterization on the method-independent observation-legal
# target locks used by the frozen final OC-RAP characterization.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
REGIME="${1:-all}"
OCRAP_OUT="${OCRAP_FINAL_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"
OUT="${FINAL_EXTERNAL_BASELINE_OUT:-$BASE_OUT/external_baselines_v48_124_final_v2}"
GPU_LIST="${GPU_LIST:-0,1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
PROFILE_LATENCY="${PROFILE_LATENCY:-true}"
LATENCY_GPU="${LATENCY_GPU:-0}"
WOMD_ROLE="${WOMD_ROLE:-validation}"
# Optional path to a previous external-baseline root. Validated learned
# checkpoints are copied into the final run so only closed-loop testing is redone.
# Missing/invalid methods (e.g. a failed GameFormer training) are trained normally.
PRETRAINED_BASELINE_ROOT="${PRETRAINED_BASELINE_ROOT:-}"

ensure_target_locks() {
  if [[ ! -s "$OCRAP_OUT/target_keys/safe.json" || ! -s "$OCRAP_OUT/target_keys/near.json" || ! -s "$OCRAP_OUT/target_keys/contact.json" ]]; then
    echo "[FINAL BASELINE] target locks missing; building them directly from frozen bucket+WOMD provenance"
    env BASE_OUT="$BASE_OUT" OCRAP_FINAL_CHARACTERIZATION_OUT="$OCRAP_OUT" WOMD_ROLE="$WOMD_ROLE" \
      bash scripts/build_final_observation_legal_target_locks.sh
  fi
}
ensure_target_locks

run_one() {
  local r="$1" keyfile="$OCRAP_OUT/target_keys/$1.json"
  [[ -s "$keyfile" ]] || { echo "missing observation-legal target-key lock: $keyfile" >&2; exit 30; }
  if [[ -n "$PRETRAINED_BASELINE_ROOT" && -d "$PRETRAINED_BASELINE_ROOT/$r/checkpoints" ]]; then
    mkdir -p "$OUT/$r/checkpoints"
    cp -a "$PRETRAINED_BASELINE_ROOT/$r/checkpoints/." "$OUT/$r/checkpoints/"
    if [[ "$r" == near && -f "$PRETRAINED_BASELINE_ROOT/near/conformal_calibration.json" ]]; then
      cp -f "$PRETRAINED_BASELINE_ROOT/near/conformal_calibration.json" "$OUT/near/conformal_calibration.json"
    fi
    echo "[FINAL BASELINE] staged validated checkpoint candidates from $PRETRAINED_BASELINE_ROOT/$r/checkpoints"
  fi
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
