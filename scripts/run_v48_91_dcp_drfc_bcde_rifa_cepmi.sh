#!/usr/bin/env bash
# V48.91 OC-CEPMI: Common-Exogenous Physical-Margin Identifiability.
# V48.91.4 engineering-only replay fix: final-layer canonical V48.14 sample-local
# balanced-pass reconstruction, fail-fast identity guard, resumable exact replay,
# plus the V48.91.2 sparse/history-cache/2-GPU acceleration.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
V90_INDEX="${V4891_V90_INDEX:-$BASE_OUT/OC-RAP-v48.90-partition-transport-audit.jsonl}"
V90_SUMMARY="${V4891_V90_SUMMARY:-$BASE_OUT/OC-RAP-v48.90-partition-transport-audit-summary.json}"
V90_COMPARE="${V4891_V90_COMPARE:-$BASE_OUT/OC-RAP-v48.90-DCP-DRFC-BCDE-RIFA-OC-CEPT-comparison.json}"
REPLAY_CONFIG="${V4891_REPLAY_CONFIG:-}"
WOMD_SOURCE_PATTERN="${V4891_WOMD_SOURCE:-${WOMD_VAL:-}}"
REPLAY_WORKERS="${V4891_REPLAY_WORKERS:-1}"
PROGRESS_EVERY="${V4891_PROGRESS_EVERY:-25}"
REPLAY_RESUME="${V4891_REPLAY_RESUME:-1}"
CLEAR_REPLAY_CHECKPOINTS="${V4891_CLEAR_REPLAY_CHECKPOINTS:-0}"
GPU0="${GPU0:-}"
GPU1="${GPU1:-}"
RUNTIME="$BASE_OUT/OC-RAP-v48.91-runtime-code-contract.json"
SIDECAR="$BASE_OUT/OC-RAP-v48.91-common-exogenous-future-physical-sidecar.jsonl.gz"
SIDECAR_SUMMARY="$BASE_OUT/OC-RAP-v48.91-common-exogenous-future-physical-sidecar-summary.json"
AUDIT="$BASE_OUT/OC-RAP-v48.91-common-exogenous-physical-response-audit.jsonl"
SUMMARY="$BASE_OUT/OC-RAP-v48.91-common-exogenous-physical-response-audit-summary.json"
COMPARE="$BASE_OUT/OC-RAP-v48.91-DCP-DRFC-BCDE-RIFA-OC-CEPMI-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.91-PIPELINE_COMPLETE.json"
AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.91-OC-CEPMI-audits.zip"
mkdir -p "$BASE_OUT"
rm -f "$RUNTIME" "$SIDECAR" "$SIDECAR_SUMMARY" "$AUDIT" "$SUMMARY" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"
rm -f "$BASE_OUT"/OC-RAP-v48.91-sidecar.part*.jsonl.gz "$BASE_OUT"/OC-RAP-v48.91-sidecar.part*.summary.json "$BASE_OUT"/OC-RAP-v48.91-sidecar.worker*.log
if [[ "$CLEAR_REPLAY_CHECKPOINTS" == 1 ]]; then rm -f "$BASE_OUT"/OC-RAP-v48.91-sidecar.worker*.checkpoint.jsonl; fi

python tools/check_v48_91_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V90_INDEX" "$V90_SUMMARY" "$V90_COMPARE" <<'PY'
import json,pathlib,sys
for p in map(pathlib.Path,sys.argv[1:]):
    if not p.is_file(): raise SystemExit(f'missing V48.90 prerequisite: {p}')
c=json.loads(pathlib.Path(sys.argv[3]).read_text()); q=c.get('preregistered_decision') or {}
if not(c.get('valid') and q.get('exogenous_partition_transport_go') and q.get('partition_stability_directional_relevance_go') and not q.get('transport_physical_response_identifiability_go')):
    raise SystemExit('V48.90 transport-GO / physical-response-STOP prerequisite missing')
PY
REPLAY_ARGS=()
if [[ -n "$REPLAY_CONFIG" ]]; then REPLAY_ARGS+=(--replay-config "$REPLAY_CONFIG"); fi
if [[ -n "$WOMD_SOURCE_PATTERN" ]]; then REPLAY_ARGS+=(--womd-source-pattern "$WOMD_SOURCE_PATTERN"); fi
if [[ "$REPLAY_RESUME" == 1 ]]; then REPLAY_ARGS+=(--resume-checkpoint); else REPLAY_ARGS+=(--no-resume-checkpoint); fi

if ! [[ "$REPLAY_WORKERS" =~ ^[0-9]+$ ]] || (( REPLAY_WORKERS < 1 || REPLAY_WORKERS > 2 )); then
  echo "V4891_REPLAY_WORKERS must be 1 or 2; got $REPLAY_WORKERS" >&2
  exit 30
fi

if (( REPLAY_WORKERS == 1 )); then
  echo "[v48.91-perf] single replay worker; set V4891_REPLAY_WORKERS=2 with GPU0/GPU1 for two-GPU replay" >&2
  if [[ -n "$GPU0" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU0" python tools/build_v48_91_common_exogenous_physical_sidecar.py \
      --v48-90-audit "$V90_INDEX" --output "$SIDECAR" --summary "$SIDECAR_SUMMARY" \
      --num-workers 1 --worker-index 0 --progress-every "$PROGRESS_EVERY" --checkpoint "$BASE_OUT/OC-RAP-v48.91-sidecar.worker0.checkpoint.jsonl" "${REPLAY_ARGS[@]}"
  else
    python tools/build_v48_91_common_exogenous_physical_sidecar.py \
      --v48-90-audit "$V90_INDEX" --output "$SIDECAR" --summary "$SIDECAR_SUMMARY" \
      --num-workers 1 --worker-index 0 --progress-every "$PROGRESS_EVERY" --checkpoint "$BASE_OUT/OC-RAP-v48.91-sidecar.worker0.checkpoint.jsonl" "${REPLAY_ARGS[@]}"
  fi
else
  GPU0="${GPU0:-0}"
  GPU1="${GPU1:-1}"
  echo "[v48.91-perf] launching two exact replay shards on GPU${GPU0} and GPU${GPU1}" >&2
  pids=()
  for w in 0 1; do
    if (( w == 0 )); then gpu="$GPU0"; else gpu="$GPU1"; fi
    part="$BASE_OUT/OC-RAP-v48.91-sidecar.part${w}.jsonl.gz"
    psum="$BASE_OUT/OC-RAP-v48.91-sidecar.part${w}.summary.json"
    log="$BASE_OUT/OC-RAP-v48.91-sidecar.worker${w}.log"
    (
      set -o pipefail
      CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python tools/build_v48_91_common_exogenous_physical_sidecar.py \
          --v48-90-audit "$V90_INDEX" --output "$part" --summary "$psum" \
          --num-workers 2 --worker-index "$w" --progress-every "$PROGRESS_EVERY" --checkpoint "$BASE_OUT/OC-RAP-v48.91-sidecar.worker${w}.checkpoint.jsonl" "${REPLAY_ARGS[@]}" \
        2>&1 | sed -u "s/^/[v48.91-w${w}] /" | tee "$log"
    ) &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then rc=30; fi
  done
  (( rc == 0 )) || exit "$rc"
  python tools/merge_v48_91_common_exogenous_physical_sidecar_parts.py \
    --part "$BASE_OUT/OC-RAP-v48.91-sidecar.part0.jsonl.gz" \
    --part "$BASE_OUT/OC-RAP-v48.91-sidecar.part1.jsonl.gz" \
    --part-summary "$BASE_OUT/OC-RAP-v48.91-sidecar.part0.summary.json" \
    --part-summary "$BASE_OUT/OC-RAP-v48.91-sidecar.part1.summary.json" \
    --output "$SIDECAR" --summary "$SIDECAR_SUMMARY"
fi

python tools/build_v48_91_common_exogenous_physical_response_audit.py \
  --v48-90-audit "$V90_INDEX" --sidecar "$SIDECAR" --sidecar-summary "$SIDECAR_SUMMARY" \
  --output "$AUDIT" --summary "$SUMMARY"
python tools/compare_v48_91_cepmi.py --summary "$SUMMARY" --v48-90-summary "$V90_SUMMARY" --v48-90-comparison "$V90_COMPARE" --output "$COMPARE"
python tools/check_v48_91_pipeline_complete.py \
  --runtime "$RUNTIME" --sidecar "$SIDECAR" --sidecar-summary "$SIDECAR_SUMMARY" --audit "$AUDIT" --audit-summary "$SUMMARY" \
  --comparison "$COMPARE" --v48-90-comparison "$V90_COMPARE" --output "$COMPLETE"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$SIDECAR_SUMMARY" "$SUMMARY" "$COMPARE" "$COMPLETE"
printf 'V48.91 complete. Upload:\n%s\n%s\n%s\n' "$AUDITS_ZIP" "$AUDIT" "$SIDECAR_SUMMARY"
