#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
OCRAP_OUT="${OCRAP_FINAL_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"
OCRAP_LATENCY_OUT="${OCRAP_LATENCY_OUT:-$BASE_OUT/ocrap_v48_124_latency_isolated}"
BASELINE_OUT="${FINAL_EXTERNAL_BASELINE_OUT:-$BASE_OUT/external_baselines_v48_124_final_v2}"
BASELINE_LATENCY_OUT="${FINAL_EXTERNAL_BASELINE_LATENCY_OUT:-${BASELINE_OUT}_latency_isolated}"
TABLE_OUT="${FINAL_TABLE_OUT:-$BASE_OUT/final_regime_comparison_tables_v48_124}"
python tools/build_submission_external_baseline_tables.py \
  --ocrap-run "$OCRAP_OUT/ocrap" \
  --ocrap-latency-run "$OCRAP_LATENCY_OUT" \
  --safe-run "$BASELINE_OUT/safe" --near-run "$BASELINE_OUT/near" --contact-run "$BASELINE_OUT/contact" \
  --safe-latency-run "$BASELINE_LATENCY_OUT/safe" --near-latency-run "$BASELINE_LATENCY_OUT/near" --contact-latency-run "$BASELINE_LATENCY_OUT/contact" \
  --variants balanced,precision --output-dir "$TABLE_OUT"
echo "final tables: $TABLE_OUT"
