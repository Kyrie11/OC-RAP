#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
OCRAP_OUT="${OCRAP_FINAL_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"
BASELINE_OUT="${FINAL_EXTERNAL_BASELINE_OUT:-$BASE_OUT/external_baselines_v48_124_final}"
TABLE_OUT="${FINAL_TABLE_OUT:-$BASE_OUT/final_regime_comparison_tables_v48_124}"
USE_ISOLATED_LATENCY="${USE_ISOLATED_LATENCY:-true}"
SOURCE="$BASELINE_OUT"
if [[ "${USE_ISOLATED_LATENCY,,}" == true || "${USE_ISOLATED_LATENCY,,}" == 1 || "${USE_ISOLATED_LATENCY,,}" == yes ]]; then
  SOURCE="${BASELINE_OUT}_latency_isolated"
fi
python tools/build_submission_external_baseline_tables.py \
  --ocrap-run "$OCRAP_OUT/ocrap" \
  --safe-run "$SOURCE/safe" \
  --near-run "$SOURCE/near" \
  --contact-run "$SOURCE/contact" \
  --variants balanced,precision \
  --output-dir "$TABLE_OUT"
echo "final tables: $TABLE_OUT"
