#!/usr/bin/env bash
# V48.124.10.7.2 terminal internal closure.
# No GPU work, no training, no selector/model/checkpoint/candidate/recovery change.
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
ONE_SHOT_RESULTS="${OCRAP_ONE_SHOT_RESULTS:-$BASE_OUT/OC-RAP-v48.124.10.7.1-ONE-SHOT-ACTION-REALIZATION-results.zip}"
OUT_JSON="${OCRAP_TERMINAL_CLOSURE_JSON:-$BASE_OUT/OC-RAP-v48.124.10.7.2-TERMINAL-INTERNAL-CLOSURE.json}"

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

[[ -s "$ONE_SHOT_RESULTS" ]] || { echo "missing authoritative V48.124.10.7.1 one-shot results: $ONE_SHOT_RESULTS" >&2; exit 30; }
python tools/adjudicate_terminal_internal_closure.py \
  --one-shot-results "$ONE_SHOT_RESULTS" \
  --output "$OUT_JSON"

echo "terminal internal closure: $OUT_JSON"
