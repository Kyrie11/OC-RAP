#!/usr/bin/env bash
# One-command final OC-RAP evaluation:
#   1) build/freeze observation-legal Safe/Near/Contact target locks and Contact anchor,
#   2) run balanced/precision (and nominal by default) three-regime closed-loop characterization,
#   3) run publication latency under the isolated single-process/single-GPU contract.
#
# This is intentionally a thin compatibility wrapper around the existing final launcher.
# The historical three commands remain supported.
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
LATENCY_GPU="${LATENCY_GPU:-$GPU0}"
OCRAP_FINAL_CHARACTERIZATION_OUT="${OCRAP_FINAL_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"
OCRAP_LATENCY_OUT="${OCRAP_LATENCY_OUT:-$BASE_OUT/ocrap_v48_124_latency_isolated}"

exec env \
  BASE_OUT="$BASE_OUT" \
  GPU0="$GPU0" GPU1="$GPU1" LATENCY_GPU="$LATENCY_GPU" \
  OCRAP_FINAL_CHARACTERIZATION_OUT="$OCRAP_FINAL_CHARACTERIZATION_OUT" \
  OCRAP_LATENCY_OUT="$OCRAP_LATENCY_OUT" \
  BUILD_TARGET_LOCKS=true PROFILE_LATENCY=true \
  bash scripts/run_final_locked_three_regime_characterization.sh
