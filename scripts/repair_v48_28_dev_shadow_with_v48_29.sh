#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="${OUT:?OUT must point to the existing v48.28 dedicated run}"
rm -rf "$OUT/dev_shadow_closed_loop_v48_29_repair"
SHADOW_ROOT_SUFFIX=_v48_29_repair \
SHADOW_LABEL_MODE="${SHADOW_LABEL_MODE:-fast}" \
SHADOW_AUDIT_LABELS="${SHADOW_AUDIT_LABELS:-0}" \
OUT="$OUT" OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}" \
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}/calibration_v48_14_prism_4814}" \
GPU0="${GPU0:-0}" GPU1="${GPU1:-1}" \
DEV_SHADOW_WOMD_SOURCE="${DEV_SHADOW_WOMD_SOURCE:-${WOMD_VAL:-}}" \
DEV_SHADOW_RAW_MAX_SCENARIOS="${DEV_SHADOW_RAW_MAX_SCENARIOS:-0}" \
CL_RESUME=0 bash scripts/run_v48_29_dev_shadow_closed_loop.sh
