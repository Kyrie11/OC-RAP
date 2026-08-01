#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="${OUT:?Set OUT to the existing v48.27 dedicated run}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
WOMD_VAL="${WOMD_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
GPU0="${GPU0:-0}" GPU1="${GPU1:-1}" \
DEV_SHADOW_WOMD_SOURCE="$WOMD_VAL" DEV_SHADOW_RAW_MAX_SCENARIOS=0 \
CL_RESUME=0 OUT="$OUT" OCRAP_ROOT="$OCRAP_ROOT" PROTOCOL_ROOT="$PROTOCOL_ROOT" \
  bash scripts/run_v48_28_dev_shadow_closed_loop.sh
