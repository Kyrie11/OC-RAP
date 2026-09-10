#!/usr/bin/env bash
# Rebuild the paper videos from the corrected population metrics/traces.
# Intentionally does NOT call repair_submission_visualization_stale_replays_v54.sh.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

: "${BASE_OUT:=/home/senzeyu2/code/OC-RAP/runs}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${CUDA_DEVICES:=0,1}"
: "${NUM_SCENES:=3}"
: "${MAX_SELECTED_TIER_RANK:=0}"
: "${FPS:=10}"
: "${TRACE_MAX_STEPS:=60}"
: "${CAMERA_MODE:=fixed}"
: "${VIEW_RADIUS_M:=35}"
: "${VIDEO_FORMAT:=mp4}"
: "${JOBS_PER_GPU:=3}"
: "${MAX_PARALLEL:=6}"
: "${OUT:=$BASE_OUT/submission_visualization_v54}"

if [[ -e "$OUT" ]]; then
  stamp="$(date +%Y%m%d-%H%M%S)"
  dst="$BASE_OUT/submission_visualization_backup_v55_$stamp"
  mv "$OUT" "$dst"
  echo "[ARCHIVE] old visualization -> $dst"
fi

BASE_OUT="$BASE_OUT" OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
NUM_SCENES="$NUM_SCENES" MAX_SELECTED_TIER_RANK="$MAX_SELECTED_TIER_RANK" \
FPS="$FPS" TRACE_MAX_STEPS="$TRACE_MAX_STEPS" CAMERA_MODE="$CAMERA_MODE" \
VIEW_RADIUS_M="$VIEW_RADIUS_M" VIDEO_FORMAT="$VIDEO_FORMAT" \
JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" OUT="$OUT" \
bash scripts/build_submission_visualization_v54.sh

echo "[DONE] corrected visualization: $OUT/videos/REGIME_VIDEO_INDEX.json"
