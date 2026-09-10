#!/usr/bin/env bash
# Final paper-facing three-regime qualitative-video pipeline.
# Population selection uses all six baselines.  Each primary video shows OC-RAP
# against the strongest/hardest per-scene external comparator using compact
# paper display names; canonical method identifiers remain unchanged on disk.
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

: "${BASE_OUT:=/home/senzeyu2/code/OC-RAP/runs}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${MODEL_VARIANT:=balanced}"
: "${OCRAP_MODEL_RUN:=$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
: "${OCRAP_RESULTS_ROOT:=$BASE_OUT/ocrap_v48_111_submission_three_regime/$MODEL_VARIANT}"
: "${SAFE_EXTERNAL_ROOT:=$BASE_OUT/safe_external}"
: "${NEAR_EXTERNAL_ROOT:=$BASE_OUT/near_external}"
: "${CONTACT_EXTERNAL_ROOT:=$BASE_OUT/contact_external}"
: "${OUT:=$BASE_OUT/submission_visualization_v54}"
: "${CUDA_DEVICES:=0,1}"
: "${NUM_SCENES:=3}"
: "${MAX_SELECTED_TIER_RANK:=0}"
: "${MIN_VIDEO_DURATION_S:=5.0}"
: "${FALLBACK_MIN_VIDEO_DURATION_S:=3.0}"
: "${TRACE_MAX_STEPS:=60}"
: "${FPS:=10}"
: "${VIDEO_FORMAT:=mp4}"
: "${CAMERA_MODE:=fixed}"
: "${VIEW_RADIUS_M:=35}"

mkdir -p "$OUT"

echo '[1/3] population-level fail-closed selection'
OCRAP_RESULTS_ROOT="$OCRAP_RESULTS_ROOT" \
OCRAP_MODEL_RUN="$OCRAP_MODEL_RUN" \
OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" MODEL_VARIANT="$MODEL_VARIANT" \
SAFE_EXTERNAL_ROOT="$SAFE_EXTERNAL_ROOT" NEAR_EXTERNAL_ROOT="$NEAR_EXTERNAL_ROOT" CONTACT_EXTERNAL_ROOT="$CONTACT_EXTERNAL_ROOT" \
OUT="$OUT" NUM_SCENES="$NUM_SCENES" MAX_SELECTED_TIER_RANK="$MAX_SELECTED_TIER_RANK" \
MIN_VIDEO_DURATION_S="$MIN_VIDEO_DURATION_S" FALLBACK_MIN_VIDEO_DURATION_S="$FALLBACK_MIN_VIDEO_DURATION_S" \
bash scripts/build_regime_visualization_selection.sh

echo '[2/3] rerun only selected targets with synchronized traces'
SELECTION_ROOT="$OUT/selection" \
INPUT_CONTRACT="$OUT/provenance/VISUALIZATION_INPUT_CONTRACT.json" \
OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" OCRAP_MODEL_RUN="$OCRAP_MODEL_RUN" MODEL_VARIANT="$MODEL_VARIANT" \
SAFE_EXTERNAL_ROOT="$SAFE_EXTERNAL_ROOT" NEAR_EXTERNAL_ROOT="$NEAR_EXTERNAL_ROOT" CONTACT_EXTERNAL_ROOT="$CONTACT_EXTERNAL_ROOT" \
CUDA_DEVICES="$CUDA_DEVICES" TRACE_MAX_STEPS="$TRACE_MAX_STEPS" ALLOW_DIAGNOSTIC_RC20=1 \
OUT="$OUT/selective_traces" \
bash scripts/run_selected_regime_visualization_traces.sh

echo '[3/3] render primary paper-facing pair videos'
TRACE_ROOT="$OUT/selective_traces" \
SELECTION_ROOT="$OUT/selection" \
OUT="$OUT/videos" \
FPS="$FPS" FORMAT="$VIDEO_FORMAT" CAMERA_MODE="$CAMERA_MODE" VIEW_RADIUS_M="$VIEW_RADIUS_M" \
INCLUDE_SINGLES=false INCLUDE_GLOBAL_STRONGEST_PAIR=false INCLUDE_WORST_PAIR=false \
bash scripts/build_regime_visualization_videos.sh

python - "$OUT/videos/REGIME_VIDEO_INDEX.json" <<'PY'
import json,sys
p=sys.argv[1]; d=json.load(open(p))
print(json.dumps({"event":"submission_visualization_v54_complete","num_videos":d.get("num_videos"),"index":p},indent=2))
PY
