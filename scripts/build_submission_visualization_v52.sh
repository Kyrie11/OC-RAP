#!/usr/bin/env bash
# Two-stage by default: always preflight + select; opt into expensive traces/video.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
: "${OUT:=/home/senzeyu2/code/OC-RAP/runs/submission_visualization_v52}"
: "${BUILD_TRACES:=false}"; : "${BUILD_VIDEOS:=false}"
OUT="$OUT" bash scripts/build_regime_visualization_selection.sh
if [[ "$BUILD_TRACES" == true || "$BUILD_VIDEOS" == true ]]; then
  SELECTION_ROOT="$OUT/selection" INPUT_CONTRACT="$OUT/provenance/VISUALIZATION_INPUT_CONTRACT.json" OUT="$OUT/selective_traces" \
    bash scripts/run_selected_regime_visualization_traces.sh
fi
if [[ "$BUILD_VIDEOS" == true ]]; then
  TRACE_ROOT="$OUT/selective_traces" SELECTION_ROOT="$OUT/selection" OUT="$OUT/videos" \
    bash scripts/build_regime_visualization_videos.sh
fi
