#!/usr/bin/env bash
# Validate and render already-complete selected traces. Never runs closed-loop.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/tools:$REPO${PYTHONPATH:+:$PYTHONPATH}"
: "${ROOT:?set ROOT to the regime_visualization output directory}"
: "${TRACE_MAX_STEPS:=60}"
: "${FPS:=10}"
: "${CAMERA_MODE:=fixed}"
: "${VIEW_RADIUS_M:=35}"
: "${VIDEO_FORMAT:=mp4}"
TRACE_ROOT="$ROOT/selective_traces"
SELECTION_ROOT="$ROOT/selection"
mkdir -p "$ROOT/logs"

# Non-destructive inventory: current traces must all be reusable.  Do not clean.
python tools/prepare_selected_trace_reruns.py \
  --trace-root "$TRACE_ROOT" --selection-root "$SELECTION_ROOT" \
  --output "$TRACE_ROOT/TRACE_FINALIZE_PRECHECK.json" \
  | tee "$ROOT/logs/02a_finalize_trace_inventory.log"
python - "$TRACE_ROOT/TRACE_FINALIZE_PRECHECK.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
bad=[x for x in d.get('methods',[]) if not x.get('valid_reusable')]
if bad:
    for x in bad:
        print(f"[FINALIZE][INVALID] {x['regime']}/{x['method']}: {'; '.join(x.get('errors') or [])}", file=sys.stderr)
    raise SystemExit(f"Cannot render: {len(bad)} of {len(d.get('methods',[]))} trace families are not reusable")
print(f"[FINALIZE] all {len(d.get('methods',[]))} selected-trace families are reusable")
PY

python tools/check_selected_trace_contract.py \
  --trace-root "$TRACE_ROOT" --selection-root "$SELECTION_ROOT" \
  --trace-max-steps "$TRACE_MAX_STEPS" --output "$TRACE_ROOT/TRACE_CONTRACT.json" \
  | tee "$ROOT/logs/02b_trace_contract.log"

TRACE_ROOT="$TRACE_ROOT" SELECTION_ROOT="$SELECTION_ROOT" OUT="$ROOT/paper_figures" VIEW_RADIUS_M="$VIEW_RADIUS_M" \
  bash scripts/render_regime_paper_figures.sh 2>&1 | tee "$ROOT/logs/03_paper_figures.log"

if [[ "$VIDEO_FORMAT" == mp4 ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[FINALIZE][WARN] ffmpeg not found; using auto/GIF video output" >&2
  VIDEO_FORMAT=auto
fi
TRACE_ROOT="$TRACE_ROOT" SELECTION_ROOT="$SELECTION_ROOT" OUT="$ROOT/videos" \
  FPS="$FPS" FORMAT="$VIDEO_FORMAT" CAMERA_MODE="$CAMERA_MODE" VIEW_RADIUS_M="$VIEW_RADIUS_M" \
  bash scripts/render_regime_videos.sh 2>&1 | tee "$ROOT/logs/04_videos.log"

python tools/audit_regime_visualization_outputs.py --root "$ROOT" --expected-scenes 3 \
  | tee "$ROOT/logs/05_output_audit.log"
echo "[FINALIZE] paper figures: $ROOT/paper_figures/PAPER_FIGURE_INDEX.json"
echo "[FINALIZE] videos: $ROOT/videos/REGIME_VIDEO_INDEX.json"
