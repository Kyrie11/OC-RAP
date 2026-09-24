#!/usr/bin/env bash
# Validate and render already-complete selected traces. Never runs closed-loop.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/tools:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
: "${ROOT:?set ROOT to the regime_visualization output directory}"
: "${TRACE_MAX_STEPS:=60}"
: "${FPS:=10}"
: "${CAMERA_MODE:=fixed}"
: "${VIEW_RADIUS_M:=35}"
: "${VIDEO_FORMAT:=mp4}"
: "${FIGURE_REGIME_JOBS:=3}"
: "${VIDEO_REGIME_JOBS:=2}"
: "${VIS_PROGRESS_INTERVAL_S:=15}"
: "${VIS_RENDER_PROGRESS_EVERY:=10}"
: "${FORCE_RENDER:=false}"
export FIGURE_REGIME_JOBS VIDEO_REGIME_JOBS VIS_RENDER_PROGRESS_EVERY FORCE_RENDER
TRACE_ROOT="$ROOT/selective_traces"
SELECTION_ROOT="$ROOT/selection"
mkdir -p "$ROOT/logs"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
count_outputs() {
  local dir="$1"
  [[ -d "$dir" ]] || { echo "files=0 size=0MiB"; return; }
  local count bytes
  count=$(find "$dir" -type f \( -name '*.png' -o -name '*.pdf' -o -name '*.mp4' -o -name '*.gif' \) 2>/dev/null | wc -l | tr -d ' ')
  bytes=$(find "$dir" -type f \( -name '*.png' -o -name '*.pdf' -o -name '*.mp4' -o -name '*.gif' \) -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.1f", s/1048576}')
  echo "files=$count size=${bytes:-0.0}MiB"
}
run_stage() {
  local label="$1" outdir="$2" logfile="$3"; shift 3
  echo "[FINALIZE][STAGE-START] $(stamp) stage=$label jobs=$([[ $label == figures ]] && echo "$FIGURE_REGIME_JOBS" || echo "$VIDEO_REGIME_JOBS")"
  ( "$@" ) > >(tee "$logfile") 2>&1 &
  local pid=$! started=$SECONDS
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$VIS_PROGRESS_INTERVAL_S"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[FINALIZE][HEARTBEAT] $(stamp) stage=$label elapsed=$((SECONDS-started))s $(count_outputs "$outdir")"
      find "$outdir" -type f \( -name '*.png' -o -name '*.pdf' -o -name '*.mp4' -o -name '*.gif' \) -printf '[FINALIZE][OUTPUT] %p %s bytes\n' 2>/dev/null | sort | tail -6 || true
    fi
  done
  local rc=0
  set +e
  wait "$pid"
  rc=$?
  set -e
  if (( rc != 0 )); then
    echo "[FINALIZE][ERROR] $(stamp) stage=$label rc=$rc" >&2
    return "$rc"
  fi
  echo "[FINALIZE][STAGE-DONE] $(stamp) stage=$label elapsed=$((SECONDS-started))s $(count_outputs "$outdir")"
}

# Non-destructive inventory: current traces must all be reusable. Do not clean.
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

echo "[FINALIZE] rendering figures first; existing valid PNG/PDF files are reused unless FORCE_RENDER=true"
run_stage figures "$ROOT/paper_figures" "$ROOT/logs/03_paper_figures.log" \
  env TRACE_ROOT="$TRACE_ROOT" SELECTION_ROOT="$SELECTION_ROOT" OUT="$ROOT/paper_figures" VIEW_RADIUS_M="$VIEW_RADIUS_M" \
      FIGURE_REGIME_JOBS="$FIGURE_REGIME_JOBS" FORCE_RENDER="$FORCE_RENDER" \
      bash scripts/render_regime_paper_figures.sh

if [[ "$VIDEO_FORMAT" == mp4 ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[FINALIZE][WARN] ffmpeg not found; using auto/GIF video output" >&2
  VIDEO_FORMAT=auto
fi
echo "[FINALIZE] rendering videos; existing valid videos are reused unless FORCE_RENDER=true"
run_stage videos "$ROOT/videos" "$ROOT/logs/04_videos.log" \
  env TRACE_ROOT="$TRACE_ROOT" SELECTION_ROOT="$SELECTION_ROOT" OUT="$ROOT/videos" \
      FPS="$FPS" FORMAT="$VIDEO_FORMAT" CAMERA_MODE="$CAMERA_MODE" VIEW_RADIUS_M="$VIEW_RADIUS_M" \
      VIDEO_REGIME_JOBS="$VIDEO_REGIME_JOBS" VIS_RENDER_PROGRESS_EVERY="$VIS_RENDER_PROGRESS_EVERY" FORCE_RENDER="$FORCE_RENDER" \
      bash scripts/render_regime_videos.sh

python tools/audit_regime_visualization_outputs.py --root "$ROOT" --expected-scenes 3 \
  | tee "$ROOT/logs/05_output_audit.log"
echo "[FINALIZE] paper figures: $ROOT/paper_figures/PAPER_FIGURE_INDEX.json"
echo "[FINALIZE] videos: $ROOT/videos/REGIME_VIDEO_INDEX.json"
