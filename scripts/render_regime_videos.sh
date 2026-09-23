#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
: "${TRACE_ROOT:?set TRACE_ROOT to selective_traces}"
: "${SELECTION_ROOT:?set SELECTION_ROOT}"
: "${OUT:=$TRACE_ROOT/videos}"
: "${FPS:=10}"; : "${FORMAT:=mp4}"; : "${CAMERA_MODE:=fixed}"; : "${VIEW_RADIUS_M:=35}"
: "${INCLUDE_SINGLES:=false}"; : "${INCLUDE_GLOBAL_STRONGEST_PAIR:=false}"; : "${INCLUDE_WORST_PAIR:=false}"; : "${INCLUDE_ALL_METHOD_MONTAGE:=true}"
: "${VIDEO_REGIME_JOBS:=2}"; : "${FORCE_RENDER:=false}"; : "${VIS_RENDER_PROGRESS_EVERY:=10}"
export VIS_RENDER_PROGRESS_EVERY
mkdir -p "$OUT"

render_one() {
  local regime="$1"; shift
  local -a methods=("$@") trace_args=() optional=()
  local m
  trace_args+=(--trace "ocrap=$TRACE_ROOT/ocrap/$regime/closed_loop_ocrap.json.scenes.jsonl")
  for m in "${methods[@]}"; do trace_args+=(--trace "$m=$TRACE_ROOT/external/$regime/closed_loop_${m}.json.scenes.jsonl"); done
  [[ "$INCLUDE_SINGLES" == true ]] && optional+=(--include-singles)
  [[ "$INCLUDE_GLOBAL_STRONGEST_PAIR" == true ]] && optional+=(--include-global-strongest-pair)
  [[ "$INCLUDE_WORST_PAIR" == true ]] && optional+=(--include-worst-pair)
  [[ "$INCLUDE_ALL_METHOD_MONTAGE" == true ]] && optional+=(--include-all-method-montage)
  [[ "$FORCE_RENDER" == true ]] && optional+=(--force)
  echo "[VIDEO][REGIME-START] regime=$regime pid=$$"
  python tools/render_regime_visualization_videos.py \
    "${trace_args[@]}" --selection "$SELECTION_ROOT/${regime}_selection.json" --output-dir "$OUT" \
    --fps "$FPS" --format "$FORMAT" --camera-mode "$CAMERA_MODE" --view-radius-m "$VIEW_RADIUS_M" "${optional[@]}"
  echo "[VIDEO][REGIME-DONE] regime=$regime"
}

declare -a PIDS=() NAMES=()
launch() {
  local name="$1"; shift
  while (( ${#PIDS[@]} >= VIDEO_REGIME_JOBS )); do
    local pid="${PIDS[0]}" n="${NAMES[0]}"
    if ! wait "$pid"; then echo "[VIDEO][ERROR] regime=$n failed" >&2; exit 2; fi
    PIDS=("${PIDS[@]:1}"); NAMES=("${NAMES[@]:1}")
  done
  ( render_one "$name" "$@" ) &
  PIDS+=("$!"); NAMES+=("$name")
}
launch safe gameformer_lite plantf pluto pdm_closed pdm_hybrid idm diffusion_planner
launch near marc_lite racp_lite robust_scenario_mpc predictive_safety_filter dr_cvar_safety_filter conformal_predictive_safety_filter flow_planner plan_r1 betopnet
launch contact postimpact_mpc_lite post_crash_braking postimpact_motion_tvlqr post_collision_restoration compensatory_postimpact_mpc robust_postimpact_control
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then echo "[VIDEO][ERROR] regime=${NAMES[$i]} failed" >&2; exit 2; fi
done

python - "$OUT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); docs=[]
for r in ('SAFE','NEAR','CONTACT'):
 p=root/f'{r}_VIDEO_INDEX.json'; docs.append(json.loads(p.read_text()))
out={'event':'regime_video_index','num_videos':sum(d['num_videos'] for d in docs),'regimes':{d['regime']:d for d in docs}}
(root/'REGIME_VIDEO_INDEX.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({'event':out['event'],'num_videos':out['num_videos'],'index':str(root/'REGIME_VIDEO_INDEX.json')}), flush=True)
PY
find "$OUT" -type f \( -name '*.mp4' -o -name '*.gif' \) -printf '%p %s bytes\n' | sort > "$OUT/VIDEO_FILES.txt"
wc -l "$OUT/VIDEO_FILES.txt"
