#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/tools:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
: "${TRACE_ROOT:?set TRACE_ROOT to selective_traces}"
: "${SELECTION_ROOT:?set SELECTION_ROOT}"
: "${OUT:=$TRACE_ROOT/paper_figures}"
: "${VIEW_RADIUS_M:=35}"
: "${FIGURE_REGIME_JOBS:=3}"
: "${FORCE_RENDER:=false}"
mkdir -p "$OUT"

render_one() {
  local regime="$1"; shift
  local -a methods=("$@") trace_args=() extra=()
  local m
  trace_args+=(--trace "ocrap=$TRACE_ROOT/ocrap/$regime/closed_loop_ocrap.json.scenes.jsonl")
  for m in "${methods[@]}"; do trace_args+=(--trace "$m=$TRACE_ROOT/external/$regime/closed_loop_${m}.json.scenes.jsonl"); done
  [[ "$FORCE_RENDER" == true ]] && extra+=(--force)
  echo "[FIG][REGIME-START] regime=$regime pid=$$"
  python tools/render_regime_paper_figures.py \
    "${trace_args[@]}" \
    --selection "$SELECTION_ROOT/${regime}_selection.json" \
    --output-dir "$OUT" \
    --view-radius-m "$VIEW_RADIUS_M" "${extra[@]}"
  echo "[FIG][REGIME-DONE] regime=$regime"
}

# Run regimes concurrently without wait -p/wait -n; ordinary PID wait works on old Bash.
declare -a PIDS=() NAMES=()
launch() {
  local name="$1"; shift
  while (( ${#PIDS[@]} >= FIGURE_REGIME_JOBS )); do
    local pid="${PIDS[0]}" n="${NAMES[0]}"
    if ! wait "$pid"; then echo "[FIG][ERROR] regime=$n failed" >&2; exit 2; fi
    PIDS=("${PIDS[@]:1}"); NAMES=("${NAMES[@]:1}")
  done
  ( render_one "$name" "$@" ) &
  PIDS+=("$!"); NAMES+=("$name")
}
launch safe gameformer_lite plantf pluto pdm_closed pdm_hybrid idm
launch near marc_lite racp_lite robust_scenario_mpc predictive_safety_filter dr_cvar_safety_filter conformal_predictive_safety_filter
launch contact postimpact_mpc_lite post_crash_braking postimpact_motion_tvlqr post_collision_restoration compensatory_postimpact_mpc robust_postimpact_control
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then echo "[FIG][ERROR] regime=${NAMES[$i]} failed" >&2; exit 2; fi
done

python - "$OUT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); docs=[]
for r in ('SAFE','NEAR','CONTACT'):
    docs.append(json.loads((root/f'{r}_PAPER_FIGURE_INDEX.json').read_text()))
out={'event':'paper_figure_index_v124','regimes':{d['regime']:d for d in docs}}
(root/'PAPER_FIGURE_INDEX.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'event':out['event'],'index':str(root/'PAPER_FIGURE_INDEX.json')}), flush=True)
PY
