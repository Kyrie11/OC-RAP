#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

: "${OCRAP_RESULTS_ROOT:=runs/ocrap_three_regime_closed_loop}"
: "${EXTERNAL_RESULTS_ROOT:=runs/all_regime_external_baselines_v49}"
: "${OUT:=runs/external_comparison_v49}"
: "${BUILD_VIDEOS:=true}"
: "${FPS:=10}"
mkdir -p "$OUT/tables" "$OUT/selection"

require_file() { [[ -f "$1" ]] || { echo "Missing required result: $1" >&2; exit 2; }; }
scene_journal() {
  local p="$1"
  if [[ -f "${p}.scenes.jsonl" ]]; then printf '%s\n' "${p}.scenes.jsonl"
  elif [[ -f "${p%.*}.scenes.jsonl" ]]; then printf '%s\n' "${p%.*}.scenes.jsonl"
  else echo "Missing scene journal for $p (rerun with closed_loop.save_partial=true)" >&2; return 2
  fi
}

OCRAP_SAFE="$OCRAP_RESULTS_ROOT/safe/closed_loop_ocrap.json"
OCRAP_NEAR="$OCRAP_RESULTS_ROOT/near/closed_loop_ocrap.json"
OCRAP_CONTACT="$OCRAP_RESULTS_ROOT/contact/closed_loop_ocrap.json"
for p in "$OCRAP_SAFE" "$OCRAP_NEAR" "$OCRAP_CONTACT"; do require_file "$p"; done

SAFE_INPUTS=(
  "ocrap=$OCRAP_SAFE"
  "nominal_replay=$EXTERNAL_RESULTS_ROOT/safe/closed_loop_nominal_replay.json"
  "wayformer_bc=$EXTERNAL_RESULTS_ROOT/safe/closed_loop_wayformer_bc.json"
  "gameformer_lite=$EXTERNAL_RESULTS_ROOT/safe/closed_loop_gameformer_lite.json"
  "betopnet_lite=$EXTERNAL_RESULTS_ROOT/safe/closed_loop_betopnet_lite.json"
)
NEAR_INPUTS=(
  "ocrap=$OCRAP_NEAR"
  "gameformer_lite=$EXTERNAL_RESULTS_ROOT/near/closed_loop_gameformer_lite.json"
  "marc_lite=$EXTERNAL_RESULTS_ROOT/near/closed_loop_marc_lite.json"
  "racp_lite=$EXTERNAL_RESULTS_ROOT/near/closed_loop_racp_lite.json"
  "expected_risk_filter=$EXTERNAL_RESULTS_ROOT/near/closed_loop_expected_risk_filter.json"
  "cvar_risk_filter=$EXTERNAL_RESULTS_ROOT/near/closed_loop_cvar_risk_filter.json"
  "dro_cvar_filter=$EXTERNAL_RESULTS_ROOT/near/closed_loop_dro_cvar_filter.json"
  "predictive_safety_filter=$EXTERNAL_RESULTS_ROOT/near/closed_loop_predictive_safety_filter.json"
)
CONTACT_INPUTS=(
  "ocrap=$OCRAP_CONTACT"
  "postimpact_mpc_lite=$EXTERNAL_RESULTS_ROOT/contact/closed_loop_postimpact_mpc_lite.json"
  "post_crash_braking=$EXTERNAL_RESULTS_ROOT/contact/closed_loop_post_crash_braking.json"
  "post_collision_restoration=$EXTERNAL_RESULTS_ROOT/contact/closed_loop_post_collision_restoration.json"
  "severity_minimization=$EXTERNAL_RESULTS_ROOT/contact/closed_loop_severity_minimization.json"
)

for spec in "${SAFE_INPUTS[@]}" "${NEAR_INPUTS[@]}" "${CONTACT_INPUTS[@]}"; do require_file "${spec#*=}"; done

run_table() {
  local regime="$1"; shift
  local args=() spec
  for spec in "$@"; do args+=(--input "$spec"); done
  PYTHONPATH=src python tools/build_regime_comparison_tables.py \
    --regime "$regime" --output-dir "$OUT/tables" "${args[@]}"
}
run_selection() {
  local regime="$1" output="$2"; shift 2
  local args=() spec
  for spec in "$@"; do args+=(--input "$spec"); done
  PYTHONPATH=src python tools/select_best_external_baseline.py \
    --regime "$regime" --output "$output" "${args[@]}"
}

run_table safe "${SAFE_INPUTS[@]}"
run_table near "${NEAR_INPUTS[@]}"
run_table contact "${CONTACT_INPUTS[@]}"

NEAR_EXTERNAL=("${NEAR_INPUTS[@]:1}")
CONTACT_EXTERNAL=("${CONTACT_INPUTS[@]:1}")
run_selection near "$OUT/selection/near_best_external.json" "${NEAR_EXTERNAL[@]}"
run_selection contact "$OUT/selection/contact_best_external.json" "${CONTACT_EXTERNAL[@]}"

# Keep the non-deployable teacher oracle out of the main external-baseline table.
ORACLE="$EXTERNAL_RESULTS_ROOT/near/closed_loop_oracle_recovery_filter.json"
if [[ -f "$ORACLE" ]]; then
  PYTHONPATH=src python tools/build_regime_comparison_tables.py --regime near \
    --input "ocrap=$OCRAP_NEAR" --input "oracle_recovery_filter=$ORACLE" \
    --output-dir "$OUT/oracle_diagnostic"
fi

if [[ "$BUILD_VIDEOS" == true ]]; then
  eval "$(python - "$OUT/selection/near_best_external.json" "$OUT/selection/contact_best_external.json" <<'PY'
import json, shlex, sys
for prefix, path in zip(("NEAR", "CONTACT"), sys.argv[1:]):
    best=json.load(open(path, encoding="utf-8"))["best"]
    print(f'{prefix}_BEST_METHOD={shlex.quote(best["method"])}')
    print(f'{prefix}_BEST_RESULT={shlex.quote(best["path"])}')
PY
)"
  NEAR_OCRAP_SCENES="$(scene_journal "$OCRAP_NEAR")" \
  NEAR_BASELINE_SCENES="$(scene_journal "$NEAR_BEST_RESULT")" \
  CONTACT_OCRAP_SCENES="$(scene_journal "$OCRAP_CONTACT")" \
  CONTACT_BASELINE_SCENES="$(scene_journal "$CONTACT_BEST_RESULT")" \
  NEAR_BASELINE_NAME="$NEAR_BEST_METHOD" CONTACT_BASELINE_NAME="$CONTACT_BEST_METHOD" \
  FPS="$FPS" OUT="$OUT/top10_recovery_videos" \
    bash scripts/build_top10_recovery_videos.sh
fi

python - "$OCRAP_RESULTS_ROOT" "$EXTERNAL_RESULTS_ROOT" "$OUT" "$BUILD_VIDEOS" <<'PY'
import json, pathlib, sys
ocrap = pathlib.Path(sys.argv[1])
external = pathlib.Path(sys.argv[2])
out = pathlib.Path(sys.argv[3])
videos = sys.argv[4].lower() == "true"
files={
 "safe_table": out/"tables/safe_comparison.json",
 "near_table": out/"tables/near_comparison.json",
 "contact_table": out/"tables/contact_comparison.json",
 "near_best_external": out/"selection/near_best_external.json",
 "contact_best_external": out/"selection/contact_best_external.json",
}
if videos: files["top10_video_index"]=out/"top10_recovery_videos/TOP10_VIDEO_INDEX.json"
doc={"event":"external_comparison_artifacts_v49","ocrap_results_root":str(ocrap),"external_results_root":str(external),"paired_target_set_required":True,"contact_protocol":"physical post-contact metrics only","artifacts":{k:str(v) for k,v in files.items()}}
(out/"COMPARISON_INDEX.json").write_text(json.dumps(doc,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"event":doc["event"],"output":str(out),"videos":videos}))
PY
