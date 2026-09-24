#!/usr/bin/env bash
# Contact-only target-display renderer.  Uses existing real traces, never edits
# vehicle states, and writes into a separate nested output tree so the original
# empirical/reviewer-safe media are untouched.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/tools:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

: "${BASE_OUT:=runs}"
: "${MAIN_VIS_ROOT:=$BASE_OUT/regime_visualization_v48_124_final_optimized}"
: "${SOURCE_SUPPLEMENT_ROOT:=$BASE_OUT/contact_qualitative_supplements/supplement_01}"
: "${SOURCE_TRACE_ROOT:=$SOURCE_SUPPLEMENT_ROOT/traces}"
: "${SOURCE_CANDIDATE_SELECTION:=$SOURCE_SUPPLEMENT_ROOT/selection_candidates/contact_selection.json}"
: "${SOURCE_CONTACT_ANCHOR_MANIFEST:=$SOURCE_SUPPLEMENT_ROOT/contact_anchor/contact_anchor_manifest.json}"
: "${PREFERRED_REAL_SELECTION:=$MAIN_VIS_ROOT/selection/contact_selection.json}"
: "${CONTACT_TARGET_NAME:=rank5_target_01}"
: "${CONTACT_TARGET_NUM_SCENES:=5}"
: "${CONTACT_TARGET_MIN_CLIP_S:=2.5}"
: "${CONTACT_TARGET_MAX_CLIP_S:=4.0}"
: "${CONTACT_TARGET_CLIP_STEP_S:=0.1}"
: "${CONTACT_TARGET_MIN_COMPARATIVE_EVIDENCE:=1}"
: "${CONTACT_TARGET_FPS:=10}"
: "${CONTACT_TARGET_VIEW_RADIUS_M:=35}"
: "${CONTACT_TARGET_CAMERA:=fixed}"
: "${CONTACT_TARGET_VIDEO_FORMAT:=mp4}"
: "${CONTACT_TARGET_FORCE_RENDER:=true}"

[[ "$CONTACT_TARGET_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid CONTACT_TARGET_NAME=$CONTACT_TARGET_NAME" >&2; exit 2; }
[[ "$CONTACT_TARGET_NUM_SCENES" =~ ^[0-9]+$ && "$CONTACT_TARGET_NUM_SCENES" -gt 0 ]] || { echo "CONTACT_TARGET_NUM_SCENES must be positive" >&2; exit 2; }
[[ -s "$SOURCE_CANDIDATE_SELECTION" ]] || { echo "missing source candidate selection: $SOURCE_CANDIDATE_SELECTION" >&2; exit 30; }
[[ -d "$SOURCE_TRACE_ROOT" ]] || { echo "missing source trace root: $SOURCE_TRACE_ROOT" >&2; exit 30; }

WORK="$BASE_OUT/contact_target_displays/$CONTACT_TARGET_NAME"
BROAD_SELECTION="$WORK/selection/contact_candidate_broad.json"
RAW_SELECTION="$WORK/selection/contact_selection_raw.json"
SELECTION="$WORK/selection/contact_selection.json"
AUDIT="$WORK/selection/contact_selection_audit.json"
DISPLAY_TRACE_ROOT="$WORK/traces"
LOG_DIR="$WORK/logs"
mkdir -p "$WORK/selection" "$DISPLAY_TRACE_ROOT" "$LOG_DIR"

BASELINES=(
  postimpact_mpc_lite
  post_crash_braking
  postimpact_motion_tvlqr
  post_collision_restoration
  compensatory_postimpact_mpc
  robust_postimpact_control
)

# Rebuild a broad metric candidate pool from the already-computed traces when
# the exact-a0 manifest is available.  This is zero-GPU and prevents the target
# display from being artificially limited by an earlier tier<=2 candidate pool.
CANDIDATE_SELECTION="$SOURCE_CANDIDATE_SELECTION"
if [[ -s "$SOURCE_CONTACT_ANCHOR_MANIFEST" ]]; then
  ANCHOR_COUNT="$(python - "$SOURCE_CONTACT_ANCHOR_MANIFEST" <<'PYCOUNT'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print(int(d.get('num_selected_anchors') or len(d.get('anchors') or [])))
PYCOUNT
)"
  if (( ANCHOR_COUNT > 0 )); then
    BROAD_ARGS=(
      --regime contact
      --ocrap-scenes "$SOURCE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
      --contact-anchor-manifest "$SOURCE_CONTACT_ANCHOR_MANIFEST"
      --output "$BROAD_SELECTION"
      --target-keys-output "$WORK/selection/contact_candidate_broad_keys.json"
      --num-scenes "$ANCHOR_COUNT"
      --min-duration-s "$CONTACT_TARGET_MAX_CLIP_S"
      --fallback-min-duration-s "$CONTACT_TARGET_MIN_CLIP_S"
      --max-selected-tier-rank 3
      --allow-fewer-scenes
    )
    for m in "${BASELINES[@]}"; do
      BROAD_ARGS+=(--baseline "$m=$SOURCE_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl")
    done
    python tools/select_regime_visualization_scenes.py "${BROAD_ARGS[@]}" \
      2>&1 | tee "$LOG_DIR/00_broad_candidate_pool.log"
    CANDIDATE_SELECTION="$BROAD_SELECTION"
  fi
fi

SELECT_ARGS=(
  --candidate-selection "$CANDIDATE_SELECTION"
  --trace-root "$SOURCE_TRACE_ROOT"
  --output "$RAW_SELECTION"
  --audit-output "$AUDIT"
  --num-scenes "$CONTACT_TARGET_NUM_SCENES"
  --min-clip-duration-s "$CONTACT_TARGET_MIN_CLIP_S"
  --max-clip-duration-s "$CONTACT_TARGET_MAX_CLIP_S"
  --clip-step-s "$CONTACT_TARGET_CLIP_STEP_S"
  --min-comparative-evidence-methods "$CONTACT_TARGET_MIN_COMPARATIVE_EVIDENCE"
  --require-exact-count
)
[[ -s "$PREFERRED_REAL_SELECTION" ]] && SELECT_ARGS+=(--preferred-selection "$PREFERRED_REAL_SELECTION")

python tools/select_contact_target_display_scenes.py "${SELECT_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/01_select.log"

python tools/materialize_contact_target_display.py \
  --selection "$RAW_SELECTION" \
  --source-trace-root "$SOURCE_TRACE_ROOT" \
  --output-trace-root "$DISPLAY_TRACE_ROOT" \
  --output-selection "$SELECTION" \
  2>&1 | tee "$LOG_DIR/02_materialize.log"

TRACE_ARGS=(--trace "ocrap=$DISPLAY_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl")
for m in "${BASELINES[@]}"; do
  TRACE_ARGS+=(--trace "$m=$DISPLAY_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl")
done

FORCE_ARG=()
[[ "$CONTACT_TARGET_FORCE_RENDER" == true ]] && FORCE_ARG+=(--force)

# 2x4 pair and 7x3 all-method figures.  This is the unchanged paper renderer.
python tools/render_regime_paper_figures.py \
  "${TRACE_ARGS[@]}" \
  --selection "$SELECTION" \
  --output-dir "$MAIN_VIS_ROOT/paper_figures" \
  --supplement-name "$CONTACT_TARGET_NAME" \
  --view-radius-m "$CONTACT_TARGET_VIEW_RADIUS_M" \
  "${FORCE_ARG[@]}" \
  2>&1 | tee "$LOG_DIR/03_figures.log"

# Primary pair + all-method montage videos.  No style/layout modification and
# no automatic slowdown, keeping presentation identical to the real renderer.
python tools/render_regime_visualization_videos.py \
  "${TRACE_ARGS[@]}" \
  --selection "$SELECTION" \
  --output-dir "$MAIN_VIS_ROOT/videos" \
  --supplement-name "$CONTACT_TARGET_NAME" \
  --fps "$CONTACT_TARGET_FPS" \
  --format "$CONTACT_TARGET_VIDEO_FORMAT" \
  --camera-mode "$CONTACT_TARGET_CAMERA" \
  --view-radius-m "$CONTACT_TARGET_VIEW_RADIUS_M" \
  --playback-slowdown 1.0 \
  --include-all-method-montage \
  "${FORCE_ARG[@]}" \
  2>&1 | tee "$LOG_DIR/04_videos.log"

python - "$WORK" "$MAIN_VIS_ROOT" "$CONTACT_TARGET_NAME" "$SELECTION" <<'PY'
import json,pathlib,sys
work=pathlib.Path(sys.argv[1]); main=pathlib.Path(sys.argv[2]); name=sys.argv[3]; sel=pathlib.Path(sys.argv[4])
d=json.loads(sel.read_text())
out={
  "event":"contact_target_display_complete_v1",
  "num_scenes":len(d.get("selected") or []),
  "selection":str(sel),
  "audit":str(work/"selection/contact_selection_audit.json"),
  "provenance":str(work/"selection/PROVENANCE.json"),
  "videos":str(main/"videos/contact"/name),
  "paper_figures":str(main/"paper_figures/contact"/name),
  "trajectory_states_modified":False,
  "metrics_recomputed_on_visible_clip":True,
  "original_empirical_media_overwritten":False,
}
(work/"TARGET_DISPLAY_SUMMARY.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
PY

echo "[CONTACT-TARGET][DONE] videos: $MAIN_VIS_ROOT/videos/contact/$CONTACT_TARGET_NAME"
echo "[CONTACT-TARGET][DONE] figures: $MAIN_VIS_ROOT/paper_figures/contact/$CONTACT_TARGET_NAME"
echo "[CONTACT-TARGET][DONE] selection: $SELECTION"
