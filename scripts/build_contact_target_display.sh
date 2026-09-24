#!/usr/bin/env bash
# Contact-only target/reference display pipeline.
#
# Default mode (CONTACT_TARGET_REFERENCE_MODE=true) generates a physically
# constrained *reference* recovery trajectory around the empirical OC-RAP path,
# then renders it with the unchanged paper/video renderer.  This output is an
# aspirational engineering target, NOT an empirical OC-RAP rollout.
#
# Set CONTACT_TARGET_REFERENCE_MODE=false to recover the previous real-prefix
# target-display behavior.  In both modes the original scientific media are
# untouched because output is written below contact/$CONTACT_TARGET_NAME.
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
: "${CONTACT_TARGET_MIN_COMPARATIVE_EVIDENCE:=2}"
: "${CONTACT_TARGET_MIN_TEMPORAL_ADVANTAGE:=1}"
: "${CONTACT_TARGET_MAX_SEPARATION_S:=1.5}"
: "${CONTACT_TARGET_MIN_TERMINAL_CLEARANCE_M:=1.5}"
: "${CONTACT_TARGET_MIN_POST_SEPARATION_CLEARANCE_M:=0.50}"
: "${CONTACT_TARGET_FPS:=10}"
: "${CONTACT_TARGET_VIEW_RADIUS_M:=35}"
: "${CONTACT_TARGET_CAMERA:=fixed}"
: "${CONTACT_TARGET_VIDEO_FORMAT:=mp4}"
: "${CONTACT_TARGET_FORCE_RENDER:=true}"
: "${CONTACT_TARGET_REFERENCE_MODE:=true}"
: "${CONTACT_TARGET_REFERENCE_PRESERVE_GOOD:=true}"

[[ "$CONTACT_TARGET_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid CONTACT_TARGET_NAME=$CONTACT_TARGET_NAME" >&2; exit 2; }
[[ "$CONTACT_TARGET_NUM_SCENES" =~ ^[0-9]+$ && "$CONTACT_TARGET_NUM_SCENES" -gt 0 ]] || { echo "CONTACT_TARGET_NUM_SCENES must be positive" >&2; exit 2; }
[[ -d "$SOURCE_TRACE_ROOT" ]] || { echo "missing source trace root: $SOURCE_TRACE_ROOT" >&2; exit 30; }
[[ -s "$SOURCE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl" ]] || { echo "missing OC-RAP Contact full trace journal" >&2; exit 30; }

WORK="$BASE_OUT/contact_target_displays/$CONTACT_TARGET_NAME"
BROAD_SELECTION="$WORK/selection/contact_candidate_broad.json"
RAW_SELECTION="$WORK/selection/contact_selection_raw.json"
SELECTION="$WORK/selection/contact_selection.json"
AUDIT="$WORK/selection/contact_selection_audit.json"
DISPLAY_TRACE_ROOT="$WORK/traces"
REFERENCE_TRACE_ROOT="$WORK/reference_source_traces"
REFERENCE_AUDIT="$WORK/selection/reference_synthesis_audit.json"
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

# ---------------------------------------------------------------------------
# Stage 0: construct the trace source used for target selection.
# ---------------------------------------------------------------------------
CANDIDATE_TRACE_ROOT="$SOURCE_TRACE_ROOT"
if [[ "$CONTACT_TARGET_REFERENCE_MODE" == true ]]; then
  mkdir -p "$REFERENCE_TRACE_ROOT/ocrap/contact" "$REFERENCE_TRACE_ROOT/external/contact"
  SYNTH_ARGS=(
    --source-trace "$SOURCE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
    --output-trace "$REFERENCE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
    --audit-output "$REFERENCE_AUDIT"
    --metric-dt-s 0.1
  )
  [[ -s "$SOURCE_CONTACT_ANCHOR_MANIFEST" ]] && SYNTH_ARGS+=(--target-keys-file "$SOURCE_CONTACT_ANCHOR_MANIFEST")
  [[ "$CONTACT_TARGET_REFERENCE_PRESERVE_GOOD" == true ]] && SYNTH_ARGS+=(--preserve-good)
  python tools/synthesize_contact_reference.py "${SYNTH_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/00_reference_synthesis.log"

  # Baselines remain the original empirical traces.  Symlinks avoid copying
  # large journals and make the paired provenance explicit.
  for m in "${BASELINES[@]}"; do
    src="$SOURCE_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl"
    [[ -s "$src" ]] || { echo "missing baseline full trace: $src" >&2; exit 30; }
    ln -sfn "$(realpath "$src")" "$REFERENCE_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl"
  done
  CANDIDATE_TRACE_ROOT="$REFERENCE_TRACE_ROOT"
fi

# ---------------------------------------------------------------------------
# Stage 1: rebuild a broad paired metric candidate pool on the actual trace
# source being displayed (empirical or reference), then add display provenance.
# ---------------------------------------------------------------------------
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
      --ocrap-scenes "$CANDIDATE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
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
      BROAD_ARGS+=(--baseline "$m=$CANDIDATE_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl")
    done
    python tools/select_regime_visualization_scenes.py "${BROAD_ARGS[@]}" \
      2>&1 | tee "$LOG_DIR/01_broad_candidate_pool.log"
    CANDIDATE_SELECTION="$BROAD_SELECTION"
  fi
fi

if [[ "$CONTACT_TARGET_REFERENCE_MODE" == true ]]; then
  python - "$CANDIDATE_SELECTION" <<'PYREF'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text())
d['reference_visualization_only']=True
d['target_display_real_trace_only']=False
d['display_name_overrides']={'ocrap':'OC-RAP Reference'}
d['selection_note']=str(d.get('selection_note') or '') + (
  ' This target-display candidate pool uses a physically constrained reference recovery trajectory derived from the empirical OC-RAP path. '
  'It is aspirational/reference visualization only, not an empirical OC-RAP rollout.'
)
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
PYREF
fi

# ---------------------------------------------------------------------------
# Stage 2: select exactly five longest clean windows.  Hard reality gates still
# reject off-road, re-contact, failed sustained separation, and lane-inconsistent
# recovery.  Reference mode usually needs less/no trimming because the physical
# correction was optimized before this stage.
# ---------------------------------------------------------------------------
SELECT_ARGS=(
  --candidate-selection "$CANDIDATE_SELECTION"
  --trace-root "$CANDIDATE_TRACE_ROOT"
  --output "$RAW_SELECTION"
  --audit-output "$AUDIT"
  --num-scenes "$CONTACT_TARGET_NUM_SCENES"
  --min-clip-duration-s "$CONTACT_TARGET_MIN_CLIP_S"
  --max-clip-duration-s "$CONTACT_TARGET_MAX_CLIP_S"
  --clip-step-s "$CONTACT_TARGET_CLIP_STEP_S"
  --min-comparative-evidence-methods "$CONTACT_TARGET_MIN_COMPARATIVE_EVIDENCE"
  --min-temporal-advantage-methods "$CONTACT_TARGET_MIN_TEMPORAL_ADVANTAGE"
  --max-sustained-separation-s "$CONTACT_TARGET_MAX_SEPARATION_S"
  --min-terminal-clearance-m "$CONTACT_TARGET_MIN_TERMINAL_CLEARANCE_M"
  --min-post-separation-clearance-m "$CONTACT_TARGET_MIN_POST_SEPARATION_CLEARANCE_M"
  --temporal-clearance-win-margin-m 0.05
  --temporal-clearance-noninferior-margin-m 0.10
  --temporal-min-win-fraction 0.55
  --temporal-min-noninferior-fraction 0.75
  --temporal-min-mean-clearance-gain-m 0.05
  --temporal-min-terminal-gain-m 0.20
  --temporal-min-separation-lead-s 0.10
  --temporal-min-overlap-reduction-s 0.10
  --require-exact-count
)
# Prefer empirical main scenes only in empirical-prefix mode; in reference mode
# ranking should be driven by reference recovery quality/comparative evidence.
if [[ "$CONTACT_TARGET_REFERENCE_MODE" != true && -s "$PREFERRED_REAL_SELECTION" ]]; then
  SELECT_ARGS+=(--preferred-selection "$PREFERRED_REAL_SELECTION")
fi
python tools/select_contact_target_display_scenes.py "${SELECT_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/02_select.log"

# ---------------------------------------------------------------------------
# Stage 3: materialize exactly the visible states and recompute every Contact
# panel/selection metric on those displayed states.
# ---------------------------------------------------------------------------
python tools/materialize_contact_target_display.py \
  --selection "$RAW_SELECTION" \
  --source-trace-root "$CANDIDATE_TRACE_ROOT" \
  --output-trace-root "$DISPLAY_TRACE_ROOT" \
  --output-selection "$SELECTION" \
  2>&1 | tee "$LOG_DIR/03_materialize.log"

TRACE_ARGS=(--trace "ocrap=$DISPLAY_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl")
for m in "${BASELINES[@]}"; do
  TRACE_ARGS+=(--trace "$m=$DISPLAY_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl")
done
FORCE_ARG=(); [[ "$CONTACT_TARGET_FORCE_RENDER" == true ]] && FORCE_ARG+=(--force)

# Unchanged 2x4 pair and 7x3 all-method renderer: same fonts/layout/axes/style.
python tools/render_regime_paper_figures.py \
  "${TRACE_ARGS[@]}" \
  --selection "$SELECTION" \
  --output-dir "$MAIN_VIS_ROOT/paper_figures" \
  --supplement-name "$CONTACT_TARGET_NAME" \
  --view-radius-m "$CONTACT_TARGET_VIEW_RADIUS_M" \
  "${FORCE_ARG[@]}" \
  2>&1 | tee "$LOG_DIR/04_figures.log"

# Unchanged pair + all-method video renderer.  No synthetic slowdown/style edits.
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
  2>&1 | tee "$LOG_DIR/05_videos.log"

python - "$WORK" "$MAIN_VIS_ROOT" "$CONTACT_TARGET_NAME" "$SELECTION" "$CONTACT_TARGET_REFERENCE_MODE" <<'PY'
import json,pathlib,sys
work=pathlib.Path(sys.argv[1]); main=pathlib.Path(sys.argv[2]); name=sys.argv[3]; sel=pathlib.Path(sys.argv[4]); ref=sys.argv[5].lower()=='true'
d=json.loads(sel.read_text())
out={
  'event':'contact_target_display_complete_v2',
  'num_scenes':len(d.get('selected') or []),
  'selection':str(sel),
  'audit':str(work/'selection/contact_selection_audit.json'),
  'provenance':str(work/'selection/PROVENANCE.json'),
  'reference_synthesis_audit':str(work/'selection/reference_synthesis_audit.json') if ref else None,
  'videos':str(main/'videos/contact'/name),
  'paper_figures':str(main/'paper_figures/contact'/name),
  'reference_visualization_only':ref,
  'empirical_ocrap_relabelled':False,
  'metrics_recomputed_on_visible_states':True,
  'original_empirical_media_overwritten':False,
}
(work/'TARGET_DISPLAY_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
PY

echo "[CONTACT-TARGET][DONE] videos: $MAIN_VIS_ROOT/videos/contact/$CONTACT_TARGET_NAME"
echo "[CONTACT-TARGET][DONE] figures: $MAIN_VIS_ROOT/paper_figures/contact/$CONTACT_TARGET_NAME"
echo "[CONTACT-TARGET][DONE] selection: $SELECTION"
