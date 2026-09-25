#!/usr/bin/env bash
# Contact-only target/reference display pipeline.
#
# Default mode (CONTACT_TARGET_REFERENCE_MODE=true) generates a physically
# constrained target recovery trajectory around the empirical OC-RAP path,
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

: "${CONTACT_TARGET_REFERENCE_MAX_SYNTH_SCENES:=14}"
: "${CONTACT_TARGET_REFERENCE_JOBS:=4}"
: "${CONTACT_TARGET_REFERENCE_TOP_K_EXACT_ACTIONS:=12}"
: "${CONTACT_TARGET_REFERENCE_REUSE:=true}"
: "${CONTACT_TARGET_PRESERVE_PREFERRED_COUNT:=1}"
: "${CONTACT_TARGET_DISPLAY_LABEL:=OC-RAP}"
: "${CONTACT_TARGET_MIN_DOMINANCE_METHODS:=4}"
: "${CONTACT_TARGET_MIN_TERMINAL_ADVANTAGE_METHODS:=2}"
: "${CONTACT_TARGET_MIN_OVERLAP_ADVANTAGE_METHODS:=2}"
: "${CONTACT_TARGET_MIN_SEPARATION_ADVANTAGE_METHODS:=2}"
: "${CONTACT_TARGET_TERMINAL_ADVANTAGE_MARGIN_M:=0.25}"
: "${CONTACT_TARGET_OVERLAP_ADVANTAGE_MARGIN_S:=0.10}"
: "${CONTACT_TARGET_SEPARATION_ADVANTAGE_MARGIN_S:=0.10}"
: "${CONTACT_TARGET_MAX_EMPIRICAL_SOURCE_OFFROAD_FRACTION:=0.15}"
: "${CONTACT_TARGET_LANE_TERMINAL_MAX_M:=2.5}"
: "${CONTACT_TARGET_LANE_P90_MAX_M:=3.5}"
: "${CONTACT_TARGET_LANE_OFFCENTER_FRACTION_MAX:=0.20}"
: "${CONTACT_TARGET_LANE_HEADING_TERMINAL_MAX_DEG:=35}"
: "${CONTACT_TARGET_LANE_HEADING_P90_MAX_DEG:=40}"

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
REFERENCE_SUBSET_KEYS="$WORK/selection/reference_synthesis_keys.json"
REFERENCE_SUBSET_MANIFEST="$WORK/selection/reference_synthesis_anchor_manifest.json"
REFERENCE_SUBSET_AUDIT="$WORK/selection/reference_synthesis_subset_audit.json"
PREFERRED_PRESERVE_KEYS="$WORK/selection/preferred_preserve_keys.json"
REFERENCE_CACHE_KEY="$WORK/selection/reference_synthesis_cache_key.txt"
LOG_DIR="$WORK/logs"
# If this target-display name already has a selection, preserve its current best
# rank first; otherwise fall back to the original reviewer-safe Contact ranks.
PREFERRED_SELECTION_SOURCE="$PREFERRED_REAL_SELECTION"
if [[ -s "$SELECTION" ]]; then PREFERRED_SELECTION_SOURCE="$SELECTION"; fi
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
ACTIVE_CONTACT_ANCHOR_MANIFEST="$SOURCE_CONTACT_ANCHOR_MANIFEST"
if [[ "$CONTACT_TARGET_REFERENCE_MODE" == true ]]; then
  mkdir -p "$REFERENCE_TRACE_ROOT/ocrap/contact" "$REFERENCE_TRACE_ROOT/external/contact"

  # Preserve the strongest existing empirical scenes (rank01 by default) and
  # synthesize only a small set of critical/fixable scenes.  This is a compute
  # prefilter only; all downstream hard reality gates remain unchanged.
  python - "$PREFERRED_SELECTION_SOURCE" "$CONTACT_TARGET_PRESERVE_PREFERRED_COUNT" "$PREFERRED_PRESERVE_KEYS" <<'PYPREF'
import json,pathlib,sys
src=pathlib.Path(sys.argv[1]); n=max(0,int(sys.argv[2])); out=pathlib.Path(sys.argv[3])
keys=[]
if src.is_file():
    d=json.loads(src.read_text())
    keys=[str(x.get('target_key')) for x in (d.get('selected') or []) if isinstance(x,dict) and x.get('target_key')][:n]
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(keys,indent=2)+'\n')
print(json.dumps({'event':'preferred_contact_preserve_keys','keys':keys}))
PYPREF

  PREFILTER_ARGS=(
    --ocrap-trace "$SOURCE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
    --anchor-manifest "$SOURCE_CONTACT_ANCHOR_MANIFEST"
    --preferred-selection "$PREFERRED_SELECTION_SOURCE"
    --preserve-preferred-count "$CONTACT_TARGET_PRESERVE_PREFERRED_COUNT"
    --max-scenes "$CONTACT_TARGET_REFERENCE_MAX_SYNTH_SCENES"
    --max-source-offroad-fraction "$CONTACT_TARGET_MAX_EMPIRICAL_SOURCE_OFFROAD_FRACTION"
    --output-target-keys "$REFERENCE_SUBSET_KEYS"
    --output-anchor-manifest "$REFERENCE_SUBSET_MANIFEST"
    --output-audit "$REFERENCE_SUBSET_AUDIT"
  )
  for m in "${BASELINES[@]}"; do
    src="$SOURCE_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl"
    [[ -s "$src" ]] || { echo "missing baseline full trace: $src" >&2; exit 30; }
    PREFILTER_ARGS+=(--baseline "$m=$src")
  done
  python tools/select_contact_reference_synthesis_subset.py "${PREFILTER_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/00a_reference_subset.log"

  SYNTH_ARGS=(
    --source-trace "$SOURCE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
    --output-trace "$REFERENCE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
    --audit-output "$REFERENCE_AUDIT"
    --metric-dt-s 0.1
    --target-keys-file "$REFERENCE_SUBSET_KEYS"
    --force-preserve-keys-file "$PREFERRED_PRESERVE_KEYS"
    --jobs "$CONTACT_TARGET_REFERENCE_JOBS"
    --top-k-exact-actions "$CONTACT_TARGET_REFERENCE_TOP_K_EXACT_ACTIONS"
  )
  [[ "$CONTACT_TARGET_REFERENCE_PRESERVE_GOOD" == true ]] && SYNTH_ARGS+=(--preserve-good)
  for m in "${BASELINES[@]}"; do
    src="$SOURCE_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl"
    SYNTH_ARGS+=(--baseline "$m=$src")
  done
  # Safe resumable cache: key includes the selected target subset, planner and
  # prefilter source code, source-file identity, and planner search settings.
  NEW_CACHE_KEY="$(python - "$REFERENCE_SUBSET_KEYS" "$SOURCE_TRACE_ROOT" "$CONTACT_TARGET_REFERENCE_TOP_K_EXACT_ACTIONS" "$CONTACT_TARGET_REFERENCE_PRESERVE_GOOD" <<'PYCACHE'
import hashlib,pathlib,sys
keys=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2]); topk=sys.argv[3]; preserve=sys.argv[4]
h=hashlib.sha256(); h.update(keys.read_bytes()); h.update(topk.encode()); h.update(preserve.encode())
repo=pathlib.Path.cwd()
for code in (repo/'tools/synthesize_contact_reference.py',repo/'tools/select_contact_reference_synthesis_subset.py'):
    h.update(code.read_bytes())
paths=[root/'ocrap/contact/closed_loop_ocrap.json.scenes.jsonl']+sorted((root/'external/contact').glob('closed_loop_*.json.scenes.jsonl'))
for p in paths:
    st=p.stat(); h.update(str(p.resolve()).encode()); h.update(str(st.st_size).encode()); h.update(str(st.st_mtime_ns).encode())
print(h.hexdigest())
PYCACHE
)"
  OLD_CACHE_KEY="$(cat "$REFERENCE_CACHE_KEY" 2>/dev/null || true)"
  if [[ "$CONTACT_TARGET_REFERENCE_REUSE" == true && -s "$REFERENCE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl" && -s "$REFERENCE_AUDIT" && "$NEW_CACHE_KEY" == "$OLD_CACHE_KEY" ]]; then
    echo "[REF][REUSE] cached reference synthesis is valid: $REFERENCE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl" | tee "$LOG_DIR/00_reference_synthesis.log"
  else
    python tools/synthesize_contact_reference.py "${SYNTH_ARGS[@]}" \
      2>&1 | tee "$LOG_DIR/00_reference_synthesis.log"
    printf '%s\n' "$NEW_CACHE_KEY" > "$REFERENCE_CACHE_KEY"
  fi

  # Baselines remain the original empirical traces.  Symlinks avoid copying
  # large journals and make the paired provenance explicit.
  for m in "${BASELINES[@]}"; do
    src="$SOURCE_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl"
    [[ -s "$src" ]] || { echo "missing baseline full trace: $src" >&2; exit 30; }
    ln -sfn "$(realpath "$src")" "$REFERENCE_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl"
  done
  CANDIDATE_TRACE_ROOT="$REFERENCE_TRACE_ROOT"
  ACTIVE_CONTACT_ANCHOR_MANIFEST="$REFERENCE_SUBSET_MANIFEST"
fi

# ---------------------------------------------------------------------------
# Stage 1: rebuild a broad paired metric candidate pool on the actual trace
# source being displayed (empirical or reference), then add display provenance.
# ---------------------------------------------------------------------------
CANDIDATE_SELECTION="$SOURCE_CANDIDATE_SELECTION"
if [[ -s "$ACTIVE_CONTACT_ANCHOR_MANIFEST" ]]; then
  ANCHOR_COUNT="$(python - "$ACTIVE_CONTACT_ANCHOR_MANIFEST" <<'PYCOUNT'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print(int(d.get('num_selected_anchors') or len(d.get('anchors') or [])))
PYCOUNT
)"
  if (( ANCHOR_COUNT > 0 )); then
    BROAD_ARGS=(
      --regime contact
      --ocrap-scenes "$CANDIDATE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl"
      --contact-anchor-manifest "$ACTIVE_CONTACT_ANCHOR_MANIFEST"
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
  python - "$CANDIDATE_SELECTION" "$CONTACT_TARGET_DISPLAY_LABEL" <<'PYREF'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text())
d['reference_visualization_only']=True
d['target_display_real_trace_only']=False
d['display_name_overrides']={'ocrap':str(sys.argv[2])}
d['selection_note']=str(d.get('selection_note') or '') + (
  ' This target-display candidate pool uses a physically constrained target recovery trajectory derived from the empirical OC-RAP path. '
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
  --min-dominance-methods "$CONTACT_TARGET_MIN_DOMINANCE_METHODS"
  --min-terminal-advantage-methods "$CONTACT_TARGET_MIN_TERMINAL_ADVANTAGE_METHODS"
  --min-overlap-advantage-methods "$CONTACT_TARGET_MIN_OVERLAP_ADVANTAGE_METHODS"
  --min-separation-advantage-methods "$CONTACT_TARGET_MIN_SEPARATION_ADVANTAGE_METHODS"
  --terminal-advantage-margin-m "$CONTACT_TARGET_TERMINAL_ADVANTAGE_MARGIN_M"
  --overlap-advantage-margin-s "$CONTACT_TARGET_OVERLAP_ADVANTAGE_MARGIN_S"
  --separation-advantage-margin-s "$CONTACT_TARGET_SEPARATION_ADVANTAGE_MARGIN_S"
  --max-empirical-source-offroad-fraction "$CONTACT_TARGET_MAX_EMPIRICAL_SOURCE_OFFROAD_FRACTION"
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
  --lane-terminal-max-m "$CONTACT_TARGET_LANE_TERMINAL_MAX_M"
  --lane-p90-max-m "$CONTACT_TARGET_LANE_P90_MAX_M"
  --lane-offcenter-fraction-max "$CONTACT_TARGET_LANE_OFFCENTER_FRACTION_MAX"
  --lane-heading-terminal-max-deg "$CONTACT_TARGET_LANE_HEADING_TERMINAL_MAX_DEG"
  --lane-heading-p90-max-deg "$CONTACT_TARGET_LANE_HEADING_P90_MAX_DEG"
  --lane-recovery-terminal-max-m "$CONTACT_TARGET_LANE_TERMINAL_MAX_M"
  --lane-recovery-p90-max-m "$CONTACT_TARGET_LANE_P90_MAX_M"
  --lane-recovery-offcenter-fraction-max "$CONTACT_TARGET_LANE_OFFCENTER_FRACTION_MAX"
  --require-exact-count
)
# Keep the already-confirmed strongest empirical scene(s) at the front when
# they still pass every current hard gate.  Weak preferred scenes are not forced.
if [[ -s "$PREFERRED_PRESERVE_KEYS" ]]; then
  SELECT_ARGS+=(--preferred-selection "$PREFERRED_PRESERVE_KEYS")
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
