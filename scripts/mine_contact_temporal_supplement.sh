#!/usr/bin/env bash
# Mine additional reviewer-safe Contact examples from an already completed
# expanded exact-a0 supplement trace pool. No closed-loop rerun is performed.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/tools:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

: "${BASE_OUT:=runs}"
: "${SOURCE_SUPPLEMENT_NAME:=supplement_01}"
: "${CONTACT_SUPPLEMENT_NAME:=supplement_02}"
: "${MAIN_VIS_ROOT:=$BASE_OUT/regime_visualization_v48_124_final_optimized}"
: "${CONTACT_TEMPORAL_MAX_TIER_RANK:=3}"
: "${CONTACT_TEMPORAL_MIN_POST_STEPS:=25}"
: "${CONTACT_TEMPORAL_MAX_CLIP_S:=4.0}"
: "${CONTACT_TEMPORAL_MIN_COMPARATIVE_METHODS:=1}"
: "${CONTACT_TEMPORAL_WIN_MARGIN_M:=0.10}"
: "${CONTACT_TEMPORAL_NONINFERIOR_MARGIN_M:=0.10}"
: "${CONTACT_TEMPORAL_MIN_WIN_FRACTION:=0.55}"
: "${CONTACT_TEMPORAL_MIN_NONINFERIOR_FRACTION:=0.75}"
: "${CONTACT_TEMPORAL_MIN_MEAN_CLEARANCE_GAIN_M:=0.10}"
: "${CONTACT_TEMPORAL_MIN_TERMINAL_GAIN_M:=0.30}"
: "${CONTACT_TEMPORAL_MIN_SEPARATION_LEAD_S:=0.20}"
: "${CONTACT_TEMPORAL_MIN_OVERLAP_REDUCTION_S:=0.20}"
: "${CONTACT_TEMPORAL_MIN_VALID_FRAMES:=8}"
: "${CONTACT_TEMPORAL_TARGET_RENDERED_DURATION_S:=3.5}"
: "${CONTACT_TEMPORAL_MAX_PLAYBACK_SLOWDOWN:=1.4}"
: "${CONTACT_TEMPORAL_FPS:=10}"
: "${CONTACT_TEMPORAL_VIEW_RADIUS_M:=35}"
: "${CONTACT_TEMPORAL_CAMERA:=fixed}"
: "${CONTACT_TEMPORAL_VIDEO_FORMAT:=mp4}"

[[ "$CONTACT_SUPPLEMENT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid CONTACT_SUPPLEMENT_NAME=$CONTACT_SUPPLEMENT_NAME" >&2; exit 2; }
[[ "$SOURCE_SUPPLEMENT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid SOURCE_SUPPLEMENT_NAME=$SOURCE_SUPPLEMENT_NAME" >&2; exit 2; }
[[ "$CONTACT_SUPPLEMENT_NAME" != "$SOURCE_SUPPLEMENT_NAME" ]] || { echo "Use a different CONTACT_SUPPLEMENT_NAME so the strict source supplement is preserved." >&2; exit 2; }

SOURCE_WORK="$BASE_OUT/contact_qualitative_supplements/$SOURCE_SUPPLEMENT_NAME"
SOURCE_TRACE_ROOT="$SOURCE_WORK/traces"
SOURCE_MANIFEST="$SOURCE_WORK/contact_anchor/contact_anchor_manifest.json"
SOURCE_SELECTION="$SOURCE_WORK/selection/contact_selection.json"
MAIN_SELECTION="$MAIN_VIS_ROOT/selection/contact_selection.json"

WORK="$BASE_OUT/contact_qualitative_supplements/$CONTACT_SUPPLEMENT_NAME"
SELECTION_CANDIDATES="$WORK/selection_candidates"
SELECTION_ROOT="$WORK/selection"
LOG_DIR="$WORK/logs"
mkdir -p "$SELECTION_CANDIDATES" "$SELECTION_ROOT" "$LOG_DIR"

[[ -s "$SOURCE_MANIFEST" ]] || { echo "missing source manifest: $SOURCE_MANIFEST" >&2; exit 30; }
[[ -s "$SOURCE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl" ]] || { echo "missing source OC-RAP trace pool: $SOURCE_TRACE_ROOT" >&2; exit 30; }

BASELINES=(postimpact_mpc_lite post_crash_braking postimpact_motion_tvlqr post_collision_restoration compensatory_postimpact_mpc robust_postimpact_control)
for m in "${BASELINES[@]}"; do
  [[ -s "$SOURCE_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl" ]] || { echo "missing completed source trace: $m" >&2; exit 30; }
done

MIN_CLIP_S="$(python - "$CONTACT_TEMPORAL_MIN_POST_STEPS" <<'PY'
import sys
print(f"{int(sys.argv[1])*0.1:.6g}")
PY
)"
ANCHOR_COUNT="$(python - "$SOURCE_MANIFEST" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
print(int(x.get('num_selected_anchors') or 0))
PY
)"
(( ANCHOR_COUNT > 0 )) || { echo "source supplement anchor pool is empty" >&2; exit 30; }

if command -v flock >/dev/null 2>&1; then
  exec 9>"$WORK/.temporal_mining.lock"
  flock -n 9 || { echo "another temporal supplement build is already using $WORK" >&2; exit 31; }
fi

cat > "$WORK/OUTPUT_LOCATIONS.txt" <<EOF
Source reusable traces:
  $SOURCE_TRACE_ROOT

Temporal-majority selection/audit:
  $SELECTION_ROOT

Final videos:
  $MAIN_VIS_ROOT/videos/contact/$CONTACT_SUPPLEMENT_NAME

Final paper figures:
  $MAIN_VIS_ROOT/paper_figures/contact/$CONTACT_SUPPLEMENT_NAME
EOF

echo "[TEMPORAL-SUPPLEMENT] source=$SOURCE_SUPPLEMENT_NAME output=$CONTACT_SUPPLEMENT_NAME anchors=$ANCHOR_COUNT"
echo "[TEMPORAL-SUPPLEMENT] no closed-loop rerun; hard OC-RAP offroad/recontact/terminal-overlap/lane gates remain unchanged"

# Metric stage is deliberately broad (tier <=3). A tier-3 scene is not accepted
# merely because it exists; it must independently pass the trace-level majority
# comparison below plus all hard realism/recovery constraints.
SEL_ARGS=()
for m in "${BASELINES[@]}"; do
  SEL_ARGS+=(--baseline "$m=$SOURCE_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl")
done
python tools/select_regime_visualization_scenes.py \
  --regime contact \
  --ocrap-scenes "$SOURCE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl" \
  "${SEL_ARGS[@]}" \
  --contact-anchor-manifest "$SOURCE_MANIFEST" \
  --output "$SELECTION_CANDIDATES/contact_selection.json" \
  --target-keys-output "$SELECTION_CANDIDATES/contact_target_keys.json" \
  --num-scenes "$ANCHOR_COUNT" \
  --min-duration-s "$CONTACT_TEMPORAL_MAX_CLIP_S" \
  --fallback-min-duration-s "$MIN_CLIP_S" \
  --max-selected-tier-rank "$CONTACT_TEMPORAL_MAX_TIER_RANK" \
  --allow-fewer-scenes \
  2>&1 | tee "$LOG_DIR/01_metric_discovery.log"

FINAL_ARGS=(
  --candidate-selection "$SELECTION_CANDIDATES/contact_selection.json"
  --trace-root "$SOURCE_TRACE_ROOT"
  --output "$SELECTION_ROOT/contact_selection.json"
  --audit-output "$SELECTION_ROOT/contact_selection_audit.json"
  --comparative-mode failure_or_temporal
  --min-external-recovery-failures 0
  --min-comparative-evidence-methods "$CONTACT_TEMPORAL_MIN_COMPARATIVE_METHODS"
  --min-clip-duration-s "$MIN_CLIP_S"
  --max-clip-duration-s "$CONTACT_TEMPORAL_MAX_CLIP_S"
  --temporal-clearance-win-margin-m "$CONTACT_TEMPORAL_WIN_MARGIN_M"
  --temporal-clearance-noninferior-margin-m "$CONTACT_TEMPORAL_NONINFERIOR_MARGIN_M"
  --temporal-min-win-fraction "$CONTACT_TEMPORAL_MIN_WIN_FRACTION"
  --temporal-min-noninferior-fraction "$CONTACT_TEMPORAL_MIN_NONINFERIOR_FRACTION"
  --temporal-min-mean-clearance-gain-m "$CONTACT_TEMPORAL_MIN_MEAN_CLEARANCE_GAIN_M"
  --temporal-min-terminal-gain-m "$CONTACT_TEMPORAL_MIN_TERMINAL_GAIN_M"
  --temporal-min-separation-lead-s "$CONTACT_TEMPORAL_MIN_SEPARATION_LEAD_S"
  --temporal-min-overlap-reduction-s "$CONTACT_TEMPORAL_MIN_OVERLAP_REDUCTION_S"
  --temporal-min-valid-frames "$CONTACT_TEMPORAL_MIN_VALID_FRAMES"
)
[[ -s "$MAIN_SELECTION" ]] && FINAL_ARGS+=(--exclude-selection "$MAIN_SELECTION")
[[ -s "$SOURCE_SELECTION" ]] && FINAL_ARGS+=(--exclude-selection "$SOURCE_SELECTION")
python tools/finalize_contact_supplement_selection.py "${FINAL_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/02_temporal_realism_filter.log"

# Human-readable discovery table. The audit JSON remains authoritative.
python - "$SELECTION_ROOT/contact_selection_audit.json" "$WORK/TEMPORAL_CANDIDATES.tsv" <<'PY'
import json, pathlib, sys
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); d=json.loads(src.read_text())
rows=[]
for x in d.get('candidates') or []:
    q=x.get('quality') or {}; tev=q.get('external_temporal_advantage') or {}
    winners=[]
    for m,e in tev.items():
        if e.get('temporal_majority_advantage'):
            winners.append((float(e.get('clearance_win_fraction') or 0), m, float(e.get('clearance_mean_gain_m') or 0)))
    winners.sort(reverse=True)
    best=winners[0] if winners else (0.0,'',0.0)
    rows.append({
      'accepted':bool(x.get('accepted')),
      'target_key':x.get('target_key',''),
      'tier':x.get('selection_tier',''),
      'failure_methods':','.join(x.get('failure_evidence_methods') or []),
      'temporal_methods':','.join(x.get('temporal_majority_methods') or []),
      'best_temporal_method':best[1],
      'best_win_fraction':best[0],
      'best_mean_clearance_gain_m':best[2],
      'rejection_reasons':','.join(x.get('rejection_reasons') or []),
    })
rows.sort(key=lambda r:(not r['accepted'],-r['best_win_fraction'],r['target_key']))
cols=['accepted','target_key','tier','failure_methods','temporal_methods','best_temporal_method','best_win_fraction','best_mean_clearance_gain_m','rejection_reasons']
out.write_text('\t'.join(cols)+'\n'+'\n'.join('\t'.join(str(r[c]) for c in cols) for r in rows)+'\n')
print(f"[TEMPORAL-SUPPLEMENT] discovery table: {out}")
PY

ACCEPTED_COUNT="$(python - "$SELECTION_ROOT/contact_selection.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8')); print(len(x.get('selected') or []))
PY
)"
echo "[TEMPORAL-SUPPLEMENT] accepted new scenes=$ACCEPTED_COUNT"

TRACE_ARGS=(--trace "ocrap=$SOURCE_TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl")
for m in "${BASELINES[@]}"; do TRACE_ARGS+=(--trace "$m=$SOURCE_TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl"); done

python tools/render_regime_paper_figures.py \
  "${TRACE_ARGS[@]}" \
  --selection "$SELECTION_ROOT/contact_selection.json" \
  --output-dir "$MAIN_VIS_ROOT/paper_figures" \
  --supplement-name "$CONTACT_SUPPLEMENT_NAME" \
  --view-radius-m "$CONTACT_TEMPORAL_VIEW_RADIUS_M" --force \
  2>&1 | tee "$LOG_DIR/03_figures.log"

python tools/render_regime_visualization_videos.py \
  "${TRACE_ARGS[@]}" \
  --selection "$SELECTION_ROOT/contact_selection.json" \
  --output-dir "$MAIN_VIS_ROOT/videos" \
  --supplement-name "$CONTACT_SUPPLEMENT_NAME" \
  --fps "$CONTACT_TEMPORAL_FPS" --format "$CONTACT_TEMPORAL_VIDEO_FORMAT" \
  --camera-mode "$CONTACT_TEMPORAL_CAMERA" --view-radius-m "$CONTACT_TEMPORAL_VIEW_RADIUS_M" \
  --playback-slowdown 1.0 \
  --target-rendered-duration-s "$CONTACT_TEMPORAL_TARGET_RENDERED_DURATION_S" \
  --max-playback-slowdown "$CONTACT_TEMPORAL_MAX_PLAYBACK_SLOWDOWN" \
  --include-all-method-montage --force \
  2>&1 | tee "$LOG_DIR/04_videos.log"

python - "$WORK" "$MAIN_VIS_ROOT" "$CONTACT_SUPPLEMENT_NAME" "$SOURCE_SUPPLEMENT_NAME" "$ANCHOR_COUNT" "$ACCEPTED_COUNT" <<'PY'
import json,pathlib,sys
work=pathlib.Path(sys.argv[1]); main=pathlib.Path(sys.argv[2]); name=sys.argv[3]; source=sys.argv[4]
d={
 'event':'contact_temporal_majority_supplement_complete_v1',
 'supplement_name':name,
 'source_supplement_name':source,
 'qualitative_only':True,
 'publication_contact_cohort_modified':False,
 'closed_loop_rerun_performed':False,
 'source_anchor_count':int(sys.argv[5]),
 'accepted_new_scene_count':int(sys.argv[6]),
 'selection':str(work/'selection/contact_selection.json'),
 'audit':str(work/'selection/contact_selection_audit.json'),
 'discovery_table':str(work/'TEMPORAL_CANDIDATES.tsv'),
 'videos':str(main/'videos/contact'/name),
 'paper_figures':str(main/'paper_figures/contact'/name),
}
(work/'SUPPLEMENT_SUMMARY.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d,indent=2))
PY

# Update supplement indexes without touching main rank_* or the strict source supplement.
python - "$MAIN_VIS_ROOT" "$CONTACT_SUPPLEMENT_NAME" "$ACCEPTED_COUNT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); name=sys.argv[2]; n=int(sys.argv[3])
for kind, local_index in (("videos","VIDEO_INDEX.json"),("paper_figures","PAPER_FIGURE_INDEX.json")):
    croot=root/kind/"contact"; croot.mkdir(parents=True,exist_ok=True)
    p=croot/"SUPPLEMENTS_INDEX.json"
    try:d=json.loads(p.read_text()) if p.is_file() else {}
    except Exception:d={}
    rows={str(x.get('name')):x for x in (d.get('supplements') or []) if isinstance(x,dict) and x.get('name')}
    rows[name]={"name":name,"accepted_scenes":n,"index":str(croot/name/local_index),"selection_mode":"temporal_majority"}
    out={"event":"contact_visualization_supplements_index_v1","supplements":[rows[k] for k in sorted(rows)]}
    p.write_text(json.dumps(out,indent=2)+'\n')
PY

echo "[TEMPORAL-SUPPLEMENT][DONE] videos: $MAIN_VIS_ROOT/videos/contact/$CONTACT_SUPPLEMENT_NAME"
echo "[TEMPORAL-SUPPLEMENT][DONE] figures: $MAIN_VIS_ROOT/paper_figures/contact/$CONTACT_SUPPLEMENT_NAME"
echo "[TEMPORAL-SUPPLEMENT][DONE] audit: $SELECTION_ROOT/contact_selection_audit.json"
echo "[TEMPORAL-SUPPLEMENT][DONE] table: $WORK/TEMPORAL_CANDIDATES.tsv"
