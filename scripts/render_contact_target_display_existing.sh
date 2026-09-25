#!/usr/bin/env bash
# Re-render an already-materialized Contact target display without re-running
# reference synthesis / broad selection / metric materialization.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/tools:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

: "${BASE_OUT:=runs}"
: "${MAIN_VIS_ROOT:=$BASE_OUT/regime_visualization_v48_124_final_optimized}"
: "${CONTACT_TARGET_NAME:=rank5_target_01}"
: "${CONTACT_TARGET_FPS:=10}"
: "${CONTACT_TARGET_VIEW_RADIUS_M:=35}"
: "${CONTACT_TARGET_CAMERA:=fixed}"
: "${CONTACT_TARGET_VIDEO_FORMAT:=mp4}"
: "${CONTACT_TARGET_FORCE_RENDER:=true}"
: "${CONTACT_TARGET_MIRROR_MEDIA_IN_WORK:=true}"
: "${CONTACT_TARGET_DISPLAY_LABEL:=OC-RAP}"

WORK="$BASE_OUT/contact_target_displays/$CONTACT_TARGET_NAME"
SELECTION="$WORK/selection/contact_selection.json"
TRACE_ROOT="$WORK/traces"
LOG_DIR="$WORK/logs"
mkdir -p "$LOG_DIR"

[[ -s "$SELECTION" ]] || { echo "missing materialized selection: $SELECTION" >&2; exit 30; }
python - "$SELECTION" "$CONTACT_TARGET_DISPLAY_LABEL" <<'PYSEL'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
d=json.loads(p.read_text(encoding='utf-8'))
overrides=dict(d.get('display_name_overrides') or {})
overrides['ocrap']=str(sys.argv[2])
d['display_name_overrides']=overrides
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PYSEL
[[ -s "$TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl" ]] || { echo "missing materialized OC-RAP trace" >&2; exit 30; }
BASELINES=(postimpact_mpc_lite post_crash_braking postimpact_motion_tvlqr post_collision_restoration compensatory_postimpact_mpc robust_postimpact_control)
TRACE_ARGS=(--trace "ocrap=$TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl")
for m in "${BASELINES[@]}"; do
  p="$TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl"
  [[ -s "$p" ]] || { echo "missing materialized baseline trace: $p" >&2; exit 30; }
  TRACE_ARGS+=(--trace "$m=$p")
done
FORCE_ARG=(); [[ "$CONTACT_TARGET_FORCE_RENDER" == true ]] && FORCE_ARG+=(--force)

# Archive previous re-render logs only.
STAMP="$(date +%Y%m%d_%H%M%S)"
for f in "$LOG_DIR/90_rerender_figures.log" "$LOG_DIR/91_rerender_videos.log"; do
  if [[ -f "$f" ]]; then mkdir -p "$LOG_DIR/history/$STAMP"; mv "$f" "$LOG_DIR/history/$STAMP/"; fi
done

python tools/render_regime_paper_figures.py \
  "${TRACE_ARGS[@]}" \
  --selection "$SELECTION" \
  --output-dir "$MAIN_VIS_ROOT/paper_figures" \
  --supplement-name "$CONTACT_TARGET_NAME" \
  --view-radius-m "$CONTACT_TARGET_VIEW_RADIUS_M" \
  "${FORCE_ARG[@]}" \
  2>&1 | tee "$LOG_DIR/90_rerender_figures.log"

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
  2>&1 | tee "$LOG_DIR/91_rerender_videos.log"

python - "$WORK" "$MAIN_VIS_ROOT" "$CONTACT_TARGET_NAME" "$SELECTION" "$CONTACT_TARGET_MIRROR_MEDIA_IN_WORK" <<'PY'
import json,pathlib,shutil,sys
work=pathlib.Path(sys.argv[1]); main=pathlib.Path(sys.argv[2]); name=sys.argv[3]; sel=pathlib.Path(sys.argv[4]); mirror=sys.argv[5].lower()=='true'
d=json.loads(sel.read_text(encoding='utf-8')); n=len(d.get('selected') or []); expected=n*2
vr=main/'videos'/'contact'/name; fr=main/'paper_figures'/'contact'/name
if not vr.is_dir(): raise SystemExit(f'missing video output dir: {vr}')
if not fr.is_dir(): raise SystemExit(f'missing figure output dir: {fr}')
mp4=sorted(vr.rglob('*.mp4')); png=sorted(fr.rglob('*.png')); pdf=sorted(fr.rglob('*.pdf'))
if len(mp4)<expected or len(png)<expected or len(pdf)<expected:
    raise SystemExit(f'incomplete render expected>={expected} each: mp4={len(mp4)} png={len(png)} pdf={len(pdf)}')
for p in [*mp4,*png,*pdf]:
    if p.stat().st_size<=1024: raise SystemExit(f'tiny/corrupt output: {p} size={p.stat().st_size}')
if mirror:
    mv=work/'media'/'videos'; mf=work/'media'/'paper_figures'
    if mv.exists(): shutil.rmtree(mv)
    if mf.exists(): shutil.rmtree(mf)
    shutil.copytree(vr,mv); shutil.copytree(fr,mf)
idx={'event':'contact_target_display_rerender_complete_v1','num_scenes':n,'num_videos':len(mp4),'num_png_figures':len(png),'num_pdf_figures':len(pdf),'video_root':str(vr),'figure_root':str(fr),'mirrored':mirror}
(work/'TARGET_MEDIA_INDEX.json').write_text(json.dumps(idx,indent=2)+'\n',encoding='utf-8')
print(json.dumps(idx,indent=2))
PY

echo "[RERENDER][DONE] videos=$MAIN_VIS_ROOT/videos/contact/$CONTACT_TARGET_NAME"
echo "[RERENDER][DONE] figures=$MAIN_VIS_ROOT/paper_figures/contact/$CONTACT_TARGET_NAME"
