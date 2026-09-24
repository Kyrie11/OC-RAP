#!/usr/bin/env bash
# Build an expanded exact-a0 Contact qualitative supplement without modifying
# the frozen publication Contact cohort or the main reviewer-safe visualization.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/tools:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

: "${BASE_OUT:=runs}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${OCRAP_MODEL_RUN:=$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
: "${MODEL_VARIANT:=balanced}"
: "${MAIN_CHARACTERIZATION_ROOT:=$BASE_OUT/ocrap_v48_124_final_characterization}"
: "${MAIN_VIS_ROOT:=$BASE_OUT/regime_visualization_v48_124_final_optimized}"
: "${CONTACT_SUPPLEMENT_NAME:=supplement_01}"
: "${CONTACT_SUPPLEMENT_MIN_POST_STEPS:=25}"
: "${CONTACT_SUPPLEMENT_MAX_CLIP_S:=4.0}"
: "${CONTACT_SUPPLEMENT_MAX_TIER_RANK:=2}"
: "${CONTACT_SUPPLEMENT_MIN_EXTERNAL_FAILURES:=1}"
: "${CONTACT_SUPPLEMENT_TARGET_RENDERED_DURATION_S:=3.5}"
: "${CONTACT_SUPPLEMENT_MAX_PLAYBACK_SLOWDOWN:=1.4}"
: "${CONTACT_SUPPLEMENT_TRACE_MAX_STEPS:=40}"
: "${CONTACT_SUPPLEMENT_FPS:=10}"
: "${CONTACT_SUPPLEMENT_VIEW_RADIUS_M:=35}"
: "${CONTACT_SUPPLEMENT_CAMERA:=fixed}"
: "${CONTACT_SUPPLEMENT_VIDEO_FORMAT:=mp4}"
: "${CUDA_DEVICES:=0,1}"
: "${JOBS_PER_GPU:=1}"
: "${MAX_PARALLEL:=2}"
: "${USE_DYNAMIC_SCHEDULER:=auto}"

[[ "$CONTACT_SUPPLEMENT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid CONTACT_SUPPLEMENT_NAME=$CONTACT_SUPPLEMENT_NAME" >&2; exit 2; }
[[ "$CONTACT_SUPPLEMENT_MIN_POST_STEPS" =~ ^[0-9]+$ && "$CONTACT_SUPPLEMENT_MIN_POST_STEPS" -ge 20 ]] || { echo "CONTACT_SUPPLEMENT_MIN_POST_STEPS must be an integer >=20" >&2; exit 2; }
[[ "$CONTACT_SUPPLEMENT_TRACE_MAX_STEPS" =~ ^[0-9]+$ && "$CONTACT_SUPPLEMENT_TRACE_MAX_STEPS" -ge "$CONTACT_SUPPLEMENT_MIN_POST_STEPS" ]] || { echo "TRACE_MAX_STEPS must be >= MIN_POST_STEPS" >&2; exit 2; }

MIN_CLIP_S="$(python - "$CONTACT_SUPPLEMENT_MIN_POST_STEPS" <<'PY'
import sys
print(f"{int(sys.argv[1])*0.1:.6g}")
PY
)"
MINING_RESULT="$MAIN_CHARACTERIZATION_ROOT/contact_anchor/mining/closed_loop_nominal.json"
MAIN_CONTACT_SELECTION="$MAIN_VIS_ROOT/selection/contact_selection.json"
WORK="$BASE_OUT/contact_qualitative_supplements/$CONTACT_SUPPLEMENT_NAME"
ANCHOR_DIR="$WORK/contact_anchor"
TARGET_DIR="$WORK/target_keys"
TRACE_ROOT="$WORK/traces"
SELECTION_CANDIDATES="$WORK/selection_candidates"
SELECTION_ROOT="$WORK/selection"
LOG_DIR="$WORK/logs"
MANIFEST="$ANCHOR_DIR/contact_anchor_manifest.json"
TARGET_KEYS="$TARGET_DIR/contact.json"

[[ -s "$MINING_RESULT" ]] || { echo "missing Contact nominal mining result: $MINING_RESULT" >&2; exit 30; }
[[ -f "$OCRAP_MODEL_RUN/candidates/$MODEL_VARIANT/model_v48_trac_sr/best.pt" || -f "$OCRAP_MODEL_RUN/dedicated_candidates/$MODEL_VARIANT/model_v48_trac_sr/best.pt" ]] || {
  echo "cannot find OC-RAP checkpoint under model run: $OCRAP_MODEL_RUN" >&2; exit 30;
}
mkdir -p "$ANCHOR_DIR" "$TARGET_DIR" "$TRACE_ROOT/ocrap" "$TRACE_ROOT/external/contact" "$SELECTION_CANDIDATES" "$SELECTION_ROOT" "$LOG_DIR"

# Exclusive supplement writer. Re-running the same name is resumable; running a
# second supplement should use CONTACT_SUPPLEMENT_NAME=supplement_02.
if command -v flock >/dev/null 2>&1; then
  exec 9>"$WORK/.supplement.lock"
  flock -n 9 || { echo "another supplement build is already using $WORK" >&2; exit 31; }
fi

echo "[SUPPLEMENT] name=$CONTACT_SUPPLEMENT_NAME min_post_steps=$CONTACT_SUPPLEMENT_MIN_POST_STEPS min_clip=${MIN_CLIP_S}s max_clip=${CONTACT_SUPPLEMENT_MAX_CLIP_S}s"
echo "[SUPPLEMENT] publication Contact cohort is not modified; outputs append under $MAIN_VIS_ROOT/{videos,paper_figures}/contact/$CONTACT_SUPPLEMENT_NAME"

# 1) Expanded qualitative-only exact-a0, scene-disjoint anchor pool.
python tools/build_contact_anchor_manifest.py \
  --mining-result "$MINING_RESULT" \
  --min-post-steps "$CONTACT_SUPPLEMENT_MIN_POST_STEPS" \
  --output "$MANIFEST" \
  --target-keys-output "$TARGET_KEYS" \
  2>&1 | tee "$LOG_DIR/01_build_anchor_pool.log"
ANCHOR_COUNT="$(python - "$MANIFEST" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
print(int(x.get('num_selected_anchors') or 0))
PY
)"
(( ANCHOR_COUNT > 0 )) || { echo "expanded Contact anchor pool is empty" >&2; exit 30; }

# 2) OC-RAP once, with full render traces. This result is supplement-only and
# never replaces the frozen publication result root.
RUN_SAFE=0 RUN_NEAR=0 RUN_CONTACT=1 \
MODEL_RUN="$OCRAP_MODEL_RUN" MODEL_VARIANT="$MODEL_VARIANT" \
OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" \
OUT="$TRACE_ROOT/ocrap" CUDA_DEVICES="$CUDA_DEVICES" \
MAX_SCENARIOS=0 MAX_STEPS="$CONTACT_SUPPLEMENT_TRACE_MAX_STEPS" \
CONTACT_TARGET_KEYS_FILE="$TARGET_KEYS" \
CONTACT_ANCHOR_PRELUDE_ENABLED=true CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60 \
CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1 CONTACT_ANCHOR_REQUIRE_FOUND=true \
CONTACT_ANCHOR_MANIFEST_FILE="$MANIFEST" \
RENDER_CONTACT=true SCENE_JOURNAL_DETAIL=full RESULT_SCENE_DETAIL=metrics \
CONTACT_LABEL_MODE=fast SKIP_COMPLETE_REGIMES=true RESUME=true RESUME_FORCE=false \
bash scripts/run_ocrap_three_regime_evaluation.sh \
  2>&1 | tee "$LOG_DIR/02_ocrap_contact.log"

# 3) All six Contact paper-table baselines, also once with full render traces.
RUN="$TRACE_ROOT/external/contact" \
OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_LEGACY_CONTACT=false \
CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$CONTACT_SUPPLEMENT_TRACE_MAX_STEPS" \
CL_TARGET_KEYS_FILE="$TARGET_KEYS" CL_RENDER_TRACE=true CL_SCENE_JOURNAL_DETAIL=full \
CL_CONTACT_ANCHOR_PRELUDE_ENABLED=true CL_CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60 \
CL_CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1 CL_CONTACT_ANCHOR_REQUIRE_FOUND=true \
CL_CONTACT_ANCHOR_MANIFEST_FILE="$MANIFEST" \
JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" USE_DYNAMIC_SCHEDULER="$USE_DYNAMIC_SCHEDULER" \
SKIP_COMPLETE_METHODS=true CL_RESUME=true CL_RESUME_FORCE=false \
bash scripts/run_external_baselines_contact.sh \
  2>&1 | tee "$LOG_DIR/03_external_contact.log"

# 4) Metric-stage filter. Tier <=2 means we keep strict/majority/all-nonregressive
# examples but exclude tier-3 best-available cases with a comparative regression.
BASELINES=(postimpact_mpc_lite post_crash_braking postimpact_motion_tvlqr post_collision_restoration compensatory_postimpact_mpc robust_postimpact_control)
SEL_ARGS=()
for m in "${BASELINES[@]}"; do
  SEL_ARGS+=(--baseline "$m=$TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl")
done
python tools/select_regime_visualization_scenes.py \
  --regime contact \
  --ocrap-scenes "$TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl" \
  "${SEL_ARGS[@]}" \
  --contact-anchor-manifest "$MANIFEST" \
  --output "$SELECTION_CANDIDATES/contact_selection.json" \
  --target-keys-output "$SELECTION_CANDIDATES/contact_target_keys.json" \
  --num-scenes "$ANCHOR_COUNT" \
  --min-duration-s "$CONTACT_SUPPLEMENT_MAX_CLIP_S" \
  --fallback-min-duration-s "$MIN_CLIP_S" \
  --max-selected-tier-rank "$CONTACT_SUPPLEMENT_MAX_TIER_RANK" \
  --allow-fewer-scenes \
  2>&1 | tee "$LOG_DIR/04_metric_filter.log"

# 5) Trace-aware realism gate. Keep ALL accepted new scenes; no arbitrary quota.
FINAL_ARGS=(
  --candidate-selection "$SELECTION_CANDIDATES/contact_selection.json"
  --trace-root "$TRACE_ROOT"
  --output "$SELECTION_ROOT/contact_selection.json"
  --audit-output "$SELECTION_ROOT/contact_selection_audit.json"
  --min-external-recovery-failures "$CONTACT_SUPPLEMENT_MIN_EXTERNAL_FAILURES"
  --min-clip-duration-s "$MIN_CLIP_S"
  --max-clip-duration-s "$CONTACT_SUPPLEMENT_MAX_CLIP_S"
)
[[ -s "$MAIN_CONTACT_SELECTION" ]] && FINAL_ARGS+=(--exclude-selection "$MAIN_CONTACT_SELECTION")
python tools/finalize_contact_supplement_selection.py "${FINAL_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/05_trace_realism_filter.log"

ACCEPTED_COUNT="$(python - "$SELECTION_ROOT/contact_selection.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
print(len(x.get('selected') or []))
PY
)"
echo "[SUPPLEMENT] accepted new scenes=$ACCEPTED_COUNT"

# 6) Render figures directly into the existing main visualization tree, nested
# under contact/<supplement_name>; main rank_01/rank_02 etc. are untouched.
TRACE_ARGS=(--trace "ocrap=$TRACE_ROOT/ocrap/contact/closed_loop_ocrap.json.scenes.jsonl")
for m in "${BASELINES[@]}"; do TRACE_ARGS+=(--trace "$m=$TRACE_ROOT/external/contact/closed_loop_${m}.json.scenes.jsonl"); done
python tools/render_regime_paper_figures.py \
  "${TRACE_ARGS[@]}" \
  --selection "$SELECTION_ROOT/contact_selection.json" \
  --output-dir "$MAIN_VIS_ROOT/paper_figures" \
  --supplement-name "$CONTACT_SUPPLEMENT_NAME" \
  --view-radius-m "$CONTACT_SUPPLEMENT_VIEW_RADIUS_M" --force \
  2>&1 | tee "$LOG_DIR/06_figures.log"

# 7) Render pair + all-method videos. Short clips are slowed only for viewing;
# simulation time/span and all metrics are unchanged, and each slowed video is
# explicitly labeled with its real-time playback factor.
python tools/render_regime_visualization_videos.py \
  "${TRACE_ARGS[@]}" \
  --selection "$SELECTION_ROOT/contact_selection.json" \
  --output-dir "$MAIN_VIS_ROOT/videos" \
  --supplement-name "$CONTACT_SUPPLEMENT_NAME" \
  --fps "$CONTACT_SUPPLEMENT_FPS" --format "$CONTACT_SUPPLEMENT_VIDEO_FORMAT" \
  --camera-mode "$CONTACT_SUPPLEMENT_CAMERA" --view-radius-m "$CONTACT_SUPPLEMENT_VIEW_RADIUS_M" \
  --playback-slowdown 1.0 \
  --target-rendered-duration-s "$CONTACT_SUPPLEMENT_TARGET_RENDERED_DURATION_S" \
  --max-playback-slowdown "$CONTACT_SUPPLEMENT_MAX_PLAYBACK_SLOWDOWN" \
  --include-all-method-montage --force \
  2>&1 | tee "$LOG_DIR/07_videos.log"

# Remove stale files inside this supplement only after both renderers succeed.
python - "$MAIN_VIS_ROOT" "$CONTACT_SUPPLEMENT_NAME" <<'PYPRUNE'
import json,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]); name=sys.argv[2]
for kind,index_name,file_keys in (
    ("videos","VIDEO_INDEX.json",("videos",)),
    ("paper_figures","PAPER_FIGURE_INDEX.json",("pair_files","all_method_files")),
):
    base=root/kind/"contact"/name; idx=base/index_name
    if not idx.is_file():
        continue
    d=json.loads(idx.read_text())
    keep=set()
    for rec in d.get("records") or []:
        rank=int(rec.get("rank") or 0); rd=f"rank_{rank:02d}"
        for key in file_keys:
            vals=rec.get(key) or []
            if key=="videos":
                vals=[x.get("path") for x in vals if isinstance(x,dict)]
            elif isinstance(vals,str):
                vals=[vals]
            for value in vals:
                if value: keep.add((rd,pathlib.Path(str(value)).name))
    for rd in base.glob("rank_*"):
        if not rd.is_dir(): continue
        for f in rd.iterdir():
            if f.is_file() and (rd.name,f.name) not in keep:
                f.unlink()
        if not any(rd.iterdir()): rd.rmdir()
PYPRUNE

python - "$WORK" "$MAIN_VIS_ROOT" "$CONTACT_SUPPLEMENT_NAME" "$ANCHOR_COUNT" "$ACCEPTED_COUNT" "$CONTACT_SUPPLEMENT_MIN_POST_STEPS" <<'PY'
import json,pathlib,sys
work=pathlib.Path(sys.argv[1]); main=pathlib.Path(sys.argv[2]); name=sys.argv[3]
d={
  'event':'contact_qualitative_supplement_complete_v1',
  'supplement_name':name,
  'qualitative_only':True,
  'publication_contact_cohort_modified':False,
  'expanded_anchor_count':int(sys.argv[4]),
  'accepted_new_scene_count':int(sys.argv[5]),
  'min_post_steps':int(sys.argv[6]),
  'selection':str(work/'selection/contact_selection.json'),
  'audit':str(work/'selection/contact_selection_audit.json'),
  'videos':str(main/'videos/contact'/name),
  'paper_figures':str(main/'paper_figures/contact'/name),
}
(work/'SUPPLEMENT_SUMMARY.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d,indent=2))
PY


python - "$MAIN_VIS_ROOT" "$CONTACT_SUPPLEMENT_NAME" "$ACCEPTED_COUNT" <<'PYIDX'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); name=sys.argv[2]; n=int(sys.argv[3])
for kind, local_index in (("videos","VIDEO_INDEX.json"),("paper_figures","PAPER_FIGURE_INDEX.json")):
    croot=root/kind/"contact"; croot.mkdir(parents=True,exist_ok=True)
    p=croot/"SUPPLEMENTS_INDEX.json"
    try:d=json.loads(p.read_text()) if p.is_file() else {}
    except Exception:d={}
    rows={str(x.get("name")):x for x in (d.get("supplements") or []) if isinstance(x,dict) and x.get("name")}
    rows[name]={"name":name,"accepted_scenes":n,"index":str(croot/name/local_index)}
    out={"event":"contact_visualization_supplements_index_v1","supplements":[rows[k] for k in sorted(rows)]}
    p.write_text(json.dumps(out,indent=2)+"\n")
PYIDX
echo "[SUPPLEMENT][DONE] videos: $MAIN_VIS_ROOT/videos/contact/$CONTACT_SUPPLEMENT_NAME"
echo "[SUPPLEMENT][DONE] figures: $MAIN_VIS_ROOT/paper_figures/contact/$CONTACT_SUPPLEMENT_NAME"
echo "[SUPPLEMENT][DONE] audit: $SELECTION_ROOT/contact_selection_audit.json"
