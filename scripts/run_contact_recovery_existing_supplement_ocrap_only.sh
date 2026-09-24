#!/usr/bin/env bash
# Re-run only OC-RAP on an already-computed expanded Contact qualitative cohort,
# then reuse the source supplement's six external full traces for paired visualization.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/tools:$REPO${PYTHONPATH:+:$PYTHONPATH}"
: "${BASE_OUT:=runs}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${OCRAP_MODEL_RUN:=$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
: "${MAIN_VIS_ROOT:=$BASE_OUT/regime_visualization_v48_124_final_optimized}"
: "${SOURCE_SUPPLEMENT_NAME:=supplement_01}"
: "${OUTPUT_SUPPLEMENT_NAME:=supplement_recovery_01}"
: "${QUICK_SUMMARY:=$BASE_OUT/contact_recovery_quickdiag_v135/QUICK_DIAGNOSTIC_SUMMARY.json}"
: "${CONTACT_RECOVERY_PROFILE:=guarded_fallback}"
: "${GPU:=0}"
: "${FPS:=10}"
: "${TARGET_RENDERED_DURATION_S:=3.5}"
: "${MAX_PLAYBACK_SLOWDOWN:=1.4}"
SOURCE="$BASE_OUT/contact_qualitative_supplements/$SOURCE_SUPPLEMENT_NAME"
MANIFEST="$SOURCE/contact_anchor/contact_anchor_manifest.json"; KEYS="$SOURCE/target_keys/contact.json"
EXT="$SOURCE/traces/external/contact"
[[ -s "$MANIFEST" && -s "$KEYS" && -d "$EXT" ]] || { echo "source supplement is incomplete: $SOURCE" >&2; exit 30; }
python - "$QUICK_SUMMARY" "$CONTACT_RECOVERY_PROFILE" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); p=sys.argv[2]
if p not in (x.get('promising_profiles') or []): raise SystemExit(f'{p} not promising in quick validation diagnostic')
PY
case "$CONTACT_RECOVERY_PROFILE" in
  guarded_fallback) SELECTOR=lcb_constrained; REQUIRE_ABS=false ;;
  calibrated_guarded) SELECTOR=calibrated_constrained; REQUIRE_ABS=false ;;
  *) echo "unsupported profile" >&2; exit 2 ;;
esac
WORK="$BASE_OUT/contact_recovery_supplements/$OUTPUT_SUPPLEMENT_NAME"
OCR="$WORK/traces/ocrap"; COMBINED="$WORK/combined_traces"; CAND="$WORK/selection_candidates"; SEL="$WORK/selection"; LOG="$WORK/logs"
mkdir -p "$OCR" "$COMBINED" "$CAND" "$SEL" "$LOG"
RUN_SAFE=0 RUN_NEAR=0 RUN_CONTACT=1 MODEL_RUN="$OCRAP_MODEL_RUN" MODEL_VARIANT=balanced \
OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" OUT="$OCR" CUDA_DEVICES="$GPU" \
CONTACT_BUCKET="$OCRAP_ROOT/test_contact" BUCKET_SPLIT=test MAX_SCENARIOS=0 MAX_STEPS=40 \
CONTACT_TARGET_KEYS_FILE="$KEYS" CONTACT_ANCHOR_PRELUDE_ENABLED=true CONTACT_ANCHOR_MANIFEST_FILE="$MANIFEST" \
CONTACT_REQUIRE_ABSOLUTE_ADMISSION_FOR_INTERVENTION="$REQUIRE_ABS" CONTACT_OCRAP_SELECTOR="$SELECTOR" \
CONTACT_LABEL_MODE=fast RENDER_CONTACT=true SCENE_JOURNAL_DETAIL=full RESULT_SCENE_DETAIL=metrics \
RESUME=true RESUME_FORCE=false bash scripts/run_ocrap_three_regime_evaluation.sh 2>&1 | tee "$LOG/01_ocrap_recovery.log"
rm -rf "$COMBINED/ocrap" "$COMBINED/external"; mkdir -p "$COMBINED/ocrap" "$COMBINED/external"
ln -s "$OCR/contact" "$COMBINED/ocrap/contact"
ln -s "$EXT" "$COMBINED/external/contact"
BASELINES=(postimpact_mpc_lite post_crash_braking postimpact_motion_tvlqr post_collision_restoration compensatory_postimpact_mpc robust_postimpact_control)
ARGS=(); for m in "${BASELINES[@]}"; do ARGS+=(--baseline "$m=$EXT/closed_loop_${m}.json.scenes.jsonl"); done
N="$(python - "$MANIFEST" <<'PY'
import json,sys; print(int(json.load(open(sys.argv[1])).get('num_selected_anchors') or 0))
PY
)"
python tools/select_regime_visualization_scenes.py --regime contact \
  --ocrap-scenes "$OCR/contact/closed_loop_ocrap.json.scenes.jsonl" "${ARGS[@]}" --contact-anchor-manifest "$MANIFEST" \
  --output "$CAND/contact_selection.json" --target-keys-output "$CAND/contact_target_keys.json" --num-scenes "$N" \
  --min-duration-s 4.0 --fallback-min-duration-s 2.5 --max-selected-tier-rank 3 --allow-fewer-scenes | tee "$LOG/02_metric_filter.log"
FINAL=(--candidate-selection "$CAND/contact_selection.json" --trace-root "$COMBINED" --output "$SEL/contact_selection.json" --audit-output "$SEL/contact_selection_audit.json" --comparative-mode failure_or_temporal --min-external-recovery-failures 0 --min-comparative-evidence-methods 1 --min-clip-duration-s 2.5 --max-clip-duration-s 4.0)
[[ -s "$MAIN_VIS_ROOT/selection/contact_selection.json" ]] && FINAL+=(--exclude-selection "$MAIN_VIS_ROOT/selection/contact_selection.json")
[[ -s "$SOURCE/selection/contact_selection.json" ]] && FINAL+=(--exclude-selection "$SOURCE/selection/contact_selection.json")
python tools/finalize_contact_supplement_selection.py "${FINAL[@]}" | tee "$LOG/03_realism_temporal_filter.log"
TRACE_ARGS=(--trace "ocrap=$OCR/contact/closed_loop_ocrap.json.scenes.jsonl"); for m in "${BASELINES[@]}"; do TRACE_ARGS+=(--trace "$m=$EXT/closed_loop_${m}.json.scenes.jsonl"); done
python tools/render_regime_paper_figures.py "${TRACE_ARGS[@]}" --selection "$SEL/contact_selection.json" --output-dir "$MAIN_VIS_ROOT/paper_figures" --supplement-name "$OUTPUT_SUPPLEMENT_NAME" --view-radius-m 35 --force | tee "$LOG/04_figures.log"
python tools/render_regime_visualization_videos.py "${TRACE_ARGS[@]}" --selection "$SEL/contact_selection.json" --output-dir "$MAIN_VIS_ROOT/videos" --supplement-name "$OUTPUT_SUPPLEMENT_NAME" --fps "$FPS" --format mp4 --camera-mode fixed --view-radius-m 35 --target-rendered-duration-s "$TARGET_RENDERED_DURATION_S" --max-playback-slowdown "$MAX_PLAYBACK_SLOWDOWN" --include-all-method-montage --force | tee "$LOG/05_videos.log"
echo "[CONTACT-RECOVERY-SUPPLEMENT][DONE] $MAIN_VIS_ROOT/videos/contact/$OUTPUT_SUPPLEMENT_NAME"
