#!/usr/bin/env bash
# One clean entry point: select from completed metrics, generate selected traces, render videos.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
usage(){ cat <<'EOF'
Usage: scripts/build_regime_visualizations.sh [options]
  --ocrap-results DIR    variant root containing safe/near/contact metric runs
  --model-run DIR        frozen OC-RAP checkpoint/calibration owner
  --external-root DIR    root with safe/near/contact external-baseline results
  --target-lock-root DIR final observation-legal target locks (safe.json/near.json/contact.json)
  --variant NAME         default balanced
  --out DIR              default $BASE_OUT/regime_visualization_v48_124_final
  --num-scenes N         default 3 per regime
  --gpus LIST            default 0,1
  --fps N                default 10
  --trace-steps N        default 60 (6 s at 0.1 s/step)
  --min-duration-s S     default 6.0; final clips must have this much continuous future horizon
  --camera fixed|dynamic default fixed
  --view-radius M        default 35
This command never uses validation_interactive unless your already-completed result provenance explicitly says so;
for the publication buckets the input checker requires standard validation provenance.
EOF
}
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
OCRAP_RESULTS_ROOT="${OCRAP_RESULTS_ROOT:-}"; OCRAP_MODEL_RUN="${OCRAP_MODEL_RUN:-}"
EXTERNAL_RESULTS_ROOT="${EXTERNAL_RESULTS_ROOT:-$BASE_OUT/external_baselines_v48_124_final_v2}"
TARGET_LOCK_ROOT="${TARGET_LOCK_ROOT:-$BASE_OUT/ocrap_v48_124_final_characterization/target_keys}"
MODEL_VARIANT="${MODEL_VARIANT:-balanced}"; OUT="${OUT:-$BASE_OUT/regime_visualization_v48_124_final}"; NUM_SCENES="${NUM_SCENES:-3}"; CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
FPS="${FPS:-10}"; TRACE_MAX_STEPS="${TRACE_MAX_STEPS:-60}"; CAMERA_MODE="${CAMERA_MODE:-fixed}"; VIEW_RADIUS_M="${VIEW_RADIUS_M:-35}"; VIDEO_FORMAT="${VIDEO_FORMAT:-mp4}"
MIN_VIDEO_DURATION_S="${MIN_VIDEO_DURATION_S:-6.0}"; FALLBACK_MIN_VIDEO_DURATION_S="${FALLBACK_MIN_VIDEO_DURATION_S:-$MIN_VIDEO_DURATION_S}"
VIS_CONTACT_MIN_POST_STEPS="${VIS_CONTACT_MIN_POST_STEPS:-$TRACE_MAX_STEPS}"
while (($#)); do case "$1" in
 --ocrap-results) OCRAP_RESULTS_ROOT="$2";shift 2;; --model-run) OCRAP_MODEL_RUN="$2";shift 2;; --external-root) EXTERNAL_RESULTS_ROOT="$2";shift 2;;
 --target-lock-root) TARGET_LOCK_ROOT="$2";shift 2;; --variant) MODEL_VARIANT="$2";shift 2;; --out) OUT="$2";shift 2;; --num-scenes) NUM_SCENES="$2";shift 2;; --gpus) CUDA_DEVICES="$2";shift 2;;
 --fps) FPS="$2";shift 2;; --trace-steps) TRACE_MAX_STEPS="$2";shift 2;; --min-duration-s) MIN_VIDEO_DURATION_S="$2"; FALLBACK_MIN_VIDEO_DURATION_S="$2"; shift 2;; --camera) CAMERA_MODE="$2";shift 2;; --view-radius) VIEW_RADIUS_M="$2";shift 2;;
 -h|--help) usage;exit 0;; *) echo "unknown option $1" >&2;usage >&2;exit 2;; esac;done
[[ -n "$OCRAP_RESULTS_ROOT" && -n "$OCRAP_MODEL_RUN" ]] || { echo '--ocrap-results and --model-run are required' >&2; exit 2; }
[[ "$TRACE_MAX_STEPS" =~ ^[0-9]+$ && "$TRACE_MAX_STEPS" -gt 0 ]] || { echo '--trace-steps must be a positive integer' >&2; exit 2; }
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}" WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
export SAFE_EXTERNAL_ROOT="$EXTERNAL_RESULTS_ROOT/safe" NEAR_EXTERNAL_ROOT="$EXTERNAL_RESULTS_ROOT/near" CONTACT_EXTERNAL_ROOT="$EXTERNAL_RESULTS_ROOT/contact"
export OCRAP_RESULTS_ROOT OCRAP_MODEL_RUN TARGET_LOCK_ROOT MODEL_VARIANT OUT NUM_SCENES CUDA_DEVICES FPS TRACE_MAX_STEPS CAMERA_MODE VIEW_RADIUS_M VIDEO_FORMAT
export MAX_SELECTED_TIER_RANK="${MAX_SELECTED_TIER_RANK:-1}" JOBS_PER_GPU="${JOBS_PER_GPU:-3}" MAX_PARALLEL="${MAX_PARALLEL:-6}"
export MIN_VIDEO_DURATION_S FALLBACK_MIN_VIDEO_DURATION_S

# Publication Contact is anchored at the first exact-a0 observed overlap.  The
# main metric cohort guarantees only the 40-step evaluation horizon, while a
# six-second qualitative clip needs 60 post-anchor steps.  Re-select, from the
# *same treatment-independent mining result*, only anchors with enough remaining
# simulator horizon.  This does not change the paper metric cohort.
CHAR_ROOT="$(cd "$TARGET_LOCK_ROOT/.." && pwd)"
CONTACT_MINING_JSON="$CHAR_ROOT/contact_anchor/mining/closed_loop_nominal.json"
VIS_CONTACT_ANCHOR_MANIFEST_FILE="$OUT/provenance/contact_anchor_${VIS_CONTACT_MIN_POST_STEPS}step_manifest.json"
CONTACT_ALLOWED_TARGET_KEYS_FILE="$OUT/provenance/contact_anchor_${VIS_CONTACT_MIN_POST_STEPS}step_target_keys.json"
[[ -s "$CONTACT_MINING_JSON" ]] || { echo "missing Contact anchor mining result required for continuous visualization: $CONTACT_MINING_JSON" >&2; exit 30; }
mkdir -p "$OUT/provenance"
python tools/build_contact_anchor_manifest.py \
  --mining-result "$CONTACT_MINING_JSON" --min-post-steps "$VIS_CONTACT_MIN_POST_STEPS" \
  --output "$VIS_CONTACT_ANCHOR_MANIFEST_FILE" --target-keys-output "$CONTACT_ALLOWED_TARGET_KEYS_FILE"
export VIS_CONTACT_ANCHOR_MANIFEST_FILE CONTACT_ALLOWED_TARGET_KEYS_FILE

bash scripts/select_regime_visualizations.sh
SELECTION_ROOT="$OUT/selection" OUT="$OUT/selective_traces" bash scripts/generate_selected_regime_traces.sh
TRACE_ROOT="$OUT/selective_traces" SELECTION_ROOT="$OUT/selection" OUT="$OUT/videos" FORMAT="$VIDEO_FORMAT" bash scripts/render_regime_videos.sh
TRACE_ROOT="$OUT/selective_traces" SELECTION_ROOT="$OUT/selection" OUT="$OUT/paper_figures" bash scripts/render_regime_paper_figures.sh
printf 'Visualization complete: %s\n' "$OUT/videos/REGIME_VIDEO_INDEX.json"
printf 'Paper figures complete: %s\n' "$OUT/paper_figures/PAPER_FIGURE_INDEX.json"
