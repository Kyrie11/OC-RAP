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
  --num-scenes N         default 5 final ranks per regime
  Per-regime overrides are also available through SAFE_FINAL_NUM_SCENES,
  NEAR_FINAL_NUM_SCENES, and CONTACT_FINAL_NUM_SCENES environment variables.
  --gpus LIST            default 0,1
  --fps N                default 10
  --trace-steps N        default 60 (6 s at 0.1 s/step)
  --min-duration-s S     preferred clip duration, default 6.0
  --fallback-min-duration-s S  Safe/Near fallback, default 4.0
  --contact-fallback-min-duration-s S  Contact fallback inside locked anchor cohort, default 4.0
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
MODEL_VARIANT="${MODEL_VARIANT:-balanced}"; OUT="${OUT:-$BASE_OUT/regime_visualization_v48_124_final}"; NUM_SCENES="${NUM_SCENES:-5}"; CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
FPS="${FPS:-10}"; TRACE_MAX_STEPS="${TRACE_MAX_STEPS:-60}"; CAMERA_MODE="${CAMERA_MODE:-fixed}"; VIEW_RADIUS_M="${VIEW_RADIUS_M:-35}"; VIDEO_FORMAT="${VIDEO_FORMAT:-mp4}"
MIN_VIDEO_DURATION_S="${MIN_VIDEO_DURATION_S:-6.0}"
# Six seconds is preferred.  Four seconds is the principled fallback because the
# frozen Contact metric cohort itself is constructed with a 40-step post-anchor
# treatment horizon.  Safe/Near keep 6 s whenever enough candidates exist.
FALLBACK_MIN_VIDEO_DURATION_S="${FALLBACK_MIN_VIDEO_DURATION_S:-4.0}"
CONTACT_FALLBACK_MIN_VIDEO_DURATION_S="${CONTACT_FALLBACK_MIN_VIDEO_DURATION_S:-4.0}"

# Reviewer-facing trace-aware gates. Safe/Near keep the strict global lane
# contract. Contact-specific overrides default to the same values, but can be
# relaxed independently after inspecting selection/contact_selection_audit.json.
VIS_LANE_TERMINAL_MAX_M="${VIS_LANE_TERMINAL_MAX_M:-5.0}"
VIS_LANE_P90_MAX_M="${VIS_LANE_P90_MAX_M:-5.5}"
VIS_LANE_OFFCENTER_THRESHOLD_M="${VIS_LANE_OFFCENTER_THRESHOLD_M:-4.5}"
VIS_LANE_OFFCENTER_FRACTION_MAX="${VIS_LANE_OFFCENTER_FRACTION_MAX:-0.30}"
VIS_LANE_HEADING_TERMINAL_MAX_DEG="${VIS_LANE_HEADING_TERMINAL_MAX_DEG:-50.0}"
VIS_LANE_HEADING_P90_MAX_DEG="${VIS_LANE_HEADING_P90_MAX_DEG:-55.0}"
CONTACT_LANE_TERMINAL_MAX_M="${CONTACT_LANE_TERMINAL_MAX_M:-$VIS_LANE_TERMINAL_MAX_M}"
CONTACT_LANE_P90_MAX_M="${CONTACT_LANE_P90_MAX_M:-$VIS_LANE_P90_MAX_M}"
CONTACT_LANE_OFFCENTER_THRESHOLD_M="${CONTACT_LANE_OFFCENTER_THRESHOLD_M:-$VIS_LANE_OFFCENTER_THRESHOLD_M}"
CONTACT_LANE_OFFCENTER_FRACTION_MAX="${CONTACT_LANE_OFFCENTER_FRACTION_MAX:-$VIS_LANE_OFFCENTER_FRACTION_MAX}"
CONTACT_LANE_HEADING_TERMINAL_MAX_DEG="${CONTACT_LANE_HEADING_TERMINAL_MAX_DEG:-$VIS_LANE_HEADING_TERMINAL_MAX_DEG}"
CONTACT_LANE_HEADING_P90_MAX_DEG="${CONTACT_LANE_HEADING_P90_MAX_DEG:-$VIS_LANE_HEADING_P90_MAX_DEG}"
# Short-horizon Contact fallback: never permits off-road/re-contact/terminal
# overlap. It only accepts a lane-distance miss when the vehicle remains under
# explicit caps and is measurably converging back toward a vehicle lane.
CONTACT_LANE_RECOVERY_TERMINAL_MAX_M="${CONTACT_LANE_RECOVERY_TERMINAL_MAX_M:-6.5}"
CONTACT_LANE_RECOVERY_P90_MAX_M="${CONTACT_LANE_RECOVERY_P90_MAX_M:-7.0}"
CONTACT_LANE_RECOVERY_OFFCENTER_FRACTION_MAX="${CONTACT_LANE_RECOVERY_OFFCENTER_FRACTION_MAX:-0.45}"
CONTACT_LANE_PEAK_IMPROVEMENT_MIN_M="${CONTACT_LANE_PEAK_IMPROVEMENT_MIN_M:-1.0}"
CONTACT_LANE_RECENT_RECOVERY_MIN_M="${CONTACT_LANE_RECENT_RECOVERY_MIN_M:-0.35}"
NEAR_MIN_EXTERNAL_SEVERE_COUNT="${NEAR_MIN_EXTERNAL_SEVERE_COUNT:-2}"
CONTACT_SEPARATION_CLEARANCE_M="${CONTACT_SEPARATION_CLEARANCE_M:-0.50}"
CONTACT_SEPARATION_HOLD_S="${CONTACT_SEPARATION_HOLD_S:-0.30}"
PRESERVE_ACCEPTED_EXISTING_SELECTION="${PRESERVE_ACCEPTED_EXISTING_SELECTION:-true}"

# Trace-aware finalization needs a modest over-selected pool so late Safe tail
# failures can be replaced and Near-Contact can favor consensus-failure/dense
# scenes without changing the locked quantitative population.
VIS_CANDIDATE_MULTIPLIER="${VIS_CANDIDATE_MULTIPLIER:-2}"
[[ "$VIS_CANDIDATE_MULTIPLIER" =~ ^[0-9]+$ && "$VIS_CANDIDATE_MULTIPLIER" -ge 1 ]] || { echo 'VIS_CANDIDATE_MULTIPLIER must be a positive integer' >&2; exit 2; }
CONTACT_VIS_CANDIDATE_MULTIPLIER="${CONTACT_VIS_CANDIDATE_MULTIPLIER:-3}"
[[ "$CONTACT_VIS_CANDIDATE_MULTIPLIER" =~ ^[0-9]+$ && "$CONTACT_VIS_CANDIDATE_MULTIPLIER" -ge 1 ]] || { echo 'CONTACT_VIS_CANDIDATE_MULTIPLIER must be a positive integer' >&2; exit 2; }
# Capture optional per-regime count overrides before parsing --num-scenes; if
# unset, they inherit the *parsed* global count below rather than the initial
# default value.
_SAFE_FINAL_NUM_SCENES_ENV="${SAFE_FINAL_NUM_SCENES:-}"
_NEAR_FINAL_NUM_SCENES_ENV="${NEAR_FINAL_NUM_SCENES:-}"
_CONTACT_FINAL_NUM_SCENES_ENV="${CONTACT_FINAL_NUM_SCENES:-}"
while (($#)); do case "$1" in
 --ocrap-results) OCRAP_RESULTS_ROOT="$2";shift 2;; --model-run) OCRAP_MODEL_RUN="$2";shift 2;; --external-root) EXTERNAL_RESULTS_ROOT="$2";shift 2;;
 --target-lock-root) TARGET_LOCK_ROOT="$2";shift 2;; --variant) MODEL_VARIANT="$2";shift 2;; --out) OUT="$2";shift 2;; --num-scenes) NUM_SCENES="$2";shift 2;; --gpus) CUDA_DEVICES="$2";shift 2;;
 --fps) FPS="$2";shift 2;; --trace-steps) TRACE_MAX_STEPS="$2";shift 2;; --min-duration-s) MIN_VIDEO_DURATION_S="$2"; shift 2;; --fallback-min-duration-s) FALLBACK_MIN_VIDEO_DURATION_S="$2"; shift 2;; --contact-fallback-min-duration-s) CONTACT_FALLBACK_MIN_VIDEO_DURATION_S="$2"; shift 2;; --camera) CAMERA_MODE="$2";shift 2;; --view-radius) VIEW_RADIUS_M="$2";shift 2;;
 -h|--help) usage;exit 0;; *) echo "unknown option $1" >&2;usage >&2;exit 2;; esac;done
[[ -n "$OCRAP_RESULTS_ROOT" && -n "$OCRAP_MODEL_RUN" ]] || { echo '--ocrap-results and --model-run are required' >&2; exit 2; }
[[ "$TRACE_MAX_STEPS" =~ ^[0-9]+$ && "$TRACE_MAX_STEPS" -gt 0 ]] || { echo '--trace-steps must be a positive integer' >&2; exit 2; }
SAFE_FINAL_NUM_SCENES="${_SAFE_FINAL_NUM_SCENES_ENV:-$NUM_SCENES}"
NEAR_FINAL_NUM_SCENES="${_NEAR_FINAL_NUM_SCENES_ENV:-$NUM_SCENES}"
CONTACT_FINAL_NUM_SCENES="${_CONTACT_FINAL_NUM_SCENES_ENV:-$NUM_SCENES}"
for _n in "$SAFE_FINAL_NUM_SCENES" "$NEAR_FINAL_NUM_SCENES" "$CONTACT_FINAL_NUM_SCENES"; do
  [[ "$_n" =~ ^[0-9]+$ && "$_n" -ge 1 ]] || { echo 'per-regime final scene counts must be positive integers' >&2; exit 2; }
done
SAFE_CANDIDATE_SCENES=$((SAFE_FINAL_NUM_SCENES * VIS_CANDIDATE_MULTIPLIER))
NEAR_CANDIDATE_SCENES=$((NEAR_FINAL_NUM_SCENES * VIS_CANDIDATE_MULTIPLIER))
# Contact has only eight locked exact-a0 anchors. Request substantially more
# than the final rank count (ALLOW_FEWER_SCENES downstream safely returns all
# eligible anchors) so the trace-aware realism gate can replace a metric-good
# but visually implausible escape trajectory instead of being forced to keep it.
CONTACT_CANDIDATE_SCENES=$((CONTACT_FINAL_NUM_SCENES * CONTACT_VIS_CANDIDATE_MULTIPLIER))
# Make scheduler choice configurable from the top-level command.  The default
# remains auto; this variable affects only execution scheduling, never method
# definitions, checkpoints, target locks, or scientific metrics.
: "${USE_DYNAMIC_SCHEDULER:=auto}"

# One visualization build owns an output root exclusively.  Two overlapping
# builds can otherwise append the same selected target to a JSONL journal at
# the same time, producing scientifically identical but structurally duplicate
# rows that the strict trace contract correctly rejects.
mkdir -p "$OUT"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$OUT/.build_regime_visualizations.lock"
  if ! flock -n 9; then
    echo "[ERROR] another build_regime_visualizations.sh process is already using OUT=$OUT" >&2
    echo "Wait for that process to finish, or choose a different --out directory." >&2
    exit 31
  fi
else
  echo "[WARN] flock is unavailable; do not run two visualization builds against the same --out directory." >&2
fi
if [[ "$VIDEO_FORMAT" == mp4 ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[WARN] ffmpeg not found; static paper figures will still be generated and video format falls back to auto/GIF." >&2
  VIDEO_FORMAT=auto
fi
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}" WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
export SAFE_EXTERNAL_ROOT="$EXTERNAL_RESULTS_ROOT/safe" NEAR_EXTERNAL_ROOT="$EXTERNAL_RESULTS_ROOT/near" CONTACT_EXTERNAL_ROOT="$EXTERNAL_RESULTS_ROOT/contact"
export OCRAP_RESULTS_ROOT OCRAP_MODEL_RUN TARGET_LOCK_ROOT MODEL_VARIANT OUT NUM_SCENES CUDA_DEVICES FPS TRACE_MAX_STEPS CAMERA_MODE VIEW_RADIUS_M VIDEO_FORMAT
export VIS_CANDIDATE_MULTIPLIER CONTACT_VIS_CANDIDATE_MULTIPLIER SAFE_CANDIDATE_SCENES NEAR_CANDIDATE_SCENES CONTACT_CANDIDATE_SCENES
export SAFE_FINAL_NUM_SCENES NEAR_FINAL_NUM_SCENES CONTACT_FINAL_NUM_SCENES
export MAX_SELECTED_TIER_RANK="${MAX_SELECTED_TIER_RANK:-1}" JOBS_PER_GPU="${JOBS_PER_GPU:-3}" MAX_PARALLEL="${MAX_PARALLEL:-6}" USE_DYNAMIC_SCHEDULER
export SAFE_MAX_SELECTED_TIER_RANK="${SAFE_MAX_SELECTED_TIER_RANK:-$MAX_SELECTED_TIER_RANK}"
export NEAR_MAX_SELECTED_TIER_RANK="${NEAR_MAX_SELECTED_TIER_RANK:-$MAX_SELECTED_TIER_RANK}"
export CONTACT_MAX_SELECTED_TIER_RANK="${CONTACT_MAX_SELECTED_TIER_RANK:-$MAX_SELECTED_TIER_RANK}"
export MIN_VIDEO_DURATION_S FALLBACK_MIN_VIDEO_DURATION_S CONTACT_FALLBACK_MIN_VIDEO_DURATION_S
export VIS_LANE_TERMINAL_MAX_M VIS_LANE_P90_MAX_M VIS_LANE_OFFCENTER_THRESHOLD_M VIS_LANE_OFFCENTER_FRACTION_MAX
export VIS_LANE_HEADING_TERMINAL_MAX_DEG VIS_LANE_HEADING_P90_MAX_DEG
export CONTACT_LANE_TERMINAL_MAX_M CONTACT_LANE_P90_MAX_M CONTACT_LANE_OFFCENTER_THRESHOLD_M CONTACT_LANE_OFFCENTER_FRACTION_MAX
export CONTACT_LANE_HEADING_TERMINAL_MAX_DEG CONTACT_LANE_HEADING_P90_MAX_DEG
export CONTACT_LANE_RECOVERY_TERMINAL_MAX_M CONTACT_LANE_RECOVERY_P90_MAX_M CONTACT_LANE_RECOVERY_OFFCENTER_FRACTION_MAX
export CONTACT_LANE_PEAK_IMPROVEMENT_MIN_M CONTACT_LANE_RECENT_RECOVERY_MIN_M
export NEAR_MIN_EXTERNAL_SEVERE_COUNT CONTACT_SEPARATION_CLEARANCE_M CONTACT_SEPARATION_HOLD_S PRESERVE_ACCEPTED_EXISTING_SELECTION

# Publication Contact is anchored at the first exact-a0 observed overlap.  Do
# not create a new 60-step target cohort for visualization: the final metric
# cohort is already treatment-independent and locked by the 40-step anchor
# manifest.  Visualization must stay inside that exact cohort.  The selector
# prefers 6 s clips and falls back to 4 s using each anchor's actual
# contact_anchor_remaining_steps.
CHAR_ROOT="$(cd "$TARGET_LOCK_ROOT/.." && pwd)"
FINAL_CONTACT_ANCHOR_MANIFEST="$CHAR_ROOT/contact_anchor/contact_anchor_manifest.json"
VIS_CONTACT_ANCHOR_MANIFEST_FILE="$OUT/provenance/contact_anchor_manifest.json"
CONTACT_HORIZON_REPORT="$OUT/provenance/CONTACT_VISUALIZATION_HORIZON.json"
[[ -s "$FINAL_CONTACT_ANCHOR_MANIFEST" ]] || { echo "missing frozen Contact anchor manifest: $FINAL_CONTACT_ANCHOR_MANIFEST" >&2; exit 30; }
mkdir -p "$OUT/provenance"
python - "$FINAL_CONTACT_ANCHOR_MANIFEST" "$TARGET_LOCK_ROOT/contact.json" "$VIS_CONTACT_ANCHOR_MANIFEST_FILE" "$CONTACT_HORIZON_REPORT" "$CONTACT_FINAL_NUM_SCENES" "$TRACE_MAX_STEPS" "$CONTACT_FALLBACK_MIN_VIDEO_DURATION_S" <<'PY'
import json, math, pathlib, shutil, sys
manifest_p, lock_p, out_p, report_p = map(pathlib.Path, sys.argv[1:5])
num_scenes=int(sys.argv[5]); preferred_steps=int(sys.argv[6]); fallback_s=float(sys.argv[7]); dt=0.1
m=json.loads(manifest_p.read_text(encoding='utf-8')); lock=json.loads(lock_p.read_text(encoding='utf-8'))
if m.get('schema')!='ocrap-contact-anchor-manifest-v1' or m.get('valid') is not True:
    raise SystemExit(f'invalid frozen Contact anchor manifest: {manifest_p}')
locked=[str(x) for x in (lock.get('target_keys') or [])]
anchors={str(a.get('target_key') or ''):a for a in (m.get('anchors') or []) if isinstance(a,dict)}
if not locked or set(anchors)!=set(locked):
    raise SystemExit(f'Contact anchor manifest/target-lock mismatch: anchors={len(anchors)} locked={len(locked)} missing={sorted(set(locked)-set(anchors))[:10]} extra={sorted(set(anchors)-set(locked))[:10]}')
rows=[]
for k in locked:
    rem=int(anchors[k].get('contact_anchor_remaining_steps') or 0)
    rows.append({'target_key':k,'remaining_steps':rem,'remaining_s':rem*dt})
fallback_steps=int(math.ceil(fallback_s/dt-1e-9))
report={
 'event':'contact_visualization_horizon_audit_v125',
 'locked_targets':len(locked),
 'requested_scenes':num_scenes,
 'preferred_steps':preferred_steps,
 'preferred_duration_s':preferred_steps*dt,
 'fallback_steps':fallback_steps,
 'fallback_duration_s':fallback_s,
 'num_preferred_eligible':sum(r['remaining_steps']>=preferred_steps for r in rows),
 'num_fallback_eligible':sum(r['remaining_steps']>=fallback_steps for r in rows),
 'per_target':rows,
}
if report['num_fallback_eligible'] < num_scenes:
    raise SystemExit(f"Contact locked cohort cannot supply {num_scenes} clips even at {fallback_s:.1f}s: {report['num_fallback_eligible']} eligible")
out_p.write_text(manifest_p.read_text(encoding='utf-8'),encoding='utf-8')
report_p.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
PY
CONTACT_ALLOWED_TARGET_KEYS_FILE=""
export VIS_CONTACT_ANCHOR_MANIFEST_FILE CONTACT_FALLBACK_MIN_VIDEO_DURATION_S CONTACT_ALLOWED_TARGET_KEYS_FILE

mkdir -p "$OUT/logs"
# Stage 1: metric-only over-selection from the locked population.
SELECTION_DIR="$OUT/selection_candidates" ALLOW_FEWER_SCENES=true \
SAFE_NUM_SCENES="$SAFE_CANDIDATE_SCENES" NEAR_NUM_SCENES="$NEAR_CANDIDATE_SCENES" CONTACT_NUM_SCENES="$CONTACT_CANDIDATE_SCENES" \
bash scripts/select_regime_visualizations.sh 2>&1 | tee "$OUT/logs/01_candidate_selection.log"

# Fail before expensive selective reruns when an explicit tier cap cannot supply
# the requested final rank count.  In the current locked Contact cohort only
# three scenes are tier<=1, so five Contact ranks require an explicit
# CONTACT_MAX_SELECTED_TIER_RANK relaxation rather than silently weakening all
# three regimes.
python - "$OUT/selection_candidates/SELECTION_INDEX.json" "$SAFE_FINAL_NUM_SCENES" "$NEAR_FINAL_NUM_SCENES" "$CONTACT_FINAL_NUM_SCENES" "$SAFE_MAX_SELECTED_TIER_RANK" "$NEAR_MAX_SELECTED_TIER_RANK" "$CONTACT_MAX_SELECTED_TIER_RANK" <<'PY'
import json,sys
p=sys.argv[1]
needs=dict(zip(("safe","near","contact"), map(int,sys.argv[2:5])))
caps=dict(zip(("safe","near","contact"), map(int,sys.argv[5:8])))
d=json.load(open(p,encoding="utf-8"))
short=[]
for r in ("safe","near","contact"):
    n=int(d["regimes"][r]["num_selected"]); need=needs[r]
    if n < need:
        short.append(f"{r}: selected={n} required={need} max_tier={caps[r]}")
if short:
    raise SystemExit("candidate selection cannot supply requested final ranks: " + "; ".join(short) + ". "
                     "Relax only the affected regime tier cap (for example CONTACT_MAX_SELECTED_TIER_RANK=3) "
                     "or lower --num-scenes.")
PY

# Stage 2: rerun full render traces for only that candidate pool.
SELECTION_ROOT="$OUT/selection_candidates" OUT="$OUT/selective_traces" bash scripts/generate_selected_regime_traces.sh 2>&1 | tee "$OUT/logs/02_selective_traces.log"

# Stage 3: trace-aware finalization. Publication-facing Safe rejects any visible
# OC-RAP overlap/off-road by default (no hiding a bad tail by cropping). All
# regimes add a roadgraph lane-realism gate when lane evidence is available.
# Near-Contact prioritizes broad external collision/low-margin consensus; Contact
# requires sustained separation plus a controlled lane-plausible recovery.
FINALIZE_ARGS=(
  --candidate-selection-root "$OUT/selection_candidates"
  --trace-root "$OUT/selective_traces"
  --output-root "$OUT/selection"
  --num-scenes "$NUM_SCENES"
  --safe-num-scenes "$SAFE_FINAL_NUM_SCENES"
  --near-num-scenes "$NEAR_FINAL_NUM_SCENES"
  --contact-num-scenes "$CONTACT_FINAL_NUM_SCENES"
  --safe-min-clip-s "$FALLBACK_MIN_VIDEO_DURATION_S"
  --near-min-external-severe-count "$NEAR_MIN_EXTERNAL_SEVERE_COUNT"
  --lane-terminal-max-m "$VIS_LANE_TERMINAL_MAX_M"
  --lane-p90-max-m "$VIS_LANE_P90_MAX_M"
  --lane-offcenter-threshold-m "$VIS_LANE_OFFCENTER_THRESHOLD_M"
  --lane-offcenter-fraction-max "$VIS_LANE_OFFCENTER_FRACTION_MAX"
  --lane-heading-terminal-max-deg "$VIS_LANE_HEADING_TERMINAL_MAX_DEG"
  --lane-heading-p90-max-deg "$VIS_LANE_HEADING_P90_MAX_DEG"
  --contact-lane-terminal-max-m "$CONTACT_LANE_TERMINAL_MAX_M"
  --contact-lane-p90-max-m "$CONTACT_LANE_P90_MAX_M"
  --contact-lane-offcenter-threshold-m "$CONTACT_LANE_OFFCENTER_THRESHOLD_M"
  --contact-lane-offcenter-fraction-max "$CONTACT_LANE_OFFCENTER_FRACTION_MAX"
  --contact-lane-heading-terminal-max-deg "$CONTACT_LANE_HEADING_TERMINAL_MAX_DEG"
  --contact-lane-heading-p90-max-deg "$CONTACT_LANE_HEADING_P90_MAX_DEG"
  --contact-lane-recovery-terminal-max-m "$CONTACT_LANE_RECOVERY_TERMINAL_MAX_M"
  --contact-lane-recovery-p90-max-m "$CONTACT_LANE_RECOVERY_P90_MAX_M"
  --contact-lane-recovery-offcenter-fraction-max "$CONTACT_LANE_RECOVERY_OFFCENTER_FRACTION_MAX"
  --contact-lane-peak-improvement-min-m "$CONTACT_LANE_PEAK_IMPROVEMENT_MIN_M"
  --contact-lane-recent-recovery-min-m "$CONTACT_LANE_RECENT_RECOVERY_MIN_M"
  --contact-separation-clearance-m "$CONTACT_SEPARATION_CLEARANCE_M"
  --contact-separation-hold-s "$CONTACT_SEPARATION_HOLD_S"
)
if [[ "${PRESERVE_ACCEPTED_EXISTING_SELECTION,,}" == "true" || "$PRESERVE_ACCEPTED_EXISTING_SELECTION" == "1" ]]; then
  FINALIZE_ARGS+=(--prefer-existing-selection)
fi
python tools/finalize_regime_visualization_selection.py "${FINALIZE_ARGS[@]}" \
  | tee "$OUT/logs/03_trace_aware_selection.log"

# Candidate journals intentionally contain extra targets. Validate the final
# five-scene subset without requiring destructive trace rewrites.
python tools/check_selected_trace_contract.py \
  --trace-root "$OUT/selective_traces" --selection-root "$OUT/selection" \
  --trace-max-steps "$TRACE_MAX_STEPS" --allow-extra-targets \
  --output "$OUT/selective_traces/TRACE_CONTRACT.json" \
  | tee "$OUT/logs/04_final_trace_contract.log"

# Static figures are rendered before videos so an encoder-specific failure can
# never suppress the paper-ready PNG/PDF outputs once the trace contract passes.
# Final selection/comparator identity can change even when the underlying trace
# journals are reusable.  Therefore never reuse an old rank_XX media file just
# because its container is valid: force only the cheap rendering stages so the
# PNG/PDF/MP4 content is guaranteed to match the current selection artifact.
TRACE_ROOT="$OUT/selective_traces" SELECTION_ROOT="$OUT/selection" OUT="$OUT/paper_figures" FORCE_RENDER=true bash scripts/render_regime_paper_figures.sh 2>&1 | tee "$OUT/logs/05_paper_figures.log"
TRACE_ROOT="$OUT/selective_traces" SELECTION_ROOT="$OUT/selection" OUT="$OUT/videos" FORMAT="$VIDEO_FORMAT" FORCE_RENDER=true bash scripts/render_regime_videos.sh 2>&1 | tee "$OUT/logs/06_videos.log"
python tools/audit_regime_visualization_outputs.py --root "$OUT" --expected-scenes "$NUM_SCENES" \
  --expected-safe-scenes "$SAFE_FINAL_NUM_SCENES" --expected-near-scenes "$NEAR_FINAL_NUM_SCENES" \
  --expected-contact-scenes "$CONTACT_FINAL_NUM_SCENES" | tee "$OUT/logs/07_output_audit.log"
printf 'Paper figures complete: %s\n' "$OUT/paper_figures/PAPER_FIGURE_INDEX.json"
printf 'Visualization complete: %s\n' "$OUT/videos/REGIME_VIDEO_INDEX.json"
