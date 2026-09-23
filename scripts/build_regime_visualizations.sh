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

# Trace-aware finalization needs a modest over-selected pool so late Safe tail
# failures can be replaced and Near-Contact can favor consensus-failure/dense
# scenes without changing the locked quantitative population.
VIS_CANDIDATE_MULTIPLIER="${VIS_CANDIDATE_MULTIPLIER:-2}"
[[ "$VIS_CANDIDATE_MULTIPLIER" =~ ^[0-9]+$ && "$VIS_CANDIDATE_MULTIPLIER" -ge 1 ]] || { echo 'VIS_CANDIDATE_MULTIPLIER must be a positive integer' >&2; exit 2; }
SAFE_CANDIDATE_SCENES=$((NUM_SCENES * VIS_CANDIDATE_MULTIPLIER))
NEAR_CANDIDATE_SCENES=$((NUM_SCENES * VIS_CANDIDATE_MULTIPLIER))
# Contact has only eight locked exact-a0 anchors and the metric selector is
# already contact-specific; do not force a larger candidate request here.
CONTACT_CANDIDATE_SCENES="$NUM_SCENES"
while (($#)); do case "$1" in
 --ocrap-results) OCRAP_RESULTS_ROOT="$2";shift 2;; --model-run) OCRAP_MODEL_RUN="$2";shift 2;; --external-root) EXTERNAL_RESULTS_ROOT="$2";shift 2;;
 --target-lock-root) TARGET_LOCK_ROOT="$2";shift 2;; --variant) MODEL_VARIANT="$2";shift 2;; --out) OUT="$2";shift 2;; --num-scenes) NUM_SCENES="$2";shift 2;; --gpus) CUDA_DEVICES="$2";shift 2;;
 --fps) FPS="$2";shift 2;; --trace-steps) TRACE_MAX_STEPS="$2";shift 2;; --min-duration-s) MIN_VIDEO_DURATION_S="$2"; shift 2;; --fallback-min-duration-s) FALLBACK_MIN_VIDEO_DURATION_S="$2"; shift 2;; --contact-fallback-min-duration-s) CONTACT_FALLBACK_MIN_VIDEO_DURATION_S="$2"; shift 2;; --camera) CAMERA_MODE="$2";shift 2;; --view-radius) VIEW_RADIUS_M="$2";shift 2;;
 -h|--help) usage;exit 0;; *) echo "unknown option $1" >&2;usage >&2;exit 2;; esac;done
[[ -n "$OCRAP_RESULTS_ROOT" && -n "$OCRAP_MODEL_RUN" ]] || { echo '--ocrap-results and --model-run are required' >&2; exit 2; }
[[ "$TRACE_MAX_STEPS" =~ ^[0-9]+$ && "$TRACE_MAX_STEPS" -gt 0 ]] || { echo '--trace-steps must be a positive integer' >&2; exit 2; }
if [[ "$VIDEO_FORMAT" == mp4 ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[WARN] ffmpeg not found; static paper figures will still be generated and video format falls back to auto/GIF." >&2
  VIDEO_FORMAT=auto
fi
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}" WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
export SAFE_EXTERNAL_ROOT="$EXTERNAL_RESULTS_ROOT/safe" NEAR_EXTERNAL_ROOT="$EXTERNAL_RESULTS_ROOT/near" CONTACT_EXTERNAL_ROOT="$EXTERNAL_RESULTS_ROOT/contact"
export OCRAP_RESULTS_ROOT OCRAP_MODEL_RUN TARGET_LOCK_ROOT MODEL_VARIANT OUT NUM_SCENES CUDA_DEVICES FPS TRACE_MAX_STEPS CAMERA_MODE VIEW_RADIUS_M VIDEO_FORMAT
export VIS_CANDIDATE_MULTIPLIER SAFE_CANDIDATE_SCENES NEAR_CANDIDATE_SCENES CONTACT_CANDIDATE_SCENES
export MAX_SELECTED_TIER_RANK="${MAX_SELECTED_TIER_RANK:-1}" JOBS_PER_GPU="${JOBS_PER_GPU:-3}" MAX_PARALLEL="${MAX_PARALLEL:-6}"
export MIN_VIDEO_DURATION_S FALLBACK_MIN_VIDEO_DURATION_S CONTACT_FALLBACK_MIN_VIDEO_DURATION_S

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
python - "$FINAL_CONTACT_ANCHOR_MANIFEST" "$TARGET_LOCK_ROOT/contact.json" "$VIS_CONTACT_ANCHOR_MANIFEST_FILE" "$CONTACT_HORIZON_REPORT" "$NUM_SCENES" "$TRACE_MAX_STEPS" "$CONTACT_FALLBACK_MIN_VIDEO_DURATION_S" <<'PY'
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

# Stage 2: rerun full render traces for only that candidate pool.
SELECTION_ROOT="$OUT/selection_candidates" OUT="$OUT/selective_traces" bash scripts/generate_selected_regime_traces.sh 2>&1 | tee "$OUT/logs/02_selective_traces.log"

# Stage 3: trace-aware finalization. Safe scenes with early visible OC-RAP
# off-road/overlap are rejected; late tail violations can only shorten the clip
# down to the configured fallback. Near-Contact prioritizes broad external
# collision/low-margin consensus and denser local traffic while keeping OC-RAP
# visibly collision/off-road free.
python tools/finalize_regime_visualization_selection.py \
  --candidate-selection-root "$OUT/selection_candidates" \
  --trace-root "$OUT/selective_traces" \
  --output-root "$OUT/selection" \
  --num-scenes "$NUM_SCENES" \
  --safe-min-clip-s "$FALLBACK_MIN_VIDEO_DURATION_S" \
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
TRACE_ROOT="$OUT/selective_traces" SELECTION_ROOT="$OUT/selection" OUT="$OUT/paper_figures" bash scripts/render_regime_paper_figures.sh 2>&1 | tee "$OUT/logs/05_paper_figures.log"
TRACE_ROOT="$OUT/selective_traces" SELECTION_ROOT="$OUT/selection" OUT="$OUT/videos" FORMAT="$VIDEO_FORMAT" bash scripts/render_regime_videos.sh 2>&1 | tee "$OUT/logs/06_videos.log"
python tools/audit_regime_visualization_outputs.py --root "$OUT" --expected-scenes "$NUM_SCENES" | tee "$OUT/logs/07_output_audit.log"
printf 'Paper figures complete: %s\n' "$OUT/paper_figures/PAPER_FIGURE_INDEX.json"
printf 'Visualization complete: %s\n' "$OUT/videos/REGIME_VIDEO_INDEX.json"
