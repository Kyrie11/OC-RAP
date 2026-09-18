#!/usr/bin/env bash
# One-command final submission suite:
# clean known-invalid final outputs -> freeze target locks -> run five accuracy
# suites concurrently -> render continuous 6 s qualitative videos -> profile all
# latency in isolation -> build tables -> package downloadable final_results.zip.
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; LATENCY_GPU="${LATENCY_GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-$GPU0,$GPU1}"
MODEL_RUN="${MODEL_RUN:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
PRETRAINED_BASELINE_ROOT="${PRETRAINED_BASELINE_ROOT:-$BASE_OUT/external_baselines_v48_111}"
OCRAP_OUT="${OCRAP_FINAL_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"
OCRAP_LATENCY_OUT="${OCRAP_LATENCY_OUT:-$BASE_OUT/ocrap_v48_124_latency_isolated}"
EXTERNAL_OUT="${FINAL_EXTERNAL_BASELINE_OUT:-$BASE_OUT/external_baselines_v48_124_final_v2}"
EXTERNAL_LATENCY_OUT="${FINAL_EXTERNAL_BASELINE_LATENCY_OUT:-${EXTERNAL_OUT}_latency_isolated}"
ABLATION_OUT="${OUT_ROOT:-$BASE_OUT/ocrap_v48_124_final_ablations}"
ABLATION_LATENCY_ROOT="${ABLATION_LATENCY_ROOT:-$ABLATION_OUT/latency_isolated}"
FINAL_TABLE_DIR="${FINAL_TABLE_OUT:-$BASE_OUT/final_regime_comparison_tables_v48_124}"
VIS_OUT="${VIS_OUT:-$BASE_OUT/regime_visualization_v48_124_final}"
FINAL_ZIP="${FINAL_RESULTS_ZIP:-$BASE_OUT/final_results.zip}"
STATUS_JSON="${FINAL_SUITE_STATUS_JSON:-$BASE_OUT/final_submission_suite_status.json}"
LOG_ROOT="${FINAL_SUITE_LOG_ROOT:-$REPO/logs/final_submission_suite}"

WOMD_ROLE="${WOMD_ROLE:-validation}"
MAX_SCENARIOS="${MAX_SCENARIOS:-0}"; MAX_STEPS="${MAX_STEPS:-40}"
NUM_CANDIDATES="${NUM_CANDIDATES:-24}"; NUM_RECOVERY_OPTIONS="${NUM_RECOVERY_OPTIONS:-12}"
VARIANTS="${VARIANTS:-balanced,precision}"; ABLATION_SET="${ABLATION_SET:-main}"
METRIC_SEMANTICS_VERSION="${METRIC_SEMANTICS_VERSION:-publication_v55_signed_clearance_unclipped_v1}"
CLEAN_OLD_RESULTS="${CLEAN_OLD_RESULTS:-true}"
BASELINE_JOBS_PER_GPU="${BASELINE_JOBS_PER_GPU:-1}"
BASELINE_MAX_PARALLEL="${BASELINE_MAX_PARALLEL:-1}"
VIS_JOBS_PER_GPU="${VIS_JOBS_PER_GPU:-3}"; VIS_MAX_PARALLEL="${VIS_MAX_PARALLEL:-6}"
VIS_NUM_SCENES="${VIS_NUM_SCENES:-3}"; VIS_TRACE_STEPS="${VIS_TRACE_STEPS:-60}"
VIS_MIN_DURATION_S="${VIS_MIN_DURATION_S:-6.0}"; VIS_FPS="${VIS_FPS:-10}"
VIS_CAMERA="${VIS_CAMERA:-fixed}"; VIS_VIEW_RADIUS="${VIS_VIEW_RADIUS:-35}"
MAX_SELECTED_TIER_RANK="${MAX_SELECTED_TIER_RANK:-1}"

bool_true(){ case "${1,,}" in 1|true|yes|on) return 0;; *) return 1;; esac; }
mkdir -p "$BASE_OUT" "$LOG_ROOT" "$REPO/logs"

status(){
  local phase="$1" state="$2" detail="${3:-}"
  python - "$STATUS_JSON" "$phase" "$state" "$detail" <<'PY'
import datetime,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); phase,state,detail=sys.argv[2:]
try: d=json.loads(p.read_text()) if p.is_file() else {}
except Exception: d={}
d.update({'schema_version':1,'phase':phase,'status':state,'detail':detail,
          'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'phase':phase,'status':state,'detail':detail},ensure_ascii=False))
PY
}

safe_rm_tree(){
  local path="$1"
  python - "$BASE_OUT" "$path" <<'PY'
import pathlib,shutil,sys
base=pathlib.Path(sys.argv[1]).resolve(); p=pathlib.Path(sys.argv[2]).resolve()
if p.parent != base:
    raise SystemExit(f'refuse cleanup outside direct runs child: base={base} path={p}')
if p == base or str(p) in {'/','.'}:
    raise SystemExit(f'refuse unsafe cleanup path: {p}')
if p.exists():
    print(f'[CLEAN] rm -rf {p}')
    shutil.rmtree(p)
else:
    print(f'[CLEAN] already absent: {p}')
PY
}

status cleanup running "remove explicitly invalid final accuracy/latency outputs"
if bool_true "$CLEAN_OLD_RESULTS"; then
  # These are the four paths explicitly identified as carrying invalid results.
  safe_rm_tree "$BASE_OUT/ocrap_v48_124_latency_isolated"
  safe_rm_tree "$BASE_OUT/ocrap_v48_124_final_characterization"
  safe_rm_tree "$BASE_OUT/external_baselines_v48_124_final_v2_latency_isolated"
  safe_rm_tree "$BASE_OUT/external_baselines_v48_124_final_v2"
  # Fresh derived presentation products must not contain stale rows/videos from
  # those removed runs.  The ablation root is deliberately NOT deleted; its
  # strict launcher archives/replaces incompatible artifacts itself.
  rm -rf -- "$FINAL_TABLE_DIR" "$VIS_OUT"
  rm -f -- "$BASE_OUT/safe_compare.csv" "$BASE_OUT/near_compare.csv" "$BASE_OUT/contact_compare.csv" \
    "$BASE_OUT/ablation_results.csv" "$BASE_OUT/ablation_results_index.json" "$FINAL_ZIP"
  rm -f -- "$BASE_OUT"/ablation_balanced_*.csv "$BASE_OUT"/ablation_precision_*.csv
fi
status cleanup complete "four invalid run directories removed; ablation root retained"

status target_locks running "freeze shared observation-legal locks and Contact exact-a0 anchor"
env BASE_OUT="$BASE_OUT" OCRAP_FINAL_CHARACTERIZATION_OUT="$OCRAP_OUT" WOMD_ROLE="$WOMD_ROLE" FINAL_MAX_STEPS="$MAX_STEPS" \
  CONTACT_ANCHOR_GPU="$GPU0" bash scripts/build_final_observation_legal_target_locks.sh \
  >"$LOG_ROOT/target_locks.log" 2>&1
status target_locks complete "$OCRAP_OUT/target_keys"

TARGET_ROOT="$OCRAP_OUT/target_keys"
ANCHOR_MANIFEST="$OCRAP_OUT/contact_anchor/contact_anchor_manifest.json"
for r in safe near contact; do [[ -s "$TARGET_ROOT/$r.json" ]] || { echo "missing target lock $TARGET_ROOT/$r.json" >&2; exit 30; }; done
[[ -s "$ANCHOR_MANIFEST" ]] || { echo "missing Contact anchor manifest $ANCHOR_MANIFEST" >&2; exit 30; }

status accuracy running "safe/near/contact external + full OC-RAP + ablation concurrently"
declare -a PIDS=() NAMES=()
launch(){ local name="$1" log="$2" pid; shift 2; ( "$@" ) >"$log" 2>&1 & pid=$!; PIDS+=("$pid"); NAMES+=("$name"); echo "[LAUNCH] $name pid=$pid log=$log"; }

launch external_safe "$LOG_ROOT/external_safe.log" env \
  BASE_OUT="$BASE_OUT" OCRAP_FINAL_CHARACTERIZATION_OUT="$OCRAP_OUT" FINAL_EXTERNAL_BASELINE_OUT="$EXTERNAL_OUT" \
  PRETRAINED_BASELINE_ROOT="$PRETRAINED_BASELINE_ROOT" GPU_LIST="$CUDA_DEVICES" JOBS_PER_GPU="$BASELINE_JOBS_PER_GPU" MAX_PARALLEL="$BASELINE_MAX_PARALLEL" \
  PROFILE_LATENCY=false WOMD_ROLE="$WOMD_ROLE" FINAL_MAX_STEPS="$MAX_STEPS" USE_DYNAMIC_SCHEDULER=auto \
  bash scripts/run_final_external_baselines.sh safe
launch external_near "$LOG_ROOT/external_near.log" env \
  BASE_OUT="$BASE_OUT" OCRAP_FINAL_CHARACTERIZATION_OUT="$OCRAP_OUT" FINAL_EXTERNAL_BASELINE_OUT="$EXTERNAL_OUT" \
  PRETRAINED_BASELINE_ROOT="$PRETRAINED_BASELINE_ROOT" GPU_LIST="$CUDA_DEVICES" JOBS_PER_GPU="$BASELINE_JOBS_PER_GPU" MAX_PARALLEL="$BASELINE_MAX_PARALLEL" \
  PROFILE_LATENCY=false WOMD_ROLE="$WOMD_ROLE" FINAL_MAX_STEPS="$MAX_STEPS" USE_DYNAMIC_SCHEDULER=auto \
  bash scripts/run_final_external_baselines.sh near
launch external_contact "$LOG_ROOT/external_contact.log" env \
  BASE_OUT="$BASE_OUT" OCRAP_FINAL_CHARACTERIZATION_OUT="$OCRAP_OUT" FINAL_EXTERNAL_BASELINE_OUT="$EXTERNAL_OUT" \
  PRETRAINED_BASELINE_ROOT="$PRETRAINED_BASELINE_ROOT" GPU_LIST="$CUDA_DEVICES" JOBS_PER_GPU="$BASELINE_JOBS_PER_GPU" MAX_PARALLEL="$BASELINE_MAX_PARALLEL" \
  PROFILE_LATENCY=false WOMD_ROLE="$WOMD_ROLE" FINAL_MAX_STEPS="$MAX_STEPS" USE_DYNAMIC_SCHEDULER=auto \
  bash scripts/run_final_external_baselines.sh contact
launch full_ocrap "$LOG_ROOT/main_full.log" env \
  BASE_OUT="$BASE_OUT" OCRAP_FINAL_CHARACTERIZATION_OUT="$OCRAP_OUT" OCRAP_LATENCY_OUT="$OCRAP_LATENCY_OUT" \
  MODEL_RUN="$MODEL_RUN" GPU0="$GPU0" GPU1="$GPU1" BUILD_TARGET_LOCKS=false PROFILE_LATENCY=false \
  WOMD_ROLE="$WOMD_ROLE" MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" \
  bash scripts/run_final_locked_three_regime_characterization.sh
launch ablation "$LOG_ROOT/ablation.log" env \
  BASE_OUT="$BASE_OUT" OUT_ROOT="$ABLATION_OUT" MODEL_RUN="$MODEL_RUN" FULL_RUN_ROOT="$OCRAP_OUT/ocrap" \
  TARGET_LOCK_CHARACTERIZATION_OUT="$OCRAP_OUT" FINAL_TARGET_LOCK_ROOT="$TARGET_ROOT" CONTACT_ANCHOR_MANIFEST_FILE="$ANCHOR_MANIFEST" \
  GPU0="$GPU0" GPU1="$GPU1" CUDA_DEVICES="$CUDA_DEVICES" VARIANTS="$VARIANTS" MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" \
  NUM_CANDIDATES="$NUM_CANDIDATES" NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" WOMD_ROLE="$WOMD_ROLE" ABLATION_SET="$ABLATION_SET" \
  BUILD_TARGET_LOCKS=false PROFILE_ISOLATED_LATENCY=false BUILD_TABLES=false SKIP_COMPLETE=true \
  bash scripts/run_submission_ablations.sh

set +e
failed=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}"; rc=$?
  echo "[DONE] ${NAMES[$i]} rc=$rc"
  if ((rc != 0)); then failed=1; fi
done
set -e
if ((failed)); then status accuracy failed "one or more concurrent suites failed; inspect $LOG_ROOT"; exit 30; fi
status accuracy complete "all five accuracy suites complete"

# Visualization is deliberately after *all* accuracy jobs so its 6-second replay
# is not competing with ablation/full/baseline workers for GPU memory.  It uses
# the same selected target for every frame of a clip; no scene stitching.
status visualization running "continuous ${VIS_MIN_DURATION_S}s clips, ${VIS_TRACE_STEPS} rollout steps"
env BASE_OUT="$BASE_OUT" JOBS_PER_GPU="$VIS_JOBS_PER_GPU" MAX_PARALLEL="$VIS_MAX_PARALLEL" MAX_SELECTED_TIER_RANK="$MAX_SELECTED_TIER_RANK" \
  MIN_VIDEO_DURATION_S="$VIS_MIN_DURATION_S" FALLBACK_MIN_VIDEO_DURATION_S="$VIS_MIN_DURATION_S" VIS_CONTACT_MIN_POST_STEPS="$VIS_TRACE_STEPS" \
  bash scripts/build_regime_visualizations.sh \
    --ocrap-results "$OCRAP_OUT/ocrap/balanced" --model-run "$MODEL_RUN" --external-root "$EXTERNAL_OUT" \
    --target-lock-root "$TARGET_ROOT" --variant balanced --out "$VIS_OUT" --num-scenes "$VIS_NUM_SCENES" \
    --gpus "$CUDA_DEVICES" --fps "$VIS_FPS" --trace-steps "$VIS_TRACE_STEPS" --min-duration-s "$VIS_MIN_DURATION_S" \
    --camera "$VIS_CAMERA" --view-radius "$VIS_VIEW_RADIUS" \
  >"$LOG_ROOT/visualization.log" 2>&1
status visualization complete "$VIS_OUT"

# Publication latency is intentionally serialized after every accuracy/visualization
# process has exited.  This preserves the single-process/single-GPU contract.
status latency running "isolated single-process/single-GPU profiling"
env BASE_OUT="$BASE_OUT" MODEL_RUN="$MODEL_RUN" OCRAP_FINAL_CHARACTERIZATION_OUT="$OCRAP_OUT" OCRAP_LATENCY_OUT="$OCRAP_LATENCY_OUT" \
  LATENCY_GPU="$LATENCY_GPU" WOMD_ROLE="$WOMD_ROLE" MAX_STEPS="$MAX_STEPS" METRIC_SEMANTICS_VERSION="$METRIC_SEMANTICS_VERSION" \
  CONTACT_ANCHOR_MANIFEST_FILE="$ANCHOR_MANIFEST" bash scripts/profile_ocrap_latency.sh all \
  >"$LOG_ROOT/latency_ocrap.log" 2>&1

# The accuracy launcher wrote its characterization index before the intentionally
# delayed isolated-latency phase.  Attach the now-complete latency index so the
# final OC-RAP index describes the same artifacts consumed by the tables.
python - "$OCRAP_OUT/FINAL_CHARACTERIZATION_INDEX.json" "$OCRAP_LATENCY_OUT/LATENCY_INDEX.json" <<'PYIDX'
import hashlib,json,pathlib,sys
idx=pathlib.Path(sys.argv[1]); lat=pathlib.Path(sys.argv[2])
if not idx.is_file() or not lat.is_file():
    raise SystemExit(f'missing characterization/latency index: {idx} {lat}')
d=json.loads(idx.read_text())
d['isolated_latency_index']={'path':str(lat.resolve()),'sha256':hashlib.sha256(lat.read_bytes()).hexdigest(),'size':lat.stat().st_size}
d['status']='FINAL_LOCKED_CHARACTERIZATION_COMPLETE_WITH_ISOLATED_LATENCY'
idx.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PYIDX

for regime in safe near contact; do
  contact_env=()
  if [[ "$regime" == contact ]]; then
    contact_env=(CL_CONTACT_ANCHOR_PRELUDE_ENABLED=true CL_CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60 CL_CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1 CL_CONTACT_ANCHOR_REQUIRE_FOUND=true CL_CONTACT_ANCHOR_MANIFEST_FILE="$ANCHOR_MANIFEST")
  fi
  env CL_TARGET_KEYS_FILE="$TARGET_ROOT/$regime.json" CL_METRIC_SEMANTICS_VERSION="$METRIC_SEMANTICS_VERSION" \
    RUN_SUPPLEMENTARY_SAFE=true RUN_SUPPLEMENTARY_NEAR=true "${contact_env[@]}" \
    bash scripts/profile_external_baselines_latency.sh --regime "$regime" --source-run "$EXTERNAL_OUT" --out "$EXTERNAL_LATENCY_OUT" \
      --gpu "$LATENCY_GPU" --max-scenarios 0 --womd-role "$WOMD_ROLE" \
    >"$LOG_ROOT/latency_external_${regime}.log" 2>&1
done

# Re-enter the ablation launcher in reuse mode: strict accuracy checks skip valid
# arms, then only the missing isolated latency artifacts are profiled and tables built.
env BASE_OUT="$BASE_OUT" OUT_ROOT="$ABLATION_OUT" LATENCY_ROOT="$ABLATION_LATENCY_ROOT" MODEL_RUN="$MODEL_RUN" FULL_RUN_ROOT="$OCRAP_OUT/ocrap" \
  TARGET_LOCK_CHARACTERIZATION_OUT="$OCRAP_OUT" FINAL_TARGET_LOCK_ROOT="$TARGET_ROOT" CONTACT_ANCHOR_MANIFEST_FILE="$ANCHOR_MANIFEST" \
  GPU0="$GPU0" GPU1="$GPU1" CUDA_DEVICES="$CUDA_DEVICES" LATENCY_GPU="$LATENCY_GPU" VARIANTS="$VARIANTS" MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" \
  NUM_CANDIDATES="$NUM_CANDIDATES" NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" WOMD_ROLE="$WOMD_ROLE" ABLATION_SET="$ABLATION_SET" \
  BUILD_TARGET_LOCKS=false PROFILE_ISOLATED_LATENCY=true BUILD_TABLES=true SKIP_COMPLETE=true \
  bash scripts/run_submission_ablations.sh >"$LOG_ROOT/latency_ablation_and_tables.log" 2>&1
status latency complete "all publication latency artifacts complete"

status tables running "build three regime comparisons and six authoritative ablation tables"
env BASE_OUT="$BASE_OUT" OCRAP_FINAL_CHARACTERIZATION_OUT="$OCRAP_OUT" OCRAP_LATENCY_OUT="$OCRAP_LATENCY_OUT" \
  FINAL_EXTERNAL_BASELINE_OUT="$EXTERNAL_OUT" FINAL_EXTERNAL_BASELINE_LATENCY_OUT="$EXTERNAL_LATENCY_OUT" FINAL_TABLE_OUT="$FINAL_TABLE_DIR" \
  bash scripts/build_final_regime_comparison_tables.sh >"$LOG_ROOT/tables_regime.log" 2>&1
cp -f "$FINAL_TABLE_DIR/safe/safe_comparison.csv" "$BASE_OUT/safe_compare.csv"
cp -f "$FINAL_TABLE_DIR/near/near_comparison.csv" "$BASE_OUT/near_compare.csv"
cp -f "$FINAL_TABLE_DIR/contact/contact_comparison.csv" "$BASE_OUT/contact_compare.csv"
python tools/export_final_ablation_csvs.py --submission-table-root "$ABLATION_OUT/submission_tables" --output-root "$BASE_OUT" \
  >"$LOG_ROOT/tables_ablation_export.log" 2>&1
status tables complete "root CSVs exported under $BASE_OUT"

status package running "$FINAL_ZIP"
python tools/package_final_submission_results.py --runs-root "$BASE_OUT" --visualization-root "$VIS_OUT" --output "$FINAL_ZIP" \
  >"$LOG_ROOT/package.log" 2>&1
status package complete "$FINAL_ZIP"

python - "$FINAL_ZIP" <<'PY'
import hashlib,pathlib,sys
p=pathlib.Path(sys.argv[1]); h=hashlib.sha256(p.read_bytes()).hexdigest()
print(f'FINAL RESULTS: {p}')
print(f'SHA256: {h}')
PY

echo "Final suite complete."
echo "  Safe table:       $BASE_OUT/safe_compare.csv"
echo "  Near table:       $BASE_OUT/near_compare.csv"
echo "  Contact table:    $BASE_OUT/contact_compare.csv"
echo "  Ablation split:   $BASE_OUT/ablation_{balanced,precision}_{safe,near,contact}.csv"
echo "  Ablation union:   $BASE_OUT/ablation_results.csv"
echo "  Visualization:    $VIS_OUT"
echo "  Download package: $FINAL_ZIP"
