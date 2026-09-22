#!/usr/bin/env bash
# Rerun only selected qualitative targets with render_trace=true.
# Population metrics are never changed.  The pipeline is resumable: complete
# trace artifacts are reused, while only missing/invalid trace families are
# removed and regenerated.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${OCRAP_MODEL_RUN:?set OCRAP_MODEL_RUN}"
: "${MODEL_VARIANT:=balanced}"
: "${SAFE_EXTERNAL_ROOT:=/home/senzeyu2/code/OC-RAP/runs/external_baselines/safe}"
: "${NEAR_EXTERNAL_ROOT:=/home/senzeyu2/code/OC-RAP/runs/external_baselines/near}"
: "${CONTACT_EXTERNAL_ROOT:=/home/senzeyu2/code/OC-RAP/runs/external_baselines/contact}"
: "${SELECTION_ROOT:?set SELECTION_ROOT to .../selection}"
: "${INPUT_CONTRACT:=$(dirname "$SELECTION_ROOT")/provenance/VISUALIZATION_INPUT_CONTRACT.json}"
: "${OUT:=$(dirname "$SELECTION_ROOT")/selective_traces}"
: "${CUDA_DEVICES:=0,1}"
: "${TRACE_MAX_STEPS:=60}"
: "${VIS_CONTACT_ANCHOR_MANIFEST_FILE:=$(dirname "$SELECTION_ROOT")/provenance/contact_anchor_manifest.json}"
# False by default: a failure in Near/Contact must not throw away already-valid
# Safe or OC-RAP traces.  Set true only when an intentional from-scratch rerun
# is desired.
: "${CLEAN_TRACE_OUTPUT:=false}"
: "${JOBS_PER_GPU:=3}"
: "${MAX_PARALLEL:=6}"

[[ -f "$INPUT_CONTRACT" ]] || { echo "Missing visualization input contract: $INPUT_CONTRACT" >&2; exit 30; }
[[ -s "$VIS_CONTACT_ANCHOR_MANIFEST_FILE" ]] || { echo "Missing frozen Contact visualization anchor manifest: $VIS_CONTACT_ANCHOR_MANIFEST_FILE" >&2; exit 30; }
python - "$INPUT_CONTRACT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d.get('valid') is True, d.get('errors')
PY

if [[ "$CLEAN_TRACE_OUTPUT" == true ]]; then
  rm -rf "$OUT/ocrap" "$OUT/external"
fi
mkdir -p "$OUT/ocrap" "$OUT/external" "$OUT/logs"

# Before launching anything, retain only trace artifacts that are actually
# renderable for the current selection.  Metrics-only journals from an earlier
# buggy qualitative run are removed as a family so the launcher cannot skip
# them merely because aggregate metrics are complete.
python tools/prepare_selected_trace_reruns.py \
  --trace-root "$OUT" --selection-root "$SELECTION_ROOT" --clean-invalid \
  --output "$OUT/TRACE_PREP.json"

# TRACE_PREP is the authority for resumability.  Do not even invoke an external
# regime launcher when all six selected-trace families are already renderable.
# This avoids unnecessary preflights and, importantly, avoids scheduler feature
# checks on older Bash installations when there is no work to schedule.
trace_pending_count() {
  local regime="$1" method_class="$2"
  python - "$OUT/TRACE_PREP.json" "$regime" "$method_class" <<'PY_PENDING'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8')); regime=sys.argv[2]; cls=sys.argv[3]
rows=[x for x in d.get('methods',[]) if (regime=='*' or x.get('regime')==regime)]
if cls=='ocrap': rows=[x for x in rows if x.get('method')=='ocrap']
elif cls=='external': rows=[x for x in rows if x.get('method')!='ocrap']
else: raise SystemExit(f'unknown method class: {cls}')
print(sum(not bool(x.get('valid_reusable')) for x in rows))
PY_PENDING
}
OCRAP_PENDING="$(trace_pending_count '*' ocrap)"
SAFE_EXTERNAL_PENDING="$(trace_pending_count safe external)"
NEAR_EXTERNAL_PENDING="$(trace_pending_count near external)"
CONTACT_EXTERNAL_PENDING="$(trace_pending_count contact external)"
echo "[VIS-TRACE] pending trace families: ocrap=$OCRAP_PENDING safe_external=$SAFE_EXTERNAL_PENDING near_external=$NEAR_EXTERNAL_PENDING contact_external=$CONTACT_EXTERNAL_PENDING"

# Replay from the collection resolved from bucket provenance.
readarray -t WOMD_SPECS < <(python - "$INPUT_CONTRACT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))['canonical_replay']
for r in ('safe','near','contact'):
    x=d[r]
    assert x.get('valid') is True and x.get('womd_spec'), (r,x)
    print(x['womd_spec'])
PY
)
SAFE_WOMD="${WOMD_SPECS[0]}"; NEAR_WOMD="${WOMD_SPECS[1]}"; CONTACT_WOMD="${WOMD_SPECS[2]}"

failures=()
run_logged() {
  local name="$1"; shift
  echo "[VIS-TRACE] stage=$name"
  if "$@" > >(tee "$OUT/logs/${name}.log") 2>&1; then
    return 0
  else
    local rc=$?
    echo "[VIS-TRACE][ERROR] stage=$name rc=$rc" >&2
    failures+=("$name:$rc")
    return 0
  fi
}

# Frozen OC-RAP model.  Explicitly request a full JSONL journal: the
# three-regime wrapper historically pre-filled metrics-only storage even when
# RENDER_* was true, which silently discarded render_trace.
if (( OCRAP_PENDING > 0 )); then
  run_logged ocrap_selected env \
    MODEL_RUN="$OCRAP_MODEL_RUN" MODEL_VARIANT="$MODEL_VARIANT" \
    OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
    OUT="$OUT/ocrap" MAX_SCENARIOS=0 MAX_STEPS="$TRACE_MAX_STEPS" \
    SAFE_WOMD="$SAFE_WOMD" NEAR_WOMD="$NEAR_WOMD" CONTACT_WOMD="$CONTACT_WOMD" \
    SAFE_TARGET_KEYS_FILE="$SELECTION_ROOT/safe_target_keys.json" \
    NEAR_TARGET_KEYS_FILE="$SELECTION_ROOT/near_target_keys.json" \
    CONTACT_TARGET_KEYS_FILE="$SELECTION_ROOT/contact_target_keys.json" \
    CONTACT_ANCHOR_PRELUDE_ENABLED=true CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60 \
    CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1 CONTACT_ANCHOR_REQUIRE_FOUND=true \
    CONTACT_ANCHOR_MANIFEST_FILE="$VIS_CONTACT_ANCHOR_MANIFEST_FILE" \
    RENDER_SAFE=true RENDER_NEAR=true RENDER_CONTACT=true \
    SCENE_JOURNAL_DETAIL=full RESULT_SCENE_DETAIL=metrics \
    SAFE_LABEL_MODE=fast NEAR_LABEL_MODE=fast CONTACT_LABEL_MODE=fast \
    SKIP_COMPLETE_REGIMES=true RESUME_FORCE=false FINALIZE_COMPLETE_JOURNALS=true \
    bash scripts/run_ocrap_three_regime_evaluation.sh
else
  echo "[VIS-TRACE][REUSE] ocrap_selected: all three regime trace families are already renderable"
fi

# Safe external baselines. Valid trace artifacts from a prior attempt are
# skipped; only families removed by TRACE_PREP are recomputed.
if (( SAFE_EXTERNAL_PENDING > 0 )); then
  run_logged external_safe env \
  RUN="$OUT/external/safe" CHECKPOINT_ROOT="$SAFE_EXTERNAL_ROOT/checkpoints" \
  OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
  DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_NOMINAL_CONTROL=false RUN_LEGACY_SAFE=false RUN_SUPPLEMENTARY_SAFE=false \
  CL_WOMD="$SAFE_WOMD" CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$TRACE_MAX_STEPS" \
  CL_TARGET_KEYS_FILE="$SELECTION_ROOT/safe_target_keys.json" CL_RENDER_TRACE=true CL_SCENE_JOURNAL_DETAIL=full \
  JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" USE_DYNAMIC_SCHEDULER=auto \
  SKIP_COMPLETE_METHODS=true CL_RESUME_FORCE=false \
  bash scripts/run_external_baselines_safe.sh
else
  echo "[VIS-TRACE][REUSE] external_safe: all six selected-trace families are already renderable"
fi

if (( NEAR_EXTERNAL_PENDING > 0 )); then
# Freeze the *policy calibration* from the population Near experiment.  The
# qualitative clip horizon may be 60 steps, but CPSF's conformal mission horizon
# was calibrated for the publication population run (40 steps by default).  Do
# not reject/refit that policy merely because visualization asks for a longer
# rollout.  Inject the exact stored intervals instead.
NEAR_CALIBRATION="$NEAR_EXTERNAL_ROOT/conformal_calibration.json"
if [[ ! -s "$NEAR_CALIBRATION" ]]; then
  echo "Missing frozen Near conformal calibration: $NEAR_CALIBRATION" >&2
  failures+=("near_calibration_missing:2")
  NEAR_CONFORMAL_INTERVALS=""
  NEAR_CONFORMAL_MISSION_HORIZON=""
else
  readarray -t _CAL < <(python - "$NEAR_CALIBRATION" "$OUT/NEAR_CPSF_VISUALIZATION_CALIBRATION.json" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); d=json.loads(p.read_text())
vals=[float(x) for x in d['conformal_prediction_intervals_m']]
T=int(d['mission_horizon']); H=int(d['prediction_horizon'])
assert len(vals)==H and all(x>=0 for x in vals)
doc={
 'event':'near_cpsf_visualization_calibration_freeze_v126',
 'source':str(p), 'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'prediction_horizon':H, 'mission_horizon':T,
 'delta':d.get('delta'), 'calibration_unit':d.get('calibration_unit'),
 'conformal_prediction_intervals_m':vals,
 'note':'Visualization rollout length is independent of the frozen CPSF policy calibration horizon.'
}
out.write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps(vals,separators=(',',':'))); print(T)
PY
)
  NEAR_CONFORMAL_INTERVALS="${_CAL[0]}"
  NEAR_CONFORMAL_MISSION_HORIZON="${_CAL[1]}"
fi

if [[ -n "$NEAR_CONFORMAL_INTERVALS" ]]; then
  run_logged external_near env \
    RUN="$OUT/external/near" OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
    DO_TRAIN=false DO_CALIBRATE=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_ORACLE_CLOSED_LOOP=false RUN_LEGACY_NEAR=false RUN_SUPPLEMENTARY_NEAR=false \
    CONFORMAL_CALIBRATION="$NEAR_CALIBRATION" CONFORMAL_INTERVALS="$NEAR_CONFORMAL_INTERVALS" \
    CONFORMAL_MISSION_HORIZON="$NEAR_CONFORMAL_MISSION_HORIZON" \
    CL_WOMD="$NEAR_WOMD" CL_LABEL_MODE=fast CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$TRACE_MAX_STEPS" \
    CL_TARGET_KEYS_FILE="$SELECTION_ROOT/near_target_keys.json" CL_RENDER_TRACE=true CL_SCENE_JOURNAL_DETAIL=full \
    JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" USE_DYNAMIC_SCHEDULER=auto \
    SKIP_COMPLETE_METHODS=true CL_RESUME_FORCE=false \
    bash scripts/run_external_baselines_near.sh
fi
else
  echo "[VIS-TRACE][REUSE] external_near: all six selected-trace families are already renderable"
fi

# Contact is independent from Near.  Run it even if Near failed so a transient
# failure cannot waste another complete rerun on the next attempt.
if (( CONTACT_EXTERNAL_PENDING > 0 )); then
  run_logged external_contact env \
  RUN="$OUT/external/contact" OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
  DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_LEGACY_CONTACT=false \
  CL_WOMD="$CONTACT_WOMD" CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$TRACE_MAX_STEPS" \
  CL_TARGET_KEYS_FILE="$SELECTION_ROOT/contact_target_keys.json" CL_RENDER_TRACE=true CL_SCENE_JOURNAL_DETAIL=full \
  CL_CONTACT_ANCHOR_PRELUDE_ENABLED=true CL_CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60 \
  CL_CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1 CL_CONTACT_ANCHOR_REQUIRE_FOUND=true \
  CL_CONTACT_ANCHOR_MANIFEST_FILE="$VIS_CONTACT_ANCHOR_MANIFEST_FILE" \
  JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" USE_DYNAMIC_SCHEDULER=auto \
  SKIP_COMPLETE_METHODS=true CL_RESUME_FORCE=false \
  bash scripts/run_external_baselines_contact.sh
else
  echo "[VIS-TRACE][REUSE] external_contact: all six selected-trace families are already renderable"
fi

python - "$OUT/TRACE_RERUN_STATUS.json" "${failures[*]-}" <<'PY'
import json,pathlib,sys
fails=[x for x in sys.argv[2].split() if x]
d={'event':'selected_trace_rerun_status_v126','valid':not fails,'failures':fails}
pathlib.Path(sys.argv[1]).write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d))
PY
if ((${#failures[@]})); then
  echo "Selected trace rerun stages failed: ${failures[*]}" >&2
  echo "Inspect $OUT/logs/*.log and $OUT/TRACE_RERUN_STATUS.json. Valid completed trace families were retained." >&2
  exit 30
fi

# Final strict contract. Rendering starts only after every method supplies a
# synchronized, full-length trace for each selected target.
python tools/check_selected_trace_contract.py \
  --trace-root "$OUT" --selection-root "$SELECTION_ROOT" \
  --trace-max-steps "$TRACE_MAX_STEPS" --output "$OUT/TRACE_CONTRACT.json"
