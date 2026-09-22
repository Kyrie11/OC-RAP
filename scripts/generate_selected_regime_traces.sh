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

# Safe external baselines. Valid trace artifacts from a prior attempt are
# skipped; only families removed by TRACE_PREP are recomputed.
run_logged external_safe env \
  RUN="$OUT/external/safe" CHECKPOINT_ROOT="$SAFE_EXTERNAL_ROOT/checkpoints" \
  OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
  DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_NOMINAL_CONTROL=false RUN_LEGACY_SAFE=false RUN_SUPPLEMENTARY_SAFE=false \
  CL_WOMD="$SAFE_WOMD" CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$TRACE_MAX_STEPS" \
  CL_TARGET_KEYS_FILE="$SELECTION_ROOT/safe_target_keys.json" CL_RENDER_TRACE=true CL_SCENE_JOURNAL_DETAIL=full \
  JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" \
  SKIP_COMPLETE_METHODS=true CL_RESUME_FORCE=false \
  bash scripts/run_external_baselines_safe.sh

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
    JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" \
    SKIP_COMPLETE_METHODS=true CL_RESUME_FORCE=false \
    bash scripts/run_external_baselines_near.sh
fi

# Contact is independent from Near.  Run it even if Near failed so a transient
# failure cannot waste another complete rerun on the next attempt.
run_logged external_contact env \
  RUN="$OUT/external/contact" OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
  DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_LEGACY_CONTACT=false \
  CL_WOMD="$CONTACT_WOMD" CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$TRACE_MAX_STEPS" \
  CL_TARGET_KEYS_FILE="$SELECTION_ROOT/contact_target_keys.json" CL_RENDER_TRACE=true CL_SCENE_JOURNAL_DETAIL=full \
  CL_CONTACT_ANCHOR_PRELUDE_ENABLED=true CL_CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60 \
  CL_CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1 CL_CONTACT_ANCHOR_REQUIRE_FOUND=true \
  CL_CONTACT_ANCHOR_MANIFEST_FILE="$VIS_CONTACT_ANCHOR_MANIFEST_FILE" \
  JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" \
  SKIP_COMPLETE_METHODS=true CL_RESUME_FORCE=false \
  bash scripts/run_external_baselines_contact.sh

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

# Final strict contract.  Rendering starts only after every method supplies a
# synchronized, full-length trace for each selected target.
python - "$OUT" "$SELECTION_ROOT" "$TRACE_MAX_STEPS" <<'PY'
import json, math, pathlib, sys
root=pathlib.Path(sys.argv[1]); sel=pathlib.Path(sys.argv[2]); trace_max_steps=int(sys.argv[3])
methods={
'safe':['gameformer_lite','plantf','pluto','pdm_closed','pdm_hybrid','idm'],
'near':['marc_lite','racp_lite','robust_scenario_mpc','predictive_safety_filter','dr_cvar_safety_filter','conformal_predictive_safety_filter'],
'contact':['postimpact_mpc_lite','post_crash_braking','postimpact_motion_tvlqr','post_collision_restoration','compensatory_postimpact_mpc','robust_postimpact_control']}
errors=[]; regimes={}
for r,ms in methods.items():
    selection=json.loads((sel/f'{r}_selection.json').read_text())
    requested=[str(x) for x in json.loads((sel/f'{r}_target_keys.json').read_text())['target_keys']]
    requested_set=set(requested); dt=float(selection.get('metric_dt_s',0.1) or 0.1)
    selected_by_key={str(x['target_key']):x for x in (selection.get('selected') or [])}; req_steps={}
    for key in requested:
        item=selected_by_key.get(key) or {}; clip=float(item.get('clip_duration_s',selection.get('selected_clip_duration_s',0.0)) or 0.0)
        steps=int(math.ceil(clip/dt-1e-9)); req_steps[key]=steps
        if steps<=0: errors.append(f'invalid selected clip duration {r}/{key}: clip={clip} dt={dt}')
        if steps>trace_max_steps: errors.append(f'selected clip exceeds trace cap {r}/{key}: required_steps={steps} cap={trace_max_steps}')
    paths={'ocrap':root/'ocrap'/r/'closed_loop_ocrap.json.scenes.jsonl', **{m:root/'external'/r/f'closed_loop_{m}.json.scenes.jsonl' for m in ms}}
    counts={}; trace_starts={k:{} for k in requested}
    for m,p in paths.items():
        if not p.is_file(): errors.append(f'missing journal {r}/{m}: {p}'); continue
        seen={}
        for lineno,line in enumerate(p.open(),1):
            if not line.strip(): continue
            x=json.loads(line); scene=x.get('scene',x); k=str(scene.get('target_key') or x.get('resume_key') or '')
            if k.startswith('target:'): k=k[len('target:'):]
            if k in seen: errors.append(f'duplicate selected target {r}/{m}/{k}: lines {seen[k]},{lineno}')
            seen[k]=lineno
            if k not in requested_set: continue
            trace=scene.get('render_trace') or []; required_frames=req_steps[k]+1
            if not trace: errors.append(f'no render_trace {r}/{m}/{k}; selected journal must be full')
            else:
                try: trace_starts[k][m]=int(trace[0]['time_index'])
                except Exception: errors.append(f'render_trace lacks integer start time {r}/{m}/{k}')
                if len(trace)<required_frames: errors.append(f'short render_trace {r}/{m}/{k}: frames={len(trace)} required>={required_frames}')
        miss=sorted(requested_set-set(seen)); extra=sorted(set(seen)-requested_set)
        if miss: errors.append(f'unresolved selected targets {r}/{m}: {miss}')
        if extra: errors.append(f'unexpected selected-trace targets {r}/{m}: {extra}')
        counts[m]=len(seen)
    for key in requested:
        starts=trace_starts.get(key,{})
        if starts and len(set(starts.values()))!=1: errors.append(f'model trace starts not synchronized {r}/{key}: {starts}')
        item=selected_by_key.get(key) or {}; field='contact_anchor_time_index' if r=='contact' else 'target_time_index'; expected=item.get(field)
        if starts and expected is not None and next(iter(starts.values()))!=int(expected):
            errors.append(f'trace start mismatch {r}/{key}: got={next(iter(starts.values()))} expected_{field}={expected}')
    regimes[r]={'requested':len(requested),'journal_counts':counts,'required_rollout_steps_by_target':req_steps,'max_required_rollout_steps':max(req_steps.values()) if req_steps else 0,'selected_clip_duration_s':selection.get('selected_clip_duration_s'),'duration_selection_mode':selection.get('duration_selection_mode'),'duration_source':selection.get('duration_source')}
doc={'event':'selected_regime_trace_contract_v126','valid':not errors,'errors':errors,'trace_max_steps':trace_max_steps,'regimes':regimes}
(root/'TRACE_CONTRACT.json').write_text(json.dumps(doc,indent=2)+'\n'); print(json.dumps(doc,indent=2)); raise SystemExit(0 if not errors else 30)
PY
