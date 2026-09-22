#!/usr/bin/env bash
# Evaluate only the selected targets with render_trace=true. Full population metrics
# remain those used by the selection journals; this longer trace is animation-only.
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
: "${CLEAN_TRACE_OUTPUT:=true}"
: "${ALLOW_DIAGNOSTIC_RC20:=0}"
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
mkdir -p "$OUT/ocrap" "$OUT/external"

# Replay from the collection resolved from the OC-RAP bucket provenance, not
# from historical launcher defaults.  The input contract has already validated
# that both full-metric sides were generated from this same role.
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

# Frozen OC-RAP submission model.  Explicit WOMD specs avoid historical launcher defaults.
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
SAFE_LABEL_MODE=fast NEAR_LABEL_MODE=fast CONTACT_LABEL_MODE=fast \
SKIP_COMPLETE_REGIMES=false RESUME_FORCE=true FINALIZE_COMPLETE_JOURNALS=true \
bash scripts/run_ocrap_three_regime_evaluation.sh

# Safe: three learned current checkpoints come from the user's independent Safe run.
RUN="$OUT/external/safe" CHECKPOINT_ROOT="$SAFE_EXTERNAL_ROOT/checkpoints" \
OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_NOMINAL_CONTROL=false RUN_LEGACY_SAFE=false RUN_SUPPLEMENTARY_SAFE=false \
CL_WOMD="$SAFE_WOMD" CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$TRACE_MAX_STEPS" \
CL_TARGET_KEYS_FILE="$SELECTION_ROOT/safe_target_keys.json" CL_RENDER_TRACE=true CL_SCENE_JOURNAL_DETAIL=full \
JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" \
SKIP_COMPLETE_METHODS=false CL_RESUME_FORCE=true \
bash scripts/run_external_baselines_safe.sh

# Near: all six main-table methods are non-neural; reuse the calibration fitted in the user's near_external run.
RUN="$OUT/external/near" OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
DO_TRAIN=false DO_CALIBRATE=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_ORACLE_CLOSED_LOOP=false RUN_LEGACY_NEAR=false RUN_SUPPLEMENTARY_NEAR=false \
CONFORMAL_CALIBRATION="$NEAR_EXTERNAL_ROOT/conformal_calibration.json" \
CL_WOMD="$NEAR_WOMD" CL_LABEL_MODE=fast CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$TRACE_MAX_STEPS" \
CL_TARGET_KEYS_FILE="$SELECTION_ROOT/near_target_keys.json" CL_RENDER_TRACE=true CL_SCENE_JOURNAL_DETAIL=full \
JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" \
SKIP_COMPLETE_METHODS=false CL_RESUME_FORCE=true \
bash scripts/run_external_baselines_near.sh

# Contact: current six methods are non-neural controllers/optimization adapters.
RUN="$OUT/external/contact" OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_LEGACY_CONTACT=false \
CL_WOMD="$CONTACT_WOMD" CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$TRACE_MAX_STEPS" \
CL_TARGET_KEYS_FILE="$SELECTION_ROOT/contact_target_keys.json" CL_RENDER_TRACE=true CL_SCENE_JOURNAL_DETAIL=full \
CL_CONTACT_ANCHOR_PRELUDE_ENABLED=true CL_CONTACT_ANCHOR_PRELUDE_MAX_STEPS=60 \
CL_CONTACT_ANCHOR_PRELUDE_REPLAN_INTERVAL=1 CL_CONTACT_ANCHOR_REQUIRE_FOUND=true \
CL_CONTACT_ANCHOR_MANIFEST_FILE="$VIS_CONTACT_ANCHOR_MANIFEST_FILE" \
JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" \
SKIP_COMPLETE_METHODS=false CL_RESUME_FORCE=true \
bash scripts/run_external_baselines_contact.sh

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
    requested_set=set(requested)
    dt=float(selection.get('metric_dt_s',0.1) or 0.1)
    selected_by_key={str(x['target_key']):x for x in (selection.get('selected') or [])}
    req_steps={}
    for key in requested:
        item=selected_by_key.get(key) or {}
        clip=float(item.get('clip_duration_s',selection.get('selected_clip_duration_s',0.0)) or 0.0)
        steps=int(math.ceil(clip/dt-1e-9))
        if steps<=0:
            errors.append(f'invalid selected clip duration {r}/{key}: clip={clip} dt={dt}')
        if steps>trace_max_steps:
            errors.append(f'selected clip exceeds trace rerun cap {r}/{key}: required_steps={steps} trace_max_steps={trace_max_steps}')
        req_steps[key]=steps
    paths={'ocrap':root/'ocrap'/r/'closed_loop_ocrap.json.scenes.jsonl', **{m:root/'external'/r/f'closed_loop_{m}.json.scenes.jsonl' for m in ms}}
    counts={}; trace_starts={k:{} for k in requested}
    for m,p in paths.items():
        if not p.is_file(): errors.append(f'missing journal {m}: {p}'); continue
        keys=set()
        for line in p.open():
            if not line.strip(): continue
            x=json.loads(line); scene=x.get('scene',x); k=str(scene.get('target_key') or x.get('resume_key') or f"{scene.get('scene_id')}::t{scene.get('target_time_index')}")
            if k.startswith('target:'): k=k[len('target:'):]
            keys.add(k)
            if k not in requested_set:
                continue
            trace=scene.get('render_trace') or []
            required_frames=req_steps[k]+1
            if not trace:
                errors.append(f'no render_trace {r}/{m}/{k}; selective rerun journal must use scene_journal_detail=full')
            else:
                try:
                    trace_starts[k][m]=int(trace[0]['time_index'])
                except Exception:
                    errors.append(f'render_trace lacks integer start time {r}/{m}/{k}')
                if len(trace) < required_frames:
                    errors.append(f'short render_trace {r}/{m}/{k}: frames={len(trace)} required>={required_frames} clip_steps={req_steps[k]}')
        miss=sorted(requested_set-keys)
        if miss: errors.append(f'unresolved selected targets {r}/{m}: {miss}')
        counts[m]=len(keys)
    for key in requested:
        starts=trace_starts.get(key,{})
        if starts and len(set(starts.values())) != 1:
            errors.append(f'model trace starts are not synchronized {r}/{key}: {starts}')
        item=selected_by_key.get(key) or {}
        expected_field='contact_anchor_time_index' if r=='contact' else 'target_time_index'
        expected=item.get(expected_field)
        if starts and expected is not None and next(iter(starts.values())) != int(expected):
            errors.append(f'trace start mismatch {r}/{key}: got={next(iter(starts.values()))} expected_{expected_field}={expected}')
    regimes[r]={
        'requested':len(requested),
        'journal_counts':counts,
        'required_rollout_steps_by_target':req_steps,
        'max_required_rollout_steps':max(req_steps.values()) if req_steps else 0,
        'selected_clip_duration_s':selection.get('selected_clip_duration_s'),
        'duration_selection_mode':selection.get('duration_selection_mode'),
        'duration_source':selection.get('duration_source'),
    }
doc={'event':'selected_regime_trace_contract_v125','valid':not errors,'errors':errors,'trace_max_steps':trace_max_steps,'regimes':regimes}
(root/'TRACE_CONTRACT.json').write_text(json.dumps(doc,indent=2)+"\n")
print(json.dumps(doc,indent=2)); raise SystemExit(0 if not errors else 30)
PY
