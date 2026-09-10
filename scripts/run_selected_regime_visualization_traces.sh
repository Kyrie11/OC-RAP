#!/usr/bin/env bash
# Rerun only selected targets with render_trace=true.  Full population metrics
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
: "${SAFE_EXTERNAL_ROOT:=/home/senzeyu2/code/OC-RAP/runs/safe_external}"
: "${NEAR_EXTERNAL_ROOT:=/home/senzeyu2/code/OC-RAP/runs/near_external}"
: "${CONTACT_EXTERNAL_ROOT:=/home/senzeyu2/code/OC-RAP/runs/contact_external}"
: "${SELECTION_ROOT:?set SELECTION_ROOT to .../selection}"
: "${INPUT_CONTRACT:=$(dirname "$SELECTION_ROOT")/provenance/VISUALIZATION_INPUT_CONTRACT.json}"
: "${OUT:=$(dirname "$SELECTION_ROOT")/selective_traces}"
: "${CUDA_DEVICES:=0,1}"
: "${TRACE_MAX_STEPS:=60}"
: "${CLEAN_TRACE_OUTPUT:=true}"
: "${ALLOW_DIAGNOSTIC_RC20:=1}"
: "${JOBS_PER_GPU:=3}"
: "${MAX_PARALLEL:=6}"

[[ -f "$INPUT_CONTRACT" ]] || { echo "Missing visualization input contract: $INPUT_CONTRACT" >&2; exit 30; }
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
RENDER_SAFE=true RENDER_NEAR=true RENDER_CONTACT=true \
SAFE_LABEL_MODE=fast NEAR_LABEL_MODE=fast CONTACT_LABEL_MODE=fast \
ALLOW_DIAGNOSTIC_RC20="$ALLOW_DIAGNOSTIC_RC20" \
SKIP_COMPLETE_REGIMES=false RESUME_FORCE=true FINALIZE_COMPLETE_JOURNALS=true \
bash scripts/run_ocrap_three_regime_closed_loop.sh

# Safe: three learned current checkpoints come from the user's independent Safe run.
RUN="$OUT/external/safe" CHECKPOINT_ROOT="$SAFE_EXTERNAL_ROOT/checkpoints" \
OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_NOMINAL_CONTROL=false RUN_LEGACY_SAFE=false \
CL_WOMD="$SAFE_WOMD" CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$TRACE_MAX_STEPS" \
CL_TARGET_KEYS_FILE="$SELECTION_ROOT/safe_target_keys.json" CL_RENDER_TRACE=true \
JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" \
SKIP_COMPLETE_METHODS=false CL_RESUME_FORCE=true \
bash scripts/run_safe_regime_external_baselines.sh

# Near: all six main-table methods are non-neural; reuse the calibration fitted in the user's near_external run.
RUN="$OUT/external/near" OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
DO_TRAIN=false DO_CALIBRATE=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_ORACLE_CLOSED_LOOP=false RUN_LEGACY_NEAR=false \
CONFORMAL_CALIBRATION="$NEAR_EXTERNAL_ROOT/conformal_calibration.json" \
CL_WOMD="$NEAR_WOMD" CL_LABEL_MODE=fast CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$TRACE_MAX_STEPS" \
CL_TARGET_KEYS_FILE="$SELECTION_ROOT/near_target_keys.json" CL_RENDER_TRACE=true \
JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" \
SKIP_COMPLETE_METHODS=false CL_RESUME_FORCE=true \
bash scripts/run_near_contact_external_baselines_2gpu_optimized.sh

# Contact: current six methods are non-neural controllers/optimization adapters.
RUN="$OUT/external/contact" OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_LEGACY_CONTACT=false \
CL_WOMD="$CONTACT_WOMD" CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$TRACE_MAX_STEPS" \
CL_TARGET_KEYS_FILE="$SELECTION_ROOT/contact_target_keys.json" CL_RENDER_TRACE=true \
JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" \
SKIP_COMPLETE_METHODS=false CL_RESUME_FORCE=true \
bash scripts/run_contact_external_baselines.sh

python - "$OUT" "$SELECTION_ROOT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); sel=pathlib.Path(sys.argv[2])
methods={
'safe':['gameformer_lite','plantf','pluto','pdm_closed','pdm_hybrid','idm'],
'near':['marc_lite','racp_lite','robust_scenario_mpc','predictive_safety_filter','dr_cvar_safety_filter','conformal_predictive_safety_filter'],
'contact':['postimpact_mpc_lite','post_crash_braking','postimpact_motion_tvlqr','post_collision_restoration','compensatory_postimpact_mpc','robust_postimpact_control']}
errors=[]; regimes={}
for r,ms in methods.items():
    requested=json.loads((sel/f'{r}_target_keys.json').read_text())['target_keys']
    paths={'ocrap':root/'ocrap'/r/'closed_loop_ocrap.json.scenes.jsonl', **{m:root/'external'/r/f'closed_loop_{m}.json.scenes.jsonl' for m in ms}}
    counts={}
    for m,p in paths.items():
        if not p.is_file(): errors.append(f'missing journal {m}: {p}'); continue
        keys=set()
        for line in p.open():
            if not line.strip(): continue
            x=json.loads(line); k=x.get('target_key') or f"{x.get('scene_id')}::{x.get('target_time_index')}"
            keys.add(str(k))
            if not x.get('render_trace'): errors.append(f'no render_trace {r}/{m}/{k}')
        miss=sorted(set(map(str,requested))-keys)
        if miss: errors.append(f'unresolved selected targets {r}/{m}: {miss}')
        counts[m]=len(keys)
    regimes[r]={'requested':len(requested),'journal_counts':counts}
doc={'event':'selected_regime_trace_contract_v52','valid':not errors,'errors':errors,'regimes':regimes}
(root/'TRACE_CONTRACT.json').write_text(json.dumps(doc,indent=2)+"\n")
print(json.dumps(doc,indent=2)); raise SystemExit(0 if not errors else 30)
PY
