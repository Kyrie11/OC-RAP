#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${OUT:=runs/all_regime_external_baselines_v49}"
: "${CUDA_DEVICES:=0,1}"
: "${MAX_SCENARIOS:=0}"
: "${MAX_STEPS:=40}"
: "${RUN_SAFE:=1}"
: "${RUN_NEAR:=1}"
: "${RUN_CONTACT:=1}"
: "${DO_TRAIN_SAFE:=true}"
: "${DO_TRAIN_NEAR:=true}"
: "${DO_OFFLINE:=true}"
: "${DO_CLOSED_LOOP:=true}"
: "${RENDER_TRACES:=true}"
: "${WOMD_VAL:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}"
: "${SAFE_CL_WOMD:=$WOMD_VAL}"
: "${NEAR_CL_WOMD:=$WOMD_VAL_INTERACTIVE}"
: "${CONTACT_CL_WOMD:=$WOMD_VAL_INTERACTIVE}"

mkdir -p "$OUT" "$OUT/safe" "$OUT/near" "$OUT/contact"
PYTHONPATH=src python tools/audit_external_baseline_fidelity.py \
  --output-json "$OUT/EXTERNAL_BASELINE_FIDELITY.json" \
  --output-md "$OUT/EXTERNAL_BASELINE_FIDELITY.md"

common=(OCRAP_ROOT="$OCRAP_ROOT" CUDA_DEVICES="$CUDA_DEVICES" CL_MAX_SCENARIOS="$MAX_SCENARIOS" CL_MAX_STEPS="$MAX_STEPS" CL_SAVE_PARTIAL=true DO_OFFLINE="$DO_OFFLINE" DO_CLOSED_LOOP="$DO_CLOSED_LOOP")

if [[ "$RUN_SAFE" == 1 ]]; then
  env "${common[@]}" RUN="$OUT/safe" DO_TRAIN="$DO_TRAIN_SAFE" CL_WOMD="$SAFE_CL_WOMD" CL_RENDER_TRACE=false \
    bash scripts/run_safe_regime_external_baselines.sh \
    > >(tee "$OUT/safe.launcher.log") 2>&1
fi
if [[ "$RUN_NEAR" == 1 ]]; then
  env "${common[@]}" RUN="$OUT/near" TRAIN_GAMEFORMER_IF_MISSING="$DO_TRAIN_NEAR" CL_WOMD="$NEAR_CL_WOMD" CL_RENDER_TRACE="$RENDER_TRACES" \
    bash scripts/run_near_contact_external_baselines_2gpu_optimized.sh \
    > >(tee "$OUT/near.launcher.log") 2>&1
fi
if [[ "$RUN_CONTACT" == 1 ]]; then
  env "${common[@]}" RUN="$OUT/contact" CL_WOMD="$CONTACT_CL_WOMD" CL_RENDER_TRACE="$RENDER_TRACES" \
    bash scripts/run_contact_external_baselines.sh \
    > >(tee "$OUT/contact.launcher.log") 2>&1
fi

python - "$OUT" <<'PY'
import glob,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); rows={}
for regime in ('safe','near','contact'):
    d=root/regime
    rows[regime]=[]
    for p in sorted(d.glob('closed_loop_*.json')) if d.is_dir() else []:
        try:x=json.load(p.open())
        except Exception:continue
        rows[regime].append({'method':x.get('method'),'path':str(p),'num_scenes':x.get('num_scenes'),'num_decisions':x.get('num_decisions')})
out=root/'EXTERNAL_BASELINE_RUN_INDEX.json'; json.dump({'event':'all_regime_external_baselines','runs':rows},out.open('w'),indent=2)
print({'event':'all_regime_external_baselines','output':str(out)})
PY
