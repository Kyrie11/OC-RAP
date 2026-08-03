#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# The RC=20 package contains two diagnostic variants.  Select one explicitly;
# no held-out result is used to choose between them.
: "${MODEL_RUN:=runs/ocrap_v48_34_barrier_crossfit_dedicated_4834}"
: "${MODEL_VARIANT:=balanced}"
: "${CHECKPOINT:=$MODEL_RUN/candidates/$MODEL_VARIANT/model_v48_trac_sr/best.pt}"
: "${GAMMA_REC_JSON:=$MODEL_RUN/candidates/$MODEL_VARIANT/calibration/gamma_rec_by_bucket_v48.json}"
[[ -f "$CHECKPOINT" ]] || { echo "Missing OC-RAP checkpoint: $CHECKPOINT" >&2; exit 2; }

# Calibration is bucket-specific.  GAMMA_REC remains a legacy fallback only.
if [[ -z "${SAFE_GAMMA_REC:-}" || -z "${NEAR_GAMMA_REC:-}" || -z "${CONTACT_GAMMA_REC:-}" ]]; then
  if [[ -f "$GAMMA_REC_JSON" ]]; then
    read -r _auto_safe _auto_near _auto_contact < <(python - "$GAMMA_REC_JSON" <<'PY2'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))["gamma_rec_by_bucket"]
print(x["test_safe"], x["test_near_contact"], x["test_contact"])
PY2
)
  elif [[ -n "${GAMMA_REC:-}" ]]; then
    _auto_safe="$GAMMA_REC"; _auto_near="$GAMMA_REC"; _auto_contact="$GAMMA_REC"
    echo "[WARN] Reusing legacy scalar GAMMA_REC across regimes; prefer GAMMA_REC_JSON." >&2
  else
    echo "Missing bucket calibration JSON: $GAMMA_REC_JSON" >&2; exit 2
  fi
fi
: "${SAFE_GAMMA_REC:=${_auto_safe:-}}"
: "${NEAR_GAMMA_REC:=${_auto_near:-}}"
: "${CONTACT_GAMMA_REC:=${_auto_contact:-}}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${OUT:=runs/ocrap_three_regime_closed_loop}"
: "${WOMD_VAL:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}"
: "${SAFE_WOMD:=$WOMD_VAL}"
: "${NEAR_WOMD:=$WOMD_VAL_INTERACTIVE}"
: "${CONTACT_WOMD:=$WOMD_VAL_INTERACTIVE}"
: "${SAFE_BUCKET:=$OCRAP_ROOT/test_safe}"
: "${NEAR_BUCKET:=$OCRAP_ROOT/test_near_contact}"
: "${CONTACT_BUCKET:=$OCRAP_ROOT/test_contact}"
: "${BUCKET_SPLIT:=test}"
: "${CUDA_DEVICES:=0,1}"
: "${MAX_SCENARIOS:=0}"
: "${MAX_STEPS:=40}"
: "${NUM_CANDIDATES:=24}"
: "${NUM_RECOVERY_OPTIONS:=12}"
: "${SAFE_LABEL_MODE:=fast}"
: "${NEAR_LABEL_MODE:=fast}"
: "${CONTACT_LABEL_MODE:=selected}"
: "${RENDER_NEAR:=true}"
: "${RENDER_CONTACT:=true}"
: "${ALLOW_DIAGNOSTIC_RC20:=0}"

[[ "$ALLOW_DIAGNOSTIC_RC20" == 1 ]] || { echo "Set ALLOW_DIAGNOSTIC_RC20=1: current model did not pass the Natural gate." >&2; exit 2; }
IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"; [[ ${#GPUS[@]} -gt 0 ]] || GPUS=(0)
mkdir -p "$OUT/safe" "$OUT/near" "$OUT/contact"

run_one() {
  local regime="$1" womd="$2" bucket="$3" label_mode="$4" render="$5" gpu="$6"
  local gamma="$7"
  RUN_DIR="$OUT/$regime" OUTPUT="$OUT/$regime/closed_loop_ocrap.json" \
  WOMD_VAL="$womd" WOMD_LIMIT=0 CHECKPOINT="$CHECKPOINT" GAMMA_REC="$gamma" GPU="$gpu" \
  MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" LABEL_MODE="$label_mode" \
  NUM_CANDIDATES="$NUM_CANDIDATES" NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
  BUCKET_DATASET="$bucket" BUCKET_SPLIT="$BUCKET_SPLIT" MAX_TARGETS_PER_SCENE=1 \
  RENDER_TRACE="$render" SAVE_PARTIAL=true RESUME_FORCE=true \
  bash scripts/run_ocrap_closed_loop_optimized.sh
}

# Two jobs first, then Contact on the first released GPU.  This avoids three JAX
# processes contending on two devices while preserving exact target sets.
run_one safe "$SAFE_WOMD" "$SAFE_BUCKET" "$SAFE_LABEL_MODE" false "${GPUS[0]}" "$SAFE_GAMMA_REC" & p_safe=$!
run_one near "$NEAR_WOMD" "$NEAR_BUCKET" "$NEAR_LABEL_MODE" "$RENDER_NEAR" "${GPUS[1]:-${GPUS[0]}}" "$NEAR_GAMMA_REC" & p_near=$!
failed=0
wait "$p_safe" || failed=1
if [[ "$failed" == 0 ]]; then
  run_one contact "$CONTACT_WOMD" "$CONTACT_BUCKET" "$CONTACT_LABEL_MODE" "$RENDER_CONTACT" "${GPUS[0]}" "$CONTACT_GAMMA_REC" || failed=1
fi
wait "$p_near" || failed=1
[[ "$failed" == 0 ]] || exit 1

python - "$OUT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); out={}
for r in ('safe','near','contact'):
 p=root/r/'closed_loop_ocrap.json'; d=json.load(p.open());
 out[r]={k:d.get(k) for k in ('method','source','num_scenes','num_decisions','collision_scene_rate','offroad_scene_rate','closed_loop_FRA_exec','closed_loop_DRS','closed_loop_ODG','closed_loop_bounded_NUP','recontact_scene_rate','post_contact_escape_scene_rate','timing')}
json.dump({'event':'ocrap_three_regime_closed_loop','regimes':out},(root/'SUMMARY.json').open('w'),indent=2)
print({'event':'ocrap_three_regime_closed_loop','output':str(root)})
PY
