#!/usr/bin/env bash
# Cheap held-out validation screen for Contact selector behavior.
# It intentionally samples only a fixed validation subset and does NOT freeze a publication profile.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
: "${BASE_OUT:=runs}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${OCRAP_MODEL_RUN:=$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
: "${MODEL_VARIANT:=balanced}"
: "${OUT:=$BASE_OUT/contact_recovery_quickdiag_v135}"
: "${CUDA_DEVICES:=0,1}"
: "${DIAG_NUM_TARGETS:=100}"
: "${DIAG_MAX_TARGETS_PER_SCENE:=2}"
: "${DIAG_SEED:=2027}"
: "${DIAG_MIN_POST_STEPS:=25}"
: "${DIAG_MAX_STEPS:=$DIAG_MIN_POST_STEPS}"
: "${RUN_CALIBRATED:=false}"
: "${REUSE_DIAGNOSTIC_ANCHORS:=true}"
IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"; ((${#GPUS[@]})) || GPUS=(0)
GPU0="${GPUS[0]}"; GPU1="${GPUS[1]:-${GPUS[0]}}"
mkdir -p "$OUT"
SUBSET="$OUT/diagnostic_target_keys.json"
ANCHOR="$OUT/anchors"; MANIFEST="$ANCHOR/contact_anchor_manifest.json"; KEYS="$ANCHOR/contact_anchor_target_keys.json"
if [[ ! -s "$SUBSET" ]]; then
  python tools/build_deterministic_target_subset.py --bucket-dataset "$OCRAP_ROOT/val_contact" --bucket-split val \
    --num-targets "$DIAG_NUM_TARGETS" --max-targets-per-scene "$DIAG_MAX_TARGETS_PER_SCENE" --seed "$DIAG_SEED" --output "$SUBSET"
fi
if [[ "${REUSE_DIAGNOSTIC_ANCHORS,,}" != true || ! -s "$MANIFEST" || ! -s "$KEYS" ]]; then
  CONTACT_BUCKET="$OCRAP_ROOT/val_contact" BUCKET_SPLIT=val WOMD_ROOT="$WOMD_ROOT" OUT="$ANCHOR" \
    SOURCE_TARGET_KEYS_FILE="$SUBSET" MIN_POST_STEPS="$DIAG_MIN_POST_STEPS" MAX_SCENARIOS=0 MAX_TARGETS_PER_SCENE=100000 \
    GPU="$GPU0" bash scripts/build_contact_anchor_cohort.sh
else
  echo "[CONTACT-QUICKDIAG][REUSE] $MANIFEST"
fi
read -r ACTUAL_TARGETS N < <(python - "$SUBSET" "$MANIFEST" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
m=json.load(open(sys.argv[2],encoding='utf-8'))
print(int(s.get('selected_num_targets') or len(s.get('target_keys') or [])), int(m.get('num_selected_anchors') or 0))
PY
)
echo "[CONTACT-QUICKDIAG] requested_targets=$DIAG_NUM_TARGETS sampled_targets=$ACTUAL_TARGETS exact_a0_anchors=$N min_post_steps=$DIAG_MIN_POST_STEPS"
if [[ "$ACTUAL_TARGETS" != "$DIAG_NUM_TARGETS" ]]; then
  echo "[CONTACT-QUICKDIAG][INFO] reusing an existing deterministic subset with $ACTUAL_TARGETS targets; DIAG_NUM_TARGETS only applies when a new subset is built."
fi
if ((N<5)); then
  echo "[CONTACT-QUICKDIAG][ERROR] only $N exact-a0 anchors. Re-run in a NEW OUT with DIAG_NUM_TARGETS=150; do not lower Contact safety/recovery gates." >&2
  exit 30
fi
run_profile(){
  local name="$1"
  local selector="$2"
  local require_abs="$3"
  local gpu="$4"
  local root="$OUT/profiles/$name"
  mkdir -p "$root"
  RUN_SAFE=0 RUN_NEAR=0 RUN_CONTACT=1 MODEL_RUN="$OCRAP_MODEL_RUN" MODEL_VARIANT="$MODEL_VARIANT" \
  OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" OUT="$root" CUDA_DEVICES="$gpu" \
  CONTACT_BUCKET="$OCRAP_ROOT/val_contact" BUCKET_SPLIT=val MAX_SCENARIOS=0 MAX_STEPS="$DIAG_MAX_STEPS" \
  CONTACT_TARGET_KEYS_FILE="$KEYS" CONTACT_ANCHOR_PRELUDE_ENABLED=true CONTACT_ANCHOR_MANIFEST_FILE="$MANIFEST" \
  CONTACT_REQUIRE_ABSOLUTE_ADMISSION_FOR_INTERVENTION="$require_abs" CONTACT_OCRAP_SELECTOR="$selector" \
  CONTACT_LABEL_MODE=fast RENDER_CONTACT=false RESUME=true RESUME_FORCE=false \
  bash scripts/run_ocrap_three_regime_evaluation.sh 2>&1 | tee "$root/run.log"
  python tools/audit_contact_recovery_execution.py --result "$root/contact/closed_loop_ocrap.json" \
    --output "$root/contact/CONTACT_RECOVERY_AUDIT.json" | tee "$root/contact/audit.log"
}
# Strict and guarded are the only profiles needed for the cheap first-pass question:
# does opening the existing guarded fallback actually cause safe useful interventions?
run_profile strict_abs lcb_constrained true "$GPU0" & P0=$!
run_profile guarded_fallback lcb_constrained false "$GPU1" & P1=$!
set +e; wait "$P0"; R0=$?; wait "$P1"; R1=$?; set -e
[[ $R0 == 0 && $R1 == 0 ]] || { echo "quick diagnostic profile failed strict=$R0 guarded=$R1" >&2; exit 30; }
CAL_ARG=()
if [[ "${RUN_CALIBRATED,,}" == true || "$RUN_CALIBRATED" == 1 ]]; then
  run_profile calibrated_guarded calibrated_constrained false "$GPU0"
  CAL_ARG=(--calibrated "$OUT/profiles/calibrated_guarded/contact/CONTACT_RECOVERY_AUDIT.json")
fi
python tools/summarize_contact_recovery_quickdiag.py \
  --strict "$OUT/profiles/strict_abs/contact/CONTACT_RECOVERY_AUDIT.json" \
  --guarded "$OUT/profiles/guarded_fallback/contact/CONTACT_RECOVERY_AUDIT.json" \
  "${CAL_ARG[@]}" --output "$OUT/QUICK_DIAGNOSTIC_SUMMARY.json" | tee "$OUT/summary.log"
echo "[CONTACT-QUICKDIAG][DONE] $OUT/QUICK_DIAGNOSTIC_SUMMARY.json"
