#!/usr/bin/env bash
# Validation-only Contact selector repair. Never uses test_contact to choose a profile.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
: "${BASE_OUT:=runs}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${OCRAP_MODEL_RUN:=$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
: "${MODEL_VARIANT:=balanced}"
: "${OUT:=$BASE_OUT/contact_recovery_validation_v134}"
: "${CUDA_DEVICES:=0,1}"
: "${VAL_MIN_POST_STEPS:=25}"
: "${VAL_MAX_STEPS:=$VAL_MIN_POST_STEPS}"
: "${REUSE_VALIDATION_ANCHORS:=true}"
ANCHOR="$OUT/anchors"
MANIFEST="$ANCHOR/contact_anchor_manifest.json"
KEYS="$ANCHOR/contact_anchor_target_keys.json"
mkdir -p "$OUT"
if [[ "${REUSE_VALIDATION_ANCHORS,,}" != true || ! -s "$MANIFEST" || ! -s "$KEYS" ]]; then
  CONTACT_BUCKET="$OCRAP_ROOT/val_contact" BUCKET_SPLIT=val WOMD_ROOT="$WOMD_ROOT" \
  OUT="$ANCHOR" MIN_POST_STEPS="$VAL_MIN_POST_STEPS" GPU="${CUDA_DEVICES%%,*}" \
  bash scripts/build_contact_anchor_cohort.sh
else
  echo "[CONTACT-VAL][REUSE] $MANIFEST"
fi
N="$(python - "$MANIFEST" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1])).get('num_selected_anchors') or 0))
PY
)"
((N>=5)) || { echo "[CONTACT-VAL][ERROR] only $N exact-a0 validation anchors; insufficient for selector selection" >&2; exit 30; }
echo "[CONTACT-VAL] anchors=$N min_post_steps=$VAL_MIN_POST_STEPS"
run_profile(){
  local name="$1" selector="$2" require_abs="$3" root="$OUT/profiles/$name"
  mkdir -p "$root"
  echo "[CONTACT-VAL][PROFILE] $name selector=$selector require_abs=$require_abs"
  RUN_SAFE=0 RUN_NEAR=0 RUN_CONTACT=1 MODEL_RUN="$OCRAP_MODEL_RUN" MODEL_VARIANT="$MODEL_VARIANT" \
  OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" OUT="$root" CUDA_DEVICES="$CUDA_DEVICES" \
  CONTACT_BUCKET="$OCRAP_ROOT/val_contact" BUCKET_SPLIT=val MAX_SCENARIOS=0 MAX_STEPS="$VAL_MAX_STEPS" \
  CONTACT_TARGET_KEYS_FILE="$KEYS" CONTACT_ANCHOR_PRELUDE_ENABLED=true CONTACT_ANCHOR_MANIFEST_FILE="$MANIFEST" \
  CONTACT_REQUIRE_ABSOLUTE_ADMISSION_FOR_INTERVENTION="$require_abs" CONTACT_OCRAP_SELECTOR="$selector" \
  CONTACT_LABEL_MODE=fast RENDER_CONTACT=false RESUME=true RESUME_FORCE=false \
  bash scripts/run_ocrap_three_regime_evaluation.sh 2>&1 | tee "$root/run.log"
  python tools/audit_contact_recovery_execution.py \
    --result "$root/contact/closed_loop_ocrap.json" --output "$root/contact/CONTACT_RECOVERY_AUDIT.json" \
    | tee "$root/contact/audit.log"
}
run_profile strict_abs lcb_constrained true
run_profile guarded_fallback lcb_constrained false
run_profile calibrated_guarded calibrated_constrained false
set +e
python tools/select_contact_recovery_profile.py \
  --audit "strict_abs=$OUT/profiles/strict_abs/contact/CONTACT_RECOVERY_AUDIT.json" \
  --audit "guarded_fallback=$OUT/profiles/guarded_fallback/contact/CONTACT_RECOVERY_AUDIT.json" \
  --audit "calibrated_guarded=$OUT/profiles/calibrated_guarded/contact/CONTACT_RECOVERY_AUDIT.json" \
  --output "$OUT/CONTACT_RECOVERY_PROFILE.json" | tee "$OUT/profile_selection.log"
rc=${PIPESTATUS[0]}
set -e
if ((rc!=0)); then
  echo "[CONTACT-VAL] No deployable selector repair passed the held-out validation rule." >&2
  echo "[CONTACT-VAL] Inspect $OUT/CONTACT_RECOVERY_PROFILE.json; do not tune on test_contact." >&2
  exit "$rc"
fi
echo "[CONTACT-VAL][DONE] frozen profile: $OUT/CONTACT_RECOVERY_PROFILE.json"
