#!/usr/bin/env bash
# Run a frozen, validation-selected Contact recovery profile on a larger exact-a0 test cohort.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
: "${BASE_OUT:=runs}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${OCRAP_MODEL_RUN:=$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
: "${MODEL_VARIANT:=balanced}"
: "${PROFILE_JSON:=$BASE_OUT/contact_recovery_validation_v134/CONTACT_RECOVERY_PROFILE.json}"
: "${OUT:=$BASE_OUT/contact_recovery_recharacterization_v134}"
: "${CUDA_DEVICES:=0,1}"
: "${JOBS_PER_GPU:=1}"
: "${MAX_PARALLEL:=2}"
: "${USE_DYNAMIC_SCHEDULER:=auto}"
: "${TEST_MIN_POST_STEPS:=30}"
: "${TEST_MAX_STEPS:=$TEST_MIN_POST_STEPS}"
[[ -s "$PROFILE_JSON" ]] || { echo "missing frozen validation profile: $PROFILE_JSON" >&2; exit 30; }
read -r SELECTOR REQUIRE_ABS VALID < <(python - "$PROFILE_JSON" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); r=x.get('selected_runtime') or {}
print(r.get('ocrap_selector',''), str(bool(r.get('require_absolute_admission_for_intervention'))).lower(), str(bool(x.get('valid'))).lower())
PY
)
[[ "$VALID" == true && -n "$SELECTOR" ]] || { echo "validation profile is not deployable: $PROFILE_JSON" >&2; exit 30; }
ANCHOR="$OUT/test_${TEST_MIN_POST_STEPS}step_anchor"
MANIFEST="$ANCHOR/contact_anchor_manifest.json"; KEYS="$ANCHOR/contact_anchor_target_keys.json"
mkdir -p "$OUT"
if [[ ! -s "$MANIFEST" || ! -s "$KEYS" ]]; then
  CONTACT_BUCKET="$OCRAP_ROOT/test_contact" BUCKET_SPLIT=test WOMD_ROOT="$WOMD_ROOT" OUT="$ANCHOR" \
  MIN_POST_STEPS="$TEST_MIN_POST_STEPS" GPU="${CUDA_DEVICES%%,*}" bash scripts/build_contact_anchor_cohort.sh
else
  echo "[CONTACT-TEST][REUSE] $MANIFEST"
fi
N="$(python - "$MANIFEST" <<'PY'
import json,sys; print(int(json.load(open(sys.argv[1])).get('num_selected_anchors') or 0))
PY
)"
echo "[CONTACT-TEST] anchors=$N profile=$SELECTOR require_abs=$REQUIRE_ABS horizon_steps=$TEST_MAX_STEPS"
OCRAP_OUT="$OUT/ocrap"
RUN_SAFE=0 RUN_NEAR=0 RUN_CONTACT=1 MODEL_RUN="$OCRAP_MODEL_RUN" MODEL_VARIANT="$MODEL_VARIANT" \
OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" OUT="$OCRAP_OUT" CUDA_DEVICES="$CUDA_DEVICES" \
CONTACT_BUCKET="$OCRAP_ROOT/test_contact" BUCKET_SPLIT=test MAX_SCENARIOS=0 MAX_STEPS="$TEST_MAX_STEPS" \
CONTACT_TARGET_KEYS_FILE="$KEYS" CONTACT_ANCHOR_PRELUDE_ENABLED=true CONTACT_ANCHOR_MANIFEST_FILE="$MANIFEST" \
CONTACT_REQUIRE_ABSOLUTE_ADMISSION_FOR_INTERVENTION="$REQUIRE_ABS" CONTACT_OCRAP_SELECTOR="$SELECTOR" \
CONTACT_LABEL_MODE=fast RENDER_CONTACT=true SCENE_JOURNAL_DETAIL=full RESULT_SCENE_DETAIL=metrics \
RESUME=true RESUME_FORCE=false bash scripts/run_ocrap_three_regime_evaluation.sh 2>&1 | tee "$OUT/ocrap.log"
python tools/audit_contact_recovery_execution.py --result "$OCRAP_OUT/contact/closed_loop_ocrap.json" \
  --output "$OCRAP_OUT/contact/CONTACT_RECOVERY_AUDIT.json" | tee "$OUT/ocrap_audit.log"
EXT_OUT="$OUT/external_contact"
RUN="$EXT_OUT" OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true RUN_LEGACY_CONTACT=false \
CL_BUCKET_DATASET="$OCRAP_ROOT/test_contact" CL_BUCKET_SPLIT=test CL_MAX_SCENARIOS=0 CL_MAX_STEPS="$TEST_MAX_STEPS" \
CL_TARGET_KEYS_FILE="$KEYS" CL_RENDER_TRACE=true CL_SCENE_JOURNAL_DETAIL=full \
CL_CONTACT_ANCHOR_PRELUDE_ENABLED=true CL_CONTACT_ANCHOR_MANIFEST_FILE="$MANIFEST" \
JOBS_PER_GPU="$JOBS_PER_GPU" MAX_PARALLEL="$MAX_PARALLEL" USE_DYNAMIC_SCHEDULER="$USE_DYNAMIC_SCHEDULER" \
SKIP_COMPLETE_METHODS=true CL_RESUME=true CL_RESUME_FORCE=false bash scripts/run_external_baselines_contact.sh \
  2>&1 | tee "$OUT/external.log"
for m in postimpact_mpc_lite post_crash_braking postimpact_motion_tvlqr post_collision_restoration compensatory_postimpact_mpc robust_postimpact_control; do
  python tools/audit_contact_recovery_execution.py --result "$EXT_OUT/closed_loop_${m}.json" \
    --output "$EXT_OUT/CONTACT_RECOVERY_AUDIT_${m}.json" >/dev/null
done
python - "$OUT" "$PROFILE_JSON" "$MANIFEST" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); prof=json.load(open(sys.argv[2])); man=json.load(open(sys.argv[3]))
methods=['postimpact_mpc_lite','post_crash_braking','postimpact_motion_tvlqr','post_collision_restoration','compensatory_postimpact_mpc','robust_postimpact_control']
rows=[]
for name,p in [('ocrap',root/'ocrap/contact/CONTACT_RECOVERY_AUDIT.json')]+[(m,root/f'external_contact/CONTACT_RECOVERY_AUDIT_{m}.json') for m in methods]:
 d=json.load(open(p)); rows.append({k:d.get(k) for k in ['method','intervention_rate','offroad_scene_rate','recontact_scene_rate','post_contact_escape_scene_rate','controlled_recovery_success_rate','mean_post_contact_overlap_duration_s','mean_terminal_clearance_capped_m','runaway_offroad_terminal_gt10m_rate']}|{'name':name})
out={'event':'contact_recovery_recharacterization_v1','profile':prof.get('selected_profile'),'profile_json':sys.argv[2],
     'min_post_steps':man.get('min_post_steps'),'num_anchors':man.get('num_selected_anchors'),'rows':rows,
     'note':'This test run uses a selector profile frozen exclusively from val_contact; all methods share the exact-a0 scene-disjoint test anchor cohort.'}
(root/'RECHARACTERIZATION_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
PY
echo "[CONTACT-TEST][DONE] $OUT/RECHARACTERIZATION_SUMMARY.json"
