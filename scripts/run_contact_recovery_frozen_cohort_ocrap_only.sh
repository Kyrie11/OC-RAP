#!/usr/bin/env bash
# Re-run only OC-RAP on the original frozen publication Contact cohort.
# Existing external-baseline Contact results are reused verbatim because the target lock,
# anchor manifest, horizon and baseline configurations are unchanged.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
: "${BASE_OUT:=runs}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${OCRAP_MODEL_RUN:=$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
: "${MAIN_CHARACTERIZATION_ROOT:=$BASE_OUT/ocrap_v48_124_final_characterization}"
: "${EXTERNAL_ROOT:=$BASE_OUT/external_baselines_v48_124_final_v2/contact}"
: "${QUICK_SUMMARY:=$BASE_OUT/contact_recovery_quickdiag_v135/QUICK_DIAGNOSTIC_SUMMARY.json}"
: "${OUT:=$BASE_OUT/contact_recovery_frozen8_v135}"
: "${GPU:=0}"
: "${CONTACT_RECOVERY_PROFILE:=guarded_fallback}"
[[ -s "$QUICK_SUMMARY" ]] || { echo "missing quick diagnostic summary: $QUICK_SUMMARY" >&2; exit 30; }
python - "$QUICK_SUMMARY" "$CONTACT_RECOVERY_PROFILE" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); p=sys.argv[2]
if p not in (x.get('promising_profiles') or []):
    raise SystemExit(f"profile {p} was not promising in held-out quick diagnostic; refusing test/frozen-cohort confirmation")
PY
case "$CONTACT_RECOVERY_PROFILE" in
  guarded_fallback) SELECTOR=lcb_constrained; REQUIRE_ABS=false ;;
  calibrated_guarded) SELECTOR=calibrated_constrained; REQUIRE_ABS=false ;;
  *) echo "unsupported CONTACT_RECOVERY_PROFILE=$CONTACT_RECOVERY_PROFILE" >&2; exit 2 ;;
esac
KEYS="$MAIN_CHARACTERIZATION_ROOT/target_keys/contact.json"
MANIFEST="$MAIN_CHARACTERIZATION_ROOT/contact_anchor/contact_anchor_manifest.json"
[[ -s "$KEYS" && -s "$MANIFEST" ]] || { echo "missing frozen publication Contact lock/manifest" >&2; exit 30; }
mkdir -p "$OUT"
RUN_SAFE=0 RUN_NEAR=0 RUN_CONTACT=1 MODEL_RUN="$OCRAP_MODEL_RUN" MODEL_VARIANT=balanced \
OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" OUT="$OUT/ocrap" CUDA_DEVICES="$GPU" \
CONTACT_BUCKET="$OCRAP_ROOT/test_contact" BUCKET_SPLIT=test MAX_SCENARIOS=0 MAX_STEPS=40 \
CONTACT_TARGET_KEYS_FILE="$KEYS" CONTACT_ANCHOR_PRELUDE_ENABLED=true CONTACT_ANCHOR_MANIFEST_FILE="$MANIFEST" \
CONTACT_REQUIRE_ABSOLUTE_ADMISSION_FOR_INTERVENTION="$REQUIRE_ABS" CONTACT_OCRAP_SELECTOR="$SELECTOR" \
CONTACT_LABEL_MODE=fast RENDER_CONTACT=false RESULT_SCENE_DETAIL=metrics SCENE_JOURNAL_DETAIL=metrics \
RESUME=true RESUME_FORCE=false bash scripts/run_ocrap_three_regime_evaluation.sh 2>&1 | tee "$OUT/ocrap.log"
python tools/audit_contact_recovery_execution.py --result "$OUT/ocrap/contact/closed_loop_ocrap.json" \
  --output "$OUT/ocrap/contact/CONTACT_RECOVERY_AUDIT.json" | tee "$OUT/ocrap_audit.log"
METHODS=(postimpact_mpc_lite post_crash_braking postimpact_motion_tvlqr post_collision_restoration compensatory_postimpact_mpc robust_postimpact_control)
mkdir -p "$OUT/reused_baseline_audits"
for m in "${METHODS[@]}"; do
  p="$EXTERNAL_ROOT/closed_loop_${m}.json"
  [[ -s "$p" ]] || { echo "missing existing baseline result: $p" >&2; exit 30; }
  python tools/audit_contact_recovery_execution.py --result "$p" --output "$OUT/reused_baseline_audits/${m}.json" >/dev/null
done
python - "$OUT" "$CONTACT_RECOVERY_PROFILE" "$EXTERNAL_ROOT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); profile=sys.argv[2]; ext=sys.argv[3]
methods=['postimpact_mpc_lite','post_crash_braking','postimpact_motion_tvlqr','post_collision_restoration','compensatory_postimpact_mpc','robust_postimpact_control']
def row(name,p):
 d=json.load(open(p)); keys=['num_scenes','intervention_rate','offroad_scene_count','offroad_scene_rate','recontact_scene_count','recontact_scene_rate','controlled_recovery_success_count','controlled_recovery_success_rate','mean_post_contact_overlap_duration_s','mean_terminal_clearance_capped_m','runaway_offroad_terminal_gt10m_count','runaway_offroad_terminal_gt10m_rate']; return {'name':name,**{k:d.get(k) for k in keys}}
rows=[row('ocrap_revised',root/'ocrap/contact/CONTACT_RECOVERY_AUDIT.json')]
for m in methods: rows.append(row(m,root/f'reused_baseline_audits/{m}.json'))
out={'event':'contact_frozen_cohort_ocrap_only_confirmation_v135','profile':profile,'baseline_results_reused':True,'external_root':ext,'rows':rows,'scientific_note':'Only OC-RAP was rerun. Existing baseline results are valid paired comparators because the frozen exact-a0 target lock, anchor manifest, 40-step horizon, simulator semantics and baseline configurations are unchanged.'}
(root/'FROZEN_COHORT_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
PY
echo "[CONTACT-FROZEN8][DONE] $OUT/FROZEN_COHORT_COMPARISON.json"
