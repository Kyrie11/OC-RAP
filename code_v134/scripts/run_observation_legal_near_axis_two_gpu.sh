#!/usr/bin/env bash
# Observation-legal WOMD v1.3.1 Near re-adjudication + downstream system-axis diagnosis.
# Engineering/protocol repair only: frozen checkpoints, calibration, candidate/recovery
# libraries, horizons, thresholds, Waymax dynamics, and recovery mechanism are unchanged.
set -Eeuo pipefail
ORIGIN_REPO="${OCRAP_ORIGIN_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
RUN_ID="${OCRAP_ROUTE_LEGAL_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}"; export OCRAP_ROUTE_LEGAL_RUN_ID="$RUN_ID"

if [[ "${OCRAP_ROUTE_LEGAL_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  SNAPSHOT_REPO="$BASE_OUT/ocrap_v48_124_10_2_execution_snapshots/$RUN_ID/OC-RAP"
  python "$ORIGIN_REPO/tools/create_fixed_main_execution_snapshot.py" --repo "$ORIGIN_REPO" --output "$SNAPSHOT_REPO" --run-id "$RUN_ID"
  exec env OCRAP_ROUTE_LEGAL_SNAPSHOT_ACTIVE=1 OCRAP_ORIGIN_REPO="$ORIGIN_REPO" OCRAP_REPO="$SNAPSHOT_REPO" OCRAP_ROUTE_LEGAL_RUN_ID="$RUN_ID" OCRAP_CONSTRAINT_AUDIT_MODE=route_legal_near_axis BASE_OUT="$BASE_OUT" GPU0="$GPU0" GPU1="$GPU1" bash "$SNAPSHOT_REPO/scripts/run_observation_legal_near_axis_two_gpu.sh"
fi
REPO="${OCRAP_REPO:?snapshot repo missing}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"

OUT_ROOT="${OCRAP_ROUTE_LEGAL_OUT_ROOT:-$BASE_OUT/ocrap_v48_124_10_2_observation_legal_near_axis}"
OUT="$OUT_ROOT/$RUN_ID"; REF="$OUT/reference_v481249"
mkdir -p "$OUT" "$REF" "$OUT/base" "$OUT/nominal" "$OUT/cohort" "$OUT/diagnostic" "$OUT/merged" "$OUT/comparisons" "$OUT/provenance" "$OUT/support" "$OUT/jax_cache" "$OUT/confirmation"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.124.10.2-OBSERVATION-LEGAL-NEAR-results.zip"
RESULTS_MANIFEST="$OUT/OC-RAP-v48.124.10.2-result-bundle-manifest.json"
package_results(){ local rc="$1"; python tools/package_near_axis_results.py --root "$OUT" --output "$RESULTS_ZIP" --manifest "$RESULTS_MANIFEST" --exit-code "$rc"; }
on_exit(){ local rc=$?; trap - EXIT; package_results "$rc" || true; exit "$rc"; }
trap on_exit EXIT
python tools/package_runtime_source_snapshot.py --repo "$REPO" --output "$OUT/provenance/runtime_source_snapshot.zip" --manifest "$OUT/provenance/runtime_source_snapshot_manifest.json"
RUNTIME="$OUT/provenance/observation_legal_runtime_contract.json"
python tools/check_observation_legal_route_contract.py --repo "$REPO" --output "$RUNTIME"
cp -f "$REPO/EXECUTION_SNAPSHOT.json" "$OUT/provenance/EXECUTION_SNAPSHOT.json"

# V48.124.9 evidence is used only to recover frozen data/checkpoint/calibration provenance.
# Its old Near scientific outcome is NOT reused after the route-legality repair.
python tools/materialize_historical_near_reference.py --base-out "$BASE_OUT" --output-dir "$REF" --historical-workdir "${OCRAP_V481249_WORK:-$BASE_OUT/ocrap_v48_124_contact_anchored_fixed_main}"
BN_OLD="$REF/OC-RAP-v48.124-balanced-near-closed-loop.json"; PN_OLD="$REF/OC-RAP-v48.124-precision-near-closed-loop.json"
BSUP="$REF/OC-RAP-v48.124-balanced-near-dataset-support.json"; PSUP="$REF/OC-RAP-v48.124-precision-near-dataset-support.json"
HIST_ADJ="$REF/OC-RAP-v48.124-fixed-main-adjudication.json"
readarray -t META < <(python - "$BN_OLD" "$PN_OLD" "$BSUP" "$PSUP" <<'PY'
import json,sys
b,p,bs,ps=[json.load(open(x)) for x in sys.argv[1:]]
assert bs['womd_pattern']==ps['womd_pattern']; assert bs['raw_source_role']==ps['raw_source_role']=='validation'
assert b['bucket_dataset']==p['bucket_dataset']==bs['dataset']==ps['dataset']; assert int(b['num_scenes'])==int(p['num_scenes'])==250
print(bs['womd_pattern']); print(b['bucket_dataset']); print(b['gamma_rec']); print(p['gamma_rec'])
PY
)
WOMD_VAL="${META[0]}"; BUCKET="${META[1]}"; BGAMMA="${META[2]}"; PGAMMA="${META[3]}"
L80_RUN="${OCRAP_ORIENTATION_MODEL_RUN:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
resolve_root(){
  local v="$1"
  local r="$L80_RUN/candidates/$v"
  [[ -f "$r/model_v48_trac_sr/best.pt" ]] || r="$L80_RUN/dedicated_candidates/$v"
  printf '%s\n' "$r"
}
BCKPT="$(resolve_root balanced)/model_v48_trac_sr/best.pt"; PCKPT="$(resolve_root precision)/model_v48_trac_sr/best.pt"
[[ -s "$BCKPT" && -s "$PCKPT" ]] || { echo 'missing frozen checkpoints' >&2; exit 30; }
python tools/check_frozen_checkpoint_contract.py --historical-adjudication "$HIST_ADJ" --balanced-checkpoint "$BCKPT" --precision-checkpoint "$PCKPT" --output "$OUT/provenance/frozen_checkpoint_contract.json"

# Shared support scan; route legality itself is enforced while decoding each target.
SUPPORT="$OUT/support/near_dataset_support.json"
python tools/check_closed_loop_dataset_support.py --dataset "$BUCKET" --split test --womd-pattern "$WOMD_VAL" --expected-source-role validation --output "$SUPPORT"

run_ocrap(){ local variant="$1" ckpt="$2" gamma="$3" gpu="$4" dir="$5" config="${6:-}" keys="${7:-}"; mkdir -p "$dir" "$OUT/jax_cache/$variant"; local extra=(); [[ -n "$config" ]] && extra+=(CONFIG="$config"); [[ -n "$keys" ]] && extra+=(TARGET_KEYS_FILE="$keys" REQUIRE_TARGET_KEYS=true); env RUN_DIR="$dir" OUTPUT="$dir/closed_loop_ocrap.json" WOMD_VAL="$WOMD_VAL" EXPECTED_WOMD_ROLE=validation CHECKPOINT="$ckpt" GAMMA_REC="$gamma" GPU="$gpu" MAX_SCENARIOS=0 MAX_STEPS=40 REPLAN_INTERVAL=1 LABEL_MODE=fast AUDIT_EVERY_N_STEPS=0 NUM_CANDIDATES=24 NUM_RECOVERY_OPTIONS=12 BUCKET_DATASET="$BUCKET" BUCKET_SPLIT=test MAX_TARGETS_PER_SCENE=1 PREFLIGHT_SUPPORT_JSON="$SUPPORT" RENDER_TRACE=false SAVE_PARTIAL=true RESUME_FORCE=true INCLUDE_SCENES_IN_RESULT=true RESULT_SCENE_DETAIL=metrics SCENE_JOURNAL_DETAIL=metrics MEMORY_SCENE_DETAIL=metrics PARTIAL_WRITE_EVERY_SCENES=32 PROGRESS_EVERY_STEPS=20 PROFILE_TIMING=true JAX_CACHE_DIR="$OUT/jax_cache/$variant" USE_SDC_PATHS=true REQUIRE_OBSERVATION_LEGAL_ROUTE=true ALLOW_LOGGED_SDC_ROUTE_FALLBACK=false ALLOW_FUTURE_ROUTE_PROXY=false "${extra[@]}" bash scripts/run_ocrap_closed_loop.sh; }

# Full observation-legal frozen-base rerun. This is mandatory because the route
# repair changes planner features and can change the very first intervention.
run_ocrap balanced "$BCKPT" "$BGAMMA" "$GPU0" "$OUT/base/balanced" & PB=$!
run_ocrap precision "$PCKPT" "$PGAMMA" "$GPU1" "$OUT/base/precision" & PP=$!
wait "$PB"; RB=$?; wait "$PP"; RP=$?; [[ $RB == 0 && $RP == 0 ]] || exit 30
BBASE="$OUT/base/balanced/closed_loop_ocrap.json"; PBASE="$OUT/base/precision/closed_loop_ocrap.json"

# Fresh exact-a0 nominal control under the same legal route source.
env OUT="$OUT/nominal" WOMD_ROOT="/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example" NEAR_WOMD="$WOMD_VAL" NEAR_BUCKET="$BUCKET" RUN_SAFE=0 RUN_NEAR=1 RUN_CONTACT=0 CUDA_DEVICES="$GPU0" MAX_SCENARIOS=0 MAX_STEPS=40 RESUME_FORCE=true INCLUDE_SCENES_IN_RESULT=true RESULT_SCENE_DETAIL=metrics USE_SDC_PATHS=true REQUIRE_OBSERVATION_LEGAL_ROUTE=true ALLOW_LOGGED_SDC_ROUTE_FALLBACK=false ALLOW_FUTURE_ROUTE_PROXY=false bash scripts/run_nominal_three_regime_control.sh
NOM="$OUT/nominal/near/closed_loop_nominal.json"

python tools/audit_observation_legal_route_results.py --result nominal="$NOM" --result balanced_base="$BBASE" --result precision_base="$PBASE" --output "$OUT/provenance/base_route_audit.json"
python tools/compare_paired_closed_loop.py "$NOM" "$BBASE" --bootstrap 5000 --seed 2027 --output "$OUT/comparisons/base_balanced_vs_nominal.json"
python tools/compare_paired_closed_loop.py "$NOM" "$PBASE" --bootstrap 5000 --seed 2027 --output "$OUT/comparisons/base_precision_vs_nominal.json"

# Variant-specific intervention cohorts; route repair may change them and they are
# deliberately NOT required to match balanced vs precision.
for v in balanced precision; do
  base="$BBASE"; [[ "$v" == precision ]] && base="$PBASE"
  python tools/build_variant_intervention_cohort.py --result "$base" --target-keys-output "$OUT/cohort/${v}_keys.json" --audit-output "$OUT/cohort/${v}_audit.json"
done

run_worker(){ local v="$1" ckpt="$2" gamma="$3" gpu="$4" base="$5" keys="$6";
  for arm in delta nested; do cfg="$REPO/configs/v48_124_10_near_relative_delta.yaml"; [[ "$arm" == nested ]] && cfg="$REPO/configs/v48_124_10_near_nested_evidence.yaml"; dir="$OUT/diagnostic/$arm/$v"; run_ocrap "$v" "$ckpt" "$gamma" "$gpu" "$dir" "$cfg" "$keys"; python tools/merge_monotone_near_subset.py --baseline-full "$base" --diagnostic-subset "$dir/closed_loop_ocrap.json" --target-keys "$keys" --output "$OUT/merged/${arm}_${v}_full250.json"; python tools/compare_paired_closed_loop.py "$NOM" "$OUT/merged/${arm}_${v}_full250.json" --bootstrap 5000 --seed 2027 --output "$OUT/comparisons/${arm}_${v}_vs_nominal.json"; python tools/compare_paired_closed_loop.py "$base" "$OUT/merged/${arm}_${v}_full250.json" --bootstrap 5000 --seed 2027 --output "$OUT/comparisons/${arm}_${v}_vs_base.json"; done; }
run_worker balanced "$BCKPT" "$BGAMMA" "$GPU0" "$BBASE" "$OUT/cohort/balanced_keys.json" & DB=$!
run_worker precision "$PCKPT" "$PGAMMA" "$GPU1" "$PBASE" "$OUT/cohort/precision_keys.json" & DP=$!
wait "$DB"; RDB=$?; wait "$DP"; RDP=$?; [[ $RDB == 0 && $RDP == 0 ]] || exit 30

python tools/audit_observation_legal_route_results.py --result delta_balanced="$OUT/merged/delta_balanced_full250.json" --result delta_precision="$OUT/merged/delta_precision_full250.json" --result nested_balanced="$OUT/merged/nested_balanced_full250.json" --result nested_precision="$OUT/merged/nested_precision_full250.json" --output "$OUT/provenance/diagnostic_route_audit.json"
ROUTE_AUDIT="$OUT/provenance/all_route_audit.json"
python tools/audit_observation_legal_route_results.py --result nominal="$NOM" --result base_balanced="$BBASE" --result base_precision="$PBASE" --result delta_balanced="$OUT/merged/delta_balanced_full250.json" --result delta_precision="$OUT/merged/delta_precision_full250.json" --result nested_balanced="$OUT/merged/nested_balanced_full250.json" --result nested_precision="$OUT/merged/nested_precision_full250.json" --output "$ROUTE_AUDIT"
ADJ="$OUT/OC-RAP-v48.124.10.2-observation-legal-near-adjudication.json"
python tools/adjudicate_observation_legal_near_axis.py --runtime-contract "$RUNTIME" --route-audit "$ROUTE_AUDIT" --nominal-full "$NOM" --base-balanced-full "$BBASE" --base-balanced-vs-nominal "$OUT/comparisons/base_balanced_vs_nominal.json" --base-precision-full "$PBASE" --base-precision-vs-nominal "$OUT/comparisons/base_precision_vs_nominal.json" --delta-balanced-full "$OUT/merged/delta_balanced_full250.json" --delta-balanced-vs-nominal "$OUT/comparisons/delta_balanced_vs_nominal.json" --delta-balanced-vs-base "$OUT/comparisons/delta_balanced_vs_base.json" --delta-precision-full "$OUT/merged/delta_precision_full250.json" --delta-precision-vs-nominal "$OUT/comparisons/delta_precision_vs_nominal.json" --delta-precision-vs-base "$OUT/comparisons/delta_precision_vs_base.json" --nested-balanced-full "$OUT/merged/nested_balanced_full250.json" --nested-balanced-vs-nominal "$OUT/comparisons/nested_balanced_vs_nominal.json" --nested-balanced-vs-base "$OUT/comparisons/nested_balanced_vs_base.json" --nested-precision-full "$OUT/merged/nested_precision_full250.json" --nested-precision-vs-nominal "$OUT/comparisons/nested_precision_vs_nominal.json" --nested-precision-vs-base "$OUT/comparisons/nested_precision_vs_base.json" --output "$ADJ"

# If an internal diagnostic arm is promoted, automatically pay the cost of one
# fresh 250-scene confirmation now, rather than requiring another analysis-only cycle.
PROMOTED="$(python - "$ADJ" <<'PY'
import json,sys; print(json.load(open(sys.argv[1])).get('promoted_arm') or '')
PY
)"
FINAL="$ADJ"
if [[ "$PROMOTED" == delta || "$PROMOTED" == nested ]]; then
  CFG="$REPO/configs/v48_124_10_near_relative_delta.yaml"; SELECTOR=lcb_constrained_relative_delta
  [[ "$PROMOTED" == nested ]] && CFG="$REPO/configs/v48_124_10_near_nested_evidence.yaml" && SELECTOR=lcb_constrained_nested_evidence
  run_ocrap balanced "$BCKPT" "$BGAMMA" "$GPU0" "$OUT/confirmation/balanced" "$CFG" & CB=$!
  run_ocrap precision "$PCKPT" "$PGAMMA" "$GPU1" "$OUT/confirmation/precision" "$CFG" & CP=$!
  wait "$CB"; RCB=$?; wait "$CP"; RCP=$?; [[ $RCB == 0 && $RCP == 0 ]] || exit 30
  CBF="$OUT/confirmation/balanced/closed_loop_ocrap.json"; CPF="$OUT/confirmation/precision/closed_loop_ocrap.json"
  python tools/audit_observation_legal_route_results.py --result balanced="$CBF" --result precision="$CPF" --output "$OUT/provenance/confirmation_route_audit.json"
  python tools/compare_paired_closed_loop.py "$NOM" "$CBF" --bootstrap 5000 --seed 2027 --output "$OUT/comparisons/confirmation_balanced_vs_nominal.json"
  python tools/compare_paired_closed_loop.py "$NOM" "$CPF" --bootstrap 5000 --seed 2027 --output "$OUT/comparisons/confirmation_precision_vs_nominal.json"
  FINAL="$OUT/OC-RAP-v48.124.10.2-fresh-250-near-confirmation.json"
  python tools/adjudicate_fresh_near_confirmation.py --selector "$SELECTOR" --nominal-full "$NOM" --balanced-full "$CBF" --precision-full "$CPF" --balanced-vs-nominal "$OUT/comparisons/confirmation_balanced_vs_nominal.json" --precision-vs-nominal "$OUT/comparisons/confirmation_precision_vs_nominal.json" --route-audit "$OUT/provenance/confirmation_route_audit.json" --output "$FINAL"
fi

python tools/check_fixed_main_execution_snapshot.py --repo "$REPO"
cp -f "$FINAL" "$BASE_OUT/OC-RAP-v48.124.10.2-OBSERVATION-LEGAL-NEAR-FINAL.json"
package_results 0
trap - EXIT
echo "V48.124.10.2 observation-legal Near run complete: $RESULTS_ZIP"
