#!/usr/bin/env bash
# V48.85 OC-SARR: Observation-Consistent State-Action Recovery Representation.
# Preregistered after V48.84 Stage-I action-observability STOP.
# Two equal-capacity arms:
#   Q85_ACTION_RESPONSE       : raw executable candidate-minus-nominal action response only
#   R85_STATE_ACTION_MAIN     : same response, parameter-free conditioned by nominal root state
# Shared Stage-I/root decoder, relative ranker, OC-MERO, truth contract and boundary transport remain frozen.
export OCRAP_V48_74_SIGNED_VIABILITY=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
REFERENCE_A="${V4885_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V4885_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V84_COMPARE="${V4885_V84_COMPARE:-$BASE_OUT/OC-RAP-v48.84-DCP-DRFC-BCDE-RIFA-OC-SAOP-comparison.json}"
Q_RUN="$BASE_OUT/ocrap_v48_85_dcp_drfc_bcde_rifa_sarr_action"
R_RUN="$BASE_OUT/ocrap_v48_85_dcp_drfc_bcde_rifa_sarr_main"
RUNTIME="$BASE_OUT/OC-RAP-v48.85-runtime-code-contract.json"
REF="$BASE_OUT/OC-RAP-v48.85-reference-reuse-contract.json"
AUDIT="$BASE_OUT/OC-RAP-v48.85-OC-SARR-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.85-DCP-DRFC-BCDE-RIFA-OC-SARR-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.85-PIPELINE_COMPLETE.json"
CACHE="${V4885_TENSOR_CACHE_DIR:-$BASE_OUT/.ocrap_v48_78_tensor_cache}"
mkdir -p "$BASE_OUT" "$CACHE"; rm -rf "$Q_RUN" "$R_RUN"; rm -f "$RUNTIME" "$REF" "$AUDIT" "$COMPARE" "$COMPLETE"

python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF"
python tools/check_v48_85_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V84_COMPARE" "$L80_RUN" <<'PY'
import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text()); d=p.get('preregistered_decision') or {}
if not(p.get('valid') and d.get('status')=='STAGE_I_ACTION_OBSERVABILITY_STOP' and not d.get('stage_i_action_observability_go')):
    raise SystemExit('V48.84 Stage-I action-observability STOP prerequisite missing')
if not pathlib.Path(sys.argv[2]).is_dir(): raise SystemExit('missing historical L80 run')
PY

CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"

# V48.85 keeps the V48.80 structural-interval truth scaffold frozen. Reuse byte-
# verified V48.82 indices when possible; rebuild the identical V48.80 truth object
# only if schema/SHA/role verification fails.
TRAIN_IDX82="$BASE_OUT/OC-RAP-v48.82-train-dev-physical-interval-truth-index.jsonl"
TRAIN_SUM82="$BASE_OUT/OC-RAP-v48.82-train-dev-physical-interval-truth-index-summary.json"
EVAL_IDX82="$BASE_OUT/OC-RAP-v48.82-dev-certificate-physical-interval-truth-index.jsonl"
EVAL_SUM82="$BASE_OUT/OC-RAP-v48.82-dev-certificate-physical-interval-truth-index-summary.json"
TRAIN_IDX="$TRAIN_IDX82"; TRAIN_SUM="$TRAIN_SUM82"; EVAL_IDX="$EVAL_IDX82"; EVAL_SUM="$EVAL_SUM82"
if ! python tools/check_v48_85_truth_index_reuse.py --index "$TRAIN_IDX82" --summary "$TRAIN_SUM82" --roles train_near,train_contact,dev_near,dev_contact >/dev/null 2>&1; then
  TRAIN_IDX="$BASE_OUT/OC-RAP-v48.85-train-dev-physical-interval-truth-index.jsonl"; TRAIN_SUM="$BASE_OUT/OC-RAP-v48.85-train-dev-physical-interval-truth-index-summary.json"
  rm -f "$TRAIN_IDX" "$TRAIN_SUM"
  python tools/build_v48_80_interval_truth_index.py --root train_near="$TRAIN_NEAR" --root train_contact="$TRAIN_CONTACT" --root dev_near="$DEV_NEAR" --root dev_contact="$DEV_CONTACT" --output "$TRAIN_IDX" --summary "$TRAIN_SUM" --workers "${V4885_TRUTH_INDEX_WORKERS:-8}"
fi
python tools/check_v48_85_truth_index_reuse.py --index "$TRAIN_IDX" --summary "$TRAIN_SUM" --roles train_near,train_contact,dev_near,dev_contact

export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false EVIDENCE_UNBOUNDED_HARM_FACTORS=false EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0 EVIDENCE_COMPONENT_SCALE=6.0 EVIDENCE_COMPONENT_HEADS=true EVIDENCE_COMPONENT_COUNT=5 EVIDENCE_DUAL_INTERACTION_BRIDGE=true EVIDENCE_ROCT_BENEFIT=true EVIDENCE_ROCT_DEPLOYABILITY=true EVIDENCE_ROCT_SCALE=3.0 EVIDENCE_ROCT_ALPHA=0.20 EVIDENCE_ROCT_BETA=0.20 EVIDENCE_ROCT_TOP_M=8 EVIDENCE_ROCT_OPTION_TEMPERATURE=0.35 EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050 EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true EVIDENCE_COMMON_MEASURE_ROOT_MASS=false EVIDENCE_ADMISSION_HEAD=false EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction PROPOSAL_TOP_K=5
export TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class EVAL_OPTION_EXECUTION_SEMANTICS=observation_class OPTION_EXECUTION_SEMANTICS=observation_class

train_one(){
  local arm="$1" v="$2" gpu="$3" state_conditioned="$4" run="$5"
  local src="$REFERENCE_A/candidates/$v" dst="$run/candidates/$v"; mkdir -p "$dst" "$run/logs"
  RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" INIT_CKPT="$src/model_v48_trac_sr/best.pt" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" \
  EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_action_response_adapter STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_action_response_adapter \
  ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT=true SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=false \
  SEMANTIC_WITNESS_ROUTE_ALIGNMENT=true SEMANTIC_WITNESS_REENTRY_ALIGNMENT=true SEMANTIC_WITNESS_CONTROL_PROJECTION=true SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false SEMANTIC_WITNESS_PROJECTION_FIDELITY=false \
  SEMANTIC_WITNESS_ACTIVE_CONSTRAINT_TYPED_SOURCE=false SEMANTIC_WITNESS_ROOT_TAIL_SOURCE=false SEMANTIC_WITNESS_TAIL_LOCALIZATION=false SEMANTIC_WITNESS_STRUCTURED_TAIL_FIELD=false SEMANTIC_WITNESS_SIGNED_TAIL_CHANNELS=false SEMANTIC_WITNESS_COUNTERFACTUAL_TAIL_RESPONSE=false \
  SEMANTIC_WITNESS_ACTION_RESPONSE_ADAPTER=true SEMANTIC_WITNESS_ACTION_RESPONSE_STATE_CONDITIONING="$state_conditioned" \
  ABSOLUTE_FEASIBILITY_WEIGHT=1.0 ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=structural_interval_bounds ABSOLUTE_FEASIBILITY_TRUTH_INDEX="$TRAIN_IDX" ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE=signed_margin_interval_huber \
  BEST_METRIC=direct_absolute_signed_margin_interval_huber BEST_METRIC_MIN_DELTA=0.00001 EVIDENCE_ADAPT_EPOCHS="${V4885_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4885_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4885_LR:-0.001}" \
  MAX_EVIDENCE_CALIBRATOR_PARAMS=100000 PERSISTENT_TENSOR_CACHE=true PERSISTENT_TENSOR_CACHE_DIR="$CACHE" PERSISTENT_TENSOR_CACHE_BUILD_WORKERS="${PERSISTENT_TENSOR_CACHE_BUILD_WORKERS:-8}" SAVE_EVERY_EPOCH=false SAVE_LATEST=false \
  OCRAP_ALGORITHM_VERSION="v48.85-OC-SARR-$arm" bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$run/logs/${arm}_${v}.log" 2>&1
  if [[ "$state_conditioned" == "true" ]]; then
    python tools/check_v48_85_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --truth-index "$TRAIN_IDX" --state-conditioned --output "$dst/V48_85_STAGE_I_STATE_ISOLATION.json"
  else
    python tools/check_v48_85_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --truth-index "$TRAIN_IDX" --output "$dst/V48_85_STAGE_I_STATE_ISOLATION.json"
  fi
}
mkdir -p "$Q_RUN/candidates" "$R_RUN/candidates"
# Pair Q/R on the two A30s for each frozen reference variant. This keeps the
# representation intervention clean while avoiding unnecessary four-run serial execution.
for v in balanced precision; do
  set +e
  train_one Q85_ACTION_RESPONSE "$v" "$GPU0" false "$Q_RUN" & p0=$!
  train_one R85_STATE_ACTION_MAIN "$v" "$GPU1" true "$R_RUN" & p1=$!
  wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
  [[ $r0 == 0 && $r1 == 0 ]] || exit 30
done

cal(){
  local run="$1" tag="$2"
  set +e
  OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.85-$tag-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.85.1-OC-SARR-ENGFIX-$tag" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_85_${tag}_certificate_controller.log" 2>&1
  rc=$?; set -e; case $rc in 0|20);;*) exit 30;;esac
}
cal "$Q_RUN" Q85
cal "$R_RUN" R85

if ! python tools/check_v48_85_truth_index_reuse.py --index "$EVAL_IDX82" --summary "$EVAL_SUM82" --roles dev_near,dev_contact,certificate_near,certificate_contact >/dev/null 2>&1; then
  EVAL_IDX="$BASE_OUT/OC-RAP-v48.85-dev-certificate-physical-interval-truth-index.jsonl"; EVAL_SUM="$BASE_OUT/OC-RAP-v48.85-dev-certificate-physical-interval-truth-index-summary.json"
  rm -f "$EVAL_IDX" "$EVAL_SUM"
  python tools/build_v48_80_interval_truth_index.py --root dev_near="$DEV_NEAR" --root dev_contact="$DEV_CONTACT" --root certificate_near="$CERT_NEAR" --root certificate_contact="$CERT_CONTACT" --output "$EVAL_IDX" --summary "$EVAL_SUM" --workers "${V4885_TRUTH_INDEX_WORKERS:-8}"
fi
python tools/check_v48_85_truth_index_reuse.py --index "$EVAL_IDX" --summary "$EVAL_SUM" --roles dev_near,dev_contact,certificate_near,certificate_contact
python tools/audit_v48_85_sarr.py --l80 "$L80_RUN" --q85 "$Q_RUN" --r85 "$R_RUN" --truth-index "$EVAL_IDX" --truth-summary "$EVAL_SUM" --output "$AUDIT"
python tools/compare_v48_85_sarr.py --audit "$AUDIT" --v84-comparison "$V84_COMPARE" --output "$COMPARE"
python tools/check_v48_85_pipeline_complete.py --q-run "$Q_RUN" --r-run "$R_RUN" --runtime "$RUNTIME" --reference "$REF" --audit "$AUDIT" --comparison "$COMPARE" --train-truth-index "$TRAIN_IDX" --eval-truth-index "$EVAL_IDX" --output "$COMPLETE"
cd "$BASE_OUT"
for r in "$Q_RUN" "$R_RUN"; do b=$(basename "$r"); rm -f "$b.zip"; zip -qr "$b.zip" "$b"; done
zip -qj OC-RAP-v48.85-OC-SARR-audits.zip "$REF" "$RUNTIME" "$TRAIN_SUM" "$EVAL_SUM" "$AUDIT" "$COMPARE" "$COMPLETE"
echo "V48.85 complete: upload $(basename "$Q_RUN").zip + $(basename "$R_RUN").zip + OC-RAP-v48.85-OC-SARR-audits.zip"
