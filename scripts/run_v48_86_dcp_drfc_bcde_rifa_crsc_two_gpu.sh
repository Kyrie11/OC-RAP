#!/usr/bin/env bash
# V48.86 OC-CRSC: Observation-Consistent Counterfactual Recovery Supervision Contract.
# Entry: V48.85 action-response representation STOP with strong Huber/Contact recovery but selectivity relapse.
# Two equal-capacity arms using the Q85 action-response representation (state gate OFF):
#   S86_RESPONSE_INTERVAL : supervise only candidate-minus-nominal physical response interval.
#   T86_SELECTIVE_RESPONSE: same + noncompensatory safe-benefit / structural-harm response constraints.
export OCRAP_V48_74_SIGNED_VIABILITY=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"; export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"; export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
REFERENCE_A="${V4886_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V4886_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
Q85_RUN="${V4886_Q85:-$BASE_OUT/ocrap_v48_85_dcp_drfc_bcde_rifa_sarr_action}"
V85_COMPARE="${V4886_V85_COMPARE:-$BASE_OUT/OC-RAP-v48.85-DCP-DRFC-BCDE-RIFA-OC-SARR-comparison.json}"
S_RUN="$BASE_OUT/ocrap_v48_86_dcp_drfc_bcde_rifa_crsc_response"; T_RUN="$BASE_OUT/ocrap_v48_86_dcp_drfc_bcde_rifa_crsc_main"
RUNTIME="$BASE_OUT/OC-RAP-v48.86-runtime-code-contract.json"; REF="$BASE_OUT/OC-RAP-v48.86-reference-reuse-contract.json"; RESP_IDX="$BASE_OUT/OC-RAP-v48.86-train-dev-action-response-truth-index.jsonl"; RESP_SUM="$BASE_OUT/OC-RAP-v48.86-train-dev-action-response-truth-index-summary.json"; AUDIT="$BASE_OUT/OC-RAP-v48.86-OC-CRSC-audit.json"; COMPARE="$BASE_OUT/OC-RAP-v48.86-DCP-DRFC-BCDE-RIFA-OC-CRSC-comparison.json"; COMPLETE="$BASE_OUT/OC-RAP-v48.86-PIPELINE_COMPLETE.json"
CACHE="${V4886_TENSOR_CACHE_DIR:-$BASE_OUT/.ocrap_v48_78_tensor_cache}"
mkdir -p "$BASE_OUT" "$CACHE"; rm -rf "$S_RUN" "$T_RUN"; rm -f "$RUNTIME" "$REF" "$RESP_IDX" "$RESP_SUM" "$AUDIT" "$COMPARE" "$COMPLETE"
python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF"
python tools/check_v48_86_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V85_COMPARE" "$L80_RUN" "$Q85_RUN" <<'PY'
import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text());d=p.get('preregistered_decision') or {}
if not(p.get('valid') and d.get('status')=='STATE_ACTION_RECOVERY_REPRESENTATION_STOP' and not d.get('action_response_representation_go')):raise SystemExit('V48.85 STOP prerequisite missing')
for x in sys.argv[2:]:
 if not pathlib.Path(x).is_dir():raise SystemExit(f'missing historical run {x}')
PY
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"; DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"; TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
TRAIN_IDX="$BASE_OUT/OC-RAP-v48.82-train-dev-physical-interval-truth-index.jsonl"; TRAIN_SUM="$BASE_OUT/OC-RAP-v48.82-train-dev-physical-interval-truth-index-summary.json"; EVAL_IDX="$BASE_OUT/OC-RAP-v48.82-dev-certificate-physical-interval-truth-index.jsonl"; EVAL_SUM="$BASE_OUT/OC-RAP-v48.82-dev-certificate-physical-interval-truth-index-summary.json"
python tools/check_v48_85_truth_index_reuse.py --index "$TRAIN_IDX" --summary "$TRAIN_SUM" --roles train_near,train_contact,dev_near,dev_contact
python tools/check_v48_85_truth_index_reuse.py --index "$EVAL_IDX" --summary "$EVAL_SUM" --roles dev_near,dev_contact,certificate_near,certificate_contact
python tools/build_v48_86_action_response_truth_index.py --absolute-index "$TRAIN_IDX" --pcd-index "$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" --pcd-index "$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" --output "$RESP_IDX" --summary "$RESP_SUM"
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false EVIDENCE_UNBOUNDED_HARM_FACTORS=false EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0 EVIDENCE_COMPONENT_SCALE=6.0 EVIDENCE_COMPONENT_HEADS=true EVIDENCE_COMPONENT_COUNT=5 EVIDENCE_DUAL_INTERACTION_BRIDGE=true EVIDENCE_ROCT_BENEFIT=true EVIDENCE_ROCT_DEPLOYABILITY=true EVIDENCE_ROCT_SCALE=3.0 EVIDENCE_ROCT_ALPHA=0.20 EVIDENCE_ROCT_BETA=0.20 EVIDENCE_ROCT_TOP_M=8 EVIDENCE_ROCT_OPTION_TEMPERATURE=0.35 EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050 EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true EVIDENCE_COMMON_MEASURE_ROOT_MASS=false EVIDENCE_ADMISSION_HEAD=false EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction PROPOSAL_TOP_K=5
export TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class EVAL_OPTION_EXECUTION_SEMANTICS=observation_class OPTION_EXECUTION_SEMANTICS=observation_class
train_one(){
 local arm="$1" v="$2" gpu="$3" objective="$4" best_metric="$5" run="$6"; local src="$REFERENCE_A/candidates/$v" dst="$run/candidates/$v"; mkdir -p "$dst" "$run/logs"
 RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" INIT_CKPT="$src/model_v48_trac_sr/best.pt" TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_action_response_adapter STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_action_response_adapter ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT=true SEMANTIC_WITNESS_ROUTE_ALIGNMENT=true SEMANTIC_WITNESS_REENTRY_ALIGNMENT=true SEMANTIC_WITNESS_CONTROL_PROJECTION=true SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false SEMANTIC_WITNESS_PROJECTION_FIDELITY=false SEMANTIC_WITNESS_ACTIVE_CONSTRAINT_TYPED_SOURCE=false SEMANTIC_WITNESS_ROOT_TAIL_SOURCE=false SEMANTIC_WITNESS_TAIL_LOCALIZATION=false SEMANTIC_WITNESS_STRUCTURED_TAIL_FIELD=false SEMANTIC_WITNESS_SIGNED_TAIL_CHANNELS=false SEMANTIC_WITNESS_COUNTERFACTUAL_TAIL_RESPONSE=false SEMANTIC_WITNESS_ACTION_RESPONSE_ADAPTER=true SEMANTIC_WITNESS_ACTION_RESPONSE_STATE_CONDITIONING=false ABSOLUTE_FEASIBILITY_WEIGHT=1.0 ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=structural_interval_bounds ABSOLUTE_FEASIBILITY_TRUTH_INDEX="$TRAIN_IDX" ACTION_RESPONSE_TRUTH_INDEX="$RESP_IDX" ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE="$objective" BEST_METRIC="$best_metric" BEST_METRIC_MIN_DELTA=0.00001 EVIDENCE_ADAPT_EPOCHS="${V4886_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4886_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4886_LR:-0.001}" MAX_EVIDENCE_CALIBRATOR_PARAMS=100000 PERSISTENT_TENSOR_CACHE=true PERSISTENT_TENSOR_CACHE_DIR="$CACHE" PERSISTENT_TENSOR_CACHE_BUILD_WORKERS="${PERSISTENT_TENSOR_CACHE_BUILD_WORKERS:-8}" SAVE_EVERY_EPOCH=false SAVE_LATEST=false OCRAP_ALGORITHM_VERSION="v48.86-OC-CRSC-$arm" bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$run/logs/${arm}_${v}.log" 2>&1
 python tools/check_v48_86_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --truth-index "$TRAIN_IDX" --response-index "$RESP_IDX" --objective "$objective" --output "$dst/V48_86_STAGE_I_STATE_ISOLATION.json"
}
mkdir -p "$S_RUN/candidates" "$T_RUN/candidates"
for v in balanced precision; do
 set +e; train_one S86_RESPONSE_INTERVAL "$v" "$GPU0" counterfactual_response_interval_huber direct_absolute_counterfactual_response_interval_huber "$S_RUN" & p0=$!; train_one T86_SELECTIVE_RESPONSE "$v" "$GPU1" counterfactual_selective_response direct_absolute_counterfactual_selective_response_loss "$T_RUN" & p1=$!; wait "$p0";r0=$?;wait "$p1";r1=$?;set -e; [[ $r0 == 0 && $r1 == 0 ]] || exit 30
done
cal(){ local run="$1" tag="$2"; set +e; OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.86-$tag-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.86.0-OC-CRSC-$tag" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_86_${tag}_certificate_controller.log" 2>&1; rc=$?;set -e;case $rc in 0|20);;*)exit 30;;esac; }
cal "$S_RUN" S86; cal "$T_RUN" T86
python tools/audit_v48_86_crsc.py --l80 "$L80_RUN" --q85 "$Q85_RUN" --s86 "$S_RUN" --t86 "$T_RUN" --truth-index "$EVAL_IDX" --truth-summary "$EVAL_SUM" --output "$AUDIT"
python tools/compare_v48_86_crsc.py --audit "$AUDIT" --v85-comparison "$V85_COMPARE" --output "$COMPARE"
python tools/check_v48_86_pipeline_complete.py --s-run "$S_RUN" --t-run "$T_RUN" --runtime "$RUNTIME" --reference "$REF" --audit "$AUDIT" --comparison "$COMPARE" --response-summary "$RESP_SUM" --output "$COMPLETE"
cd "$BASE_OUT"; for r in "$S_RUN" "$T_RUN"; do b=$(basename "$r");rm -f "$b.zip";zip -qr "$b.zip" "$b";done
zip -qj OC-RAP-v48.86-OC-CRSC-audits.zip "$REF" "$RUNTIME" "$RESP_SUM" "$EVAL_SUM" "$AUDIT" "$COMPARE" "$COMPLETE"
echo "V48.86 complete: upload $(basename "$S_RUN").zip + $(basename "$T_RUN").zip + OC-RAP-v48.86-OC-CRSC-audits.zip"
