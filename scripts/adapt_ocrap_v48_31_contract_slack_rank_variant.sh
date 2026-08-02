#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

FINAL_RUN="${RUN:?RUN is required}"
SOURCE_CKPT="${INIT_CKPT:?INIT_CKPT is required}"
GROUP_INDEX="${GROUP_INDEX:?GROUP_INDEX is required}"
FACTOR_RUN="$FINAL_RUN/factor_stage"
ENABLE_SUPPORT_RELIABILITY="${V4831_ENABLE_SUPPORT_RELIABILITY:-1}"
ENABLE_JOINT_STAGE="${V4831_ENABLE_JOINT_STAGE:-1}"
ADMISSION_RUN="$FINAL_RUN/admission_stage"
SUPPORT_JSON="$FINAL_RUN/FACTOR_SUPPORT_CONTRACT.json"
SUPPORT_ENV="$FINAL_RUN/FACTOR_SUPPORT_CONTRACT.env"
rm -rf "$FACTOR_RUN" "$ADMISSION_RUN"
mkdir -p "$FINAL_RUN" "$FACTOR_RUN" "$ADMISSION_RUN"

python tools/build_v48_31_factor_support_contract.py \
  --index "$GROUP_INDEX" --output "$SUPPORT_JSON" --env-output "$SUPPORT_ENV" \
  --macro-ids "${DIRECT_VALUE_MACRO_IDS:-2,3,5,6,7}" \
  --max-hard "${POLICY_METRIC_MAX_HARD:-1.0}" \
  --min-nominal-deviation "${POLICY_METRIC_MIN_NOMINAL_DEVIATION:-0.002}" \
  --min-positive "${FACTOR_SUPPORT_MIN_POSITIVE:-40}" \
  --drs-tolerance "${COMPONENT_HARM_DRS_TOLERANCE:-0.05}" \
  --dep-tolerance "${COMPONENT_HARM_DEP_TOLERANCE:-0.05}" \
  --gap-tolerance "${COMPONENT_HARM_GAP_TOLERANCE:-0.05}" \
  --hard-tolerance "${COMPONENT_HARM_HARD_TOLERANCE:-0.05}" \
  --proxy-tolerance "${COMPONENT_HARM_PROXY_TOLERANCE:-0.05}" \
  --require-readable-samples
# shellcheck disable=SC1090
source "$SUPPORT_ENV"
if [[ "$ENABLE_SUPPORT_RELIABILITY" != 1 ]]; then
  EVIDENCE_COMPONENT_RELIABILITY="1,1,1,1,1"
fi
export EVIDENCE_COMPONENT_RELIABILITY

# Stage 1: learn raw benefit and signed physical margins on the natural
# population.  Every scene-time group appears at most once per epoch; rare safe
# opportunities are emphasized only through loss/group weights, never by
# replacement sampling that changes the deployment prior.
RUN="$FACTOR_RUN" INIT_CKPT="$SOURCE_CKPT" \
EVIDENCE_ADMISSION_HEAD=false \
EVIDENCE_COMPONENT_COUNT="${EVIDENCE_COMPONENT_COUNT:-5}" \
EVIDENCE_ADMISSION_PRIOR_MODE=safety_slack \
ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=false \
ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT="${FACTOR_BENEFIT_LISTWISE_WEIGHT:-0.75}" \
ORDINAL_EVIDENCE_COMPONENT_TAIL_WEIGHT="${FACTOR_COMPONENT_TAIL_WEIGHT:-0.50}" \
ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_WEIGHT="${FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT:-0.75}" \
ORDINAL_EVIDENCE_SAFE_UTILITY_REGRESSION_WEIGHT=0 \
ORDINAL_EVIDENCE_SAFE_UTILITY_LISTWISE_WEIGHT=0 \
ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_WEIGHT=0 \
ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_WEIGHT=0 \
ORDINAL_EVIDENCE_ADMISSION_WEIGHT=0 SETWISE_W=0 \
SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0 \
OPPORTUNITY_ADMISSION_WEIGHT=0 HARM_ADMISSION_WEIGHT=0 \
GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false \
POSITIVE_GROUP_BOOST="${FACTOR_POSITIVE_GROUP_BOOST:-3.0}" \
POSITIVE_MACRO_BALANCE_POWER="${FACTOR_POSITIVE_MACRO_BALANCE_POWER:-0.25}" \
BEST_METRIC=direct_factor_supervised_risk EVALUATE_INITIAL_CHECKPOINT=false \
EVIDENCE_ADAPT_EPOCHS="${FACTOR_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${FACTOR_PATIENCE:-6}" \
  bash scripts/adapt_ocrap_v48_31_contract_slack_rank_single_stage.sh

FACTOR_CKPT="$FACTOR_RUN/model_v48_trac_sr/best.pt"
[[ -f "$FACTOR_CKPT" ]] || { echo "missing factor checkpoint $FACTOR_CKPT" >&2; exit 30; }

# Stage 2: learn a bounded safe-utility admission residual while preserving the
# semantic benefit and factor heads.  Checkpoint selection now uses the exact
# calibration eligibility population and requires safe top-1 support.
RUN="$ADMISSION_RUN" INIT_CKPT="$FACTOR_CKPT" \
EVIDENCE_ADMISSION_HEAD=true EVIDENCE_COMPONENT_COUNT="${EVIDENCE_COMPONENT_COUNT:-5}" \
EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_evidence_concord_admission_calibrator \
EVIDENCE_ADMISSION_PRIOR_MODE=safety_slack \
EVIDENCE_SLACK_TEMPERATURE="${EVIDENCE_SLACK_TEMPERATURE:-0.025}" \
EVIDENCE_SLACK_PENALTY="${EVIDENCE_SLACK_PENALTY:-1.0}" \
EVIDENCE_ADMISSION_SCALE="${EVIDENCE_ADMISSION_SCALE:-2.0}" \
EVIDENCE_ADMISSION_BOUNDED=true \
ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=false \
ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT=0 \
ORDINAL_EVIDENCE_COMPONENT_TAIL_WEIGHT=0 \
ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_WEIGHT=0 \
ORDINAL_EVIDENCE_SAFE_UTILITY_REGRESSION_WEIGHT="${ADMISSION_SAFE_UTILITY_REGRESSION_WEIGHT:-0.75}" \
ORDINAL_EVIDENCE_SAFE_UTILITY_LISTWISE_WEIGHT="${ADMISSION_SAFE_UTILITY_LISTWISE_WEIGHT:-0.50}" \
ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_WEIGHT="${ADMISSION_SAFE_HARD_NEGATIVE_WEIGHT:-1.50}" \
ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_MARGIN="${ADMISSION_SAFE_HARD_NEGATIVE_MARGIN:-0.05}" \
ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_WEIGHT="${ADMISSION_FRONTIER_PAIRWISE_WEIGHT:-0.15}" \
ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_MARGIN="${ADMISSION_FRONTIER_MARGIN:-0.05}" \
ORDINAL_EVIDENCE_ADMISSION_WEIGHT="${ADMISSION_BINARY_WEIGHT:-0.10}" \
ORDINAL_EVIDENCE_ADMISSION_POS_WEIGHT="${ADMISSION_POS_WEIGHT:-2.0}" \
ORDINAL_EVIDENCE_ADMISSION_HARM_NEGATIVE_WEIGHT="${ADMISSION_HARM_NEGATIVE_WEIGHT:-2.5}" \
SETWISE_W="${ADMISSION_SETWISE_WEIGHT:-0.10}" \
SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0 \
OPPORTUNITY_ADMISSION_WEIGHT=0 HARM_ADMISSION_WEIGHT=0 \
GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false \
POSITIVE_GROUP_BOOST="${ADMISSION_POSITIVE_GROUP_BOOST:-3.0}" POSITIVE_MACRO_BALANCE_POWER=0.0 \
BEST_METRIC=direct_contract_safe_rank_risk EVALUATE_INITIAL_CHECKPOINT=false \
EVIDENCE_ADAPT_EPOCHS="${ADMISSION_EPOCHS:-18}" EVIDENCE_ADAPT_PATIENCE="${ADMISSION_PATIENCE:-6}" \
  bash scripts/adapt_ocrap_v48_31_contract_slack_rank_single_stage.sh

ADMISSION_CKPT="$ADMISSION_RUN/model_v48_trac_sr/best.pt"
[[ -f "$ADMISSION_CKPT" ]] || { echo "missing admission checkpoint $ADMISSION_CKPT" >&2; exit 30; }

# Stage 3: low-rate joint semantic refinement.  v48.30 froze the factor heads,
# so a small admission residual could not repair within-group action identity.
# Only the three compact evidence calibrators are trainable here; frozen source
# experts, encoder and proposal policy remain untouched.
rm -rf "$FINAL_RUN/model_v48_trac_sr" "$FINAL_RUN/calibration"
if [[ "$ENABLE_JOINT_STAGE" == 1 ]]; then
RUN="$FINAL_RUN" INIT_CKPT="$ADMISSION_CKPT" \
EVIDENCE_ADMISSION_HEAD=true EVIDENCE_COMPONENT_COUNT="${EVIDENCE_COMPONENT_COUNT:-5}" \
EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE='direct_evidence_concord_benefit_calibrator,direct_evidence_concord_harm_calibrator,direct_evidence_concord_admission_calibrator' \
EVIDENCE_ADMISSION_PRIOR_MODE=safety_slack \
EVIDENCE_SLACK_TEMPERATURE="${EVIDENCE_SLACK_TEMPERATURE:-0.025}" \
EVIDENCE_SLACK_PENALTY="${EVIDENCE_SLACK_PENALTY:-1.0}" \
EVIDENCE_ADMISSION_SCALE="${EVIDENCE_ADMISSION_SCALE:-2.0}" EVIDENCE_ADMISSION_BOUNDED=true \
ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=false \
ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT="${JOINT_BENEFIT_LISTWISE_WEIGHT:-0.15}" \
ORDINAL_EVIDENCE_COMPONENT_TAIL_WEIGHT="${JOINT_COMPONENT_TAIL_WEIGHT:-0.15}" \
ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_WEIGHT="${JOINT_COMPONENT_MARGIN_REGRESSION_WEIGHT:-0.25}" \
ORDINAL_EVIDENCE_SAFE_UTILITY_REGRESSION_WEIGHT="${JOINT_SAFE_UTILITY_REGRESSION_WEIGHT:-0.75}" \
ORDINAL_EVIDENCE_SAFE_UTILITY_LISTWISE_WEIGHT="${JOINT_SAFE_UTILITY_LISTWISE_WEIGHT:-0.75}" \
ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_WEIGHT="${JOINT_SAFE_HARD_NEGATIVE_WEIGHT:-1.50}" \
ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_MARGIN="${JOINT_SAFE_HARD_NEGATIVE_MARGIN:-0.05}" \
ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_WEIGHT="${JOINT_FRONTIER_PAIRWISE_WEIGHT:-0.25}" \
ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_MARGIN="${JOINT_FRONTIER_MARGIN:-0.05}" \
ORDINAL_EVIDENCE_ADMISSION_WEIGHT="${JOINT_ADMISSION_BINARY_WEIGHT:-0.05}" \
ORDINAL_EVIDENCE_ADMISSION_POS_WEIGHT="${JOINT_ADMISSION_POS_WEIGHT:-2.0}" \
ORDINAL_EVIDENCE_ADMISSION_HARM_NEGATIVE_WEIGHT="${JOINT_ADMISSION_HARM_NEGATIVE_WEIGHT:-2.5}" \
SETWISE_W="${JOINT_SETWISE_WEIGHT:-0.10}" \
EVIDENCE_CALIBRATOR_ANCHOR_WEIGHT="${JOINT_CALIBRATOR_ANCHOR_WEIGHT:-0.10}" \
SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0 \
OPPORTUNITY_ADMISSION_WEIGHT=0 HARM_ADMISSION_WEIGHT=0 \
GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false \
POSITIVE_GROUP_BOOST="${JOINT_POSITIVE_GROUP_BOOST:-3.0}" POSITIVE_MACRO_BALANCE_POWER=0.0 \
EVIDENCE_ADAPT_LR="${JOINT_LR:-0.00005}" \
BEST_METRIC=direct_contract_safe_rank_risk EVALUATE_INITIAL_CHECKPOINT=true \
EVIDENCE_ADAPT_EPOCHS="${JOINT_EPOCHS:-12}" EVIDENCE_ADAPT_PATIENCE="${JOINT_PATIENCE:-5}" \
  bash scripts/adapt_ocrap_v48_31_contract_slack_rank_single_stage.sh
else
  cp -a "$ADMISSION_RUN/model_v48_trac_sr" "$FINAL_RUN/model_v48_trac_sr"
  cp "$ADMISSION_RUN/STAGE_ARCHITECTURE.json" "$FINAL_RUN/STAGE_ARCHITECTURE.json"
  cp "$ADMISSION_RUN/POLICY_CONTRACT.env" "$FINAL_RUN/POLICY_CONTRACT.env"
fi

FINAL_CKPT="$FINAL_RUN/model_v48_trac_sr/best.pt"
transfer_extra=()
if [[ "$ENABLE_JOINT_STAGE" != 1 ]]; then transfer_extra+=(--allow-no-joint); fi
python tools/check_v48_31_stage_transfer.py \
  --factor "$FACTOR_CKPT" --admission "$ADMISSION_CKPT" --final "$FINAL_CKPT" \
  --output "$FINAL_RUN/STAGE_TRANSFER_INTEGRITY.json" "${transfer_extra[@]}"

python - "$FINAL_RUN" "$SOURCE_CKPT" "$FACTOR_CKPT" "$ADMISSION_CKPT" "$FINAL_CKPT" "$SUPPORT_JSON" "$ENABLE_SUPPORT_RELIABILITY" "$ENABLE_JOINT_STAGE" <<'PY'
import hashlib,json,pathlib,sys,time
run,source,factor,admission,final,support=map(pathlib.Path,sys.argv[1:7])
support_enabled=sys.argv[7] == "1"
joint_enabled=sys.argv[8] == "1"
for p in (source,factor,admission,final,support):
    if not p.is_file(): raise SystemExit(f'missing v48.31 artifact: {p}')
doc={
 'event':'v48_31_contract_slack_rank_complete','created_unix':time.time(),
 'source_checkpoint':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'factor_checkpoint':str(factor),'factor_sha256':hashlib.sha256(factor.read_bytes()).hexdigest(),
 'admission_checkpoint':str(admission),'admission_sha256':hashlib.sha256(admission.read_bytes()).hexdigest(),
 'final_checkpoint':str(final),'final_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
 'factor_support_contract':str(support),'factor_support_sha256':hashlib.sha256(support.read_bytes()).hexdigest(),
 'stage1_population':'natural_without_replacement',
 'stage2_population':'natural_without_replacement',
 'stage3_population':'natural_without_replacement' if joint_enabled else 'disabled_ablation',
 'stage3_trainable':['benefit_calibrator','component_harm_calibrator','admission_calibrator'] if joint_enabled else [],
 'model_regime_routing':False,
 'continuous_unified_semantics':'raw benefit minus reliability-weighted positive worst safety slack',
 'independent_measured_hard_veto':True,
 'checkpoint_metric':'direct_contract_safe_rank_risk',
 'exact_eligibility_metric':True,
 'support_reliability_enabled': support_enabled,
 'joint_refinement_enabled': joint_enabled,
 'test_roots_read':False,
}
(run/'THREE_STAGE_TRAINING_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
