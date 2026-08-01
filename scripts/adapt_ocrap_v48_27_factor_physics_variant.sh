#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

FINAL_RUN="${RUN:?RUN is required}"
SOURCE_CKPT="${INIT_CKPT:?INIT_CKPT is required}"
FACTOR_RUN="$FINAL_RUN/factor_stage"
rm -rf "$FACTOR_RUN"
mkdir -p "$FACTOR_RUN"

# Stage 1: dense raw-benefit ordering + five non-compensatory harm factors.
RUN="$FACTOR_RUN" INIT_CKPT="$SOURCE_CKPT" \
EVIDENCE_ADMISSION_HEAD=false \
EVIDENCE_COMPONENT_COUNT=5 \
ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=false \
ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT="${FACTOR_BENEFIT_LISTWISE_WEIGHT:-0.50}" \
ORDINAL_EVIDENCE_SAFE_UTILITY_REGRESSION_WEIGHT=0 \
ORDINAL_EVIDENCE_SAFE_UTILITY_LISTWISE_WEIGHT=0 \
ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_WEIGHT=0 \
ORDINAL_EVIDENCE_ADMISSION_WEIGHT=0 \
SETWISE_W=0 \
SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0 \
OPPORTUNITY_ADMISSION_WEIGHT=0 HARM_ADMISSION_WEIGHT=0 \
BEST_METRIC=direct_factor_selection_risk \
EVIDENCE_ADAPT_EPOCHS="${FACTOR_EPOCHS:-${EVIDENCE_ADAPT_EPOCHS:-20}}" \
EVIDENCE_ADAPT_PATIENCE="${FACTOR_PATIENCE:-${EVIDENCE_ADAPT_PATIENCE:-6}}" \
  bash scripts/adapt_ocrap_v48_27_factor_physics_single_stage.sh

FACTOR_CKPT="$FACTOR_RUN/model_v48_trac_sr/best.pt"
[[ -f "$FACTOR_CKPT" ]] || { echo "missing factor checkpoint $FACTOR_CKPT" >&2; exit 30; }

# Stage 2: freeze factor heads and train only deployment-exact admission.
rm -rf "$FINAL_RUN/model_v48_trac_sr" "$FINAL_RUN/calibration"
RUN="$FINAL_RUN" INIT_CKPT="$FACTOR_CKPT" \
EVIDENCE_ADMISSION_HEAD=true \
EVIDENCE_COMPONENT_COUNT=5 \
EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_evidence_concord_admission_calibrator \
ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=false \
ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT=0 \
ORDINAL_EVIDENCE_COMPONENT_TAIL_WEIGHT=0 \
ORDINAL_EVIDENCE_SAFE_UTILITY_REGRESSION_WEIGHT="${ADMISSION_SAFE_UTILITY_REGRESSION_WEIGHT:-0.50}" \
ORDINAL_EVIDENCE_SAFE_UTILITY_LISTWISE_WEIGHT="${ADMISSION_SAFE_UTILITY_LISTWISE_WEIGHT:-0.25}" \
ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_WEIGHT="${ADMISSION_FRONTIER_PAIRWISE_WEIGHT:-0.25}" \
ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_MARGIN="${ADMISSION_FRONTIER_MARGIN:-0.05}" \
ORDINAL_EVIDENCE_ADMISSION_WEIGHT="${ADMISSION_BINARY_WEIGHT:-0.50}" \
SETWISE_W="${ADMISSION_SETWISE_WEIGHT:-0.50}" \
SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0 \
OPPORTUNITY_ADMISSION_WEIGHT=0 HARM_ADMISSION_WEIGHT=0 \
EVIDENCE_ADMISSION_BOUNDED=true \
BEST_METRIC=direct_integrity_selection_risk \
EVIDENCE_ADAPT_EPOCHS="${ADMISSION_EPOCHS:-${EVIDENCE_ADAPT_EPOCHS:-18}}" \
EVIDENCE_ADAPT_PATIENCE="${ADMISSION_PATIENCE:-${EVIDENCE_ADAPT_PATIENCE:-6}}" \
  bash scripts/adapt_ocrap_v48_27_factor_physics_single_stage.sh

python - "$FINAL_RUN" "$SOURCE_CKPT" "$FACTOR_CKPT" <<'PY'
import hashlib,json,pathlib,sys,time
run,source,factor=map(pathlib.Path,sys.argv[1:4])
final=run/'model_v48_trac_sr'/'best.pt'
if not final.is_file(): raise SystemExit(f'missing final checkpoint: {final}')
doc={
 'event':'v48_27_two_stage_factor_admission_complete','created_unix':time.time(),
 'source_checkpoint':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'factor_checkpoint':str(factor),'factor_sha256':hashlib.sha256(factor.read_bytes()).hexdigest(),
 'final_checkpoint':str(final),'final_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
 'stage1_trainable':['benefit_calibrator','five_component_harm_calibrator'],
 'stage2_trainable':['admission_calibrator'],
 'opportunity_semantics':'raw_benefit',
 'gate_positive_semantics':'safe_benefit',
 'component_harm_components':['drs','deployability','gap','hard_rule','harm_proxy'],
 'test_roots_read':False,
}
(run/'TWO_STAGE_TRAINING_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
