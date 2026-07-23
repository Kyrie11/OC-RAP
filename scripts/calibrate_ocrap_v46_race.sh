#!/usr/bin/env bash
set -euo pipefail
export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export BASE_RUN=${BASE_RUN:?set BASE_RUN to a trained v46 OC-RACE run}
export CKPT=${CKPT:-$BASE_RUN/model_v46_race/best.pt}
export CAL_DIR=${CAL_DIR:-$BASE_RUN/calibration}
export CAL_GPU=${CAL_GPU:-0}
export RACE_CONTRACT_MODE=${RACE_CONTRACT_MODE:-development}
mkdir -p "$CAL_DIR"
[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 2; }

case "$RACE_CONTRACT_MODE" in
  development)
    : "${RACE_MIN_GROUPS:=60}"; : "${RACE_MIN_SCENES:=20}"
    : "${RACE_MIN_VERIFY_SELECTED:=2}"; : "${RACE_MIN_VERIFY_PRECISION:=0.50}"
    : "${RACE_MAX_VERIFY_HARMFUL_UCB:=0.12}"
    : "${RACE_MAX_VERIFY_HARMFUL_SELECTED_UCB:=0.75}"
    : "${RACE_MIN_PRED_TEACHER_CORRELATION:=0.10}"
    : "${RACE_MIN_MACRO_POSITIVE_COUNT:=2}"; : "${RACE_MIN_MACRO_POSITIVE_RATE:=0.02}"
    ;;
  final)
    # Publication contract: scene-independent calibration evidence and a
    # conditional false-admission bound among actions that would execute.
    : "${RACE_MIN_GROUPS:=200}"; : "${RACE_MIN_SCENES:=100}"
    : "${RACE_MIN_VERIFY_SELECTED:=25}"; : "${RACE_MIN_VERIFY_PRECISION:=0.80}"
    : "${RACE_MAX_VERIFY_HARMFUL_UCB:=0.06}"
    : "${RACE_MAX_VERIFY_HARMFUL_SELECTED_UCB:=0.10}"
    : "${RACE_MIN_PRED_TEACHER_CORRELATION:=0.15}"
    : "${RACE_MIN_MACRO_POSITIVE_COUNT:=5}"; : "${RACE_MIN_MACRO_POSITIVE_RATE:=0.05}"
    [[ -n "${RACE_CAL_NEAR_DATA:-}" && -n "${RACE_CAL_CONTACT_DATA:-}" ]] || {
      echo "FINAL contract requires dedicated RACE_CAL_NEAR_DATA and RACE_CAL_CONTACT_DATA roots" >&2; exit 2; }
    [[ "${RACE_CAL_NEAR_DATA%/}" != "${OCRAP_ROOT%/}/val_near_contact" ]] || {
      echo "FINAL contract forbids reusing val_near_contact for calibration" >&2; exit 2; }
    [[ "${RACE_CAL_CONTACT_DATA%/}" != "${OCRAP_ROOT%/}/val_contact" ]] || {
      echo "FINAL contract forbids reusing val_contact for calibration" >&2; exit 2; }
    ;;
  *) echo "unknown RACE_CONTRACT_MODE=$RACE_CONTRACT_MODE" >&2; exit 2 ;;
esac
export RACE_MIN_GROUPS RACE_MIN_SCENES RACE_MIN_VERIFY_SELECTED RACE_MIN_VERIFY_PRECISION
export RACE_MAX_VERIFY_HARMFUL_UCB RACE_MAX_VERIFY_HARMFUL_SELECTED_UCB
export RACE_MIN_PRED_TEACHER_CORRELATION RACE_MIN_MACRO_POSITIVE_COUNT RACE_MIN_MACRO_POSITIVE_RATE

echo "OC-RACE calibration contract=$RACE_CONTRACT_MODE groups>=$RACE_MIN_GROUPS scenes>=$RACE_MIN_SCENES selected>=$RACE_MIN_VERIFY_SELECTED precision>=$RACE_MIN_VERIFY_PRECISION conditional_harm_ucb<=$RACE_MAX_VERIFY_HARMFUL_SELECTED_UCB"

for bucket in near contact; do
  if [[ "$bucket" == near ]]; then
    data=${RACE_CAL_NEAR_DATA:-$OCRAP_ROOT/val_near_contact}; harm=${NEAR_MAX_HARM:-0.55}; hard=${NEAR_MAX_HARD:-0.0}
  else
    data=${RACE_CAL_CONTACT_DATA:-$OCRAP_ROOT/val_contact}; harm=${CONTACT_MAX_HARM:-0.70}; hard=${CONTACT_MAX_HARD:-1.0}
  fi
  CUDA_VISIBLE_DEVICES="$CAL_GPU" PYTHONUNBUFFERED=1 python -u tools/calibrate_direct_value_risk_v46.py \
    --dataset "$data" --checkpoint "$CKPT" --bucket "$bucket" \
    --output "$CAL_DIR/direct_value_risk_${bucket}_v46.json" \
    --rows-output "$CAL_DIR/direct_value_risk_${bucket}_v46.rows.jsonl" \
    --contract-mode "$RACE_CONTRACT_MODE" \
    --required-min-groups "$RACE_MIN_GROUPS" \
    --required-min-scenes "$RACE_MIN_SCENES" \
    --fold-unit ${RACE_FOLD_UNIT:-scene} \
    --macro-ids ${RACE_DIRECT_MACRO_IDS:-2,3,5,7} \
    --min-macro-fit-positive-count "$RACE_MIN_MACRO_POSITIVE_COUNT" \
    --min-macro-fit-positive-rate "$RACE_MIN_MACRO_POSITIVE_RATE" \
    --min-nominal-deviation ${RACE_MIN_DEVIATION:-0.002} \
    --max-hard "$hard" --max-harm "$harm" \
    --positive-gain ${RACE_POSITIVE_GAIN:-0.015} \
    --negative-gain ${RACE_NEGATIVE_GAIN:-0.010} \
    --min-opportunity ${RACE_MIN_OPPORTUNITY:-0.0} \
    --min-score-advantage ${RACE_MIN_SCORE_ADV:--0.10} \
    --min-fit-selected ${RACE_MIN_FIT_SELECTED:-4} \
    --min-fit-precision ${RACE_MIN_FIT_PRECISION:-0.55} \
    --max-fit-harmful-selected-rate ${RACE_MAX_FIT_HARMFUL_RATE:-0.20} \
    --min-verify-selected "$RACE_MIN_VERIFY_SELECTED" \
    --min-verify-precision "$RACE_MIN_VERIFY_PRECISION" \
    --max-verify-harmful-group-ucb "$RACE_MAX_VERIFY_HARMFUL_UCB" \
    --max-verify-harmful-selected-ucb "$RACE_MAX_VERIFY_HARMFUL_SELECTED_UCB" \
    --min-pred-teacher-correlation "$RACE_MIN_PRED_TEACHER_CORRELATION" \
    2>&1 | tee "$CAL_DIR/direct_value_risk_${bucket}_v46.log"
done

python - "$CAL_DIR" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); ok=True
for bucket in ('near','contact'):
    d=json.load(open(root/f'direct_value_risk_{bucket}_v46.json'))
    verify=d.get('verify') or {}
    print(bucket, {
      'contract': d.get('contract_mode'),
      'active_valid': d.get('valid_for_active_contract'),
      'development_valid': d.get('valid_for_development'),
      'deployment_valid': d.get('valid_for_deployment'),
      'opportunity_threshold': d.get('direct_value_opportunity_threshold'),
      'score_threshold': d.get('direct_value_threshold'),
      'verify_selected': verify.get('num_selected'),
      'verify_precision': verify.get('challenge_precision'),
      'verify_harm_selected_ucb90': verify.get('harmful_selected_ucb90'),
      'verify_harm_group_ucb90': verify.get('harmful_group_exposure_ucb90'),
      'correlation': d.get('pred_teacher_advantage_correlation'),
      'supported_macro_ids': d.get('supported_macro_ids'),
      'warnings': d.get('warnings'),
    })
    ok &= bool(d.get('valid_for_active_contract', False))
raise SystemExit(0 if ok else 3)
PY
