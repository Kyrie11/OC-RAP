#!/usr/bin/env bash
set -euo pipefail
export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export BASE_RUN=${BASE_RUN:?set BASE_RUN to a trained v45 OC-RAVE run}
export CKPT=${CKPT:-$BASE_RUN/model_v45_rave/best.pt}
export CAL_DIR=${CAL_DIR:-$BASE_RUN/calibration}
export CAL_GPU=${CAL_GPU:-0}
mkdir -p "$CAL_DIR"
[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 2; }

for bucket in near contact; do
  if [[ "$bucket" == near ]]; then
    data=${RAVE_CAL_NEAR_DATA:-$OCRAP_ROOT/val_near_contact}; harm=${NEAR_MAX_HARM:-0.55}; hard=${NEAR_MAX_HARD:-0.0}
  else
    data=${RAVE_CAL_CONTACT_DATA:-$OCRAP_ROOT/val_contact}; harm=${CONTACT_MAX_HARM:-0.70}; hard=${CONTACT_MAX_HARD:-1.0}
  fi
  CUDA_VISIBLE_DEVICES="$CAL_GPU" PYTHONUNBUFFERED=1 python -u tools/calibrate_direct_value_risk_v45.py \
    --dataset "$data" --checkpoint "$CKPT" --bucket "$bucket" \
    --output "$CAL_DIR/direct_value_risk_${bucket}_v45.json" \
    --rows-output "$CAL_DIR/direct_value_risk_${bucket}_v45.rows.jsonl" \
    --required-min-groups ${RAVE_MIN_GROUPS:-60} \
    --macro-ids ${RAVE_DIRECT_MACRO_IDS:-2,3,5,6,7} \
    --min-nominal-deviation ${RAVE_MIN_DEVIATION:-0.002} \
    --max-hard "$hard" --max-harm "$harm" \
    --positive-gain ${RAVE_POSITIVE_GAIN:-0.015} \
    --min-opportunity ${RAVE_MIN_OPPORTUNITY:-0.0} \
    --min-score-advantage ${RAVE_MIN_SCORE_ADV:--0.10} \
    --min-fit-selected ${RAVE_MIN_FIT_SELECTED:-4} \
    --min-fit-precision ${RAVE_MIN_FIT_PRECISION:-0.55} \
    --max-fit-harmful-selected-rate ${RAVE_MAX_FIT_HARMFUL_RATE:-0.20} \
    --min-verify-selected ${RAVE_MIN_VERIFY_SELECTED:-2} \
    --min-verify-precision ${RAVE_MIN_VERIFY_PRECISION:-0.50} \
    --max-verify-harmful-group-ucb ${RAVE_MAX_VERIFY_HARMFUL_UCB:-0.06} \
    2>&1 | tee "$CAL_DIR/direct_value_risk_${bucket}_v45.log"
done

python - "$CAL_DIR" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); ok=True
for bucket in ('near','contact'):
    d=json.load(open(root/f'direct_value_risk_{bucket}_v45.json'))
    print(bucket, {
      'valid': d.get('valid_for_deployment'),
      'opportunity_threshold': d.get('direct_value_opportunity_threshold'),
      'score_threshold': d.get('direct_value_threshold'),
      'fit_selected': (d.get('fit') or {}).get('num_selected'),
      'verify_selected': (d.get('verify') or {}).get('num_selected'),
      'verify_precision': (d.get('verify') or {}).get('challenge_precision'),
      'verify_harm_ucb': (d.get('verify') or {}).get('harmful_group_exposure_ucb90'),
      'warnings': d.get('warnings'),
    })
    ok &= bool(d.get('valid_for_deployment', False))
raise SystemExit(0 if ok else 3)
PY
