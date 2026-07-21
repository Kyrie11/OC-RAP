#!/usr/bin/env bash
set -euo pipefail
export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export BASE_RUN=${BASE_RUN:?set BASE_RUN to a trained v42/v43 run}
export CKPT=${CKPT:-}
if [[ -z "$CKPT" ]]; then
  if [[ -f "$BASE_RUN/model_v43_rsc/best.pt" ]]; then CKPT="$BASE_RUN/model_v43_rsc/best.pt";
  else CKPT="$BASE_RUN/model_v42_ocsava/best.pt"; fi
fi
export CAL_DIR=${CAL_DIR:-$BASE_RUN/calibration}
export CAL_GPU=${CAL_GPU:-0}
mkdir -p "$CAL_DIR"
[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 2; }

for bucket in near contact; do
  if [[ "$bucket" == near ]]; then
    data=${RSC_CAL_NEAR_DATA:-$OCRAP_ROOT/val_near_contact}; harm=${NEAR_MAX_HARM:-0.55}; hard=${NEAR_MAX_HARD:-0.0}
  else
    data=${RSC_CAL_CONTACT_DATA:-$OCRAP_ROOT/val_contact}; harm=${CONTACT_MAX_HARM:-0.70}; hard=${CONTACT_MAX_HARD:-1.0}
  fi
  CUDA_VISIBLE_DEVICES="$CAL_GPU" PYTHONUNBUFFERED=1 python -u tools/calibrate_direct_value_risk_v43.py \
    --dataset "$data" --checkpoint "$CKPT" \
    --output "$CAL_DIR/direct_value_risk_${bucket}_v43.json" \
    --rows-output "$CAL_DIR/direct_value_risk_${bucket}_v43.rows.jsonl" \
    --required-min-groups ${RSC_MIN_GROUPS:-60} \
    --min-nominal-deviation ${RSC_MIN_DEVIATION:-0.002} \
    --max-hard "$hard" --max-harm "$harm" \
    --positive-gain ${RSC_POSITIVE_GAIN:-0.025} \
    --min-score-advantage ${RSC_MIN_SCORE_ADV:-0.0} \
    --min-fit-selected ${RSC_MIN_FIT_SELECTED:-4} \
    --min-fit-precision ${RSC_MIN_FIT_PRECISION:-0.60} \
    --max-fit-harmful-selected-rate ${RSC_MAX_FIT_HARMFUL_RATE:-0.10} \
    --min-verify-selected ${RSC_MIN_VERIFY_SELECTED:-2} \
    --min-verify-precision ${RSC_MIN_VERIFY_PRECISION:-0.50} \
    --max-verify-harmful-group-ucb ${RSC_MAX_VERIFY_HARMFUL_UCB:-0.05} \
    2>&1 | tee "$CAL_DIR/direct_value_risk_${bucket}_v43.log"
done

python - "$CAL_DIR" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); ok=True
for bucket in ('near','contact'):
    p=root/f'direct_value_risk_{bucket}_v43.json'
    d=json.load(open(p))
    print(bucket, {
      'valid': d.get('valid_for_deployment'),
      'threshold': d.get('direct_value_threshold'),
      'fit_selected': (d.get('fit') or {}).get('num_selected'),
      'verify_selected': (d.get('verify') or {}).get('num_selected'),
      'verify_precision': (d.get('verify') or {}).get('challenge_precision'),
      'verify_harm_ucb': (d.get('verify') or {}).get('harmful_group_exposure_ucb90'),
      'warnings': d.get('warnings'),
    })
    ok &= bool(d.get('valid_for_deployment', False))
raise SystemExit(0 if ok else 3)
PY
