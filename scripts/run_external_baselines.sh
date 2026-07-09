#!/usr/bin/env bash
set -euo pipefail

export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export RUN=${RUN:-runs/external_baselines}
export TRAIN_MIX=${TRAIN_MIX:-$OCRAP_ROOT/train_safe,$OCRAP_ROOT/train_near_contact,$OCRAP_ROOT/train_contact}
export VAL_MIX=${VAL_MIX:-$OCRAP_ROOT/val_safe,$OCRAP_ROOT/val_near_contact,$OCRAP_ROOT/val_contact}
export SAFE_TEST=${SAFE_TEST:-$OCRAP_ROOT/test_safe}
export NEAR_TEST=${NEAR_TEST:-$OCRAP_ROOT/test_near_contact}
export CONTACT_TEST=${CONTACT_TEST:-$OCRAP_ROOT/test_contact}
mkdir -p "$RUN"

# 1) Safe-regime route-conditioned imitation / BC baseline.
CUDA_VISIBLE_DEVICES=${GPU_TRAIN:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli train-baseline \
  --config configs/external_baselines/route_bc_lite.yaml \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --baseline route_bc_lite \
  --output "$RUN/route_bc_lite" \
  | tee "$RUN/train_route_bc_lite.log"

# 2) Interaction-aware GameFormer-lite baseline.
CUDA_VISIBLE_DEVICES=${GPU_TRAIN:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli train-baseline \
  --config configs/external_baselines/gameformer_lite.yaml \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --baseline gameformer_lite \
  --output "$RUN/gameformer_lite" \
  | tee "$RUN/train_gameformer_lite.log"

# 3) Evaluate all external baselines. MARC/RACP/CVaR/DRO/post-impact MPC are
#    optimization-style baselines over OC-RAP's candidate/future labels; they do
#    not require a learned checkpoint. GameFormer-lite uses its checkpoint here.
for D in safe near_contact contact; do
  case "$D" in
    safe) DATASET="$SAFE_TEST" ;;
    near_contact) DATASET="$NEAR_TEST" ;;
    contact) DATASET="$CONTACT_TEST" ;;
  esac

  PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate-baseline \
    --config configs/external_baselines/all_external_baselines.yaml \
    --dataset "$DATASET" \
    --checkpoint "$RUN/gameformer_lite/best.pt" \
    --split test \
    --output "$RUN/eval_${D}_external_all.json" \
    --baselines route_bc_lite,gameformer_lite,marc_lite,racp_lite,expected_risk_filter,cvar_risk_filter,dro_cvar_filter,postimpact_mpc_lite \
    | tee "$RUN/eval_${D}_external_all.log"
done

python - <<'PY' "$RUN"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
keys=['FRA_exec','FRA_cand','DRS','ODG','bounded_NUP','intervention_rate','secondary_collision_rate','stable_stop_success','route_rejoin_success','mean_harm_proxy','post_contact_deployability_score']
for p in sorted(root.glob('eval_*_external_all.json')):
    d=json.load(open(p))
    print('\n', p.name)
    for m, r in d.get('methods', {}).items():
        vals=' '.join(f'{k}={r.get(k)}' for k in keys if k in r)
        print(f'  {m:24s} {vals}')
PY
