#!/usr/bin/env bash
set -euo pipefail

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${TRAIN_NEAR:=$OCRAP_ROOT/train_near_contact}"
: "${VAL_NEAR:=$OCRAP_ROOT/val_near_contact}"
: "${TEST_NEAR:=$OCRAP_ROOT/test_near_contact}"
: "${RUN:=runs/near_contact_external_baselines}"
: "${NUM_GPUS:=1}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}"
: "${CL_WOMD:=${WOMD_VAL_INTERACTIVE}@150}"
: "${CL_MAX_SCENARIOS:=50}"
: "${CL_MAX_STEPS:=40}"

mkdir -p "$RUN"

NONLEARNED=(
  marc_lite
  racp_lite
  expected_risk_filter
  cvar_risk_filter
  dro_cvar_filter
  predictive_safety_filter
  oracle_recovery_filter
)
ALL_METHODS="marc_lite,racp_lite,expected_risk_filter,cvar_risk_filter,dro_cvar_filter,predictive_safety_filter,oracle_recovery_filter,gameformer_lite"

# These are optimization/filter baselines, so train-baseline performs a
# deterministic registration/sanity pass over the grouped OC-RAP dataset.
for method in "${NONLEARNED[@]}"; do
  python -u -m ocrap.cli train-baseline \
    --config configs/external_baselines/near_contact_external_baselines.yaml \
    --dataset "$TRAIN_NEAR" \
    --val-dataset "$VAL_NEAR" \
    --baseline "$method" \
    --output "$RUN/$method" \
    2>&1 | tee "$RUN/train_${method}.log"
done

# GameFormer remains a learning baseline.
if [ "$NUM_GPUS" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NUM_GPUS" -m ocrap.cli train-baseline \
    --config configs/external_baselines/near_contact_gameformer_lite.yaml \
    --dataset "$TRAIN_NEAR" \
    --val-dataset "$VAL_NEAR" \
    --baseline gameformer_lite \
    --output "$RUN/gameformer_lite" \
    2>&1 | tee "$RUN/train_gameformer_lite.log"
else
  python -u -m ocrap.cli train-baseline \
    --config configs/external_baselines/near_contact_gameformer_lite.yaml \
    --dataset "$TRAIN_NEAR" \
    --val-dataset "$VAL_NEAR" \
    --baseline gameformer_lite \
    --output "$RUN/gameformer_lite" \
    2>&1 | tee "$RUN/train_gameformer_lite.log"
fi

# Offline grouped candidate-set test, useful before expensive closed-loop runs.
python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/near_contact_external_baselines.yaml \
  --dataset "$TEST_NEAR" \
  --split test \
  --output "$RUN/eval_near_contact_nonlearned.json" \
  --baselines "marc_lite,racp_lite,expected_risk_filter,cvar_risk_filter,dro_cvar_filter,predictive_safety_filter,oracle_recovery_filter" \
  2>&1 | tee "$RUN/eval_near_contact_nonlearned.log"

python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/near_contact_gameformer_lite.yaml \
  --dataset "$TEST_NEAR" \
  --checkpoint "$RUN/gameformer_lite/best.pt" \
  --split test \
  --output "$RUN/eval_near_contact_gameformer_lite.json" \
  --baselines gameformer_lite \
  2>&1 | tee "$RUN/eval_near_contact_gameformer_lite.log"

# True Waymax receding-horizon closed-loop runs. Non-learning baselines use
# online teacher labels because their certificates are defined on branch/root
# counterfactuals. GameFormer uses the learned checkpoint and selected-candidate
# auditing for closed-loop DRS/FRA diagnostics.
for method in "${NONLEARNED[@]}"; do
  python -u -m ocrap.cli closed-loop \
    --config configs/external_baselines/near_contact_external_baselines.yaml \
    --dataset "$CL_WOMD" \
    --output "$RUN/closed_loop_${method}.json" \
    --set closed_loop.method="$method" \
    --set closed_loop.max_scenarios="$CL_MAX_SCENARIOS" \
    --set closed_loop.max_steps="$CL_MAX_STEPS" \
    --set closed_loop.replan_interval_steps=1 \
    --set closed_loop.label_mode=all \
    --set closed_loop.num_candidate_prefixes=24 \
    --set waymax.compute_future_metrics=true \
    2>&1 | tee "$RUN/closed_loop_${method}.log"
done

python -u -m ocrap.cli closed-loop \
  --config configs/external_baselines/near_contact_gameformer_lite.yaml \
  --dataset "$CL_WOMD" \
  --checkpoint "$RUN/gameformer_lite/best.pt" \
  --output "$RUN/closed_loop_gameformer_lite.json" \
  --set closed_loop.method=gameformer_lite \
  --set closed_loop.max_scenarios="$CL_MAX_SCENARIOS" \
  --set closed_loop.max_steps="$CL_MAX_STEPS" \
  --set closed_loop.replan_interval_steps=1 \
  --set closed_loop.label_mode=selected \
  --set closed_loop.num_candidate_prefixes=24 \
  --set waymax.compute_future_metrics=true \
  2>&1 | tee "$RUN/closed_loop_gameformer_lite.log"

python - <<'PY'
import json, os, glob
run=os.environ.get('RUN','runs/near_contact_external_baselines')
rows=[]
for p in sorted(glob.glob(os.path.join(run,'closed_loop_*.json'))):
    try:
        d=json.load(open(p))
    except Exception:
        continue
    rows.append({k:d.get(k) for k in ['method','num_scenes','num_decisions','mean_overlap_rate','mean_offroad_rate','mean_kinematic_infeasibility_rate','mean_drs','mean_fra_exec','mean_nup']})
out=os.path.join(run,'closed_loop_summary.json')
json.dump(rows, open(out,'w'), indent=2)
print({'event':'near_contact_closed_loop_summary','output':out,'num_methods':len(rows)})
PY
