#!/usr/bin/env bash
set -euo pipefail

export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export TRAIN_MIX=${TRAIN_MIX:-"$OCRAP_ROOT/train_safe,$OCRAP_ROOT/train_near_contact,$OCRAP_ROOT/train_contact"}
export VAL_MIX=${VAL_MIX:-"$OCRAP_ROOT/val_safe,$OCRAP_ROOT/val_near_contact,$OCRAP_ROOT/val_contact"}
export SAFE_TEST=${SAFE_TEST:-"$OCRAP_ROOT/test_safe"}
export NEAR_TEST=${NEAR_TEST:-"$OCRAP_ROOT/test_near_contact"}
export CONTACT_TEST=${CONTACT_TEST:-"$OCRAP_ROOT/test_contact"}
export RUN=${RUN:-runs/external_baselines_paper}
export NUM_GPUS=${NUM_GPUS:-2}
mkdir -p "$RUN"

# A30 dual-card training.  The train-baseline entrypoint auto-detects
# WORLD_SIZE/LOCAL_RANK from torchrun and wraps the model in DDP.
torchrun --standalone --nproc_per_node="$NUM_GPUS" -m ocrap.cli train-baseline \
  --config configs/external_baselines/route_bc_lite.yaml \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --baseline route_bc_lite \
  --output "$RUN/route_bc_wayformer"

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m ocrap.cli train-baseline \
  --config configs/external_baselines/gameformer_lite.yaml \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --baseline gameformer_lite \
  --output "$RUN/gameformer_lite"

# Evaluate with the checkpoint matching the learned method.  Rule-based MPC and
# risk filters can be evaluated with or without a checkpoint; passing the
# GameFormer checkpoint lets route_bc/gameformer share learned score heads in one
# summary file.


python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/route_bc_lite.yaml \
  --dataset "$SAFE_TEST" \
  --checkpoint "$RUN/route_bc_wayformer/best.pt" \
  --split test \
  --output "$RUN/eval_safe_route_bc_wayformer.json" \
  --baselines route_bc_lite

python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/all_external_baselines.yaml \
  --dataset "$NEAR_TEST" \
  --checkpoint "$RUN/gameformer_lite/best.pt" \
  --split test \
  --output "$RUN/eval_near_contact_external_all.json" \
  --baselines route_bc_lite,gameformer_lite,postimpact_mpc_lite

python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/all_external_baselines.yaml \
  --dataset "$CONTACT_TEST" \
  --checkpoint "$RUN/gameformer_lite/best.pt" \
  --split test \
  --output "$RUN/eval_contact_external_all.json" \
  --baselines route_bc_lite,gameformer_lite,postimpact_mpc_lite
