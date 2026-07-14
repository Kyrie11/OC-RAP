#!/usr/bin/env bash
set -euo pipefail

export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export TRAIN_SAFE=${TRAIN_SAFE:-$OCRAP_ROOT/train_safe}
export VAL_SAFE=${VAL_SAFE:-$OCRAP_ROOT/val_safe}
export TEST_SAFE=${TEST_SAFE:-$OCRAP_ROOT/test_safe}
export RUN=${RUN:-runs/safe_external_baselines}
export NUM_GPUS=${NUM_GPUS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
mkdir -p "$RUN"

# 0) Rule-based nominal/log replay: no training, evaluates the logged nominal prefix.
python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/nominal_log_replay.yaml \
  --dataset "$TEST_SAFE" \
  --split test \
  --output "$RUN/eval_safe_nominal_log_replay.json" \
  --baselines nominal_replay,log_replay

# 1) Waymax route-conditioned BC / Wayformer-style BC.
torchrun --standalone --nproc_per_node="$NUM_GPUS" -m ocrap.cli train-baseline \
  --config configs/external_baselines/wayformer_bc.yaml \
  --dataset "$TRAIN_SAFE" \
  --val-dataset "$VAL_SAFE" \
  --baseline wayformer_bc \
  --output "$RUN/wayformer_bc"

python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/wayformer_bc.yaml \
  --dataset "$TEST_SAFE" \
  --checkpoint "$RUN/wayformer_bc/best.pt" \
  --split test \
  --output "$RUN/eval_safe_wayformer_bc.json" \
  --baselines wayformer_bc

# 2) GameFormer level-k adapter.
torchrun --standalone --nproc_per_node="$NUM_GPUS" -m ocrap.cli train-baseline \
  --config configs/external_baselines/gameformer_lite.yaml \
  --dataset "$TRAIN_SAFE" \
  --val-dataset "$VAL_SAFE" \
  --baseline gameformer_lite \
  --output "$RUN/gameformer_lite"

python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/gameformer_lite.yaml \
  --dataset "$TEST_SAFE" \
  --checkpoint "$RUN/gameformer_lite/best.pt" \
  --split test \
  --output "$RUN/eval_safe_gameformer_lite.json" \
  --baselines gameformer_lite

# 3) BeTop / BeTopNet-lite adapter.
torchrun --standalone --nproc_per_node="$NUM_GPUS" -m ocrap.cli train-baseline \
  --config configs/external_baselines/betopnet_lite.yaml \
  --dataset "$TRAIN_SAFE" \
  --val-dataset "$VAL_SAFE" \
  --baseline betopnet_lite \
  --output "$RUN/betopnet_lite"

python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/betopnet_lite.yaml \
  --dataset "$TEST_SAFE" \
  --checkpoint "$RUN/betopnet_lite/best.pt" \
  --split test \
  --output "$RUN/eval_safe_betopnet_lite.json" \
  --baselines betopnet_lite

