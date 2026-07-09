# External baselines for OC-RAP

This package implements paper-faithful adapters for the three main comparison baselines used in the safe / near-contact / contact regimes.

## 1. Safe regime: Waymax route-conditioned BC

`route_bc_lite` is now a Waymax/Wayformer-style behavior cloning baseline rather than a single candidate scorer.  The implementation uses:

- early-fusion attention over OC-RAP candidate-prefix tokens;
- optional latent-query attention for Wayformer-like scene compression;
- a four-layer residual MLP policy head;
- cross-entropy on the logged / nominal candidate, which corresponds to the nearest candidate action in the OC-RAP prefix lattice.

This follows the Waymax planning benchmark design: route-conditioned BC reuses a Wayformer encoder, then applies a residual MLP / softmax action head over a discrete action space.

## 2. Near-contact regime: GameFormer level-k

`gameformer_lite` has been upgraded to `GameFormerLevelK`.  It keeps the core GameFormer structure:

- Transformer scene encoder;
- learnable modality / candidate queries;
- level-0 cross-attention decoder;
- iterative level-k response decoders;
- branch/future interaction features from OC-RAP hidden roots, targeted futures, root probabilities, and `m_star` recovery margins;
- deep supervision on every reasoning level and response-consistency regularization.

Because OC-RAP does not store raw multi-agent future trajectories for every external paper format, the branch tensors produced by the OC-RAP dataset builder are used as the future proxy.  This preserves the key comparison target: GameFormer can model branch-specific interaction futures but does not enforce OC-RAP's observation-consistent shared recovery admission.

## 3. Contact regime: post-impact MPC

`postimpact_mpc_lite` is now a finite-lattice planning-integrated MPC adapter.  The original post-impact MPC optimizes stability recovery and secondary collision avoidance using vehicle dynamics and road/obstacle constraints.  In OC-RAP, the available action space is a candidate prefix / recovery-option lattice, so the MPC objective is evaluated over that finite horizon lattice:

- yaw-rate and yaw-acceleration damping;
- terminal stable-stop speed;
- acceleration / steering / jerk effort;
- hard violation and harm / secondary-collision proxy;
- route rejoin utility;
- shared recovery option success.

## Dual-A30 training

Use `torchrun`; DDP is auto-detected from `WORLD_SIZE` and `LOCAL_RANK`:

```bash
export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP
export TRAIN_MIX="$OCRAP_ROOT/train_safe,$OCRAP_ROOT/train_near_contact,$OCRAP_ROOT/train_contact"
export VAL_MIX="$OCRAP_ROOT/val_safe,$OCRAP_ROOT/val_near_contact,$OCRAP_ROOT/val_contact"
export RUN=runs/external_baselines_paper
mkdir -p "$RUN"

torchrun --standalone --nproc_per_node=2 -m ocrap.cli train-baseline \
  --config configs/external_baselines/route_bc_lite.yaml \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --baseline route_bc_lite \
  --output "$RUN/route_bc_wayformer"

torchrun --standalone --nproc_per_node=2 -m ocrap.cli train-baseline \
  --config configs/external_baselines/gameformer_lite.yaml \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --baseline gameformer_lite \
  --output "$RUN/gameformer_levelk"
```

Only rank 0 writes checkpoints and JSON summaries.  TQDM progress bars are enabled on rank 0 by default.

## Evaluation

```bash
python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/all_external_baselines.yaml \
  --dataset "$OCRAP_ROOT/test_near_contact" \
  --checkpoint "$RUN/gameformer_levelk/best.pt" \
  --split test \
  --output "$RUN/eval_near_contact_external_all.json" \
  --baselines route_bc_lite,gameformer_lite,marc_lite,racp_lite,expected_risk_filter,cvar_risk_filter,dro_cvar_filter,postimpact_mpc_lite
```

For convenience, run:

```bash
bash scripts/run_external_baselines.sh
```
