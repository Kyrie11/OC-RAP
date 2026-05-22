# ReCAP: Recoverability-Centered Planning for MetaDrive/CARLA BEV Roots

ReCAP implements the paper pipeline:

```text
BEV history + ego state + route command
  -> action prefix proposals
  -> scene-conditioned recovery options
  -> CARE evidence prediction
  -> MERO profile computation
  -> calibrated constrained selector
  -> shared low-level tracking controller
  -> MetaDrive env.step([steer, throttle])
```

The main method is a BEV-level privileged-state planner, not raw camera/LiDAR perception and not a scalar collision-risk planner. The code keeps simulator-specific state access behind adapters and unit-tests the algorithmic invariants without requiring MetaDrive to be installed.

## 1. Project overview

Core modules:

- `metadrive_recovery/raster`: privileged BEV rasterization with fixed channel order.
- `metadrive_recovery/proposals`: lattice bootstrap, deterministic projection, recovery options.
- `metadrive_recovery/teacher`: root-shared modes, synthetic/MetaDrive rollout hooks, margins, labels.
- `metadrive_recovery/models`: CARE, MERO, selector, neural proposal head, ablation scalar critic.
- `metadrive_recovery/evaluation`: paper metrics and baselines.
- `scripts`: dataset construction, training, calibration, evaluation, ablations, table export.

## 2. Installation

```bash
pip install -r requirements.txt
pip install -e .
python -m metadrive.examples.profile_metadrive
```

The last command verifies a real MetaDrive installation. The CI/smoke path works without MetaDrive by using synthetic roots with the same tensor schema.

## 3. Dataset construction

MVP smoke test:

```bash
python scripts/collect_roots.py \
  --config configs/ablations/mvp_fast_debug.yaml \
  --output data/debug/roots

python scripts/rasterize_bev.py \
  --root-dir data/debug/roots \
  --split debug \
  --bev-config configs/bev_160_debug.yaml \
  --output data/debug/bev.zarr \
  --write-channel-png true

python scripts/build_teacher_labels.py \
  --config configs/ablations/mvp_fast_debug.yaml \
  --split debug \
  --root-dir data/debug/roots \
  --bev-dir data/debug/bev.zarr \
  --output data/debug/labels.zarr

python scripts/offline_eval.py \
  --config configs/ablations/mvp_fast_debug.yaml \
  --dataset data/debug/labels.zarr \
  --method oracle \
  --output outputs/ci/oracle
```

## 4. BEV rasterization

The main BEV is produced by `metadrive_recovery/raster/bev_builder.py`, not MetaDrive `TopDownObservation`.

```bash
python scripts/rasterize_bev.py \
  --root-dir data/recap/roots_raw \
  --split train \
  --bev-config configs/bev_256.yaml \
  --channels compact \
  --history-steps 10 \
  --num-workers 8 \
  --output data/recap/bev/train.zarr \
  --save-debug 100
```

Single-root channel export:

```bash
python scripts/rasterize_bev.py \
  --root-dir data/recap/roots_raw \
  --root-id ROOT_ID_HERE \
  --bev-config configs/bev_256.yaml \
  --output data/debug/single_root_bev.zarr \
  --write-channel-png true \
  --debug-dir outputs/debug_bev/ROOT_ID_HERE
```

## 5. Teacher rollout and label generation

```bash
python scripts/build_teacher_labels.py \
  --config configs/dataset_metadrive.yaml \
  --split train \
  --root-dir data/recap/roots_raw \
  --bev-dir data/recap/bev/train.zarr \
  --output data/recap/train.zarr

python scripts/build_teacher_labels.py \
  --config configs/dataset_metadrive.yaml \
  --split calib \
  --root-dir data/recap/roots_raw \
  --bev-dir data/recap/bev/calib.zarr \
  --output data/recap/calib.zarr

python scripts/build_teacher_labels.py \
  --config configs/dataset_metadrive.yaml \
  --split test \
  --root-dir data/recap/roots_raw \
  --bev-dir data/recap/bev/test.zarr \
  --output data/recap/test.zarr
```

Teacher labels enforce: same-root latent context, fixed prefix/recovery boundary, first-contact harm separated from R, post-contact recovery not killed by first-contact collision clearance.

## 6. CARE training

```bash
python scripts/train_action_proposal.py \
  --config configs/train_action_proposal.yaml \
  --dataset data/recap/train.zarr \
  --output checkpoints/action_proposal

python scripts/train_care.py \
  --config configs/train_care.yaml \
  --dataset data/recap/train.zarr \
  --proposal-checkpoint checkpoints/action_proposal/best.pt \
  --output checkpoints/care
```

## 7. MERO calibration

```bash
python scripts/calibrate.py \
  --config configs/train_care.yaml \
  --dataset data/recap/calib.zarr \
  --checkpoint checkpoints/care/best.pt \
  --split calib \
  --output outputs/calibration
```

`q_values.json` contains `q_R`, `q_H`, Clopper-Pearson UCBs, split information, and calibration failure flags.

## 8. Closed-loop evaluation

```bash
python scripts/eval_closed_loop.py \
  --config configs/eval_closed_loop.yaml \
  --dataset data/recap/test.zarr \
  --checkpoint checkpoints/care/best.pt \
  --calibration outputs/calibration/q_values.json \
  --method ours \
  --split test \
  --output outputs/eval/ours
```

If the simulator backend is unavailable, the script falls back to offline teacher-label evaluation while keeping the same selected-action metrics.

## 9. Offline evaluation

```bash
python scripts/offline_eval.py --dataset data/recap/test.zarr --method oracle --output outputs/offline/oracle
python scripts/offline_eval.py --dataset data/recap/test.zarr --method nominal --output outputs/offline/nominal
python scripts/offline_eval.py --dataset data/recap/test.zarr --method risk_aware --output outputs/offline/risk_aware
```

## 10. Baselines

```bash
python scripts/eval_closed_loop.py --config configs/eval_closed_loop.yaml --dataset data/recap/test.zarr --method nominal --split test --output outputs/eval/nominal
python scripts/eval_closed_loop.py --config configs/eval_closed_loop.yaml --dataset data/recap/test.zarr --method risk_aware --split test --output outputs/eval/risk_aware
python scripts/eval_closed_loop.py --config configs/eval_closed_loop.yaml --dataset data/recap/test.zarr --method backup_filter --split test --output outputs/eval/backup_filter
python scripts/eval_closed_loop.py --config configs/eval_closed_loop.yaml --dataset data/recap/test.zarr --method direct_scalar_critic --split test --output outputs/eval/direct_scalar_critic
python scripts/eval_closed_loop.py --config configs/eval_closed_loop.yaml --dataset data/recap/test.zarr --method oracle --split test --output outputs/eval/oracle
```

All baselines share `PurePursuitPID`/`TrackingController` and the same candidate prefixes.

## 11. Ablations

```bash
python scripts/run_ablation.py \
  --base-config configs/ablations/final_main.yaml \
  --ablation mean_over_options \
  --dataset data/recap/test.zarr \
  --checkpoint checkpoints/care/best.pt \
  --output outputs/ablations/mean_over_options

python scripts/run_all_experiments.py \
  --experiment-list configs/experiment_suite.yaml \
  --dataset data/recap \
  --output outputs/full_suite
```

Ablation flags live in config files and are saved as `ablation_flags.json`. `--method ours` must not silently enable ablations.

## 12. Debug visualization

```bash
python scripts/rasterize_bev.py \
  --root-dir data/debug/roots \
  --split debug \
  --bev-config configs/bev_160_debug.yaml \
  --output data/debug/bev_debug.zarr \
  --save-debug all \
  --write-channel-png true
```

## 13. Unit tests

```bash
pytest tests -q
pytest tests/test_mero_monotonicity.py -q
pytest tests/test_selector_logic.py -q
pytest tests/test_bev_raster.py -q
```

CI smoke test:

```bash
python scripts/collect_roots.py --config configs/ablations/mvp_fast_debug.yaml --max-roots 8 --output data/ci/roots
python scripts/rasterize_bev.py --root-dir data/ci/roots --split debug --bev-config configs/bev_160_debug.yaml --output data/ci/bev.zarr
python scripts/build_teacher_labels.py --config configs/ablations/mvp_fast_debug.yaml --split debug --root-dir data/ci/roots --bev-dir data/ci/bev.zarr --max-roots 8 --output data/ci/labels.zarr
python scripts/offline_eval.py --dataset data/ci/labels.zarr --method oracle --output outputs/ci/oracle
```

## 14. Reproducibility checklist

Every run should log `implementation_level`, config hash, dataset version, split, ablation flags, calibration values, and `alignment_report.json`. Final main-table runs require all final alignment fields to be true.

## 15. Dataset schema

See `docs/DATASET_SCHEMA.md`. Main arrays include BEV, ego info, route command, actions, options, masks, mode probabilities, debug-only mode seed params, margins, evidence labels, `R_star`, and witness labels.

## 16. Expected outputs and table export

```bash
python scripts/export_tables.py \
  --eval-dirs outputs/eval/ours outputs/eval/nominal outputs/eval/risk_aware outputs/eval/backup_filter outputs/eval/direct_scalar_critic outputs/eval/oracle \
  --output outputs/tables
```

## 17. MVP vs Final mode

`implementation_level: mvp` allows lattice proposal, heuristic affordances, M=4, L=6, and synthetic teacher smoke tests.

`implementation_level: final` requires neural action proposal + deterministic projection, full CARE architecture, all seven recovery options, M=8, calibrated affordance energy, learned CARE inference, monotone MERO, q calibration, baselines, metrics, and ablations.


## CARLA recorder extension

CARLA roots must reuse the same ReCAP schema. Recorder data can reconstruct roots or replay to a root tick, but it cannot be fed to CARE as actor tables or future trajectories. Counterfactual MERO teacher labels require fork support under the same Traffic Manager latent context.

```bash
python scripts/collect_carla_roots.py \
  --recorder-file path/to/recording.log \
  --root-frame 1200 \
  --map-name Town05 \
  --carla-version 0.9.x \
  --traffic-manager-seed 42 \
  --fork-support \
  --output data/carla_recovery/roots_raw
```

If `--fork-support` is omitted, exported CARLA roots are marked valid for BEV pretraining/replay evaluation only, not MERO teacher-label generation.

## 18. Known limitations

This generated code includes a synthetic teacher fallback so tests and smoke commands run without MetaDrive. For paper-grade results, replace or extend the adapter extraction in `MetaDriveStateAdapter`, run real MetaDrive root snapshots, and verify `test_root_restore_determinism` on the installed simulator. CARLA support is documented as a schema-compatible backend boundary; counterfactual fork support must be implemented per CARLA scenario setup.
