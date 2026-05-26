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

**Important:** `scripts/collect_roots.py` currently supports the synthetic smoke-test backend only. It is useful for checking tensor schemas and algorithmic invariants, but it does not yet collect real MetaDrive root snapshots or real closed-loop teacher rollouts. Do not use synthetic outputs as the paper-final MetaDrive-Recovery dataset.

## 1. Project overview

Core modules:

- `recap/raster`: privileged BEV rasterization with fixed channel order.
- `recap/proposals`: lattice bootstrap, deterministic projection, recovery options.
- `recap/teacher`: root-shared modes, synthetic/MetaDrive rollout hooks, margins, labels.
- `recap/models`: CARE, MERO, selector, neural proposal head, ablation scalar critic.
- `recap/evaluation`: paper metrics and baselines.
- `scripts`: dataset construction, training, calibration, evaluation, ablations, table export.

## 2. Installation

```bash
pip install -r requirements.txt
pip install -e .
python -m metadrive.examples.profile_metadrive
```

The last command verifies a real MetaDrive installation. The CI/smoke path works without MetaDrive by using synthetic roots with the same tensor schema.

## 3. Dataset construction

```bash
export WOMD=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/scenario
export RECAP=/data0/senzeyu2/dataset/ReCAP
export SN=$RECAP/scenarionet/womd_1_3_1_full
export FULL=$RECAP/full_womd131_recap_v1
```

```bash
python scripts/convert_womd_to_scenarionet.py \
  --womd-root $WOMD \
  --output-root $SN \
  --splits training validation testing_interactive \
  --num-workers 24
```

```bash
python scripts/collect_metadrive_roots.py \
  --scenario-dir $SN/training \
  --output $FULL/roots_raw \
  --split-name train \
  --max-roots 6000 \
  --target-regime-counts normal=1500,low=1500,near=2250,contact=750 \
  --max-scenarios-to-scan 300000 \
  --history-steps 10 \
  --max-samples-per-log 1 \
  --append

python scripts/collect_metadrive_roots.py \
  --scenario-dir $SN/validation \
  --output $FULL/roots_raw \
  --split-name val \
  --max-roots 1000 \
  --target-regime-counts normal=250,low=250,near=375,contact=125 \
  --max-scenarios-to-scan 120000 \
  --history-steps 10 \
  --max-samples-per-log 1 \
  --append
  
  
python scripts/collect_metadrive_roots.py \
    --scenario-dir $SN/validation \
    --output $FULL/roots_raw \
    --split-name calib \
    --max-roots 1000 \
    --target-regime-counts normal=250,low=250,near=375,contact=125 \
    --max-scenarios-to-scan 120000 \
    --history-steps 10 \
    --max-samples-per-log 1 \
    --append
  
python scripts/collect_metadrive_roots.py \
  --scenario-dir $SN/testing_interactive \
  --output $FULL/roots_raw \
  --split-name test \
  --max-roots 1000 \
  --target-regime-counts normal=250,low=250,near=375,contact=125 \
  --max-scenarios-to-scan 120000 \
  --history-steps 10 \
  --max-samples-per-log 1 \
  --append
  
for SPLIT in train val calib test; do
  python scripts/rasterize_bev.py \
    --root-dir $FULL/roots_raw \
    --split $SPLIT \
    --bev-config configs/bev_256.yaml \
    --channels compact \
    --history-steps 10 \
    --output $FULL/bev/${SPLIT}.zarr \
    --shard-size 16 \
    --save-debug 16 \
    --write-channel-png \
    --debug-dir outputs/diagnostics/full_bev_${SPLIT}
done

```

## 4. BEV rasterization

The main BEV is produced by `recap/raster/bev_builder.py`, not MetaDrive `TopDownObservation`.

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
  --write-channel-png \
  --debug-dir outputs/debug_bev/ROOT_ID_HERE
```

## 5. Teacher rollout and label generation

```bash
build_labels_split () {
  local SPLIT=$1
  local SCENARIO_DIR=$2
  local PARTS=$3
  local N
  N=$(count_split "$SPLIT")

  echo "Building teacher labels for split=$SPLIT roots=$N parts=$PARTS"

  rm -rf "$CHECK/${SPLIT}_parts" "$CHECK/${SPLIT}.zarr"
  mkdir -p "$CHECK/${SPLIT}_parts"

  local pids=()

  for i in $(seq 0 $((PARTS-1))); do
    local START=$(( i * N / PARTS ))
    local END=$(( (i + 1) * N / PARTS ))

    if [ "$START" -ge "$END" ]; then
      continue
    fi

    (
      export CUDA_VISIBLE_DEVICES=""
      export OMP_NUM_THREADS=2
      export MKL_NUM_THREADS=1

      python scripts/build_teacher_labels.py \
        --config configs/dataset_metadrive_paper_check.yaml \
        --split $SPLIT \
        --root-dir $CHECK/roots_raw \
        --bev-dir $CHECK/bev/${SPLIT}.zarr \
        --output $CHECK/${SPLIT}_parts/part_${i}.zarr \
        --rollout-backend metadrive \
        --scenario-dir $SCENARIO_DIR \
        --metadrive-reactive-traffic true \
        --allow-temporal-root-rollout \
        --root-start $START \
        --root-end $END \
        --root-stride 1 \
        --shard-size 1
    ) > "$CHECK/logs/build_${SPLIT}_part_${i}.log" 2>&1 &

    pids+=($!)
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  if [ "$failed" -ne 0 ]; then
    echo "ERROR: split=$SPLIT 有分片失败，请检查 $CHECK/logs/build_${SPLIT}_part_*.log"
    exit 1
  fi

  python scripts/merge_sharded_datasets.py \
    --inputs $CHECK/${SPLIT}_parts/part_*.zarr \
    --output $CHECK/${SPLIT}.zarr \
    --shard-size 4
}



python scripts/parallel_build_teacher_labels.py \
  --config configs/dataset_metadrive_paper_check.yaml \
  --split train \
  --root-dir $CHECK/roots_raw \
  --bev-dir $CHECK/bev/train.zarr \
  --output $CHECK/train.zarr \
  --rollout-backend metadrive \
  --scenario-dir $SN/training \
  --metadrive-reactive-traffic true \
  --allow-temporal-root-rollout \
  --parts 24 \
  --shard-size 1 \
  --merge-shard-size 4 \
  --log-dir $CHECK/logs
  
  
python scripts/parallel_build_teacher_labels.py \
  --config configs/dataset_metadrive_paper_check.yaml \
  --split val \
  --root-dir $CHECK/roots_raw \
  --bev-dir $CHECK/bev/val.zarr \
  --output $CHECK/val.zarr \
  --rollout-backend metadrive \
  --scenario-dir $SN/validation \
  --metadrive-reactive-traffic true \
  --allow-temporal-root-rollout \
  --parts 12 \
  --shard-size 1 \
  --merge-shard-size 4 \
  --log-dir $CHECK/logs
  
  
  
  
python scripts/parallel_build_teacher_labels.py \
  --config configs/dataset_metadrive_paper_check.yaml \
  --split calib \
  --root-dir $CHECK/roots_raw \
  --bev-dir $CHECK/bev/calib.zarr \
  --output $CHECK/calib.zarr \
  --rollout-backend metadrive \
  --scenario-dir $SN/validation \
  --metadrive-reactive-traffic true \
  --allow-temporal-root-rollout \
  --parts 12 \
  --shard-size 1 \
  --merge-shard-size 4 \
  --log-dir $CHECK/logs
  
  
python scripts/parallel_build_teacher_labels.py \
  --config configs/dataset_metadrive_paper_check.yaml \
  --split calib \
  --root-dir $CHECK/roots_raw \
  --bev-dir $CHECK/bev/calib.zarr \
  --output $CHECK/calib.zarr \
  --rollout-backend metadrive \
  --scenario-dir $SN/validation \
  --metadrive-reactive-traffic true \
  --allow-temporal-root-rollout \
  --parts 12 \
  --shard-size 1 \
  --merge-shard-size 4 \
  --log-dir $CHECK/logs
```

Teacher labels enforce[dataset_construction.diff](../dataset_construction.diff): same-root latent context, fixed prefix/recovery boundary, first-contact harm separated from R, post-contact recovery not killed by first-contact collision clearance.

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
  --write-channel-png
```

## 13. Diagnose Dataset 

```bash
mkdir -p outputs/diagnostics

for SPLIT in train val calib test; do
  python scripts/diagnose_recap_dataset.py \
    --dataset $CHECK/${SPLIT}.zarr \
    --roots $CHECK/roots_raw \
    --output outputs/diagnostics/papercheck_${SPLIT}_report.json \
    --sample-roots 512
done

python - <<'PY'
import json
from pathlib import Path

for split in ["train", "val", "calib", "test"]:
    p = Path(f"outputs/diagnostics/papercheck_{split}_report.json")
    print("\n===", split, "===")
    if not p.exists():
        print("missing", p)
        continue
    r = json.loads(p.read_text())
    for k in [
        "num_samples",
        "paper_quality_gate",
        "positive_recovery_root_rate",
        "nontrivial_action_ranking_rate",
        "mode_label_disagreement_mean",
        "mode_best_margin_disagreement_mean",
    ]:
        if k in r:
            print(k, "=", r[k])
PY
```

## 14. Unit tests

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

## 15. Reproducibility checklist

Every run should log `implementation_level`, config hash, dataset version, split, ablation flags, calibration values, and `alignment_report.json`. Final main-table runs require all final alignment fields to be true.

## 16. Dataset schema

See `docs/DATASET_SCHEMA.md`. Main arrays include BEV, ego info, route command, actions, options, masks, mode probabilities, debug-only mode seed params, margins, evidence labels, `R_star`, and witness labels.

## 16. Expected outputs and table export

```bash
python scripts/export_tables.py \
  --eval-dirs outputs/eval/ours outputs/eval/nominal outputs/eval/risk_aware outputs/eval/backup_filter outputs/eval/direct_scalar_critic outputs/eval/oracle \
  --output outputs/tables
```

## 18. MVP vs Final mode

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

## 19. Known limitations

This generated code includes a synthetic teacher fallback so tests and smoke commands run without MetaDrive. For paper-grade results, replace or extend the adapter extraction in `MetaDriveStateAdapter`, run real MetaDrive root snapshots, and verify `test_root_restore_determinism` on the installed simulator. CARLA support is documented as a schema-compatible backend boundary; counterfactual fork support must be implemented per CARLA scenario setup.

