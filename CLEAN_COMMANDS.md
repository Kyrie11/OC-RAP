# OC-RAP cleaned workspace — current commands

> Publication WOMD role: **standard `validation`** for validation/test/calibration replay.  `validation_interactive` is supported only when explicitly requested for a separate diagnostic dataset whose provenance says so.

## 0. Common environment

```bash
cd /home/senzeyu2/code/OC-RAP

export BASE_OUT=/home/senzeyu2/code/OC-RAP/runs
export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP
export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example
export CUDA_DEVICES=0,1
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
```

## 0. Current V48.119 scientific reference contract

The stable unversioned orientation launcher now runs **V48.119 OC-VOP**. It treats the completed V48.118 OC-VSE STOP artifacts as immutable scientific inputs and verifies their exact SHA/status/registered next branch before any V48.119 audit starts. Python implementation filenames remain semantic/unversioned; scientific inputs and outputs remain versioned.

V48.118 showed that the same-option joint viability construction is active and exact, but a single rank-1 `max` envelope is not population-stable: the winning option identity switches across almost the entire recovery library, while full-set breadth helps Reserve but hurts Near Support. V48.119 therefore tests the preregistered **viability order profile** hypothesis: transferable recoverability may require the order/redundancy structure of joint viable options, not only the current best extreme.

## 1. Current orientation audit — V48.119 OC-VOP

V48.119 is **audit-only**. The 156-D frozen candidate response, actuator-projected executable constraints, recovery horizon/library, cohorts, candidate shuffle and 220-D closed-form ridge capacity remain fixed. For each recovery option it retains V48.118's same-option joint prefix/suffix viability margin, then summarizes the option set by exact fractional upper-tail means at the fixed preregistered masses `0.25, 0.50, 0.75, 1.00`.

The primary `full_profile` uses all common valid recovery options. The equal-capacity `exposed_profile` control uses only the support (not the weights) of V48.117's frozen weak-root zero-boundary witnesses. The four masses are fixed by the inherited 64-D geometry budget and are **not tuned**. The `gamma=1` coordinate is not evaluated as a standalone uniform-mean family; uniform averaging remains closed as a complete carrier. No option identity, regime label, teacher future value, learned set encoder, threshold, horizon, option-count, capacity, source, LR or epoch sweep is introduced.

Run exactly:

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 \
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
bash scripts/run_constraint_native_orientation_audit.sh
```

Upload only:

```text
/home/senzeyu2/code/OC-RAP/runs/OC-RAP-v48.119-OC-VOP-results.zip
```

### About OC-RAP training in V48.119

There is no registered V48.119 planner/root/source training stage. Running generic `python -m ocrap.cli train` would not reproduce this audit. A Main/carrier integration is authorized only if one preregistered order-profile family passes absolute Support+Reserve, historical improvement and activity gates.

## 2. Direct OC-RAP three-regime closed-loop test

Evaluate both frozen model variants on Safe / Near-Contact / Contact, using standard WOMD validation replay and all bucket targets:

```bash
bash scripts/run_ocrap_evaluation.sh \
  --model-run "$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main" \
  --variants balanced,precision \
  --out "$BASE_OUT/ocrap_v48_111_submission_three_regime" \
  --gpus 0,1 \
  --max-scenarios 0 \
  --womd-role validation
```

Completed regime artifacts are reused automatically after contract validation.

## 3. External baselines — all regimes

### Default: resume-aware train/registration/calibration + test

```bash
bash scripts/run_external_baselines.sh \
  --regime all \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 \
  --max-scenarios 0 \
  --womd-role validation
```

Default behavior:

- learned Safe baselines: reuse only when `best.pt` is valid **and** `train_summary.json` proves the configured epoch budget is complete; otherwise train;
- non-learning baselines: reuse compatible registration summaries without rescanning train/val; otherwise register;
- Near CPSF: reuse only a compatible calibration artifact; otherwise calibrate;
- then run closed-loop test.

### Force learned retraining / non-learning re-registration and CPSF recalibration

```bash
bash scripts/run_external_baselines.sh \
  --regime all \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 \
  --max-scenarios 0 \
  --womd-role validation \
  --retrain \
  --recalibrate
```

### Strict test-only mode

```bash
bash scripts/run_external_baselines.sh \
  --regime all \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 \
  --max-scenarios 0 \
  --womd-role validation \
  --test-only
```

`--test-only` fails closed if a required learned checkpoint or Near calibration artifact is missing/incompatible.

## 4. External baselines — one regime at a time

Safe:

```bash
bash scripts/run_external_baselines.sh \
  --regime safe \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 --max-scenarios 0 --womd-role validation
```

Near-Contact:

```bash
bash scripts/run_external_baselines.sh \
  --regime near \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 --max-scenarios 0 --womd-role validation
```

Contact:

```bash
bash scripts/run_external_baselines.sh \
  --regime contact \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 --max-scenarios 0 --womd-role validation
```

Append `--retrain` to force retraining/re-registration; append `--recalibrate` for the Near CPSF artifact.

## 5. Frozen-module ablations

The main set keeps the existing fixed-module ablations:

- no observation consistency;
- mean instead of lower tail;
- no actuator projection;
- no persistent re-entry;
- no RIFA absolute admission.

```bash
GPU0=0 GPU1=1 CUDA_DEVICES=0,1 \
BASE_OUT="$BASE_OUT" \
OUT_ROOT="$BASE_OUT/ocrap_v48_111_submission_ablations" \
MODEL_RUN="$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main" \
FULL_RUN_ROOT="$BASE_OUT/ocrap_v48_111_submission_three_regime" \
VARIANTS=balanced,precision \
MAX_SCENARIOS=0 \
MAX_STEPS=40 \
NUM_CANDIDATES=24 \
NUM_RECOVERY_OPTIONS=12 \
WOMD_ROLE=validation \
ABLATION_SET=main \
bash scripts/run_submission_ablations.sh
```

For supplementary `no_active_set_alignment` and `no_route_alignment` as well:

```bash
ABLATION_SET=all bash scripts/run_submission_ablations.sh
```

Keep the same environment variables from the main command when running `ABLATION_SET=all`.

## 6. Three-regime qualitative videos

Prerequisites: first complete the OC-RAP three-regime evaluation and external baseline evaluation above.  The visualization selector uses their corrected metric provenance, generates traces **only for selected scenes**, then renders MP4 videos.

```bash
JOBS_PER_GPU=3 MAX_PARALLEL=6 \
bash scripts/build_regime_visualizations.sh \
  --ocrap-results "$BASE_OUT/ocrap_v48_111_submission_three_regime/balanced" \
  --model-run "$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main" \
  --external-root "$BASE_OUT/external_baselines_v48_111" \
  --variant balanced \
  --out "$BASE_OUT/regime_visualization_v48_111" \
  --num-scenes 3 \
  --gpus 0,1 \
  --fps 10 \
  --trace-steps 60 \
  --camera fixed \
  --view-radius 35
```

Final index:

```text
$BASE_OUT/regime_visualization_v48_111/videos/REGIME_VIDEO_INDEX.json
```

## 7. Toy-example selection and figures

This remains dataset-only and does not depend on a model checkpoint:

```bash
python tools/select_recovery_toy_examples.py \
  --near "$OCRAP_ROOT/val_near_contact" \
  --contact "$OCRAP_ROOT/val_contact" \
  --output "$BASE_OUT/paper_toy_examples_val" \
  --oracle-gap-count 24 \
  --near-reserve-count 8 \
  --contact-debt-count 8 \
  --min-compat 0.95 \
  --max-hard 0 \
  --max-harm 0.05 \
  --dpi 180 \
  --resume
```

## 8. Dataset construction

### Train buckets

These retain the actual Safe / Near / Contact training recipes from the supplied construction instructions and use WOMD `training`.

All three:

```bash
bash scripts/build_training_datasets.sh --regime all
```

Or separately:

```bash
bash scripts/build_training_datasets.sh --regime safe
bash scripts/build_training_datasets.sh --regime near
bash scripts/build_training_datasets.sh --regime contact
```

### Validation + held-out test buckets

The cleaned builder deliberately uses **standard WOMD validation** for all six buckets and deterministic scene-disjoint partitions; it does not use `validation_interactive` for Near/Contact publication test buckets.

```bash
GPU0=0 GPU1=1 RESUME=1 \
OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" \
bash scripts/build_validation_test_datasets.sh
```

### Calibration buckets

Build in a temporary root and merge into `$OCRAP_ROOT/calibration_*` with scene-overlap checks:

```bash
GPU0=0 GPU1=1 \
OCRAP_ROOT="$OCRAP_ROOT" WOMD_ROOT="$WOMD_ROOT" \
CALIBRATION_BUILD_ROOT="$OCRAP_ROOT/.calibration_build" \
CALIBRATION_OVERWRITE=0 \
bash scripts/build_ocrap_datasets.sh --role calibration
```

Set `CALIBRATION_OVERWRITE=1` only after intentionally deciding to replace existing final calibration roots.

### Fresh full rebuild

Only for a fresh/empty target root:

```bash
GPU0=0 GPU1=1 \
OCRAP_ROOT=/path/to/fresh/OCRAP \
WOMD_ROOT="$WOMD_ROOT" \
bash scripts/build_ocrap_datasets.sh --all
```

Do not run a full rebuild over the currently audited dataset merely to use this cleaned code.


## 10. Dataset property + construction provenance audit

Paper-grade full scan of the standard 12 OC-RAP buckets:

```bash
cd /home/senzeyu2/code/OC-RAP
OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
OUT=/home/senzeyu2/code/OC-RAP/runs/dataset_properties_v48_111 \
bash scripts/analyze_dataset_properties.sh
```

Upload this file for build-parameter reconstruction and property analysis:

```text
/home/senzeyu2/code/OC-RAP/runs/dataset_properties_v48_111/dataset_properties_bundle.zip
```

For a fast statistical smoke test only:

```bash
MAX_SAMPLES=500 \
OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
OUT=/home/senzeyu2/code/OC-RAP/runs/dataset_properties_v48_111_quick \
bash scripts/analyze_dataset_properties.sh
```

The exporter is read-only. It captures the full `resume_contract.json`/`semantic_config`, `dataset_summary.json`, manifest hashes/columns/WOMD-role counts and the standard dataset diagnostics. Standard publication test buckets should resolve to WOMD `validation`, not `validation_interactive`.
