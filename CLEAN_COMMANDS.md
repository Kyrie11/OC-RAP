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

## 0. Current V48.121 scientific reference contract

The stable unversioned orientation launcher now runs **V48.121 OC-VRPC**. It treats the completed V48.120 OC-VRT STOP artifacts as immutable scientific inputs and verifies their exact SHA/status/registered rank-persistence next branch before any V48.121 audit starts. Python implementation filenames remain semantic/unversioned; scientific inputs and outputs remain versioned.

V48.120 established an important structural point: the set functional can simultaneously preserve a candidate-independent **nominal recovery ordering** and **same-option candidate causal correspondence**. Production transport and rank reassignment were active in every unique role, and Contact Reserve retained strong directional signal. Nevertheless the instantaneous nominal-rank transport did not transfer across populations, especially on Support. The remaining preregistered question is whether the same option must retain its nominal rank ownership over the relevant prefix/suffix for candidate displacement to become population stable.

## 1. Current orientation audit — V48.121 OC-VRPC

V48.121 is **audit-only**. The 156-D frozen candidate response, actuator-projected executable constraints, same-option joint prefix/suffix viability margins, recovery horizon/library, cohorts, whole-row candidate shuffle and 220-D closed-form ridge capacity remain fixed. Candidate values never define the rank coordinate.

For each recovery option, V48.121 computes its exact candidate-independent nominal midrank at every time. Exact nominal ties share one midrank. Directional persistence is threshold-free:

```text
prefix defect(t) = mean_{s<=t} |r_l(t) - r_l(s)|
suffix defect(t) = mean_{s>=t} |r_l(t) - r_l(s)|
```

The same option's signed viability displacement `d_l(t)=q_l^a(t)-q_l^0(t)` is coupled to four fixed modes:

```text
1                 global signed viability shift
x = 2r - 1        nominal-rank tilt
p                 rank-persistence defect
x * p             rank-by-persistence interaction
```

These modes are fixed before evaluation. There is no rank cut, quantile mass, temperature, decay/window tuning, option-count sweep or learned set encoder.

The primary `full_persistence` uses all common valid recovery options. The equal-capacity `exposed_persistence` control uses only the support (not weights) of V48.117 frozen weak-root zero-boundary witnesses. Option identity is used only internally to maintain same-option trajectories and is never exported to the readout.

```text
8 bins x 2 temporal channels x 4 fixed coupling modes = 64-D
156-D frozen candidate response + 64-D = 220-D
```

No regime label, teacher future value, boundary transport, threshold, horizon, source, capacity, LR or epoch sweep is introduced.

Run exactly:

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 \
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
bash scripts/run_constraint_native_orientation_audit.sh
```

Upload only:

```text
/home/senzeyu2/code/OC-RAP/runs/OC-RAP-v48.121-OC-VRPC-results.zip
```

### About OC-RAP training in V48.121

There is no registered V48.121 planner/root/source training stage. Running generic `python -m ocrap.cli train` would not reproduce this audit. Main/carrier integration is authorized only if a preregistered persistence family passes absolute Support+Reserve, historical improvement, and exact persistence/reassignment/re-entry activity gates.

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
