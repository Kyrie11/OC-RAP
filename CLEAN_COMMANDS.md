# 0. Current Near localization — V48.124.10.6 non-floor admission seed screen

V48.124.10.5 is attribution-ready but its apparent 69 non-trigger PCD opportunities are dominated by the exact `R_dep*=0.5` structural plateau. Only seven non-floor positive decisions remain, in two scenes, all before the first historical Base intervention. Non-floor does **not** imply fully point-identified physical truth under the V48.80 structural-interval contract; this is only a cleaner falsification subset. Do **not** train/recalibrate an absolute source and do not sweep thresholds yet.

Keep the V48.124.10.5 result ZIP in `BASE_OUT`, then run:

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_constraint_native_orientation_audit.sh
```

Equivalent explicit launcher:

```bash
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_near_nonfloor_admission_seed_screen_two_gpu.sh
```

Expected artifact:

```text
$BASE_OUT/OC-RAP-v48.124.10.6-NONFLOOR-ADMISSION-SEED-SCREEN-results.zip
```

The screen fresh-replays only the two non-floor seed scenes per robustness variant. Before each preregistered seed start, frozen Base must remain exact nominal. From the seed onward a privileged oracle may bypass learned absolute admission only for candidates with teacher `R_dep*>0`, `R_dep* != 0.5` within `1e-8`, and teacher PCD strictly above exact nominal. This is a **falsification screen**, not a 250-scene population GO test. `PROMISING` means only that the admission-repair hypothesis merits a broader small scene-disjoint prevalence audit; it does not authorize V48.125.

---

# 0. Current Near STOP localization — V48.124.10.5

V48.124.10.4 established `PCD_ORACLE_CEILING_STOP`, so do **not** train/recalibrate a relative head or sweep RIFA thresholds. Keep the 10.4 result ZIP in `BASE_OUT`, then run the small all-state support scan:

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_constraint_native_orientation_audit.sh
```

This replays only the same 11 historical Base-intervention scenes per robustness variant. Base actions remain unchanged; after each action is selected, all 24 frozen candidates are teacher-labeled for diagnostic PCD support. Expected artifact:

```text
/home/senzeyu2/code/OC-RAP/runs/OC-RAP-v48.124.10.5-ALL-STATE-SUPPORT-LOCALIZATION-results.zip
```

This diagnostic determines whether Base-nominal decisions on the frozen intervention cohort contain positive **absolute-admitted** PCD opportunities. It is not a deployed arm and is not publication evidence.

# Current next step — V48.124.10.4 privileged PCD-oracle Near ceiling

V48.124.10.3.1 is attribution-ready and localizes a relative-evidence alignment bottleneck, but no closed-loop oracle ceiling has yet been run. The default stable command now fresh-replays only the 11 historical Base-intervention scenes for balanced/precision in parallel. The other 239 Base-zero-intervention Near scenes are reused exactly by the trigger-gating induction proof.

Prerequisites:

```text
$BASE_OUT/OC-RAP-v48.124.10.2-OBSERVATION-LEGAL-NEAR-results.zip
$BASE_OUT/OC-RAP-v48.124.10.3-CANDIDATE-QUALITY-AUDIT-results.zip
```

Run:

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_constraint_native_orientation_audit.sh
```

Equivalent explicit launcher:

```bash
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_near_pcd_oracle_ceiling_two_gpu.sh
```

Outputs:

```text
$BASE_OUT/OC-RAP-v48.124.10.4-PCD-ORACLE-CEILING-results.zip
$BASE_OUT/OC-RAP-v48.124.10.4-PCD-ORACLE-CEILING.json
```

The privileged oracle is a causal-sufficiency ceiling, not publication evidence and not a deployable selector. Do not start final external-baseline or three-regime Main-table evaluation until a deployable selector is subsequently frozen.

---

# Current next step — small Near candidate-quality audit

The attribution-ready observation-legal V48.124.10.2 run is a scientific Near STOP. Do **not** rerun 250 scenes and do not start V48.125 yet. The stable command now replays only the 11 fresh intervention scenes and computes privileged teacher labels only after the frozen base action is selected.

Prerequisite:

```text
$BASE_OUT/OC-RAP-v48.124.10.2-OBSERVATION-LEGAL-NEAR-results.zip
```

Run:

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_constraint_native_orientation_audit.sh
```

Equivalent explicit launcher:

```bash
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_near_candidate_quality_audit_two_gpu.sh
```

Output:

```text
$BASE_OUT/OC-RAP-v48.124.10.3-CANDIDATE-QUALITY-AUDIT-results.zip
$BASE_OUT/OC-RAP-v48.124.10.3-CANDIDATE-QUALITY-AUDIT.json
```

This is diagnostic-only; external-baseline/final three-regime evaluation remains blocked until a deployed selector is frozen.

---

# V48.124.10 Near deployed-system-axis diagnostic

V48.124.9 is attribution-ready but Near is STOP. Recovery-mechanism / representation search remains frozen. The next command therefore runs two preregistered selector-only diagnostic arms on the exact 35-scene historical Near intervention cohort; it does not train, recalibrate, sweep thresholds, change horizon, or modify the recovery library.

Run exactly the same command as before:

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_constraint_native_orientation_audit.sh
```

Canonical output for the next analysis is:

```text
/home/senzeyu2/code/OC-RAP/runs/OC-RAP-v48.124.10-NEAR-RIFA-SYSTEM-AXIS-results.zip
```

The default run is intentionally diagnostic-scale (35 historical intervention scenes per arm/variant). If one arm is promoted, the next scientific confirmation must rerun a fresh full 250-scene Near population; reconstructed 35+215 artifacts are explicitly marked internal-diagnostic-only and are not publication Main results. To reproduce the historical V48.124.9 full audit instead, set `OCRAP_CONSTRAINT_AUDIT_MODE=full`.

---

## V48.124.7 provenance re-adjudication fix

V48.124.7 is engineering-only. It does not authorize V48.125 mechanism search and does not change the frozen Main. It permits exact SHA-pinned reuse of the authoritative V48.124.5 full-population runtime evidence during V48.124.7 re-adjudication; arbitrary stale runtime contracts still fail closed.

The stable command remains:

```bash
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_constraint_native_orientation_audit.sh
```

## V48.124.6 Contact construct-validity engineering fix

V48.124.6 does not change the planner. It changes scientific adjudication only: Contact post-contact endpoints are not entered unless nominal/balanced/precision all start from the same simulator-observed contact anchor at rollout step 0, before policy action. The existing `test_contact` counterfactual-surrogate cohort therefore fails closed for post-contact causal adjudication; do not rerun the long V48.124 population command on that cohort expecting a valid Contact gate.

To re-audit an existing extracted V48.124.5 result bundle without GPU work:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/audit_fixed_main_scientific_validity.py \
  --bundle-dir /path/to/extracted/OC-RAP-v48.124-OC-FMSA-results \
  --output /path/to/V48.124.5-scientific-validity-reaudit.json
```

A new Contact experiment must first construct a method-independent pre-treatment contact-anchor cohort; theory/mechanism search remains frozen.

# OC-RAP cleaned workspace — current commands

> Publication WOMD role: all publication validation/test/calibration buckets use standard WOMD `validation`. V48.124 explicitly requests `validation` and fails closed if bucket provenance disagrees. `validation_interactive` is not a publication/test source. Any TeX text claiming otherwise is a paper error and must be corrected to `validation`; do not change the code/data source to match the typo.

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

## 1. Current scientific stage — V48.124 OC-FMSA

V48.123 passed reliability/scientific attribution but formally STOPped as a complete zero-boundary transition carrier. Its exact preregistered successor freezes the recovery-set mechanism family. V48.124 therefore does **not** train, recalibrate, integrate a V48.123 carrier, or define another recovery statistic. It evaluates the existing frozen L80 Main with balanced/precision robustness variants against same-target nominal controls.

V48.124 preregisters five gates: exact full-bucket coverage/source identity, deterministic sentinel replay, Safe non-interference, Near closed-loop validity, and Contact recovery validity. Paired inference uses 5000 bootstrap draws with seed 2027 and zero non-interference margin. If all five GO, the Main is frozen and final external-baseline comparison is authorized. If any gate STOPs, the recovery mechanism family remains frozen and only that failed system/provenance axis may be diagnosed.

Run exactly (V48.124.8 automatically mines and freezes the causally valid Contact anchor cohort before Contact treatment):

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 \
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
bash scripts/run_constraint_native_orientation_audit.sh
```

The stable launcher uses `scripts/build_contact_anchor_cohort.sh` internally. Do not manually select Contact scenes from treatment outcomes. The generated `contact_anchor_manifest.json` is part of the scientific evidence and fixes one pre-treatment observed-overlap state per scene.

Upload only:

```text
/home/senzeyu2/code/OC-RAP/runs/OC-RAP-v48.124-OC-FMSA-results.zip
```

### What the stable command does in V48.124

1. verifies exact authoritative V48.123 STOP/freeze-branch SHA and runtime contract before long GPU work;
2. runs full same-target nominal Safe/Near/Contact controls, resolving WOMD from bucket provenance;
3. runs frozen L80 balanced and precision Safe/Near/Contact evaluation with scenes embedded;
4. computes six fixed paired-bootstrap comparisons;
5. replays one lexicographically fixed sentinel target per regime/variant and checks deterministic scientific output;
6. applies the preregistered coverage / determinism / Safe / Near / Contact gates;
7. packages the canonical V48.124 adjudication bundle.

There is no registered V48.124 recovery-model training stage and no V48.125 recovery mechanism successor. A V48.124 STOP keeps the mechanism family frozen.


### V48.124.9 execution-source isolation

The stable V48.124 command now snapshots the repository at launch and runs the entire fixed-Main audit from that immutable snapshot. You may edit the original checkout for external-baseline work after the command starts; those edits will not affect the running scientific audit. If an existing fixed-Main work directory was produced by different runtime hashes, it is archived automatically and rebuilt instead of being mixed with the new run.

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
USE_DYNAMIC_SCHEDULER=auto
bash scripts/run_external_baselines.sh \
  --regime safe \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 --jobs-per-gpu 3 --max-parallel 6 --max-scenarios 0 --womd-role validation
```

Near-Contact:

```bash
USE_DYNAMIC_SCHEDULER=auto
bash scripts/run_external_baselines.sh \
  --regime near \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 --jobs-per-gpu 3 --max-parallel 6 --max-scenarios 0 --womd-role validation
```

Contact:

```bash
USE_DYNAMIC_SCHEDULER=auto
bash scripts/run_external_baselines.sh \
  --regime contact \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 --jobs-per-gpu 3 --max-parallel 6 --max-scenarios 0 --womd-role validation
```

Each one-regime launcher now uses six concurrent worker slots by default: three jobs on GPU 0 and three jobs on GPU 1. A slot belongs to one baseline for its full pipeline (checkpoint preparation/training or registration reuse, optional offline evaluation, then closed-loop test); as soon as that baseline finishes, the next queued baseline is launched on the GPU whose slot became free. This removes the old global train/test phase barrier. Safe additionally enables Diffusion Planner; Near-Contact additionally enables Flow Planner, Plan-R1, and the expanded BeTopNet adapter. Contact keeps its six post-contact baselines. Dynamic refill is the default and requires Bash with `wait -n -p` support (Bash 5.1+ is recommended).

**Latency exception.** The commands above are throughput-oriented and intentionally run multiple processes per GPU, so their timing must not be reported as uncontended publication latency. After the accuracy/metric run is complete, profile a regime serially on one GPU while reusing its checkpoints (and the Near CPSF calibration artifact):

```bash
bash scripts/profile_external_baselines_latency.sh \
  --regime safe \
  --source-run "$BASE_OUT/external_baselines_v48_111" \
  --gpu 0 --max-scenarios 0 --womd-role validation
```

Use `--regime near` or `--regime contact` analogously. The closed-loop summary records warm-up-excluded steady-state mean/p50/p95 deployed-planner latency; the comparison-table builder prefers that steady-state mean when present.

Append `--retrain` to force retraining/re-registration; append `--recalibrate` for the Near CPSF artifact. Set `RUN_SUPPLEMENTARY_SAFE=false` or `RUN_SUPPLEMENTARY_NEAR=false` only when reproducing the historical main-table-only suite.

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

## V48.124.2 fixed-Main adjudication resume after sentinel/provenance hotfix

The user-facing command is unchanged:

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 \
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
bash scripts/run_constraint_native_orientation_audit.sh
```

If the completed V48.124 nominal/balanced/precision Safe/Near/Contact artifacts remain under
`$BASE_OUT/ocrap_v48_124_fixed_main_stability`, the launcher reuses them and only recomputes paired comparisons, six one-target sentinel replays, adjudication, pipeline closure and packaging. Do not delete the completed full-population artifacts before this resume.

Upload only the final canonical bundle after the launcher prints success:

```text
/home/senzeyu2/code/OC-RAP/runs/OC-RAP-v48.124-OC-FMSA-results.zip
```

## V48.124.3 resume after journal-finalize scene-preservation hotfix

Do **not** delete `runs/ocrap_v48_124_fixed_main_stability/`. The six balanced/precision population `.scenes.jsonl` journals are the authoritative completed scene evidence and are reused. V48.124.3 reconstructs the requested embedded metric scenes from those journals, preserving bucket provenance, then recomputes paired comparisons and runs only the missing deterministic one-target sentinels plus adjudication/package stages.

Stable command remains unchanged:

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 \
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
bash scripts/run_constraint_native_orientation_audit.sh
```

Expected successful terminal artifact:

```text
/home/senzeyu2/code/OC-RAP/runs/OC-RAP-v48.124-OC-FMSA-results.zip
```

Upload that canonical ZIP for the reliability/scientific-attribution audit. Until it exists and closes, do not interpret the stale paired-comparison files as V48.124 scientific evidence and do not authorize external-baseline comparison from V48.124.



V48.124.4 keeps the stable launcher command unchanged and enforces `selection.require_absolute_admission_for_intervention=true` inside the publication closed-loop launcher. Do not reuse V48.124.1-.3 population artifacts for the repaired Main.


## V48.124.5 exact-nominal control conformance

V48.124.5 keeps the scientific version `v48.124-OC-FMSA` and the stable launcher command unchanged. It repairs only the control arm: `method=nominal` and `log_replay` execute the explicit upstream nominal anchor `a0` even when that prefix is marked infeasible. They never feasibility-substitute another candidate. The adjudicator requires top-level and scene-level nominal intervention rate to be exactly zero and `selection_reason_counts={"nominal_prefix_exact_a0": ...}`.

The active runtime source closure now includes `src/ocrap/evaluation/baselines.py`. Because this active source and `closed_loop_runner.py` changed, V48.124.4 population artifacts are not scientific evidence for V48.124.5. Use the fresh work directory `ocrap_v48_124_exact_nominal_fixed_main`.

For a fresh two-GPU run, balanced and precision robustness variants are execution-independent and are launched concurrently, one complete variant per GPU. Each variant still runs Safe/Near/Contact sequentially on its assigned GPU; no checkpoint, RNG state, cache, or rollout state is shared across variants. This is an orchestration-only acceleration.

Stable command:

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 \
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
bash scripts/run_constraint_native_orientation_audit.sh
```

Upload only `runs/OC-RAP-v48.124-OC-FMSA-results.zip` after completion.
