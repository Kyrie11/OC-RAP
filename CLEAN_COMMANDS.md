# 0. Final evaluation contract — V48.124.10.7.5 (baseline/full-cohort/latency contract; no OC-RAP algorithm change)

V48.124.10.7.2 closed internal mechanism search, V48.124.10.7.3 locked the submitted Main for final reporting, and V48.124.10.7.4 repaired method-independent observation-legal target eligibility. V48.124.10.7.5 still does **not** change the OC-RAP planner. It closes three final-baseline engineering gaps: `--max-scenarios 0` now reaches the low-level regime launchers as the full frozen cohort rather than silently reverting to 50 scenes; completed artifacts are accepted only when their scene journal exactly matches the frozen target-key lock; and publication latency is measured for OC-RAP and every baseline under the same isolated single-process/single-GPU contract and exact target set. It also repairs native validation for Diffusion Planner and Flow Planner and retains the FP32 GameFormer-lite training fix.

Build the method-independent observation-legal target locks first. This step is cheap compared with closed-loop rollout and can be run before OC-RAP or any baseline:

```bash
cd /home/senzeyu2/code/OC-RAP
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/build_final_observation_legal_target_locks.sh
```

The lock files are:

```text
$BASE_OUT/ocrap_v48_124_final_characterization/target_keys/safe.json
$BASE_OUT/ocrap_v48_124_final_characterization/target_keys/near.json
$BASE_OUT/ocrap_v48_124_final_characterization/target_keys/contact.json
```

Resume/finalize the frozen OC-RAP characterization with the same command as V48.124.10.7.3:

```bash
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_final_locked_three_regime_characterization.sh
```

The launcher reuses already-complete Near/Contact artifacts and the existing Safe scene journals. It does **not** backfill a route-ineligible Safe target from a different scene. The final paper cohort is the predeclared target cohort intersected with method-independent observation-legal route eligibility; the exclusion count and provenance are serialized in the target lock and closed-loop result.

External baselines no longer depend on OC-RAP finishing. As soon as the target locks exist, the three regimes may be launched independently. To reuse compatible learned checkpoints from a previous baseline root while rerunning the final observation-legal test protocol, set `PRETRAINED_BASELINE_ROOT`:

```bash
PRETRAINED_BASELINE_ROOT=/home/senzeyu2/code/OC-RAP/runs/external_baselines_v48_111 \
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs bash scripts/run_final_external_baselines.sh safe

PRETRAINED_BASELINE_ROOT=/home/senzeyu2/code/OC-RAP/runs/external_baselines_v48_111 \
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs bash scripts/run_final_external_baselines.sh near

PRETRAINED_BASELINE_ROOT=/home/senzeyu2/code/OC-RAP/runs/external_baselines_v48_111 \
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs bash scripts/run_final_external_baselines.sh contact
```

The final wrapper always reruns closed-loop testing on the exact target lock and follows the throughput accuracy run with an isolated one-GPU latency profile. Do **not** resume historical 50-scene/no-target-lock closed-loop journals into the final-v2 root: their information contract differs. Compatible training checkpoints may be staged, but final closed-loop results start in the clean `external_baselines_v48_124_final_v2` root. Three learned baselines must be retrained under the current contract: **GameFormer-lite** (the uploaded AMP run produced non-finite recurrent gradients; final config is FP32), **Diffusion Planner**, and **Flow Planner** (their historical eval path compared a generated prediction with its detached copy, producing identically zero validation loss and invalid `best.pt` selection). PlanTF, PLUTO, Plan-R1, and BeTopNet checkpoints may be reused when the checker validates their training/implementation contract.

Build final comparison tables only from the new final roots:

```bash
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/build_final_regime_comparison_tables.sh
```

The historical V48.124 deployment-acceptance freeze remains **NO**; the immutable evaluation/artifact freeze remains **YES**. No V48.125, threshold/capacity sweep, or recovery-mechanism search is authorized.

---

# 0. Terminal internal closure — V48.124.10.7.2 (offline only)

V48.124.10.7.1 is attribution-ready and `ONE_SHOT_ACTION_REALIZATION_MIXED`. The repeated privileged-trajectory confound has been removed for the registered causal question: each seed scene follows exact nominal to the preregistered seed, executes exactly one historical strongest candidate, then returns to exact nominal forever. One scene is locally positive and one worsens Near extremal endpoints. Therefore do **not** open another recovery mechanism, admission retraining, threshold/capacity sweep, or another realization-ceiling experiment.

Keep the authoritative 10.7.1 bundle in `BASE_OUT`, then run the stable command:

```bash
cd /home/senzeyu2/code/OC-RAP
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_constraint_native_orientation_audit.sh
```

Expected artifact:

```text
$BASE_OUT/OC-RAP-v48.124.10.7.2-TERMINAL-INTERNAL-CLOSURE.json
```

This is an offline bookkeeping closure, not a new experiment. It freezes internal mechanism search but explicitly withholds deployed-Main freeze and final three-regime/external-baseline authorization because the V48.124 Near system gate was not repaired and the deployed Main was not modified. Historical one-shot reproduction remains available only via `OCRAP_CONSTRAINT_AUDIT_MODE=one_shot_action_realization`.

---

# 0. Current terminal Near diagnostic — V48.124.10.7.1 one-shot action realization

V48.124.10.6 is attribution-ready and `NONFLOOR_ADMISSION_SEED_SCREEN_NOT_PROMISING`. Do not interpret a repeated privileged PCD trajectory as the isolated effect of the original seed action. The terminal diagnostic follows exact nominal until the preregistered strongest V48.124.10.5 non-floor seed, executes that candidate once, then returns to exact nominal forever. No training, recalibration, threshold sweep, candidate/recovery change, or new mechanism is authorized.

Keep the V48.124.10.6 result ZIP in `BASE_OUT`, then run the stable command:

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_constraint_native_orientation_audit.sh
```

Equivalent explicit launcher:

```bash
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_near_nonfloor_one_shot_realization_two_gpu.sh
```

Expected artifact:

```text
$BASE_OUT/OC-RAP-v48.124.10.7.1-ONE-SHOT-ACTION-REALIZATION-results.zip
```

This is a terminal diagnostic ceiling, not publication evidence and not a deployable arm. Scientific attribution is entered only if the result bundle proves exactly one seed intervention per scene, exactly two teacher labels at the seed, exact nominal elsewhere, valid observation-legal route provenance, and the authoritative 10.6 predecessor contract.

---

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

**Final-paper path.** First build the observation-legal target locks (Section 0). These locks depend only on the frozen bucket provenance and WOMD connectivity geometry, not on OC-RAP outputs, so baseline testing can run in parallel with the resumed OC-RAP characterization.

Preferred final commands:

```bash
# Safe
PRETRAINED_BASELINE_ROOT="$BASE_OUT/external_baselines_v48_111" \
  bash scripts/run_final_external_baselines.sh safe

# Near-Contact
PRETRAINED_BASELINE_ROOT="$BASE_OUT/external_baselines_v48_111" \
  bash scripts/run_final_external_baselines.sh near

# Contact
PRETRAINED_BASELINE_ROOT="$BASE_OUT/external_baselines_v48_111" \
  bash scripts/run_final_external_baselines.sh contact
```

`PRETRAINED_BASELINE_ROOT` is optional. When supplied, checkpoint candidates are staged but are reused only after `check_external_training_complete.py` verifies epoch budget, AMP mode, method identity, model validation version, and implementation contract. **Retrain from scratch:** GameFormer-lite, Diffusion Planner, and Flow Planner. Their old checkpoints are intentionally invalid under V48.124.10.7.5. **Checkpoint reuse allowed if validated:** PlanTF, PLUTO, Plan-R1, and BeTopNet. The remaining Near/Contact filters/controllers are non-learning registrations/calibrations and have no neural checkpoint to retrain. Manual deletion of old checkpoints is not required when using the new final-v2 root; deleting only the three invalid learned checkpoint directories is optional if a clean training tree is preferred.

The wrapper enforces standard WOMD `validation`, the exact regime target lock, max 40 closed-loop steps, observation-legal `sdc_paths`, and no logged/future SDC route fallback. It runs the complete current method set: Safe includes Diffusion Planner; Near includes Flow Planner, Plan-R1, and BeTopNet; Contact includes all six post-impact baselines.

Accuracy/metric evaluation may use the throughput scheduler (three jobs per GPU by default). **Do not report latency from that throughput run.** By default the wrapper immediately reruns the same regime in isolated single-GPU test-only mode and writes the publication timing root `${FINAL_EXTERNAL_BASELINE_OUT}_latency_isolated`. Run OC-RAP latency symmetrically with `bash scripts/profile_ocrap_latency.sh all`. The table builder refuses a latency artifact unless `timing.execution_contract=isolated_single_process_single_gpu` and its scene-key journal exactly equals that method's accuracy scene-key journal.

The lower-level `scripts/run_external_baselines.sh` commands remain supported for debugging/reproduction, but their outputs are not final-paper evidence unless they use the same target lock and observation-legal route contract.

## 5. Frozen-module ablations

The final ablation suite is tied to the **frozen submitted Main** (`lcb_constrained`) and to the same observation-legal target locks used in the main tables. It tests modules only where their semantics are active. The main set is:

| Ablation | Safe | Near | Contact | What is removed |
|---|:---:|:---:|:---:|---|
| `no_obs_consistency` | -- | ✓ | ✓ | observation-compatible common-option recovery aggregation |
| `mean_tail` | -- | ✓ | ✓ | lower-tail aggregation, replacing it with a mean |
| `no_actuator_projection` | -- | ✓ | ✓ | actuator-feasible recovery projection |
| `no_persistent_reentry` | -- | -- | ✓ | post-contact persistent safe re-entry semantics |
| `no_rifa_absolute_admission` | ✓ | ✓ | ✓ | final Main absolute recovery-admission predicate (now wired to `lcb_constrained`) |
| `no_nominal_abstention` | ✓ | ✓ | ✓ | exact-nominal fail-closed abstention / nominal priority |
| `no_route_alignment` | -- | ✓ | ✓ | route-alignment semantics in recovery witness/certification |

The supplementary `no_active_set_alignment` ablation is evaluated in Near and Contact. Structural/physical teacher separation, split/merge counterfactual correspondence, the signed reserve/debt interpretation, and the absence of a learned regime router are **not** represented as local runtime knockouts: they are training/data/theory contracts and should be supported by offline audits or structural analysis rather than by a misleading single-flag ablation.

Run the main frozen-module suite:

```bash
GPU0=0 GPU1=1 CUDA_DEVICES=0,1 \
BASE_OUT="$BASE_OUT" \
OUT_ROOT="$BASE_OUT/ocrap_v48_124_final_ablations" \
MODEL_RUN="$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main" \
FULL_RUN_ROOT="$BASE_OUT/ocrap_v48_124_final_characterization/ocrap" \
VARIANTS=balanced,precision \
MAX_SCENARIOS=0 MAX_STEPS=40 NUM_CANDIDATES=24 NUM_RECOVERY_OPTIONS=12 \
WOMD_ROLE=validation ABLATION_SET=main \
bash scripts/run_submission_ablations.sh
```

For the supplementary active-set knockout as well:

```bash
ABLATION_SET=all bash scripts/run_submission_ablations.sh
```

The launcher automatically builds missing final target locks, fails closed if a regime cannot match the lock, keeps checkpoint/calibration fixed, and writes one independent job per GPU. Do not compare an ablation evaluated on a different target set with the frozen Main.

## 6. Three-regime qualitative videos

Prerequisites: first complete the OC-RAP three-regime evaluation and external baseline evaluation above.  The visualization selector uses their corrected metric provenance, generates traces **only for selected scenes**, then renders MP4 videos.

```bash
JOBS_PER_GPU=3 MAX_PARALLEL=6 \
bash scripts/build_regime_visualizations.sh \
  --ocrap-results "$BASE_OUT/ocrap_v48_111_submission_three_regime/balanced" \
  --model-run "$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main" \
  --external-root "$BASE_OUT/external_baselines_v48_111" \
  --variant balanced \
  --out "$BASE_OUT/regime_visualization" \
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
