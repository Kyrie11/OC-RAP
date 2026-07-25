# OC-RAP Algorithm Changelog

## v48.4 — OC-TRAC-SRGR (2026-07-25)

### Evidence motivating this iteration

The uploaded v48.3 proxy-calibration run did **not** resolve the policy-level failure.
Candidate-positive AUC remained informative (Near 0.7249–0.7268; Contact 0.7634–0.7906),
but group top-1 correlation stayed negative (Near about -0.02; Contact -0.14 to -0.068),
and every fit/verify policy selected zero recovery actions. Relative to v48.1, v48.3 improved
only balanced-Near candidate AUC and slightly reduced the precision-Contact top-1 error; it
regressed precision-Near top-1 and Contact candidate AUC. Natural-gate abstention remained
correct, but no Near/Contact recovery benefit was demonstrated.

Two implementation defects were found in the executed v48.3 path:

- NASC used a non-zero random residual at warm start (`sigmoid(-1.5)≈0.18`), so loading the
  v48.1 checkpoint did not initially reproduce the inherited selector.
- `training.best_metric=loss_direct_recovery_value_worst` was never emitted by validation;
  the trainer silently fell back to total loss, so checkpoint selection did not optimize
  the intended worst-regime policy objective.

### New algorithmic contribution: Shift-Robust Groupwise Recovery (SRGR)

- **ZI-NASC:** zero-initialized nominal-anchored set context. The inherited pointwise policy
  is now an exact initialization, while the set residual learns only evidence-supported
  corrections. The gate is also made more conservative.
- **DRA-RCD:** decoupled ranking-admission regret distillation. Value-only logits learn the
  within-group teacher ordering and expected regret; opportunity/harm logits remain in a
  separate admission distribution. A weak/non-transferable harm head can therefore block
  unsafe execution without corrupting candidate ranking gradients.
- **Soft opportunity/downside supervision:** continuous teacher advantage is converted to
  soft labels around the positive/negative margins, reducing contradictory labels caused
  by small train/dev contract shifts.
- **Pseudo-environment GroupDRO:** group losses are robustly aggregated over
  `(regime, nominal-severity bin, opportunity state, teacher-best macro)` environments.
  This reduces domination by train-specific severity or macro pockets without using
  calibration/test distributions for training.
- **Policy-regret checkpointing:** validation now reports exact teacher-PCD group regret,
  top-1 accuracy, positive recall and harmful-switch rate for Near and Contact. Early
  stopping uses worst-regime mean regret and raises an error if the configured metric is
  absent; silent fallback is removed.

### Engineering and experiment protocol

- The v48.1 precision checkpoint loads into v48.4 with no shape mismatch; training from
  scratch or from v47 is not recommended for this iteration.
- Added `scripts/run_v48_4_core_ablations.sh` with four explicit runs: SRC reference,
  ZI-NASC only, DRA-RCD only, and full SRGR. Each run has its own output directory.
- Added `scripts/recalibrate_v48_4_multiseed.sh`. `CALIBRATION_SEED` values 4801/4802/4803
  produce separate proxy splits and separate output roots while reusing the same trained
  checkpoint; checkpoints are not retrained for calibration-seed robustness.
- Added aggregation tools for ablation and multi-seed summaries.
- Direct-only and full training paths now call the same direct-value loss helper.

### Required decision gates before closed loop

- Near and Contact group top-1 correlation should become positive and preferably exceed 0.10
  in screening; publication readiness remains >=0.20.
- At least one variant must produce non-zero verify selections in both regimes with finite
  precision/harm bounds.
- Candidate AUC should not fall more than 0.03 below v48.1 while group regret improves.
- The same checkpoint should be stable across calibration seeds 4801/4802/4803; output
  directories must not be shared.

### Non-repetition note

This iteration does not repeat threshold relaxation, handwritten rescue rules, ordinary
pairwise/listwise ranking, or bucket-conditioned routing. Its new contribution is the
combination of warm-start-safe set interaction, ranking/admission gradient separation,
shift-robust pseudo-environment optimization, and policy-regret checkpoint selection.

## v48.3 — OC-TRAC-NASC/RCD (2026-07-25)

### Evidence motivating this iteration

The uploaded screening run completed training and calibration diagnostics despite its
`v48_1` output-directory name. Checkpoint configs show the v48.2 SRC, encoder anchor,
exact teacher-PCD sampler and robust experts were active. Candidate AUC remained useful
(Near 0.696–0.729; Contact 0.786–0.822), but unconstrained group top-1 correlation was
near zero or negative and every calibrated policy abstained. Therefore this is not a
threshold problem: the pointwise direct head still lacks explicit candidate-set context.

### New algorithmic contribution

- **NASC (Nominal-Anchored Set Context):** a permutation-equivariant adapter compares
  every recovery candidate with the nominal embedding and exchangeable mean/max
  summaries of the recovery set. A learned conservative residual gate preserves the
  prior pointwise solution at initialization.
- **RCD (Regret-Consistent Distillation):** the composite admission policy is trained
  against the full teacher advantage distribution, not only a hard argmax, and directly
  minimizes expected teacher top-1 regret while retaining SRC harmful-mass and coverage
  constraints.
- Calibration now scores a complete scene-time candidate set in one batched call, matching
  training and closed-loop deployment semantics. Singleton APIs deliberately fall back to
  the legacy pointwise path.

### Engineering changes

- Added checkpoint/config support for the NASC adapter.
- Passed `(bucket, scene, time)` group keys and nominal masks through training.
- Updated `calibrate_policy_risk_v48.py` to batch each group via `predict_samples`.
- Enabled NASC/RCD in `train_ocrap_v48_trac_sr.sh`; existing v48.2 can be reproduced by
  setting `model.direct_recovery_set_context=false`, `POLICY_DISTILL_WEIGHT=0`, and
  `POLICY_REGRET_WEIGHT=0`.

### Required ablations

1. v48.2 SRC baseline.
2. NASC only.
3. RCD only.
4. NASC + RCD (main).
5. Remove harm/SRC constraint from NASC + RCD.

### Non-repetition note

This does not repeat prior pointwise, pairwise, listwise, top-rank, expert-routing, or
threshold-search attempts. The new element is architectural cross-candidate interaction
plus policy-level expected-regret supervision.

This root log is the canonical index for future iterations. Historical detail is retained in `ALGORITHM_CHANGELOG_V48.md` and `ALGORITHM_CHANGELOG_V48_1.md`; do not repeat an item below unless its implementation or experimental conclusion changed.

## v48.2 — OC-TRAC-SRC (2026-07-24)

### Why this iteration was necessary

An earlier static audit of the pre-fix v48.1 source found a missing `os` import and sampler-key mismatch. The uploaded screening artifacts now prove those fixes were already present in the executed job: both variants trained, the exact teacher-PCD sampler reported positive-group coverage, and v48.2 SRC settings were stored in the checkpoints. The historical engineering fixes below therefore describe prerequisites that were effective in this run, not a failure of the uploaded run itself.

### Engineering correctness fixes

- Added the missing `import os` in `src/ocrap/cli/train.py`.
- Made `group_batch_positive_advantage_{macro_ids,bucket_ids,gain_min}` the canonical sampler keys, retaining legacy aliases only for backward compatibility.
- Fixed the positive-group scanner to use the bucket stored in the exact teacher-PCD index instead of re-inferring it from the file path.
- Added `training.group_batch_require_positive_advantage_groups`; v48.2 training fails before epoch 1 when a requested positive boost resolves to zero groups.
- Added regression tests covering the canonical sampler configuration and exact-index bucket routing.
- Added atomic `calibration_build_status.json` breadcrumbs for `preflight`, `build_safe`, `build_near_contact`, `build_contact`, `merge_filter_audit`, failure, and completion.
- Added `START_STAGE=safe|near|contact|merge` to resume a failed dedicated calibration build without unnecessarily restarting completed stages; merge preflight now requires all six shard manifests.
- Replaced per-sample Safe diagnostic spam about missing targeted futures with regime-aware source requirements and one aggregate warning. Safe/nominal datasets require replay+reactive; targeted futures are required only when configured.

### New algorithm: Selective Risk-Coverage regularization (SRC)

- Added a differentiable policy distribution over the explicit nominal abstention class and all recovery candidates using the same score/opportunity/harm composition as setwise admission.
- Added a harmful-selection probability budget: probability mass assigned to teacher-harmful recovery candidates is penalized above `direct_value_selective_harm_budget`.
- Added a positive-group recovery-coverage floor so risk minimization cannot collapse to always selecting nominal.
- New controls:
  - `training.direct_value_selective_risk_weight`
  - `training.direct_value_selective_harm_budget`
  - `training.direct_value_selective_coverage_weight`
  - `training.direct_value_selective_coverage_target`
- Default v48.2 settings are risk weight 2.0, harm budget 0.05, coverage weight 1.0, and positive-group coverage target 0.65.
- The contribution is policy-level rather than another candidate classifier: it optimizes the calibrated risk/coverage trade-off under explicit abstention and complements the existing tri-state, harm-head, and robust-expert design.

### Required ablation protocol

Run both of the following from the same initialization and data split:

1. Fixed v48 baseline: `SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0`.
2. v48.2 OC-TRAC-SRC: defaults enabled.

Do not attribute gains to SRC unless both runs pass the same fit/verify Natural gate and are compared on scene-paired closed-loop evaluation.

### Local validation

- Targeted v48/v46 tests pass, including new SRC and sampler-regression tests.
- Full test-suite, compile, and shell validation status is recorded in `V48_2_VALIDATION_STATUS.txt` in the delivered package.
- Real WOMD/JAX/GPU results are not available in the local audit environment.

## v48.1 — Existing-data-first and calibration isolation

See `ALGORITHM_CHANGELOG_V48_1.md`. Key items already tried: proxy scene-disjoint calibration/dev split, dedicated validation-tail calibration construction, exact teacher-PCD coverage indexing, manifest repair, and existing-data-first screening.

## v48 — OC-TRAC-SR

See `ALGORITHM_CHANGELOG_V48.md`. Key items already tried: tri-state supervision, nominal setwise abstention, harm head, conservative two-expert aggregation, encoder fine-tuning, exact teacher-PCD alignment, joint calibration, and disabling handwritten rescue rules in the main v48 policy.
- Added `tools/inspect_calibration_build_v48.py` to classify the first incomplete stage from shard manifests and explain whether contact logs are expected.
- Added an explicit `SEED` override to the v48.2 training command so multi-seed publication experiments are reproducible rather than relying on an implicit config default.
- Added normalized L2-SP encoder anchoring during direct-only fine-tuning (`training.encoder_anchor_weight`, default 0.02). This limits drift of the shared representation away from the loaded OC-MERO/root-margin model while still allowing policy-level adaptation; without it, zero-weight root/margin losses and an unfrozen encoder could silently invalidate the pretrained core heads.
- Added an output-root `flock` guard to the dedicated calibration controller. The two commands in the supplied request are identical; launching both concurrently against the same shard/log paths can corrupt or race the build, so v48.2 rejects a second controller.
