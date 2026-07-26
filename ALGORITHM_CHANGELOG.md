# OC-RAP Algorithm Changelog

## v48.6 — OC-TRAC-RPGC (2026-07-26)

### Evidence from the completed v48.5 experiments

The completed main run, four controlled ablations, and fixed-checkpoint calibration seeds
4801/4802/4803 change the diagnosis from “ranking is uniformly broken” to a more specific
two-stage failure. The independent ECPR preference design is directionally valid: Contact
within-group top-1 correlation is positive for both variants on every calibration seed
(approximately 0.124–0.174), whereas v48.4 Contact ranking was negative on every seed.
Near remains split-sensitive (-0.092 to 0.062 balanced and -0.070 to 0.028 precision), and
no calibrated rule selects a recovery action. Candidate-positive AUC remains useful but is
not sufficient (multi-seed mean about 0.682/0.728 for Near and 0.781/0.796 for Contact).
The calibrated downside AUC is only moderate (roughly 0.54–0.62), and the closest Contact
rules still have low precision, high harmful selection, and >0.93 single-macro share.

The ablations identify the source of the ranking improvement. `C_exact_ecpr` is the only
standalone module that makes Contact top-1 correlation positive. Legacy NASC alone does not
improve ranking, and combining shared NASC with ECPR restores candidate AUC but drives
policy top-1 back toward zero or negative. Therefore v48.6 retains independent preference
learning and removes legacy shared-feature set context from the main path.

### Engineering conclusions and fixes

- Verified that the executed v48.5 configuration used `direct_value_output_mode=score` in
  both validation and calibration. A suspected raw-logit/probability mismatch was ruled out
  as the root cause; the generic validation path is nevertheless made mode-explicit.
- The v48.5 “delta distribution” subtracts two absolute value predictions and adds their
  variances as if their errors were independent. Candidate and nominal share the same scene
  encoder, so this approximation can overestimate uncertainty and collapse opportunity/harm
  gates to zero coverage.
- Calibration now enforces the macro-concentration constraint on the fit fold, not only as a
  held-out warning. Near-miss optimisation also includes macro-share deficit.
- Added explicit rank-margin abstention to calibration and runtime, plus diagnostics for
  rank-margin correctness AUC, direct-risk harm AUC, and the legacy Harm-head AUC.
- Checkpoint selection uses the worst scene-hash fold across Near and Contact rather than a
  single pooled development mean, reducing sensitivity to the proxy calibration split.

### New algorithmic contribution: Relative Preference and Gain Certificate (RPGC)

- **Preference-only relative context:** nominal-relative, recovery-mean, and recovery-max
  context augments only the independent rank residual. The absolute value representation is
  no longer rewritten by NASC. The new residual projection is zero-initialized, preserving
  the v48.5 checkpoint exactly at warm start.
- **Direct relative-gain distribution:** a dedicated head predicts
  `PCD(candidate)-PCD(nominal)` mean and log-variance from paired relative features. This
  replaces the independence approximation formed by subtracting two absolute predictions.
  The same output drives delta NLL training, checkpoint admission metrics, calibration, and
  closed-loop risk control.
- **Confidence-paced listwise preference and rank-gap calibration:** exact teacher-PCD
  supervises the complete recovery ordering and the best-vs-runner-up margin. Near-ties are
  downweighted; high-confidence positive groups receive stronger gradients. The learned
  margin becomes a deployable abstention certificate rather than an uncalibrated score.
- **Exact-opportunity macro-balanced sampling:** positive groups are reweighted by inverse
  teacher-best-macro frequency using only the training teacher index. This addresses the
  observed macro-5 shortcut without the high variance of minibatch GroupDRO or any use of
  validation/test distribution statistics.
- **Fold-robust policy checkpointing:** early stopping minimises the worst Near/Contact
  scene-hash-fold policy risk, combining positive-group exact regret, harmful switches, and
  false interventions.

### Required attribution protocol

1. `A_v485_ecpr_reference`: effective v48.5 ECPR path, no legacy NASC and no direct delta.
2. `B_preference_context_only`: A plus preference-only relative context.
3. `C_direct_delta_only`: A plus the direct candidate-vs-nominal gain distribution.
4. `D_full_rpgc`: preference context plus direct gain distribution (v48.6 main).

All runs use the same exact teacher-PCD, initialization, scene split, macro-balanced
positive sampler, and Natural-gate constraints. The main, multi-seed calibration, ablations,
and closed-loop probes must run sequentially and only immutable completed checkpoints may
be compared.

### Decision gates

- Near and Contact top-1 correlation should be positive on all three proxy seeds; initial
  target >0.10 and publication target >=0.20.
- Rank-margin correctness AUC should exceed 0.65 before margin-based coverage is trusted.
- At least one variant must produce non-zero held-out selections in both regimes with
  precision LCB90 >=0.40 during development, then >=0.60 for paper readiness.
- Positive recall should reach >=0.35, harmful-selection UCB90 <=0.10, and selected macro
  share <=0.85 before development closed loop.
- Safe, Near, and Contact paper targets remain closed-loop requirements; zero-action
  abstention is safe but supplies no evidence of recovery benefit.

### Non-repetition note

This iteration does not repeat shared NASC, threshold relaxation, Harm-head-driven ranking,
minibatch GroupDRO, or another generic pairwise loss. It deepens the experimentally supported
independent preference idea and makes relative gain, uncertainty, confidence, and macro
coverage separately identifiable.

## v48.5 — OC-TRAC-ECPR (2026-07-25)

### Evidence motivating this iteration

The re-uploaded v48.4 artifacts confirm a persistent candidate-to-policy gap. Across proxy calibration seeds 4801/4802/4803, candidate-positive AUC remains useful (Near roughly 0.750–0.791; Contact 0.732–0.880), but Contact group top-1 correlation is negative for every seed (-0.094 to -0.072), Near top-1 is unstable (-0.044 to 0.115), Harm AUC remains near random (about 0.512–0.561), and every verified policy selects zero actions. The completed A/SRC-reference ablation also selects zero actions. The uploaded archive contains seven complete main-training epochs and an interrupted eighth epoch; only ablation A is complete, B is partial, and C/D are absent, so v48.4 component attribution is limited to supported evidence rather than inferred from unfinished runs.

### Engineering isolation fixes

- Unified training targets, validation checkpoint metrics, and calibration on the same **exact teacher-PCD shared-option contract**. v48.4 trained/checkpointed against a differentiable soft shared-success approximation but calibrated against exact best-shared-option PCD; this objective mismatch could reverse within-group order.
- Added an independent zero-initialized preference head. Warm start exactly preserves the inherited value ranking while allowing policy ordering to specialize without changing the calibrated gain scale.
- Calibration and runtime now use preference logits for recovery-candidate top-1 and value mean/std for candidate-vs-nominal admission. Deployment therefore matches training and calibration semantics.
- Added output-root and GPU locks, per-variant and aggregate training-completion markers, immutable checkpoint SHA256 manifests, strict multi-seed source checks, and a completion auditor. Multi-seed calibration can no longer consume an actively changing checkpoint.
- Fixed the candidate-selection result key from `harmful_rate_selected` to `harmful_selected_rate`.
- Added a calibration near-miss frontier so zero-selection failures report which statistical constraint is closest to passing.

### New algorithmic contribution: Exact-Contract Preference Recovery (ECPR)

- **Exact Policy Contract:** the exact OC-MERO q table chooses one globally shared option; hard success is evaluated on teacher margins and converted to PCD identically in training, validation, calibration, and deployment diagnostics.
- **Independent Preference Ranking:** a dedicated set-aware rank residual learns which recovery candidate is best; the value distribution is reserved for whether that candidate should challenge nominal. Candidate AUC and policy top-1 are no longer forced through one scalar.
- **Confidence-paced best-vs-rest preference:** only exact-teacher ordering gaps above a minimum are supervised strongly. Near-ties are downweighted, reducing sensitivity to train/validation contract drift and ambiguous teacher order.
- **Expected preference regret:** the rank distribution is penalized by exact teacher advantage regret on positive-opportunity groups.
- **Distributional candidate-minus-nominal gain:** value mean/log-variance model the gain delta. Opportunity and downside probabilities are derived analytically from the delta distribution, replacing the non-transferable Harm head as the main deployment risk source. Harm/opportunity heads remain optional auxiliary diagnostics.
- **Risk-focused checkpoint selection:** early stopping minimizes the worse-regime sum of positive-group top-1 regret, harmful selected-candidate rate, and false-intervention rate. Always-nominal behavior cannot hide incorrect ranking.
- Pseudo-environment minibatch GroupDRO is disabled by default because v48.4 did not complete the required attribution and sparse minibatch domains can amplify noise.

### Required attribution protocol

1. `A_exact_pointwise`: exact teacher-PCD, pointwise value only.
2. `B_exact_zi_nasc`: A plus zero-initialized set context.
3. `C_exact_ecpr`: A plus independent preference head and confidence-paced preference regret.
4. `D_full_ecpr`: C plus set context and distributional delta NLL.

All four use the same scene split, initialization, exact teacher target, and distributional calibration. Run them sequentially and require `completion_audit.json` to report `comparable=true` before attribution.

### Decision gates

- Development target: positive-group top-1 correlation/accuracy and regret must improve before Natural-gate thresholds are changed.
- Screening target: Near and Contact top-1 correlation >0.10 initially; publication target >=0.20, verify precision LCB90 >=0.60, positive recall >=0.35, and harmful-selection UCB90 <=0.10.
- Only a frozen checkpoint stable across calibration seeds 4801/4802/4803 may enter closed loop.
- Safe remains strict nominal non-inferiority; Near and Contact must pass the existing regime-specific closed-loop gates before test evaluation.

### Non-repetition note

This iteration does not repeat threshold relaxation, handwritten rescue certificates, bucket-conditioned routing, ordinary candidate classification, or the unverified minibatch GroupDRO setting. It changes the supervised decision object and makes ranking, admission, and risk estimation separately identifiable.

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
