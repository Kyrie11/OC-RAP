## v48.11 — OC-TRAC-CASTER (2026-07-28)

### Evidence from the completed v48.10 COPE experiment

- The main v48.10 run and all eight causal ablation tasks completed. Neither variant passed the joint Near+Contact Natural gate, so no stress closed-loop result is attributable to COPE.
- Monotone ordinal evidence was the only consistently useful module. Relative to the A reference, the C evidence ablation raised Contact candidate-positive AUC from 0.724/0.670 to 0.806/0.826 for balanced/precision and improved Contact harm AUC to 0.583/0.597. The full model reached Contact candidate AUC 0.834/0.814 and policy-top1 benefit AUC 0.808/0.768.
- Conditional Option Preference did not solve groupwise ranking. Main Near/Contact top-1 correlation stayed around zero (-0.003/0.012 balanced, 0.013/0.001 precision). The conditional objective also increased non-positive false switches to 0.37-0.53 and harmful ranked switches to 0.14-0.19.
- Precision Contact was the only main result with non-zero verify coverage: 13/193 groups, precision 0.308 (LCB90 0.146), harmful rate 0.462 (UCB90 0.675), recall 0.121, mean exact-teacher advantage -0.180, and macro concentration 0.846. It is not deployable.
- The inherited candidate value remained inside the preference score as `rank_base + residual`. The trained residual changed the chosen recovery in only a small fraction of groups, so the high-AUC but groupwise-wrong candidate value continued to dominate option ordering.

### Engineering correction: policy-first, no-fallback semantics

Stage-E evidence was trained on the frozen preference top-1 candidate, but calibration and runtime first filtered candidates by evidence and then selected the highest-ranked survivor. A failed top-1 could therefore fall through to a runner-up that Stage E never trained on. v48.11 makes the policy contract explicit and identical everywhere:

1. rank physically supported recovery candidates;
2. choose the preference top-1 candidate;
3. evaluate rank margin and evidence only for that candidate;
4. abstain if it is uncertified; never fall through to rank 2.

The calibration JSON now records `direct_value_policy_first_no_fallback`, and the runtime selector exposes the same option.

### New algorithmic contribution: CASTER

**CASTER = Conditional Attention Set Tournament with Evidence Routing.**

1. **Recovery-only set tournament**
   - Replaces, rather than residualizes, the inherited candidate-level value ranking.
   - Uses a small permutation-equivariant self-attention tournament over nominal-relative recovery tokens.
   - Nominal is pinned to zero and excluded from the tournament; admission is isolated in Stage E.
   - Group scores are centered to remove an unidentifiable common offset.

2. **Policy-conditioned regime evidence**
   - Stage E freezes the complete set tournament.
   - Separate Near and Contact evidence experts consume nominal-relative candidate features plus the frozen policy score and top1-vs-runner-up gap.
   - The evidence model therefore learns the distribution actually encountered by each regime rather than averaging incompatible harm/dead boundaries.

3. **Proper ordered three-state likelihood**
   - The ordered logits induce a valid simplex: harmful, dead-zone, beneficial.
   - A class-weighted three-class NLL replaces two independent focal BCE losses.
   - Harmful examples receive the largest weight because harm-vs-dead separation is the current certification bottleneck.

4. **Strict attribution and speed**
   - Added immutable staged architecture/checkpoint contracts.
   - Added dynamic ablation summarization and a calibration-only policy-semantics ablation.
   - Four ablations are queued together with at most one model per A30; the policy-first ablation reuses the reference checkpoint and avoids duplicate training.

### Required validation order

1. Stage T: Near and Contact recovery-only top-1 correlation should both exceed 0.10 before evidence results are interpreted.
2. Stage E: policy-top1 benefit AUC should reach at least 0.70 Near / 0.75 Contact and harm AUC at least 0.60 in both regimes.
3. Natural gate must produce non-zero verify coverage with positive mean exact-teacher advantage and unchanged confidence bounds.
4. Run seeds 4801/4802/4803 only after stages 1-2 pass.
5. Run stress closed loop only when the controller creates `NEXT_COMMANDS.txt`.

### Non-repetition note

CASTER does not repeat threshold relaxation, another residual MLP on top of the candidate value, independent harm BCE, or evidence-first runner-up fallback. Its novelty is the combination of recovery-only set competition, policy-conditioned regime evidence, ordered three-state certification, and one consistent no-fallback policy contract.

## v48.10 — OC-TRAC-COPE (2026-07-27)

### Evidence from the completed v48.9 PACER experiment

- The main v48.9 run and fixed-checkpoint calibration seeds 4801/4802/4803 completed. Neither variant passed the scene-disjoint Near+Contact Natural gate, so the controller correctly did not create `NEXT_COMMANDS.txt`; no Near/Contact stress closed-loop result is attributable to PACER.
- Candidate evidence remained detectable, but policy quality did not improve enough. Main policy-top1 benefit AUC was about 0.67/0.74 for balanced Near/Contact and 0.67/0.73 for precision, while policy-top1 harm AUC was only about 0.49–0.57. Group top-1 correlation remained near zero in Near and slightly negative in Contact.
- The closest Near fit rules could reach nominal precision around 0.75 on only eight selected groups, but verify precision fell to roughly 0.38–0.50, recall to about 0.08–0.12, and selected actions were concentrated in macro 5. Contact fit-to-verify transfer was worse and could become predominantly harmful with negative mean teacher advantage.
- Policy-top1 conformal calibration did not solve certification. One-sided overprediction quantiles remained approximately 0.60–0.62, forcing zero verified coverage. Exact teacher advantage is strongly tri-modal (`harmful/dead-zone/beneficial`, with many exact zeros and boundary masses), whereas the continuous delta regressor collapsed near zero.
- The uploaded ablation suite was incomplete: only balanced A/B/C artifacts were available. More importantly, Stage C for A/B instantiated a 128-wide preference context while Stage P used width 32, causing preference-adapter checkpoint shape mismatches and discarding learned Stage-P context. Final A/B attribution is therefore invalid even though the main run did not contain this mismatch.

### What v48.9 established

- **Intervention-aware preference is useful for suppression but not sufficient for ranking.** Relative to the old nominal-inclusive objective, the Stage-P audit reduced non-positive false switches from about 0.65–0.71 to 0.12–0.15 and harmful ranked switches from about 0.21–0.23 to 0.05–0.07. However, Contact conditional recovery ordering regressed to near zero/negative correlation.
- **Policy-aligned certificate sampling is directionally useful.** Compared with all-candidate training, the available balanced ablation improved policy-top1 benefit AUC, especially in Contact, but harm discrimination remained near random and no rule passed verification.
- **Conformal calibration is not a substitute for a discriminative evidence model.** Correcting the conformal sampling scope cannot rescue a regressor whose residual scale is comparable to the full teacher-advantage range.

### Engineering corrections

1. Added `training.strict_init_prefixes`. Stage E now aborts unless the complete Stage-P preference adapter loads with exactly matching geometry; silent loss of learned preference context is forbidden.
2. The staged architecture writes `STAGE_ARCHITECTURE.json`, and completion markers include immutable preference/evidence checkpoint hashes.
3. The v48.10 ablation controller propagates the same preference width to both stages, creates one immutable `TASK_COMPLETE.json` per task, resumes completed tasks, and refuses to write the suite summary until all eight `(4 ablations × 2 variants)` tasks exist.
4. Calibration, checkpoint metrics, offline evaluation, selector semantics, and closed-loop execution now share the same conditional-recovery ranking and ordinal-evidence interpretation.

### New algorithmic contribution: COPE

**COPE = Conditional Option Preference with Monotone Ordinal Evidence.** It separates the two logically different questions that PACER still mixed inside the preference target and continuous certificate.

1. **Conditional Option Preference (COP)**
   - Stage P ranks recovery options only; nominal is excluded from the option-ordering loss and from the conditional rank margin.
   - Exact teacher-PCD defines an ambiguity-aware acceptable recovery set. The loss maximizes mass on that set and minimizes exact expected recovery regret.
   - Positive-opportunity groups receive full weight. No-opportunity and harmful groups receive a lower weight and teach only the least-bad recovery ordering; whether any recovery should be executed is deferred entirely to Stage E.
   - This preserves the experimentally useful nominal-relative low-capacity context while preventing nominal-suppression gradients from destroying Contact recovery ordering.

2. **Monotone Ordinal Evidence (MOE)**
   - Stage E freezes the complete preference policy and models the frozen policy-top1 candidate as one of three ordered states: beneficial, dead-zone, or harmful relative to nominal.
   - Two ordered cumulative logits parameterize `P(beneficial)` and `P(non-harm)`, with the architecture enforcing `P(beneficial) <= P(non-harm)` and `P(harm)=1-P(non-harm)`.
   - Focal ordinal supervision is concentrated on policy-top1 candidates, with only a weak all-candidate regularizer. This matches the deployment distribution and the tri-modal exact teacher target without regressing advantages toward zero.
   - Admission uses opportunity probability, harm probability, evidence score, conditional recovery rank margin, support, recall, and macro concentration under the unchanged fit/verify Natural gate. Thresholds are not relaxed.

### Required causal ablations

1. A: v48.9-style nominal-inclusive preference + continuous delta evidence.
2. B: conditional option preference + continuous delta evidence.
3. C: nominal-inclusive preference + monotone ordinal evidence.
4. D: full COPE.

The first attribution question is whether B improves conditional recovery top-1 and regret without relying on nominal switching. The second is whether C/D improve policy-top1 benefit and harm AUC and create transferable non-zero verification coverage. Multi-seed and stress closed loop remain forbidden until the fixed checkpoint passes the diagnostic learning gates and unchanged Natural gate.

### Local validation

- 144 tests passed.
- Python compileall passed.
- The main controller and all modified v48.10 shell scripts passed `bash -n`.
- Real WOMD/Waymax/A30 training and closed-loop evaluation are not available locally; COPE is an experimentally testable design, not a claim that the publication thresholds have already been reached.

## v48.9 — OC-TRAC-PACER (2026-07-27)

### Evidence from the completed v48.8 experiment

- The v48.8 main run, eight ablation jobs, and the paired Safe probe were audited. Natural gate failed for both variants and both stress regimes; every calibrated rule selected zero actions, so no Near/Contact closed-loop improvement can be attributed to SCOPE.
- Candidate-level signal remained usable (main candidate-positive AUC: Near 0.643–0.730, Contact 0.765–0.785), but policy ordering remained insufficient. Main top-1 correlation was 0.053/0.125 for balanced Near/Contact and 0.163/0.185 for precision Near/Contact, below the internal 0.20 readiness target.
- The conflict-free preference ablation did not improve top-1 over the engineering-fixed reference. Near/Contact correlation changed from 0.022/0.079 to 0.006/0.058 for balanced and from 0.048/0.102 to 0.006/0.058 for precision.
- The split-conformal certificate was over-conservative for the wrong reason: residuals were fitted over every recovery candidate, although deployment evaluates only the frozen preference top-1 candidate. One-sided overprediction quantiles were about 0.57–0.61, causing all scored rows to saturate at opportunity=0 and harm=1; no strict threshold grid or near-miss frontier existed.
- The paired Safe probe used only eight scenes. Collision/offroad, bounded NUP, and intervention were identical to nominal, but route progression, jerk p95, and yaw-rate p95 were unavailable. It is diagnostic only and not a paper-ready Safe claim.

### Root-cause corrections

1. **Partial-label set mass rather than uniform-set KL.** The v48.8 target forced all acceptable candidates to equal logits. PACER minimizes negative probability mass on the acceptable set, preserving ambiguity without inventing an ordering inside the set.
2. **Nominal-only target for no-opportunity groups.** Dead-zone recoveries are no longer treated as equally acceptable deployment actions. They receive a weak intervention-cost margin below nominal; materially harmful recoveries receive a stronger margin.
3. **Policy-induced certificate training.** Stage C now trains the relative-gain head strongly on the recovery candidate actually selected by the frozen Stage-P preference policy, while retaining a low-weight all-candidate regularizer. This removes the train/deploy distribution mismatch.
4. **Policy-top1 conformal scope.** Optional conformal calibration fits residuals on one frozen-policy candidate per group, not all unused candidates. The default main experiment uses empirical direct-delta admission; conformal remains a controlled ablation until it demonstrates non-zero verified coverage.
5. **Non-empty failure diagnostics.** Calibration now writes a diagnostic frontier even when all predicted opportunity/harm values violate hard bounds, while explicitly marking probability-bound deficits so diagnostic rows cannot pass the Natural gate.

### New algorithmic contribution: PACER

**PACER = Policy-Aligned Candidate Evidence for Recovery.** It couples two isolated stages through the policy-induced candidate distribution rather than through shared gradients.

- **Intervention-aware partial-label preference:** Stage P uses only nominal-relative set context. Positive groups maximize mass on the exact-teacher equivalent recovery set; no-opportunity groups choose nominal; dead-zone and harmful alternatives are separated by different margins.
- **Policy-aligned evidence:** Stage C freezes the whole preference path and learns exact candidate-minus-nominal PCD gain on Stage P's selected candidate, with smooth-L1 and tri-state sign supervision. The all-candidate loss is only a weak representation regularizer.
- **Auditable abstention:** empirical fit/verify precision, conditional harmful-switch bounds, rank margin, recall, support, and macro concentration remain unchanged. PACER does not lower gate thresholds to manufacture coverage.

### Required validation and ablations

1. Old uniform-set preference + old all-candidate certificate.
2. Intervention-aware set-mass preference only.
3. Policy-aligned certificate only.
4. Full PACER.

The first learning checkpoint is whether Near and Contact top-1 correlation improve without increasing non-positive false switches. The second is whether policy-top1 gain AUC and near-miss precision improve. Multi-seed and stress closed loop remain forbidden until a fixed checkpoint passes the held-out Natural gate.

### Engineering and speed notes

- Main and ablation runs may reuse the v48.8 proxy split and exact teacher-PCD index; no dataset rebuild is required.
- Four ablations are submitted together but the scheduler runs at most one job per A30, so two jobs execute concurrently without GPU oversubscription.
- The Stage-P adapter width is reduced from 48 to 32, Stage C uses only the delta adapter, BF16/TF32 and persistent data workers remain enabled, and calibration uses group-batched inference.

### Local validation

- 140 tests passed.
- Python compileall passed.
- Modified shell scripts passed `bash -n`.
- Real WOMD/Waymax/A30 results are not available in the local environment; no claim is made that v48.9 already passes Natural gate or closed-loop publication targets.

## v48.8 — OC-TRAC-SCOPE (2026-07-27)

### Evidence from the completed v48.7 proxy experiment

- Stage P did not reliably learn policy top-1. Candidate-level rank correlation was positive (about 0.12–0.16), but unconstrained group top-1 correlation remained slightly negative in both Near and Contact. Acceptable-set accuracy was only about 0.53–0.64 and positive-group regret remained 0.12–0.19.
- Stage C did not learn deployable execution evidence. Candidate-positive AUC was 0.66–0.77 and risk-harm AUC only 0.55–0.61. Every Natural-gate rule selected zero actions. The closest rules had low precision, high conditional harmful-switch UCB, negative or near-zero mean teacher advantage, and 0.85–1.00 maximum macro share.
- Ablations isolate two partial ideas: staged training improves Contact relative to joint single-winner training, and set-valued supervision improves both regimes under joint training. Their v48.7 combination regressed because the loss simultaneously treated near-tied candidates as equivalent and as ordered best-vs-rest competitors, while checkpoint selection was dominated by a sparse fold.
- The Safe probe used only eight nominal-locked scenes. It confirms zero intervention and NUP=1 for that probe, but does not establish paired collision/offroad non-inferiority, confidence intervals, route progression, jerk/yaw-rate, or the publication Safe target.

### Engineering corrections required for clean attribution

- Checkpoint improvement is now strict with `training.best_metric_min_delta`; equal validation metrics no longer overwrite the earlier best checkpoint.
- Fold selection is support-aware. Preference/certificate checkpointing ignores folds below a configurable positive-group floor and uses the mean of the worst supported K folds instead of a noisy single-fold maximum.
- Preference risk now includes harmful top-1 and non-positive nominal-switch penalties, rather than evaluating only positive-group regret.
- Training summaries retain the exact `trainable_param_prefixes` and metric tolerance.
- Calibration always writes unconstrained top-1 diagnostic rows even when no rule passes and evaluates each near-miss fit rule on the held-out verify fold.
- Proxy splits and exact teacher-PCD indexes can be prepared once and reused by all ablations. The controller supports one-variant jobs, shared assets, and a two-GPU queue without oversubscribing either A30.
- Safe evaluation can now run a scene-paired scalar/nominal reference and the nominal-locked model on the two A30s, then report paired bootstrap non-inferiority intervals. A duplicate-loop syntax error in the legacy summary block was also fixed; route/jerk/yaw metrics are marked unavailable rather than silently proxied.

### New algorithm: SCOPE

**SCOPE = Support-aware Conflict-free Ordinal Preference with Conformal Evidence.**

1. **Conflict-free nominal-inclusive set preference**
   - Every scene-time group is supervised, not only groups with a positive recovery opportunity.
   - Material-positive groups target a teacher-equivalent recovery set; no-opportunity groups target nominal plus only dead-zone alternatives; harmful recoveries are explicitly pushed below nominal.
   - When enabled, this objective replaces the contradictory single-winner/listwise family instead of being added on top of it.

2. **Invariant low-capacity preference context**
   - The trainable Stage-P adapter receives only candidate-minus-nominal, recovery-mean, and recovery-max relative blocks. Absolute candidate features are excluded to reduce severity/macro shortcuts under train/dev contract drift.
   - Only the context residual is trained; inherited pointwise preference remains frozen. Hidden width is reduced to 48 and the residual remains zero-initialized.

3. **Robust relative-gain learning**
   - Stage C trains only the relative delta adapter with smooth-L1 regression and soft positive/harm sign supervision. Heteroscedastic NLL is disabled, preventing the severe train-negative/validation-positive variance-collapse pattern observed in v48.7.
   - The delta log-variance remains at a fixed conservative initializer and is not treated as learned epistemic confidence.

4. **Split-conformal execution evidence**
   - On the calibration fit scenes, one-sided finite-sample residual quantiles form a lower confidence bound for candidate-minus-nominal gain. Rule search and held-out verification use this bound.
   - Selector, offline evaluator, and closed-loop runner consume the same conformal quantile and score semantics if a rule passes.

### Stepwise validation protocol

- Stage-P audit must first show positive Near/Contact top-1 correlation and acceptable-set accuracy before certificate results are interpreted.
- Stage-C discrimination then checks positive/harm AUC and regret independently of Natural-gate coverage.
- Only after both pass is fixed-checkpoint multi-seed calibration run; stress closed loop remains forbidden unless the held-out Natural gate is valid.
- Four controlled groups are required: engineering-fixed v48.7 reference, conflict-free preference only, robust/conformal certificate only, and full SCOPE. The queue runs at most two jobs concurrently on two A30 GPUs.

### Non-repetition note

SCOPE does not repeat stronger Harm-head weighting, joint preference/certificate gradients, minibatch GroupDRO, threshold relaxation, shared NASC, or another additive single-winner ranking loss. It specifically removes contradictory supervision, absolute-feature shortcuts, learned-variance collapse, and sparse-fold checkpoint noise exposed by the completed v48.7 experiment.

## v48.7 — OC-TRAC-SPIRE (2026-07-26)

### Evidence from the completed v48.6 experiment

- All v48.6 training, three-seed recalibration, and four core ablations were audited.
- Candidate recovery signal remained usable (three-seed mean AUC: Near 0.702–0.706, Contact 0.818–0.824), but the policy layer regressed: Near top-1 correlation was -0.039 to -0.002 and Contact was -0.087 to -0.054; every verify fold selected zero actions.
- The only positive ablation was preference-only relative context: it improved rank correlation and made balanced Contact top-1 slightly positive on seed 4801. Direct-delta-only and the full joint RPGC objective worsened Contact ordering, demonstrating negative transfer from certificate learning into the shared preference representation.
- Rank-margin correctness AUC was informative only for Contact (about 0.63–0.65) and weak for Near (about 0.42–0.43). The direct-delta/harm channel remained insufficiently transferable (risk-harm AUC about 0.55–0.58).

### Engineering corrections before further algorithm attribution

- Validation checkpointing now computes opportunity/harm from the same Gaussian direct-delta CDF used by calibration; the v48.6 raw-delta threshold was not deployment-equivalent.
- Harmful *population exposure* UCB and conditional harmful-switch UCB among selected actions are now reported and constrained separately. A low exposure UCB caused by zero/rare selections can no longer be mistaken for a low harmful-switch rate.
- `run_ocrap_v48_trac_sr.sh` no longer silently falls back to the obsolete `runs/ocrap_v48_trac_sr_regime_balanced` path. `BASE_RUN` or explicit checkpoint/calibration paths are mandatory.
- Natural-gate failure writes `GATE_FAILED.json` and exits before producing closed-loop commands. It does not imply that the trained candidate checkpoints are missing.
- Added a Safe-only nominal-locked probe that does not require Near/Contact certificates and cannot authorize stress-regime intervention.
- Added partial dedicated-calibration merge support so completed Safe/Near worker pairs can be filtered and atomically installed under the evaluation root before Contact is finished.
- Teacher-PCD data-quality reports now distinguish all-macro opportunities from deployable-macro opportunities; quality gates use the actual selector allowlist.
- Added parameter allow-list freezing for auditable staged optimization.

### New algorithm: SPIRE

**SPIRE = Set-valued Preference with Isolated Relative-gain Evidence.** It explicitly separates the three policy objects that v48.6 failed to identify jointly.

1. **Preference stage — who should be selected?**
   - The encoder and value surface are frozen. Only the pointwise and nominal-relative preference residuals are trainable.
   - Exact teacher-PCD supervision is ambiguity-aware: Near and Contact use regime-specific acceptable sets around the teacher optimum rather than forcing arbitrary single winners in near-tied groups.
   - The loss combines acceptable-set KL, set-versus-nominal/worse-candidate margin, confidence-paced best-vs-rest preference, exact expected regret, and a small rank-gap term.
   - Early stopping uses worst-fold tie-aware preference risk, not candidate AUC or total loss.

2. **Certificate stage — should the selected recovery be executed?**
   - The complete preference path, encoder, and value heads are frozen. Only the direct candidate-minus-nominal delta adapter is trained.
   - This removes the v48.6 negative transfer in which delta-NLL gradients degraded Contact ordering.
   - Early stopping uses a fixed deployment-aligned certificate risk based on direct-delta opportunity probability, harm probability, rank margin, harmful admitted actions, false interventions, and missed positive opportunities. Always-abstain therefore no longer receives a deceptively good checkpoint score.

3. **Evidence and gate semantics**
   - Calibration keeps exact preference top-1 and direct-delta admission separate.
   - It reports both strict single-winner accuracy and acceptable-set accuracy/tie-aware regret.
   - Proxy calibration may use development conditional-harm UCB limits; paper promotion requires the larger dedicated set and substantially tighter conditional harmful-switch bounds.

### Required ablations

1. Joint single-winner v48.6 objective.
2. Staged optimization with the old single-winner preference target.
3. Joint optimization with set-valued preference.
4. Full SPIRE: staged optimization plus set-valued preference.

This design isolates whether gains come from ambiguity-aware supervision, gradient isolation, or their combination.

### Closed-loop promotion requirements

- Stress closed loop remains forbidden unless both Near and Contact certificates pass.
- First screening target: all three seeds positive top-1 correlation, mean >=0.10, non-zero verify selections, and no uncontrolled conditional harmful-switch UCB.
- Paper-readiness remains stricter: top-1 correlation >=0.20, precision LCB90 >=0.60, positive recall >=0.35, conditional harmful-switch rate/UCB approaching the 5–10% target on a sufficiently large dedicated calibration set, and scene-paired closed-loop improvements without Safe degradation.

### Non-repetition note

SPIRE does not repeat joint value/rank/delta optimization, stronger Harm-head weighting, ordinary single-winner listwise training, GroupDRO, threshold relaxation, or handwritten rescue rules. The new contribution is ambiguity-aware set preference plus stage-isolated evidence certification under a deployment-aligned checkpoint and Natural gate.

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
