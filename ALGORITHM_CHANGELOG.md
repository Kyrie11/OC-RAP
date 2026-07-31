## v48.23 — OC-TRAC-FRONTIER-BRIDGE (2026-07-31)

### v48.22 result attribution

- The uploaded v48.22 dedicated controller completed both Balanced and Precision
  adaptations, non-empty scene-disjoint certificate fitting/verification, protocol
  and teacher-index contract checks, and the test-root seal. It records
  `pipeline_valid=true`, `gate_evaluated=true`, `test_roots_read=false`, and
  `RC=20`. This is a genuine Natural-gate rejection rather than a controller,
  protocol, empty-pool, parameter-count, unsupported-support, or partial-variant
  failure.
- The preregistered gate remains mathematically feasible, but it is intentionally
  close to an oracle-quality selective policy at the current support. Near-fit must
  select at least 10 groups with at least 8 positives and zero harmful selections;
  Near-verify needs at least 5/8 positives and zero harmful selections. Contact-fit
  needs at least 11/16 positives and at most one harmful selection; Contact-verify
  needs at least 6/10 positives and zero harmful selections. These conditions must
  not be relaxed retrospectively in the v48.22 protocol.
- v48.22 does not fail only because the gate is strict. Precision learns broad
  component risk (candidate harm AUC about 0.64--0.66), but Near/Contact safe-action
  precision remains about 0.05--0.10 at the closest fit rules. Balanced selects the
  epoch-zero identity, leaves harm at 0.5, and effectively abstains. Every main and
  ablation verify certificate has zero coverage.
- The frozen proposal remains high recall: positive-group oracle-best hit is about
  0.97--1.00. Candidate generation is not the primary bottleneck. The unresolved
  problem is converting a proposal-contained opportunity into a calibrated,
  high-benefit, non-harmful action and deciding when to leave nominal.
- Training support is extremely imbalanced. Near has 25 safe-beneficial candidates
  among 1425 deployable candidates (1.75%), across 11 groups and 7 scenes; Contact
  has 106/4086 (2.59%), across 41 groups and 17 scenes. Global harm prevalence is
  approximately 54% in Near and 45% in Contact. Broad candidate AUC is therefore
  insufficient evidence of safety-frontier transfer.

### Root engineering and objective defects found in v48.22

1. **Neutral risk was encoded as 0.5 harmful probability.** Zero-initialized
   component logits produced `P(harm)=0.5`, even though the target contract defines
   a tolerance/deadband within which a candidate is non-harmful. Balanced early
   stopping therefore preferred an artificial all-abstain identity rather than a
   semantically neutral source policy.
2. **The detached admission prior had a constant negative offset.** v48.22 used
   `benefit_logit - softplus(harm_logit)`. At the zero residual identity this
   subtracts `log(2)`, so the new admission head is not identity-preserving and is
   pushed toward abstention before any target-domain evidence is learned.
3. **The two-head fallback was also structurally over-conservative.** Its score
   `P(benefit)*(1-P(harm))-0.5` is non-negative at neutral harm only when benefit is
   nearly certain. The A ablation therefore cannot isolate the third head fairly.
4. **Noisy-OR does not match the one-action deployment event.** The group objective
   treated top-k candidates as independent Bernoulli opportunities, while runtime
   chooses exactly one candidate or nominal. Noisy-OR inflates opportunity as top-k
   grows and can be satisfied by diffuse weak scores rather than one executable
   safe action.
5. **Benefit supervision was mostly a binary sign test.** Continuous PCD magnitude
   ordering was disabled (`CENTERED/DELTA_NLL/pairwise benefit` effectively zero or
   weak), explaining high Near candidate AUC but weak or negative score correlation,
   unstable Contact ranking, and positive-group top-1 regret.
6. **Checkpoint selection emphasized broad risk instead of the high-benefit safety
   frontier.** Global harm AUC/mass is dominated by abundant dead or obviously
   harmful candidates. The deployment-critical distinction is safe high-benefit
   versus harmful high-benefit recovery.
7. **Primary-gate-only debugging is information-poor.** A zero-coverage certificate
   cannot identify whether failure comes from proposal support, labels/features,
   ranking, admission, or overly conservative finite-sample certification. A
   proposal-constrained oracle audit and adaptation-dev-only shadow closed loop are
   required, without reading held-out test/stress roots.
8. **Contact event aggregation was incorrect.** `secondary_overlap_event` was
   aggregated by maximum across scenes, making any single event report as 1.0.
   v48.23 reports scene rates for secondary contact, stable stop and sustained
   escape. A duplicate scene-quantile computation was also removed.

### v48.23 algorithm: FRONTIER-BRIDGE

**FRONTIER = Factorized Recovery Opportunity with Non-compensatory Threat Evidence,
Rank-consistent Transfer, and Intervention Evaluation.** It remains one unified
model over Safe, Near and Contact. No regime ID, bucket router, bucket-selected
calibrator, or regime-specific residual is available at inference.

1. **Semantic non-harm prior.** Component risk logits start from a configurable
   low-risk prior (`-2.0` by default) and learn bounded residuals around that prior.
   The prior represents the component-veto deadband rather than an arbitrary 0.5
   harmful probability. Exact non-compensatory `max` aggregation is retained.
2. **Centered identity-preserving admission.** The detached prior is
   `benefit - [softplus(harm)-softplus(harm_prior)]`. At zero residual the admission
   logit exactly equals the transferred benefit logit. Risk can subsequently veto
   an action without imposing a fixed pre-training abstention penalty.
3. **Categorical one-action group policy.** The primary group objective is a softmax
   over nominal plus frozen proposal top-k, matching the actual decision that one
   action (or nominal) is executed. It replaces noisy-OR in FRONTIER runs while the
   legacy path remains checkpoint-compatible.
4. **Continuous top-k benefit ranking.** A vectorized listwise/KL objective uses the
   continuous raw PCD advantages inside the exact deployment top-k. It teaches which
   beneficial action is better, not only whether its signed delta exceeds a
   threshold. Candidates outside frozen top-k receive no listwise gradient.
5. **High-benefit safety-frontier contrast.** Admission logits for safe beneficial
   candidates are trained to outrank raw-beneficial but component-harmful candidates
   in the same group. This directly targets the false-safe frontier that broad harm
   AUC misses.
6. **FRONTIER checkpoint risk.** Early stopping adds high-opportunity harmful policy
   mass and false-admission mass to the threshold-free cross-regime risk, with a
   small global-harm tie-break only. Near/Contact remain validation strata and are
   never model inputs.
7. **Proposal-constrained oracle gate audit.** Certificate artifacts now report the
   most optimistic fit/verify feasibility achievable using non-harmful opportunities
   already contained in the frozen proposal. This audit ignores macro-concentration
   constraints and is therefore necessary but not sufficient. Oracle failure means
   more training cannot pass the current proposal/label/gate contract; oracle pass
   with model failure localizes the bottleneck to representation/ranking/admission.
8. **Adaptation-dev shadow closed loop.** After `RC=20`, a separate script runs only
   on adaptation-dev roots, never certificate/test/stress roots. It is explicitly
   diagnostic and non-paper. Held-out stress remains authorization-gated by
   `NEXT_COMMANDS.txt`.
9. **Physical regime diagnostics.** Near adds minimum clearance/TTC, near-contact and
   critical-TTC exposure durations, and clearance/TTC deficit integrals. Contact
   adds overlap duration and longest run, secondary contact rate, post-contact
   clearance/free-space integral, sustained escape rate/time, and stable-stop rate.
   The paired comparator reports metric-aware improvement direction.

### Non-repeated v48.23 ablations

1. `A_semantic_prior_categorical`: semantic risk prior, centered admission identity
   and categorical one-action group objective; no continuous ranking or frontier
   contrast. This isolates the v48.22 engineering corrections.
2. `B_add_benefit_listwise`: A plus continuous top-k PCD listwise supervision. This
   isolates benefit magnitude/ranking transfer.
3. `C_add_frontier_contrast`: A plus high-benefit safe-versus-harmful admission
   contrast. This isolates safety-frontier discrimination.
4. `D_full_frontier`: semantic prior, categorical group policy, continuous benefit
   ranking, component veto and frontier contrast.

All eight tasks are launched simultaneously. Round-robin assignment places four
jobs on GPU0 and four jobs on GPU1. Each job uses one DataLoader worker, batch size
56 and bounded host math threads. Main Balanced/Precision use separate A30s, batch
size 96, three workers, pinned persistent workers, prefetching and bfloat16 AMP.
The new losses are vectorized inside the existing model forward pass and do not add
an additional encoder or duplicate proposal computation.

### Near/Contact development targets and diagnostic policy

- **Near-contact:** preserve zero collision/non-inferior nominal safety; improve
  minimum clearance by at least 0.10 m and minimum TTC by at least 0.20 s in paired
  development analysis; reduce near-contact/critical-TTC exposure and safety-margin
  deficit integrals; target PCD >= 0.54, FRA <= 0.12, DRS >= 0.88, NUP >= 0.995,
  intervention <= 0.02, intervention-episode rate <= 0.012, maximum intervention run
  <= 1, and selector miss <= 0.034 for development (<= 0.025 publication target).
- **Contact:** target PCD >= 0.52, FRA <= 0.16, DRS >= 0.84, NUP >= 0.985,
  intervention <= 0.04, intervention-episode rate <= 0.025 and maximum run <= 2;
  reduce paired secondary-overlap scene rate by at least 0.02, overlap duration/run
  and residual impact; increase post-contact free-space, sustained escape and stable
  stop by at least 0.02 while preserving route/offroad/comfort constraints.
- These are development targets, not claimed v48.23 results. Publication claims
  require the preregistered gate, held-out authorized closed loop, paired confidence
  intervals and multi-seed confirmation.

### Decision and non-repetition rules

- `RC=0`: run only the authorization-checked held-out stress/closed-loop command,
  then multi-seed confirmation and final Safe paired non-interference evaluation
  using the same selected checkpoint.
- `RC=20`: do not read test/stress. First inspect the proposal-constrained oracle
  audit. If it fails, repair proposal/label/gate support under a newly preregistered
  protocol rather than training another calibrator. If it passes, run the
  adaptation-dev shadow closed loop and A/B/C/D ablations to localize
  ranking-versus-frontier-versus-admission failure.
- A dev shadow physical improvement with primary-gate failure means the certificate
  may be too conservative for the available sample size; document this and create a
  new preregistered protocol in a later version. Never relax v48.22/v48.23 gates
  retrospectively or use dev shadow results as paper results.
- `RC=30`: no algorithm conclusion is allowed. Repair the named protocol, index,
  training, checkpoint, calibration or artifact stage first.
- Do not repeat opportunity-only noisy-OR, neutral harm at probability 0.5,
  uncentered softplus admission, binary-only benefit supervision, global-harm-only
  checkpointing, regime-specific calibrators/residuals, raw high-dimensional
  context, proposal retraining, threshold-grid-only tuning, or dataset regeneration
  in this round.

### Local validation

- `pytest`: 216 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for all shell scripts: passed.
- Missing-protocol fault injection normalizes the controller and both failure
  artifacts to `RC=30`, `pipeline_valid=false`, and `test_roots_read=false`.
- New tests cover semantic risk initialization, identity-preserving admission,
  continuous top-k ranking gradients, frontier contrast gradients, categorical
  one-action probability, FRONTIER checkpoint sensitivity, proposal-oracle/dev-
  shadow plumbing and eight-task/two-GPU assignment.
- The delivery environment has no real WOMD/Waymax data or A30 GPUs. No v48.23
  Natural-gate or closed-loop result is claimed.


## v48.22 — OC-TRAC-COVENANT-BRIDGE (2026-07-30)

### v48.21 result attribution

- The v48.21 dedicated controller is valid and records `pipeline_valid=true`,
  `test_roots_read=false`, and `RC=20`; Near/Contact fit and verify support are
  non-empty and feasible. This is a genuine Natural-gate rejection, not a data,
  parameter-count, unsupported-gate, or controller failure.
- v48.21 converted part of the Near signal into learned selector evidence, but only
  in one objective branch. Balanced Near reaches candidate benefit AUC 0.841 and
  learned frozen-top-k benefit AUC 0.805, with positive top-1 regret 0.005. Precision
  Near collapses to 0.201/0.339. The Near signal is therefore real but not stable
  under the current training objective.
- Contact shows the first partial evidence of transferable safe-benefit selection in
  one branch: Precision candidate/learned benefit AUC is 0.586/0.613 and learned
  top-1 correlation is 0.166. Balanced Contact remains 0.540/0.432 with correlation
  -0.165. Contact capability is not jointly present with the strongest Near model.
- Component-risk learning is retained: learned harm AUC is approximately 0.60--0.65
  in Near and 0.64--0.66 in Contact. However, harmful top-1 switch rates remain
  0.45--0.70. Global harm AUC mainly separates harmful candidates from abundant dead
  candidates and does not adequately separate harmful candidates from high-benefit
  safe candidates at the deployment frontier.
- Every main and ablation certificate still has zero verify coverage. The nearest
  Balanced-Near fit rule selects 12 groups with only 2 positives, precision LCB
  0.071 and harmful UCB 0.673. The nearest Precision-Contact rule selects 19 groups
  with 2 positives, precision LCB 0.045 and harmful UCB 0.292. This is not a small
  threshold miss.
- The frozen proposal remains non-limiting: positive top-k oracle hit is about
  0.97--1.00. Proposal retraining remains prohibited in this round.

### Root defects found in v48.21

1. **Three deployment semantics were compressed into two heads.** Raw PCD benefit,
   componentwise harm, and final safe admission are different hypotheses. v48.21
   trained its benefit head on safe benefit while the preregistered gate continued
   to measure raw benefit plus an independent harm veto. This made one logit serve
   incompatible training, reporting, and certificate meanings.
2. **The safe-opportunity MIL probability was not safe.** The noisy-OR group loss used
   only `sigmoid(opportunity)` and ignored harm, even though its target was “at least
   one raw-beneficial and non-harmful candidate”. It therefore rewarded high
   opportunity on positive-but-harmful candidates—the exact false-safe failure mode
   exposed by Contact.
3. **Benefit and harm were combined differently in loss, checkpoint selection,
   calibration, and runtime.** Group MIL, safe-set loss, soft dev metrics and
   deployment did not all consume one explicit admission score. Candidate AUC could
   improve without improving the action actually certified or executed.
4. **Sparse admission gradients still competed semantically with raw benefit.** The
   same benefit logit was expected to preserve raw opportunities and also encode the
   conjunction `benefit AND non-harm`. Balanced learned Near while Precision learned
   part of Contact, producing complementary specialization rather than one model
   that works in both regimes.
5. **The zero-initialized source identity was never evaluated as a checkpoint.**
   Training started checkpoint comparison after epoch 1. A useful source/consensus
   initialization could not win if the first update damaged one regime.
6. **Sampler semantics needed an explicit task choice.** Raw benefit and safe
   admission now have separate heads. The positive group sampler must deliberately
   oversample safe-admission groups without changing the raw-benefit target.
7. **Certificate diagnostics were incomplete at the safety frontier.** Global
   benefit/harm AUC concealed failure among high-opportunity candidates. Safe-positive
   AUC and conditional high-opportunity harm AUC are required diagnostics.
8. **New admission branch had an uncovered runtime error during development.** The
   loss accepted `pred_admission_logit` but initially failed to initialize the local
   `admission_logits` tensor. A gradient-level unit test exposed and fixed this before
   delivery; static compilation alone would not have caught it.

### v48.22 algorithm: COVENANT-BRIDGE

**COVENANT = Cross-regime Opportunity, Veto Evidence, and Non-regime-specific
Admission with Nominal-preserving Transfer.** It remains one unified model over Safe,
Near and Contact. No regime ID, bucket-selected calibrator, or regime-specific
residual is available at inference.

1. **Three factorized hypotheses.** The unified model now predicts:
   - raw recovery benefit, trained on total PCD improvement and used by the primary
     opportunity gate;
   - DRS/deployability/gap component harm, aggregated by exact non-compensatory
     maximum and used by the independent harm veto;
   - final safe admission, trained on `raw benefit AND no component veto` and used
     for top-k reranking and group admission.
2. **Detached conservative admission prior.** The admission logit starts from
   `detach(raw_benefit_logit) - softplus(detach(harm_logit))` plus a zero-initialized,
   bounded admission residual. Sparse admission gradients cannot overwrite benefit
   or risk heads, while the prior remains conservative and context-correctable.
3. **Correct safe-opportunity MIL.** Frozen-top-k noisy-OR now uses the explicit
   admission probability. In the two-head ablation it uses
   `P(raw benefit) * (1-P(harm))`; it never uses opportunity alone. Candidates
   outside frozen top-k receive no group-opportunity gradient.
4. **One deployment score everywhere.** Training safe-set loss, soft checkpoint
   metrics, calibration, evaluator, selector and closed-loop runtime all use the
   explicit candidate-vs-nominal admission score. Legacy checkpoints fall back to
   their historical opportunity-minus-harm score.
5. **Raw-benefit/safe-admission sampler decoupling.** Raw benefit remains the benefit
   head target, while `group_batch_safe_positive_target=true` stratifies minibatches
   using safe-positive groups. Harmful-benefit overlap groups still supervise raw
   benefit and harm tails, but are not presented as positive admission groups.
6. **Epoch-zero checkpoint evaluation.** The source/consensus identity is validated
   and saved before any optimizer step and may win early stopping.
7. **COVENANT checkpoint risk.** The threshold-free CONCORD risk is retained and adds
   worst-regime harmful policy mass and false-admission penalties. This is dev-only,
   uses no test root, and does not relax the Natural gate.
8. **Frontier diagnostics.** Certificate reports now include candidate safe-positive
   AUC, learned top-k safe-positive AUC, and harm AUC conditioned on high opportunity,
   separating broad risk discrimination from the safety-critical decision frontier.

### Non-repeated v48.22 ablations

1. `A_two_head_safe_probability`: raw benefit + component harm, corrected
   `P(benefit)*(1-P(harm))` group probability, no third admission head. This isolates
   the v48.21 MIL/score engineering correction.
2. `B_triad_candidate_only`: three heads and candidate admission BCE, but no group MIL
   or setwise objective. This isolates the third hypothesis.
3. `C_triad_group_mil_aggregate`: admission head plus group objectives with one
   aggregate harm tail. This isolates component risk heads.
4. `D_full_covenant`: raw benefit, three component veto heads, admission head,
   deployment-exact safe-set loss, safe-opportunity MIL and COVENANT checkpoint risk.

All eight tasks (four groups times Balanced/Precision) are launched together. Round-
robin assignment places four tasks on GPU0 and four tasks on GPU1. Each task defaults
to batch size 48, one DataLoader worker, and bounded host math threads to reduce A30,
CPU and disk contention.

### Decision and non-repetition rules

- `RC=0`: run only the authorization-checked stress command generated by the
  independent certificate, then multi-seed confirmation and final Safe paired
  non-interference evaluation using the same selected checkpoint.
- `RC=20`: do not read test. Use A/B/C/D to identify whether the missing factor is the
  admission hypothesis, group supervision, or component veto. Do not tune the gate
  or proposal.
- `RC=30`: no algorithm conclusion is allowed. Repair the named protocol, index,
  training, checkpoint or certificate stage first.
- Do not repeat safe-benefit-overloaded opportunity heads, opportunity-only MIL,
  shared two-head admission semantics, fixed-threshold early stopping, exact-min
  benefit transfer, regime-specific calibrators, raw high-dimensional context,
  proposal retraining, threshold-grid-only tuning, or dataset regeneration in this
  round.

### Local validation

- `pytest`: 209 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for all shell scripts: passed.
- New tests cover three-head bucket invariance/capacity, detached-gradient isolation,
  harmful-benefit MIL rejection, independent safe-positive sampling, explicit
  admission-score plumbing, epoch-zero selection, and eight-task/two-GPU assignment.
- The delivery environment has no real WOMD/Waymax data or A30 GPUs. No v48.22 gate or
  closed-loop result is claimed.


## v48.21 — OC-TRAC-CONCORD-BRIDGE (2026-07-30)

### v48.20 result attribution

- The uploaded v48.20 dedicated controller completed both Balanced and Precision adaptations, a non-empty scene-disjoint certificate, manifest/protocol checks, and the test-root seal. It records `pipeline_valid=true`, `test_roots_read=false`, and `RC=20`. Fit/verify support feasibility is true in both Near and Contact, so this is a real Natural-gate rejection rather than the historical unsupported-gate or parameter-guard failure.
- Component-risk learning improved materially. Main candidate harm AUC is approximately 0.669--0.676 in Near and 0.661 in Contact, compared with approximately random harm evidence in v48.19. The component semantic reset and exact non-compensatory maximum are therefore retained.
- Natural-gate usability did not improve: every main and ablation certificate still selects zero verify groups. Near benefit is unstable across objectives/variants (Balanced main 0.445 versus Precision main 0.790), learned proposal-evidence correlation is only -0.074--0.125, and no gate-authorized Near closed-loop result exists.
- Contact remains the limiting regime. Main candidate benefit AUC is 0.487--0.495, proposal-evidence benefit AUC is 0.430--0.439, learned correlation is -0.074--0.105, and all verify coverage is zero. The strongest Contact benefit result is only the candidate-tail Precision ablation (0.603), whose harm AUC falls to 0.419 and still yields zero certificate coverage.
- The frozen proposal remains high recall (positive-group top-k oracle hit approximately 0.982--0.991). Proposal generation is not retrained in this version.

### Root defects found in v48.20

1. **Benefit/risk negative transfer in one shared adapter.** Safe-benefit positives are only about 3% of deployable candidates, whereas component-harm positives are about 45--54%. A single shared hidden representation allowed dense risk gradients to overwrite sparse benefit structure. The ablations expose this directly: candidate-only models retain the strongest Near/Contact benefit, while component-head models improve harm AUC but damage benefit AUC.
2. **The exact minimum expert envelope is too pessimistic for transfer.** v48.20 defines base benefit as `min(expert_1, expert_2)`. One source expert that is mismatched for a candidate can erase useful Near evidence even when the other expert is well calibrated. Balanced Near benefit collapses to 0.445 while the same frozen proposal still contains the opportunity almost always.
3. **Candidate tails do not directly supervise the group event.** Deployment first asks whether the frozen top-k contains any safe recovery and only then chooses a candidate. Candidate BCE and a setwise winner objective provide an indirect and unstable signal for this rare group-level admission event.
4. **Early stopping used fixed dev thresholds that were inactive.** The v48.20 metric used fixed opportunity/harm thresholds. Positive admission recall was zero through almost all dev epochs, making the selection risk nearly constant and causing Precision to select epoch 1. This cannot choose the checkpoint most likely to have a useful fit/verify frontier.
5. **Sampler semantics disagreed with safe-benefit training.** Raw PCD-positive but component-harmful overlap groups were placed in the positive stratum even though the safe-benefit group loss labelled them negative. In the uploaded index, Near has 16 raw positive groups but only 11 safe-positive groups; Contact has 44 raw and 41 safe-positive groups.
6. **Frozen proposal diagnostics were mixed with learned selector diagnostics.** The reported 0.86--0.93 non-positive false-switch rate and approximately 0.38--0.41 harmful ranked-switch rate are frozen tournament metrics and are identical across ablations. They diagnose the need for admission evidence, but they are not the learned gate's false-intervention rate. The learned gate currently abstains everywhere.
7. **Main/D ablation reproducibility was not exact.** Main and ablation jobs used different batch sizes/worker settings and non-deterministic cuDNN behavior, weakening causal comparison.

### v48.21 algorithm: CONCORD-BRIDGE

**CONCORD = CONservative Consensus Opportunity and Non-Compensatory Risk Decoupling.** It remains one continuous, bucket-invariant mechanism across Safe, Near and Contact; regime labels are never model inputs or inference routers.

1. **Permutation-invariant expert consensus.** Replace the exact minimum with `mean(expert benefit) - lambda * expert range` (`lambda=0.15` by default). This preserves transferable evidence when one expert is locally pessimistic while retaining an explicit disagreement penalty. The representation uses symmetric expert statistics, so expert ordering cannot act as a hidden regime identifier.
2. **Decoupled benefit and component-risk adapters.** Sparse safe-benefit and dense risk supervision receive separate zero-initialized bounded MLPs. Benefit keeps consensus transfer; harm remains an absolute semantic reset with DRS/deployability/gap component heads and exact `max` veto.
3. **Safe-benefit candidate target.** Opportunity supervision is `raw benefit AND not component harmful`. A candidate may still be a raw-benefit opportunity for certificate accounting, but the trained admission score is not rewarded for unsafe benefit/harm overlap.
4. **Frozen-top-k multiple-instance opportunity objective.** A noisy-OR loss directly supervises whether the deployed proposal top-k contains any safe beneficial candidate. Candidates outside frozen top-k receive no group-opportunity gradient. The existing deployment-exact safe-set objective remains primary; candidate tails and ranking terms remain auxiliary.
5. **Safe-positive stratified sampling.** When safe-benefit training is enabled, raw-positive/component-harmful groups are no longer sampled as positive admission groups. The teacher-index audit now reports safe-positive candidate/group/scene support explicitly.
6. **Threshold-free checkpoint selection.** `direct_concord_selection_risk` uses soft top-k safe-opportunity NLL, soft recall, false-admission mass, harmful policy mass, safe-candidate mass and safe top-1 regret. Near/Contact are only robust validation strata; the model receives no regime ID. Fixed 0.65/0.30 thresholds remain diagnostic and do not drive early stopping.
7. **Primary certificate semantics preserved.** The preregistered Natural gate continues to count raw PCD benefit and independently veto component harm (`OPPORTUNITY_LABEL_MODE=raw_benefit`). `safe_benefit` is supported as a separate audit mode but is not silently substituted into the primary gate, because current Near fit/verify support is sparse.
8. **Deterministic attribution.** Main and D use the same default batch size (72), deterministic algorithms, and `cudnn.benchmark=false`. Both variants must complete; all non-0/20 lower-level failures normalize to `RC=30`.

### Non-repeated v48.21 ablations

1. `A_safe_target_legacy_trunk`: safe target, safe sampler, group MIL and soft checkpoint metric, but retain the v48.20 shared trunk/exact-min architecture. This isolates the architecture correction.
2. `B_concord_candidate_only`: consensus plus decoupled adapters, but no group MIL or safe-set objective. This tests whether architecture alone is sufficient.
3. `C_concord_group_mil_aggregate`: consensus, decoupled adapters and group objectives with one aggregate harm head. This isolates component heads.
4. `D_full_concord`: full consensus, decoupled component risk, safe-positive sampler, frozen-top-k group MIL, deployment-exact safe set and threshold-free checkpoint selection.

Balanced and Precision run in separate waves. Each wave launches four tasks concurrently, two tasks per A30 and one DataLoader worker per task. Do not repeat exact-min unified benefit, one shared benefit/harm trunk, raw-positive sampler under safe targets, fixed-threshold early stopping, threshold-grid-only tuning, proposal retraining, regime-selected calibrators, or signed-total-PCD harm.

### Decision rules

- `RC=0`: run only the authorization-checked stress command generated after the independent certificate; then run multi-seed confirmation and the final Safe paired non-interference experiment.
- `RC=20`: do not read test. Compare A/B/C/D to determine whether the remaining failure is consensus transfer, group opportunity learning, or component calibration. Do not relax the Natural gate in the same output directory.
- `RC=30`: repair the stage-specific engineering failure before making any algorithm conclusion.
- Dataset regeneration is still deferred. The index/sampler/reporting fixes do not modify the three regime datasets.

### Local validation

- `pytest`: 203 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for every `scripts/*.sh`: passed.
- The delivery environment does not contain the real WOMD/Waymax datasets or the two A30 GPUs. No v48.21 Natural-gate or closed-loop result is claimed.

## v48.20 — OC-TRAC-UNISON-BRIDGE (2026-07-30)

### v48.19 result attribution and CCF-A readiness

- The uploaded v48.19 dedicated run completed both variants, non-empty scene-disjoint certificate fitting/verification, protocol-manifest checks, support-feasibility checks, and the test-root seal. Its controller records `pipeline_valid=true`, `test_roots_read=false`, and `RC=20`. Unlike the historical v48.18 Near specification, all v48.19 fit/verify support bounds are mathematically feasible; this is a genuine certificate rejection rather than a parameter guard or impossible-gate artifact.
- The frozen recovery proposal is already high recall: oracle-best top-k hit is approximately 0.982–0.991 in the main run and approximately 0.98–1.00 across ablations. The failure is therefore downstream of candidate generation.
- Near retains a useful benefit signal (main candidate benefit AUC 0.708–0.759; the strongest shared-only ablation reaches 0.800), but harmful evidence and group admission do not generalize. Contact benefit/harm AUC remains close to random or inverted, and every main/ablation certificate has zero deployable verify selections.
- Safe is ready only for the non-interference claim: the same mechanism must remain nominal when recovery is unnecessary. Near and Contact are not yet ready for a CCF-A main-result claim because no gate-authorized closed-loop OC-RAP result exists. Overall submission readiness is therefore **not reached**.
- Available external references are used only as progress anchors. Safe nominal/log/Wayformer artifacts are complete; Contact has complete 50-scene closed-loop baselines; Near offline baselines are complete, but all uploaded Near closed-loop summaries are incomplete or count-inconsistent and are excluded from paper-ready comparisons.

### Root defects found in v48.19

1. **Candidate classification replaced the deployed group decision.** `direct_value_ordinal_evidence_balanced_replaces_erm=true` allowed candidate-level balanced BCE to replace group ERM, while setwise admission, top-k, and intragroup terms were zero or negligible. Training optimized candidate labels, whereas deployment chooses nominal versus one recovery candidate per group.
2. **Training and deployment used different action scores and candidate supports.** The old setwise path scored every recovery candidate using frozen PCD plus log-sigmoid tails. Certificate/closed-loop first freezes proposal top-k and then reranks only that set with `sigmoid(benefit)-sigmoid(harm)`. Candidate AUC could improve without improving the actual deployed action.
3. **Harm semantics changed without resetting the source prior.** FACET component-veto harm was still added as a residual to the old signed-PCD source harm logit. The old base and new target do not represent the same event, so zero initialization was not semantic identity.
4. **Factorized supervision remained partly contaminated by signed three-class masks.** Class weighting, hard mining, and intragroup harm masks still used signed total PCD labels in several paths instead of component-veto labels.
5. **Regime-conditioned routing remained in the model.** Shared plus bucket-selected residual calibrators still consumed regime/bucket identity, so the learned policy was not a single continuous mechanism across Safe/Near/Contact.
6. **One aggregate harm tail discarded the observed physical structure.** DRS, deployability, and gap degradation account for nearly all harmful examples; hard-violation and `harm_proxy` positive increments are too sparse to support learned tails.
7. **The normalized smooth envelope was not conservative.** Normalized soft-min/soft-max lies inside the input range, so it can overestimate the weakest benefit expert and dilute one high-risk component with several low-risk components.
8. **External-baseline completion was not uniformly audited.** Several Near closed-loop summaries reported totals inconsistent with progress or scene journals. These artifacts must not enter a paper table.

### v48.20 algorithm: UNISON-BRIDGE

**UNISON = Unified Non-regime-specific Intervention Selection with Observation-consistent Non-compensatory evidence.**

1. **One bucket-invariant evidence model.** Inference does not receive a regime ID and does not select a Near/Contact calibrator. Both frozen source experts are evaluated for every candidate. The shared calibrator consumes their outputs, means, absolute disagreement, frozen policy margins, and tournament context.
2. **Conservative benefit transfer.** The source benefit is the exact lower envelope `min(expert_1, expert_2)`, followed by one zero-initialized bounded shared residual. This retains transferable Near signal while treating expert disagreement as lack of confidence without first classifying the regime.
3. **Componentwise harm semantic reset.** Three explicit zero-initialized bounded heads estimate nominal-relative DRS, deployability, and gap risk. The aggregate harm logit is the exact `max` across heads. No old signed-PCD harm base is added. Hard violation and `harm_proxy` remain deterministic certificate vetoes until their positive support is sufficient.
4. **Deployment-exact safe-set admission.** The frozen tournament first forms proposal top-k. The teacher safe set is `beneficial AND not component-harmful` within that top-k. If empty, nominal is the sole group target; otherwise a temperature-weighted distribution is formed over safe recovery candidates. The loss uses the exact deployed score `sigmoid(benefit)-sigmoid(harm)` and gives no safe-set gradient to candidates outside frozen top-k.
5. **Group objective is primary.** Candidate balance, component BCE, top-k auxiliary, and intragroup ranking remain auxiliary. They can no longer replace group ERM. Global balancing is bucket-agnostic.
6. **Safe is an invariant boundary, not a routed strategy.** Nominal rows remain pinned, and stress/test remains sealed until the same Natural gate authorizes execution.
7. **Worst-regime metrics are evaluation-only.** `direct_unison_selection_risk` can select a checkpoint using the worst Near/Contact validation behavior, but regime labels are not passed to the model or used for inference routing.

### Engineering corrections

- Set `ORDINAL_EVIDENCE_BALANCED_REPLACES_ERM=false` in the v48.20 pipeline.
- Use factorized component labels for component weights, hard masks, and intragroup supervision; use strict `margin > 0` for harmful membership.
- Use exact `min` benefit and exact `max` harm envelopes, preserving legacy DUET/FACET behavior when UNISON is disabled.
- Add a float64 gradient-norm reducer before clipping to prevent finite float32 gradients from overflowing the norm reduction and silently zeroing an update.
- Persist UNISON model flags and component-head geometry in checkpoints and inference bundles.
- Keep protocol preflight, manifest SHA256 binding, teacher-index contract/rebuild, both-variant completion, normalized return codes, and test-root sealing.
- Add `tools/audit_external_baseline_artifacts_v48_20.py`; a closed-loop summary is paper-eligible only when progress is complete and progress/summary/journal scene counts agree.

### Non-repeated v48.20 ablations

- `A_candidate_tail_only`: unified experts plus component candidate tails, no safe-set group objective.
- `B_safe_set_aggregate_harm`: deployment-exact safe-set objective with aggregate harm, no component heads.
- `C_component_safe_set_no_balance`: component heads plus deployment-exact safe-set, no global auxiliary balance.
- `D_full_unison`: component heads, deployment-exact safe-set, global auxiliary balance, and robust checkpoint selection.

Run Balanced and Precision as two waves. Each wave launches four tasks concurrently, two tasks per A30, one DataLoader worker per task. Do not repeat v48.19 separate/shared/regime-residual comparisons, threshold-only tuning, signed-PCD harm, raw-context expansion, or candidate-BCE replacement.

### Decision rules

- `RC=0`: run stress/closed-loop only through the automatically generated `NEXT_COMMANDS.txt` authorization.
- `RC=20`: the v48.20 protocol and artifacts are valid but the algorithm is rejected; run the four new ablations without reading test.
- `RC=30`: inspect the stage-specific failure JSON/log and fix engineering only. Do not interpret it as an algorithm result and do not mutate protocol settings in the same output directory.

### Validation

- `pytest`: 196 passed, 5 warnings.
- `python -m compileall -q src tests tools`: passed.
- `bash -n` for every `scripts/*.sh`: passed.
- The delivery environment did not contain the real WOMD/Waymax datasets or two A30 GPUs. No v48.20 Natural-gate or closed-loop outcome is claimed.

## v48.19 — OC-TRAC-FACET-BRIDGE (2026-07-30)

### Result attribution corrected before further tuning

- A recovered v48.17 `RC=20` is an algorithmic Natural-gate rejection only when the dedicated certificate is non-empty, scene-disjoint, and the controller records `pipeline_valid=true`. The earlier missing-report/`RC=30` failure was the separate 78,630-vs-20,000 parameter-guard bug already documented in v48.18.
- The uploaded v48.18 dedicated run completed adaptation and a non-empty independent certificate, returned `RC=20`, did not read test roots, and selected no verify groups for either variant. However, that result is **not a clean algorithm-only rejection** because the historical Near-fit specification was unsupported by its own split.
- Near fit contained only eight positive opportunities but required at least twelve selections and `precision LCB90 >= 0.50`. With the historical `z=1.6448536`, even an oracle selecting all eight positives plus four non-harmful negatives has LCB `0.43149`; no model or threshold could pass.
- The code labelled those directional bounds as 90% LCB/UCB while using the central two-sided 90% critical value. v48.19 declares the convention explicitly and uses one-sided 90% Wilson bounds (`z=1.2815516`). Under the new, separately preregistered protocol, Near fit uses 10 selections (`8/10` optimistic LCB `0.60160`) and verify retains 8 (`6/8` optimistic LCB `0.52371`, zero-harm UCB `0.17033`). This is a new protocol and must not be used to retroactively relabel v48.18 as passing.

### v48.18 ablation conclusions

1. `A_dual_scalar` preserved the only robust signal: Near benefit AUC (0.817 Balanced / 0.756 Precision). Harm AUC remained near random and Contact remained weak.
2. `B_dual_tournament` modestly improved harm ordering (especially Near) but reduced benefit ordering. It did not make Contact separable and therefore does not justify the current tournament context as a standalone improvement.
3. `C_dual_tournament_balanced` was unstable: Balanced nearly reverted to A; Precision gained only small harm AUC while losing benefit AUC and increasing intervention/harm on dev.
4. `D_full_duet` selected the same epochs and produced numerically identical certificate metrics as C for both variants. The v48.18 cross-regime checkpoint metric had no observed causal effect.
5. All v48.18 variants remained all-abstain on verify. Threshold search cannot solve this because the underlying Contact evidence and harm supervision are not discriminative.

### Root algorithm defect fixed

- v48.18 made the network outputs independent but generated both labels from one signed total PCD delta. Consequently benefit and harm were still mutually exclusive in supervision, even though a Contact recovery candidate can improve total deployability while worsening DRS, hard violation, gap quality, or post-contact stability.
- v48.19 introduces **FACET-BRIDGE: Factorized Advantage and Componentwise Evidence Transfer with a shared cross-regime bridge**.
- The benefit tail remains total PCD advantage. The harm tail is a non-compensatory component veto over nominal-relative DRS, deployability-gate probability, gap discount, hard violation, and `harm_proxy`. Benefit and harm can now be simultaneously positive.
- Component harm uses strict tolerance exceedance. Equality at the tolerance boundary is non-harmful; the default normalized deadband is 0.05 for each component. This removes the previous `component_margin == 0` soft-label ambiguity.
- Near and Contact share one zero-initialized bounded calibrator, with a small bounded regime residual (`scale=0.25`). This partially pools sparse evidence across regimes while retaining phase-specific corrections. Safe is the nominal boundary condition and remains protected by the verified nominal lock.
- Default trainable evidence-correction parameters are 2,298: three 766-parameter modules (one shared plus two regime residuals), far below the v48.17 raw-context calibrator and below the architecture-aware 8,000-parameter guard.

### Engineering and statistical safeguards

1. Add train/certificate-shared target implementation in `src/ocrap/algorithms/evidence_targets.py`; no duplicated harm definition is permitted.
2. Add explicit Wilson confidence-level/bound-type implementation and optimistic certificate-support preflight. Unsupported gates return protocol/artifact failure rather than algorithm rejection.
3. Freeze `GATE_SPEC.json` before certificate scoring. It now binds the full statistical protocol and SHA256 identities of Safe/Near/Contact manifests; changing data or gate settings requires a new output directory.
4. Bind teacher-index reuse to train-root paths, manifest SHA256 values, PCD parameters, macro set, and component-veto tolerances. A stale index is automatically rebuilt.
5. Add a pre-training FACET target-support audit. Near and Contact must each contain positive and negative examples for both benefit and harm tails; absence of overlap is reported as a warning rather than fabricated.
6. Main runs require both Balanced and Precision adaptation branches by default. One failed branch yields normalized `RC=30`; partial variants are allowed only with the explicit debugging flag `ALLOW_PARTIAL_VARIANTS=1`.
7. Normalize controller semantics: `RC=0` is a valid Natural-gate pass, `RC=20` is a valid supported-protocol algorithm rejection, and every other lower-level failure becomes `RC=30` with a stage-specific JSON artifact.
8. Stress/test execution remains sealed unless an independently certified run creates `NEXT_COMMANDS.txt`.

### v48.19 non-repetition ablations

- `A_component_veto_separate`: factorized component-veto targets with separate regime calibrators.
- `B_shared_component_veto`: A plus shared cross-regime partial pooling and bounded regime residuals.
- `C_shared_only_no_regime_residual`: shared bridge only; isolates whether regime residuals are necessary.
- `D_full_facet`: B plus the FACET checkpoint metric, which prioritizes minimum cross-regime recall subject to harm/false-intervention budgets.

Run four tasks concurrently per wave: two tasks on each A30, one DataLoader worker per task, Balanced wave followed by Precision wave. Do not repeat simplex labels, signed-total-delta harm labels, raw 4,890-D context, auxiliary-only balancing, sampler-only balancing, or v48.18 C/D comparisons.

### Validation

- `pytest`: 185 passed, 5 warnings.
- `python -m compileall -q src tools`: passed.
- `bash -n scripts/*.sh`: passed.
- No real Waymax/WOMD/A30 experiment was executed in the delivery environment; no v48.19 Natural-gate or closed-loop result is claimed.

## v48.18 — OC-TRAC-DUET-BRIDGE (2026-07-30)

### v48.17 result audit and corrected Natural-gate status

- The uploaded package does not contain the main `runs/ocrap_v48_17_bridge_dedicated_4817` directory, so its controller log cannot be inspected directly. However, the main controller invokes the same adaptation script used by the uploaded B/C ablations, and all four B/C logs show completed training followed by the same post-check failure: 78,630 `direct_evidence_calibrators.*` state parameters rejected against a hard-coded maximum of 20,000. The deterministic early-exit path therefore explains the reported missing main-run artifacts.
- `run_v48_17_bridge_dedicated.sh` then exited with code 30 before invoking the certificate controller and before running `check_v48_16_learning_gates.py`. This directly explains the simultaneous absence of `learning_gates_v48_17.json` and `NEXT_COMMANDS.txt`.
- The A_simplex_scalar ablation is the only v48.17 component with a valid, non-empty, scene-disjoint certificate. Both variants were genuinely rejected: Near used 290 groups/123 scenes and Contact 764 groups/215 scenes, but both selected zero verify groups and had zero positive recall.
- B_context_simplex and C_full_bridge cannot be assigned a Natural-gate result because their completed checkpoints were blocked before certificate. Their dev curves are diagnostic only.

### v48.17 algorithm attribution

1. **Retain the frozen top-k recovery proposal and Safe nominal lock.** The proposal continues to expose useful positive candidates, and the 120-scene paired Safe experiment passed all available non-inferiority checks with zero candidate-minus-baseline deltas.
2. **Context contains useful signal, but the v48.17 representation is statistically inefficient.** Relative-context BRIDGE improved Near dev positive recall in several epochs and modestly improved Contact recall in Precision, but it fed roughly 4,890 raw relative features into a 78,630-parameter calibrator despite only 60 deployable-positive adaptation groups.
3. **The three-class simplex correction is not appropriate for the observed target ambiguity.** It forces harm, dead and benefit to compete for unit mass. In Contact, a candidate may carry both benefit evidence and unresolved harm evidence; forcing one tail down can create false-safe decisions.
4. **The advertised batch-balanced objective was only auxiliary.** v48.17 added per-regime/class-balanced loss on top of the original top-1/top-k group ERM, so dead/mixed groups still dominated the primary gradient. The implementation did not match the intended “replace dead-zone-dominated ERM” contract.
5. **Checkpoint selection remained vulnerable to one-regime collapse.** Fold-robust risk could select an epoch with Near improvement but near-zero Contact recall, or vice versa.

### Engineering corrections

1. Replace the fixed 20,000-parameter v48.17 post-check with a configurable architecture-aware cap (`MAX_EVIDENCE_CALIBRATOR_PARAMS`, default 100,000 for recovery of existing BRIDGE checkpoints).
2. Always emit a learning-gate report and controller completion record, including on adaptation failure/exit 30. A missing report is no longer overloaded with “gate failed”.
3. Add `recover_v48_17_after_param_guard.sh` to reuse already-trained v48.17 checkpoints and run the withheld certificate without retraining.
4. Persist the evidence context source and new objective flags in checkpoints/inference configuration.
5. Add tests for identity initialization, tournament-context dimensionality, independent dual tails and cross-regime checkpoint risk. Full local validation: 176 tests passed; Python compileall and all Shell syntax checks passed.

### New algorithm: DUET-BRIDGE

**DUET-BRIDGE = Dual-tail Uncoupled Evidence Transfer with frozen tournament context and balanced target adaptation.**

1. **Frozen tournament context instead of raw relative features.** The evidence calibrator consumes the 48-dimensional contextual recovery embedding already produced by the frozen set tournament. This preserves proposal semantics while reducing the default two-regime calibrator from 78,630 parameters to 1,532 parameters.
2. **Independent benefit and harm residual tails.** A zero-initialized bounded residual is added independently to source benefit and harm logits. The model is no longer forced onto a three-class simplex; ambiguous candidates may have both tails elevated and are conservatively rejected by the harm veto. Nominal rows are explicitly pinned back to zero logits after correction so trained residual biases cannot alter nominal semantics.
3. **Independent-tail supervision.** Beneficial candidates supervise `(benefit=1, harm=0)`, harmful candidates `(0,1)`, and dead-zone candidates `(0,0)` using two BCE losses.
4. **Strict per-regime/per-class balanced replacement.** In calibrator-only adaptation, the minibatch-balanced objective replaces the dead-zone-dominated evidence ERM rather than being added as a weak auxiliary term.
5. **Cross-regime feasibility checkpoint metric.** `direct_duet_selection_risk` adds the minimum Near/Contact recall shortfall and worst-regime harm/false-intervention penalties to the held-out dev certificate risk. This changes only early stopping; the final Natural gate remains unchanged and scene-disjoint.

### Required v48.18 ablations

- A_dual_scalar: independent benefit/harm tails with the four source scalar inputs.
- B_dual_tournament: A plus frozen tournament context.
- C_dual_tournament_balanced: B plus stratified batches and strict balanced replacement.
- D_full_duet: C plus cross-regime feasibility checkpoint selection.

All eight tasks (four groups × Balanced/Precision) launch together. Tasks are assigned round-robin so each A30 runs four low-memory jobs; each task defaults to one DataLoader worker to limit CPU/I/O contention.

### Decision and non-repetition rules

- Do not create `NEXT_COMMANDS.txt` or run test/stress closed loop on exit 20 or 30. Stress remains authorized only by a valid independent Near+Contact certificate.
- Do not repeat the 78,630-parameter raw-context calibrator, simplex-only target correction, “balanced as auxiliary” loss, full Evidence retraining, threshold relaxation, or test-guided tuning.
- If repaired v48.17 returns 0, run its authorized stress experiment before v48.18 and preserve it as a valid comparison. If it returns 20, treat that as a true v48.17 algorithmic rejection and proceed to v48.18 without reading test results.


## v48.14 — OC-TRAC-PRISM (2026-07-29)

### Evidence from the completed v48.13 TERRA experiment

- Neither balanced nor precision passed the joint Near+Contact Natural gate. No v48.13 stress closed-loop result is attributable to the learned policy.
- TERRA's top-k proposal objective was the clearest success: on the main split, positive-group oracle-best hit was about 0.959 Near / 0.970 Contact and any-positive hit was 1.000 / 0.985. The high-recall proposal should be retained.
- Exact top-1 remained weak (Near negative or unstable, Contact only slightly positive), but this is no longer the primary bottleneck once proposal recall is high.
- Proposal evidence did not transfer: Contact harm AUC was approximately 0.39–0.54 and evidence/teacher correlation was near zero or negative. Non-zero Contact selections had low precision, high conditional harm, and negative mean exact-teacher advantage.
- The dedicated calibration diagnostics prove a train-to-target contract shift. Near/Contact calibration roots are far closer to val/test than train in `r_dep_star`, hard violation, candidate count, recoverability, and artifact rate. The legacy `harm_proxy` is non-zero in train but identically zero in calibration/val/test.

### Engineering defects fixed before further attribution

1. **Missing standard calibration artifacts.** Staged v48.13 training used `SKIP_POST_TRAIN_CALIBRATION=1`, so `gamma_rec_by_bucket_v48.json` and the standard calibration JSONs were never produced. v48.14 atomically generates them from the independent certificate pool.
2. **Safe nominal-only dependency bug.** The runner checked gamma and calibration before entering the Safe nominal-lock branch. Safe paired non-inferiority now requires only the checkpoint; Near/Contact stress execution still requires a valid certificate.
3. **Incomplete dedicated recalibration.** The uploaded source run had no completed `dedicated_candidates` artifacts. The new finalizer writes temporary outputs, verifies every required file, atomically installs the calibration directory, and writes `CERTIFICATE_CALIBRATION_COMPLETE.json`.
4. **Invalid v48.13 ablation scheduler.** `GROUPS` is a Bash special array containing Unix group IDs; only `1012_balanced/precision` ran instead of A/B/C/D. The scheduler now uses `ABLATION_SPECS` and requires all eight task markers. Consequently, the uploaded v48.13 ablation cannot support causal algorithm claims.
5. **Ordered-NLL option propagation.** The staged script computed `ORDERED_TOP1/ORDERED_ALL` but passed unrelated fallback defaults to the generic trainer. Parameter names and effective values are now unified.

### New algorithmic contribution: PRISM

**PRISM = Proposal-aligned Risk adaptation with Independent Scene-disjoint certification Model.**

1. **Freeze the proven high-recall proposal policy.** The v48.13 recovery tournament and encoder are frozen. v48.14 does not repeat exact-winner pairwise/listwise attempts that previously degraded Near.
2. **Scene-disjoint calibration-stage evidence adaptation.** Dedicated Near/Contact calibration roots are split by scene into 45% evidence-adaptation train, 15% adaptation dev, and 40% certificate pool. Only the small regime-specific `direct_delta_adapters` are updated. Test roots remain sealed.
3. **Dynamic false-safe hard-harm mining.** Ordered three-state NLL dynamically upweights harmful proposal members that the current adapter predicts as safe, plus a weaker missed-benefit weight to prevent all-abstain collapse. Hardness weights are detached.
4. **Independent certificate pool.** Standard OC-MERO calibration/gamma, policy-rule fit, scene-disjoint verify, and Natural gate are performed only on certificate-pool scenes not used by adaptation or early stopping.
5. **Target-distribution-aligned checkpoint selection.** Adaptation early stopping uses the same top-k evidence-rerank certificate semantics as deployment.

### Required v48.14 ablations

- A: v48.13 frozen checkpoint + dedicated certificate recalibration only.
- B: dedicated target-domain evidence adaptation without dynamic hard mining.
- C: target adaptation + dynamic hard-harm/missed-benefit mining, no same-group pair objective.
- D: full PRISM, adding same-group counterfactual evidence.

Per variant, all four tasks run concurrently: A/C on GPU0 and B/D on GPU1. Balanced and precision remain separate waves to limit CPU and storage contention.

### Decision gates

- Proposal oracle-best/any-positive hit must not materially regress from v48.13.
- Contact policy harm AUC should improve to at least 0.60 and remain directionally consistent between adaptation dev and certificate verify.
- Near benefit AUC should remain at least 0.70; Contact at least 0.75.
- Natural gate still requires non-zero verify coverage, positive mean exact-teacher advantage, unchanged precision/harm confidence bounds, recall/support, and opportunity-normalized macro constraints.
- Stress closed loop is allowed only when `NEXT_COMMANDS.txt` is generated from the independent dedicated certificate pool.

### Non-repetition note

Do not repeat all-pairs recovery ranking, cross-scene bipolar evidence as the primary harm objective, conformal calibration on a non-discriminative evidence model, threshold relaxation, absolute macro caps, or full train-set reconstruction at this stage. PRISM reuses the successful top-k proposal and specifically targets the empirically proven evidence-domain shift while preserving an independent statistical certificate.


## v48.13 — OC-TRAC-TERRA (2026-07-29)

### Evidence from the completed v48.12 TRIDENT experiment

- Neither balanced nor precision passed the joint Near+Contact Natural gate, so no OC-RAP stress closed-loop result is attributable to v48.12.
- Under the correct policy-first/no-fallback contract, three-seed recovery ranking remained asymmetric: Near group top-1 correlation was negative for both variants on average (about -0.054 balanced and -0.035 precision), while Contact was consistently positive but insufficient (about 0.077 balanced and 0.101 precision, versus the internal 0.20 readiness target).
- Contact benefit detection remained strong (candidate-positive AUC about 0.82 and policy-top1 benefit AUC near 0.80), but harmful-vs-dead evidence did not transfer across scenes. Fit rules with positive mean exact-teacher advantage collapsed on verify to high harmful rates and negative mean advantage.
- Near near-miss rules were sparse but sometimes safe: selected groups could have positive mean exact-teacher advantage and no harmful actions, yet support, Wilson precision lower bounds, recall, and cross-seed stability were below the Natural gate.
- External baselines establish the eventual closed-loop bar. Safe is dominated by nominal/log replay non-intervention. Near predictive-safety filtering offers a strong DRS/FRA/ODG/NUP trade-off. Contact restoration/MPC baselines recover more aggressively but pay substantial intervention and NUP cost. v48.12 has no gate-authorized closed-loop result and therefore has not surpassed these baselines.

### Causal conclusions from the complete v48.12 ablation

1. **The standalone recovery-set tournament remains useful but exact winner supervision is underidentified.** Candidate rank correlation is positive, especially in Contact, but it does not reliably become exact top-1. The v48.12 all-pairs teacher-gap loss degraded Near and did not materially improve Contact.
2. **Bipolar cross-group evidence is not a sufficient harm solution.** It improved some Near harm AUC values, but Contact harmful-vs-dead discrimination remained near random and selected verify actions retained negative average teacher advantage. Cross-scene pairwise losses can exploit scene severity and are noisy under minibatch sampling.
3. **Opportunity-normalized macro support is an engineering-correct certificate, not the current bottleneck.** Precision/harm transfer fails before macro excess becomes decisive.
4. **Threshold relaxation remains contraindicated.** Natural-gate rejection is consistent with the observed harmful verify actions.

### Engineering defects fixed before v48.13 attribution

1. **Parent-controller policy-contract loss.** The staged child process exported policy-first/no-fallback internally, but the parent calibration process reverted to default false values. The v48.12 main run therefore calibrated a different selection contract than its multi-seed run. Every staged variant now writes `POLICY_CONTRACT.env`, and the controller sources it before calibration.
2. **Checkpoint-selection/deployment mismatch.** Stage-E early stopping evaluated only the tournament rank top-1, while TERRA deploys evidence reranking within a frozen top-k proposal. Validation now uses the same proposal and evidence-reranking candidate, including certificate regret, harm, false intervention, recall, and evidence margin.
3. **Calibration/runtime contract propagation.** Proposal size and evidence-rerank semantics are now stored in calibration JSON selector overrides and consumed by offline and closed-loop selectors.
4. **Dedicated and multi-seed recalibration parity.** Both scripts source the immutable per-variant contract and use the same support, conditional-harm, macro, and proposal settings as the main run.
5. **Legacy packaging regression.** The missing historical v47 orchestration file required by the existing regression suite was restored; all historical tests now execute.

### New algorithmic contribution: TERRA

**TERRA = Top-k Evidence-Reranked Recovery with Abstention.**

1. **Set-valued recovery proposal**
   - Retains the independent permutation-equivariant recovery tournament.
   - Replaces noisy all-pairs exact-winner supervision with a differentiable top-k inclusion objective: at least one exact-teacher acceptable recovery must enter the proposal.
   - Proposal quality is measured on positive-opportunity groups by oracle-best hit rate and any-positive hit rate, separately from exact top-1 correlation.

2. **Proposal-distribution ordinal evidence**
   - Freezes the tournament and trains regime-specific ordered harmful/dead/beneficial evidence on every member of the actual top-k proposal, with rank-decayed weights.
   - This removes the v48.12 mismatch in which only rank-1 evidence was trained although useful or harmful runner-up candidates determined failure analysis.

3. **Same-group counterfactual evidence**
   - Adds beneficial-vs-nonbeneficial and harmful-vs-nonharmful comparisons only within the same scene-time proposal.
   - Shared scene severity cancels in these comparisons, reducing the train/dev shortcut that harmed Contact cross-scene transfer.
   - The v48.12 cross-group bipolar pair loss is disabled in the TERRA main experiment.

4. **Evidence reranking with abstention**
   - Runtime order is: physical recovery candidates → frozen rank top-k proposal → evidence thresholds → choose the highest evidence member within the proposal → abstain if none passes.
   - This is not an out-of-distribution runner-up fallback because Stage E explicitly trains all proposal members.
   - The same evidence score and margin are used by checkpoint selection, calibration, offline evaluation, and closed loop.

### Required layered validation

1. **Proposal gate:** on positive-opportunity groups, proposal oracle-best hit rate should be at least 0.75 and any-positive hit rate at least 0.90 in Near and Contact. Exact top-1 remains diagnostic rather than the sole Stage-P success condition.
2. **Proposal-evidence gate:** evidence-reranked proposal top-1 benefit AUC should reach at least 0.70 Near / 0.75 Contact, harm AUC at least 0.60, and evidence/teacher correlation at least 0.10.
3. **Natural gate:** only non-zero held-out coverage with positive mean exact-teacher advantage, unchanged precision/harm confidence bounds, recall/support, and macro-excess constraints may authorize stress closed loop.
4. **Multi-seed:** run 4801/4802/4803 only on an immutable checkpoint after proposal/evidence diagnostics are promising.
5. **Closed-loop comparison:** Safe must remain nominal-noninferior; Near must improve safety/recovery relative to predictive filtering without excessive intervention; Contact must approach restoration/MPC recovery while materially improving intervention and NUP trade-offs.

### Required ablations and GPU scheduling

- A: top-1 contract baseline.
- B: top-k proposal training only, deployment remains top-1.
- C: proposal-distribution evidence and evidence reranking on the old tournament.
- D: full TERRA.

The v48.13 scheduler runs all four groups concurrently per variant wave. A/C share GPU0 and B/D share GPU1, permitting two approximately 1-GB jobs per A30 while preserving separate processes and outputs. Balanced and precision waves remain sequential to limit host I/O contention.

### Non-repetition note

Do not repeat v48.12 all-pairs recovery ordering, cross-group bipolar evidence as the main harm objective, threshold relaxation, absolute macro caps, inherited value residual ranking, or untrained runner-up fallback. TERRA changes the identifiable policy object from a noisy exact winner to a small recovery proposal and aligns evidence training with every candidate that deployment may execute.


## v48.12 — OC-TRAC-TRIDENT (2026-07-28)

### Evidence from the completed v48.11 CASTER experiment

- No balanced or precision candidate passed the joint Near+Contact Natural gate; stress closed loop was correctly withheld.
- The standalone recovery set tournament produced a real but incomplete Contact ranking gain. Across calibration seeds 4801/4802/4803, balanced Contact top-1 correlation was 0.0735/0.0850/0.1026 (mean 0.0870), and precision was 0.0208/0.0455/0.0777 (mean 0.0480). Near remained approximately zero or negative.
- Candidate recovery signal remained strong, especially Contact (three-seed candidate-positive AUC about 0.829 balanced and 0.813 precision), but policy-top1 harm discrimination remained weak and unstable (Contact candidate-harm AUC about 0.536).
- Balanced Near exposed a useful near-miss: a verify rule selected 10 groups with 0.70 point precision, no harmful selections, 0.28 positive recall, and +0.146 mean exact-teacher advantage. It was rejected partly because every selected action was macro 5. However, the teacher-positive training distribution itself was about 88% macro 5, so the old absolute 0.85 macro-share constraint confounded policy shortcut with opportunity support.
- Contact fit-to-verify transfer remained the main safety failure: representative fit rules had positive mean advantage and moderate coverage, but verify precision collapsed and harmful rate increased sharply.

### Engineering defects fixed before further algorithm attribution

1. **Conditional checkpoint semantic mismatch.** v48.11 trained a recovery-only tournament, but `PREFERENCE_CONDITIONAL_MODE` was not enabled in the staged script. Early stopping therefore penalized nominal false-switch terms even though nominal was not part of the tournament and centered recovery scores make one recovery positive by construction. v48.12 sets this flag explicitly.
2. **Ablation scheduler fail-fast bug.** Under `set -e`, failure of the first `wait` stopped the entire suite. The uploaded ablation package therefore lacked C-precision and D-precision and could not isolate all CASTER modules. The new scheduler records failures, continues all eight tasks, and creates `ABLATIONS_COMPLETE.json` only when all tasks finish.
3. **Macro certificate contract.** The absolute selected-macro cap is replaced in the main v48.12 experiment by an opportunity-normalized excess concentration: selected concentration is penalized only when it exceeds the exact-teacher positive-policy concentration by more than a configured allowance. Raw macro share remains reported.

### New algorithmic contribution: TRIDENT

**TRIDENT = Teacher-gap Recovery tournament with Inter-regime Discriminative Evidence and Normalized-support cerTification.**

1. **Teacher-gap recovery-pair tournament**
   - Retains the recovery-only, permutation-equivariant set tournament that improved Contact.
   - Adds exact-PCD gap-weighted pair supervision only when a recovery pair is materially ordered.
   - Near ties below the configured gap remain unordered, avoiding artificial winner labels; clear pairs receive direct top-1 gradients.

2. **Bipolar cross-group ordinal evidence**
   - Retains the proper harmful/dead-zone/beneficial ordered simplex and frozen policy-top1 training distribution.
   - Adds regime-local cross-group pairwise AUC surrogates for beneficial-vs-nonbeneficial and harmful-vs-nonharmful policy selections.
   - Harm separation receives the larger weight because Contact verify harm inversion, rather than benefit detection, is the current certification bottleneck.

3. **Opportunity-normalized support certificate**
   - Reports both raw macro concentration and oracle-positive macro concentration.
   - The deployability constraint uses positive-policy excess concentration by default in TRIDENT experiments, preventing an impossible diversity requirement when the available teacher opportunities are intrinsically concentrated.
   - This is not threshold relaxation: precision, harmful-switch, support, recall, positive mean advantage, and scene-disjoint verification requirements are unchanged.

4. **Layered experimental attribution**
   - A: conditional-contract fix only.
   - B: recovery-pair tournament.
   - C: bipolar evidence.
   - D: full TRIDENT.
   - All eight variant tasks are attempted with at most two concurrent single-GPU jobs.

### Required validation order

1. Stage R: Near top-1 correlation must become consistently positive and Contact should exceed the v48.11 mean; inspect exact regret as well as correlation.
2. Stage E: policy-top1 benefit AUC should be at least 0.70 Near / 0.75 Contact, and harm AUC at least 0.60 in both regimes.
3. Certificate: non-zero verify selections must have positive mean exact-teacher advantage, precision LCB and harmful UCB within the unchanged Natural gate, and macro excess within budget.
4. Only a candidate passing both Near and Contact may enter stress closed loop. Safe paired non-inferiority remains a separate experiment.

### Non-repetition note

TRIDENT does not repeat threshold relaxation, handwritten recovery rules, the inherited value-plus-residual rank, global conformal saturation, or a shared harm classifier. It deepens the only v48.11 component with positive evidence (the standalone set tournament), directly optimizes the failing cross-group harm tail, and corrects support certification relative to the opportunity distribution.

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

## v48.15 — OC-TRAC-PRISM-CC (2026-07-29)

### Evidence and correction of the v48.14 conclusion

The uploaded v48.14 ablation package did **not** evaluate the Natural gate.  Every
certificate worker terminated at shell line 23 with `variant: unbound variable`:
`local variant="$1" gpu="$2" run="$OUTPUTDIR/candidates/$variant"` expanded
`$variant` before the local assignment under `set -u`.  The controller then
misclassified missing risk JSONs as `GATE_FAILED.json`.  Consequently, absence of
`NEXT_COMMANDS.txt` in that run means *calibration artifact failure*, not a valid
algorithmic gate rejection.  v48.15 separates exit code 30 / `CALIBRATION_FAILED.json`
from exit code 20 / `GATE_FAILED.json` and provides a no-retraining recovery script.

The Safe paired package also had zero matched calibration targets because the runner
forced `closed_loop.bucket_split=test` on `calibration_safe`; it silently fell back to
eight arbitrary WOMD scenes.  v48.15 removes the forced split, requires non-empty
bucket targets, disables stale resume by default in the Safe wrapper, and emits
scene-level jerk/yaw-rate p95.  The uploaded 8-scene Safe result is therefore a
nominal-lock smoke test, not a calibration-safe non-inferiority result.

### v48.14 algorithm evidence that remains valid

- The dedicated scene-disjoint adaptation/dev/certificate protocol is retained.
- Target-domain adaptation reduced harmful-switch/false-intervention diagnostics on
  adaptation dev, but the full `direct_delta_adapters` update trained roughly 0.39M
  parameters from only 16 deployable-positive Near groups (10 scenes) and 44 Contact
  groups (17 scenes).  Positive admission recall collapsed to 0–0.33 Near and
  0–0.036 Contact, indicating overfit/over-conservative forgetting rather than a
  calibrated deployable certificate.
- Dynamic hard-harm mining reduced some false-safe diagnostics but further suppressed
  positive recall.  It remains a moderate auxiliary weight, not the main adaptation
  mechanism.
- The same-group counterfactual term produced no consistent gain over hard-harm-only
  adaptation and is disabled in the v48.15 main experiment.

### New algorithm: PRISM-CC

**PRISM-CC = Proposal-aligned Risk adaptation with Independent Scene-disjoint
certification and low-Capacity Correction.**

1. **Frozen proposal and frozen source evidence.**  The high-recall v48.13 top-k
   recovery proposal and the source ordinal-evidence experts are both frozen.
2. **Tiny regime-specific residual evidence calibrator.**  A zero-initialized MLP
   consumes the frozen source evidence center/width and frozen policy score/gap, then
   produces a bounded residual correction.  The two regime calibrators contain 132 state parameters in total, versus approximately 392k trainable parameters
   in v48.14.  Initial predictions exactly reproduce the source checkpoint.
3. **Balanced three-state correction.**  Ordered harmful/dead-zone/beneficial NLL is
   retained, but hard-harm amplification is reduced and missed-benefit importance in
   checkpoint selection is increased to avoid an always-abstain optimum.
4. **Independent certificate pool unchanged.**  Natural-gate thresholds, scene
   disjointness, Wilson bounds, harmful-selection bounds, support requirements, and
   opportunity-normalized macro checks are not relaxed.

### Engineering and attribution changes

- Fixed the certificate worker local-variable expansion bug.
- Added `VARIANTS` filtering so a single-variant ablation task does not launch or report
  a nonexistent sibling variant.
- Distinguish calibration/controller failure from a genuine Natural-gate rejection.
- Added `scripts/recover_v48_14_certificate_pool.sh` to evaluate already-trained v48.14
  checkpoints without retraining.
- Added strict Safe target matching and removed arbitrary-scene fallback.
- Added scene-level jerk and yaw-rate p95 to Safe paired output.
- Added `scripts/run_v48_15_parallel_ablations.sh`; four ablations run concurrently per
  variant wave, two processes per A30 as supported by the measured memory footprint.
- Added layered `tools/check_v48_15_learning_gates.py` diagnostics.

### Required v48.15 ablations

1. `A_source_dedicated`: fixed source checkpoint, dedicated recalibration only.
2. `B_full_adapter_prism`: v48.14 high-capacity target adaptation.
3. `C_tiny_calibrator`: low-capacity residual correction without hard mining.
4. `D_full_prism_cc`: low-capacity correction with balanced hard-harm/missed-benefit
   supervision.

### Non-repetition and stopping rule

Do not repeat all-pairs recovery ranking, shared NASC, minibatch GroupDRO, continuous
relative-gain regression, broad conformal radii, strong hard-harm weighting, or
full-adapter target adaptation unless new evidence invalidates the prior conclusions.
First recover and evaluate the already-trained v48.14 certificates.  Run stress
closed-loop only when the controller creates `NEXT_COMMANDS.txt`; no gate threshold is
lowered to force that file to appear.

## v48.16 — OC-TRAC-ANCHOR (2026-07-29)

### Correction of the v48.15 experimental conclusion

The uploaded v48.15 certificate result with `rc=20` was not a valid Natural-gate
rejection.  The dedicated partition deliberately labels samples as
`evidence_adapt_train`, `evidence_adapt_dev`, and `certificate_pool`, while both
standard calibration and policy-risk calibration accepted only literal
`calibration`/`val`.  Every certificate NPZ was therefore discarded:
`num_groups=0`, `num_scenes=0`.  The controller installed the empty JSON files and
misclassified the risk tool's failure as a gate rejection.  v48.16 introduces
protocol-aware split roles, requires non-empty scene-disjoint certificate data, and
uses exit code 30 for artifact/protocol failure.  A Natural gate is considered
evaluated only when both Near and Contact contain non-zero groups, scenes, fit folds,
and verify folds.

The uploaded Safe paired run was also invalid: 120 offline targets were loaded but
zero were matched after scanning only 2,000 raw validation scenarios.  The correct
WOMD validation shard specification is `validation_tfexample.tfrecord@150`, and
sparse dedicated target IDs require scanning the complete validation set.  v48.16
validates all 150 shard files, defaults `SAFE_RAW_MAX_SCENARIOS=0`, and hard-fails
instead of writing an empty apparently valid result when no target is matched.

### Evidence retained from the adaptation-dev ablation

Final certificate metrics are unavailable because of the split-role bug, but the
adaptation-dev results still reveal the optimization failure mode:

- the high-capacity v48.14 adapter substantially reduces admissions and often
  destroys Contact positive recall;
- the 132-parameter v48.15 calibrator preserves the source model structurally and
  lowers harmful/false interventions, but collapses positive admission recall to
  0--0.11 Near and approximately 0.036 Contact;
- the v48.15 hard-harm/hard-benefit configuration selected exactly the same best
  epoch metrics as the plain tiny calibrator, so the same weighting is not repeated;
- the frozen high-recall top-k proposal, source ordinal evidence, scene-disjoint
  adaptation/certificate protocol, and zero-initialized bounded residual correction
  remain the useful foundation.

### New algorithm: ANCHOR

**ANCHOR = Adaptation with Nominal-preserving Class-balanced Held-out Ordinal Risk.**

1. **Class-balanced ordered evidence.**  Proposal evidence loss is averaged within
   harmful, dead-zone, and beneficial classes before averaging present classes.
   Dead-zone prevalence can no longer make all-abstain the lowest-loss solution.
2. **Bipolar probability margins.**  Beneficial proposals are explicitly pushed to a
   minimum benefit probability and harmful proposals to a minimum harm probability.
   This trains both tails required by the selective certificate.
3. **Source-residual anchoring.**  The target-domain calibrator residual receives an
   L2 anchor, retaining the source evidence unless dedicated data supports a bounded
   correction.
4. **Lower-capacity correction.**  Hidden width is reduced from 8 to 4 and residual
   scale from 0.30 to 0.20.  Strong hard-harm mining is replaced by moderate
   harm/benefit weights; the missed-opportunity checkpoint penalty is increased.
5. **No proposal retraining in this round.**  The top-k recovery proposal is frozen so
   any change in Natural-gate performance is attributable to target-domain evidence
   correction rather than another ranking modification.

### Engineering and attribution changes

- Added semantic split-role aliases in `ocrap.models.data`.
- Dedicated standard calibration explicitly accepts `certificate_pool` and disables
  validation fallback.
- Policy-risk calibration accepts an explicit `--allowed-splits` contract and returns
  an artifact-failure code for empty data.
- Certificate completion now validates non-zero samples/groups/scenes and non-empty
  fit/verify folds before installing results.
- Added dedicated protocol role/scene-leakage audit.
- Main and ablation controllers distinguish exit 0 (gate pass), 20 (valid gate
  rejection), and 30 (pipeline/artifact failure), and capture adaptation log tails.
- Safe WOMD shard preflight requires 150 validation shards; complete-set scanning is
  the default for sparse target matching; zero matched targets now hard-fail.
- Generated `NEXT_COMMANDS.txt` invokes an authorization-checking stress wrapper.
- Four v48.16 ablations run concurrently per variant wave, two light jobs per A30.

### Required v48.16 ablations

1. `A_source`: fixed v48.13 source evidence plus valid dedicated certificate.
2. `B_old_tiny`: the v48.15 tiny-calibrator objective under the repaired pipeline.
3. `C_balanced_margin`: class-balanced ordinal evidence and bipolar margins.
4. `D_full_anchor`: class-balanced margins plus source-residual anchoring.

Do not claim a v48.15/v48.16 gate result unless `certificate_data_valid=true` and
both risk JSON files contain non-zero independent scenes.  Do not run test/stress
closed loop after exit 20; development-only qualitative diagnostics may be used, but
must be isolated from paper metrics and threshold selection.

## v48.17 BRIDGE — 2026-07-30

**BRIDGE: Batch-balanced Regime-conditioned Identity-preserving Discriminative Group Evidence**

### Why this version was necessary

The completed v48.16 ablation bundle contains eight valid, non-empty dedicated
certificates (four components times balanced/precision).  Every run returned a real
Natural-gate rejection (exit 20), not an artifact failure: the held-out verify folds
contained 163 Near groups with 6 positive opportunities and 380 Contact groups with
14 positive opportunities, but every accepted rule selected zero groups.  The
uploaded Safe paired run contains 120 matched scenes and is identical on its available
metrics, while route progression was not emitted and jerk/yaw-rate did not yet carry
non-inferiority margins.

The source proposal is not the dominant bottleneck.  On the balanced source
certificate, top-3 contains an oracle-best or another positive candidate for all
positive Near groups and all positive Contact groups.  Positive-group top-1 accuracy
is 0.643 for Near and 0.594 for Contact.  In contrast, Evidence has weak harmful
ranking and severe false-switch exposure: proposal-Evidence harm AUC is below 0.5 in
both regimes, and the unconstrained non-positive false-switch rate exceeds 0.90.
Contact additionally exhibits a strong fit-to-verify reversal: the closest fit rule
selected 1/20 positive and 2/20 harmful candidates, while its verify counterpart
selected 0/24 positive and 14/24 harmful candidates.

v48.16 B/C/D changed the dedicated certificate metrics only at approximately floating
point noise.  Code audit identified three reasons:

1. The target calibrator observed only four summary scalars, so candidates with
   similar source center/width and rank margins but opposite target-domain outcomes
   were conditionally indistinguishable.
2. The advertised class-balanced Evidence loss was balanced inside each scene-time
   group.  Because most groups contain a single teacher class, it often collapsed to
   ordinary NLL; dead-zone groups still dominated across the minibatch.
3. Weighted replacement increased the probability of rare groups but did not ensure
   beneficial, harmful and dead-zone evidence was simultaneously present in a batch.
   Bipolar margins and class balance were therefore frequently inactive.  Checkpoint
   selection could still prefer the early always-abstain solution.

### Algorithm changes

1. **Identity-preserving tri-simplex residual.**  Added
   `direct_recovery_evidence_calibrator_mode=simplex_context`.  A zero-initialized,
   bounded residual is added to the frozen source log-probabilities of the harmful,
   dead-zone and beneficial classes, followed by a softmax.  At initialization the
   model is exactly the source Evidence model; unlike the old center/width correction,
   the beneficial and harmful tails may be corrected independently while retaining a
   valid probability simplex.
2. **Frozen candidate-vs-nominal context.**  The small calibrator can consume the
   source relative feature vector in addition to source class summaries and proposal
   rank margins.  Context is detached by default, preserving proposal/source Evidence
   attribution and keeping target adaptation low capacity.
3. **Batch- and regime-balanced ordinal Evidence.**  Beneficial, harmful and dead-zone
   candidate losses are accumulated over the whole minibatch and separately by
   regime, then averaged over the classes/regimes that are present.  Bipolar benefit
   and harm probability margins are applied at the same batch scope.
4. **Evidence-stratified scene-time batches.**  The group sampler builds exact teacher
   strata from best candidate-vs-nominal PCD: beneficial, harmful-only and dead/mixed.
   Replacement sampling is performed within each stratum and batches are interleaved,
   with default group fractions 0.35/0.35/0.30.  Scene-time grouping remains intact.
5. **Recall-constrained checkpoint selection.**  Added a configurable minimum positive
   recall and a shortfall penalty in the direct-policy metric.  The default v48.17
   target is recall >= 0.25 on adaptation dev; this prevents an always-abstain epoch
   from winning only by avoiding harm.
6. **Conservative bounded adaptation.**  BRIDGE freezes the source model and proposal,
   uses an 8-wide calibrator, a bounded residual scale of 0.75, an L2 source anchor of
   0.02, and no selective-risk, hard-mining or pairwise objectives.  Those objectives
   were intentionally disabled because previous versions did not provide stable
   incremental evidence.

### Engineering changes

- Added full checkpoint/config compatibility for calibrator mode, context input and
  context detachment; legacy `center_width` checkpoints remain loadable.
- Added exact evidence-stratum accounting to training summaries and hard failure when
  stratification is requested without an exact scene-time group index.
- Fixed Natural-gate checker field names (`precision_wilson_lcb90` and
  `teacher_advantage_mean`) so reports no longer silently read missing metrics.
- Fixed final candidate selection to read `teacher_advantage_mean` (with legacy
  fallback), so a dual-pass run is not ranked with a silently zeroed advantage.
- Rewrote the ablation summarizer, corrected dedicated-certificate paths and proposal
  metric names, and made the reported version explicit.
- Added signed fixed-route progression at scene level.  Waymax SDC routes are used
  when available; otherwise the already constructed logged-future route proxy is
  transformed once to global coordinates and its source is reported explicitly.
- Added 5% paired non-inferiority margins for jerk and yaw-rate; the Safe paper-ready
  flag now requires route progression, jerk and yaw-rate to be available and pass.
- Added authorization-checked v48.17 stress execution and exit-code separation:
  0 = valid Natural-gate pass, 20 = valid algorithmic rejection, 30 = engineering or
  artifact failure.
- Added four focused unit tests for simplex identity/bounds, calibrator capacity,
  evidence-stratified batching and signed route progression.

### Required v48.17 experiment and ablations

Main experiment: `run_v48_17_bridge_dedicated.sh`, with balanced on GPU0 and precision
on GPU1.

Component ablations compare against the already completed v48.16 `D_full_anchor`
baseline and therefore do not rerun old failed designs:

1. `A_simplex_scalar`: tri-simplex residual with the legacy four scalar inputs.
2. `B_context_simplex`: add frozen relative context, keep the old sampler/loss scope.
3. `C_full_bridge`: add evidence-stratified batches, batch/regime balance and the
   recall-constrained checkpoint metric.

### Decision and stopping rules

- Exit 0 and `NEXT_COMMANDS.txt` present: run authorized stress closed loop, rerun Safe
  paired evaluation with route progression, then perform multi-seed confirmation.
- Exit 20: do not inspect test/stress.  Use the three component ablations and explicitly
  labelled validation-only trajectory diagnostics to determine whether the remaining
  limitation is conditional Evidence capacity or irreducible positive support.
- Exit 30: no algorithm conclusion is allowed; repair the pipeline first.
- Do not relax the Natural-gate statistical constraints merely to create coverage.
- Do not retrain the proposal unless v48.17 shows that top-3 positive-hit rate itself
  degrades under the corrected protocol; current uploaded evidence supports freezing it.
- Do not rebuild the three regime datasets in this round.  Sparse positive support is
  addressed through sampler/loss/checkpoint logic so the next result remains
  attributable to the algorithm rather than a changed dataset.
