## v48.34.1 — RC30-MODEL-CONTRACT-AND-PROGRESS-HOTFIX (2026-08-03)

### Scope

This is an engineering-only hotfix for the uploaded v48.34 run. It does not change the BARRIER-CROSSFIT model, loss weights, proposal policy, dataset, gate thresholds, or algorithm interpretation.

### v48.34 RC=30 attribution

- Both Balanced and Precision adaptation jobs completed successfully (`adaptation_exit_codes=0/0`), produced final checkpoints, and passed stage-transfer integrity.
- The controller failed at `model_inference_contract` before certificate execution. The raw subprocess exit code was 2 and the normalized pipeline code was 30.
- The v48.34 runner passed `--expect-admission-prior-mode barrier_gated_slack` to the older `check_v48_32_model_contract.py`, whose argparse choices contained only `risk_centered`, `benefit_only`, and `safety_slack`. The checker rejected the argument before reading either checkpoint.
- Consequently `certificate_executed=false`, `gate_evaluated=false`, and no algorithm conclusion is permitted from this run.

### Pipeline fixes

1. Added version-specific `check_v48_34_model_contract.py`, including `barrier_gated_slack`, checkpoint/support SHA records, five-component reliability checks, bounded admission, slack temperature/penalty, and inference-contract verification.
2. Updated the v48.34 dedicated and ablation controllers to use the v48.34 checker. The old checker also accepts the new enum for backward-compatible diagnostics.
3. Added `repair_v48_34_rc30_model_contract_with_v48_34_1.sh`. It refuses every failure signature except the observed model-contract parser failure, verifies both final checkpoint hashes and stage-transfer metadata, reruns only model/training contracts, then resumes at certificate calibration without retraining.
4. Added a clean-run wrapper `run_v48_34_1_barrier_crossfit_dedicated.sh` for environments where the server-side v48.34 checkpoints are unavailable.
5. Repair status now distinguishes certificate-controller invocation, completed certificate execution, gate evaluation, raw certificate exit code, and normalized pipeline exit code.

### Exploratory closed-loop, baseline, and visualization fixes

1. Adaptation-dev Near and Contact now use split `evidence_adapt_dev` and the standard WOMD validation source. Contact can no longer silently use `validation_interactive` or a test split.
2. Dataset preflight validates target split, source role, official scenario IDs/source indices, raw TFRecord resolution, and any explicit `@N` scan limit before Waymax execution.
3. OC-RAP Near/Contact Balanced and Precision runs execute concurrently on two GPUs. External methods execute two at a time. Physical comparison defaults to `label_mode=fast` with zero online teacher labels; expensive selected-topk teacher audits are not used for progress-only closed loop.
4. All methods must run on the exact same target-key set. Paired reports fail closed on missing, duplicate, or mismatched scene-time targets and report absolute means, raw deltas versus scalar control, oriented deltas, and paired bootstrap intervals.
5. Safe, Near, and Contact each receive compact presentation CSV/Markdown tables plus full-metric CSV tables. The scalar control is an explicit table row rather than only an implicit delta reference.
6. Critical-scene selection requires complete regime-critical metrics, an actual intervention, positive composite physical change, and no new overlap/offroad/re-contact for positive examples. Failure examples are also exported and cannot duplicate positives.
7. Video output uses unique scene-time filenames, common paired bounds, SDC trails, selection reason/continuous metrics, MP4 `veryfast` encoding when ffmpeg is available, GIF fallback, and complete JSON/CSV provenance indices.
8. Held-out test exploration requires an explicit contamination flag and writes a permanent disclosure that outputs are exploratory only and cannot be used for future checkpoint, threshold, or algorithm selection.

### Runtime implications

- Reusing the completed v48.34 checkpoints avoids repeating approximately 29–33 minutes of parallel adaptation wall time (about 62 GPU-minutes combined in the uploaded run).
- Removing online OC-MERO teacher labeling from progress-only closed loop avoids the previously observed dominant audit-label cost while preserving Waymax physical rollout metrics.
- Running the two OC-RAP variants and pairs of external methods concurrently uses the available two-GPU server without changing scene sets or metric definitions.
- Videos rerun only the auditable selected scene-time subset with render traces rather than rendering every exploratory rollout.

### Decision rule

- Repair/clean run `RC=30`: stop; do not analyze the algorithm or run ablations/shadow/test/stress.
- `RC=20`: the pipeline and certificate are valid; return the complete result for algorithm analysis. Optional same-target closed-loop/video outputs remain progress-only.
- `RC=0`: execute only generated `NEXT_COMMANDS.txt` for formal downstream evaluation.

## v48.34 — BARRIER-CROSSFIT (2026-08-03)

### v48.33 result attribution

- The uploaded v48.33 main pipeline is a valid algorithmic rejection: `pipeline_exit_code=20`, `certificate_exit_code=20`, `certificate_executed=true`, `gate_evaluated=true`, `gate_passed=false`, `pipeline_valid=true`, and `test_roots_read=false`. The operational rejection remains `development_rule_fit_rejection`.
- Unified top-5 proposal support is not the blocker. Adaptation-dev/certificate contain 8/9 Near and 17/20 Contact proposal-contained safe opportunities, and the proposal-constrained oracle is feasible.
- Near improved on adaptation-dev but not on the scene-disjoint certificate. Precision Near candidate safe-positive AUC rose to approximately `0.919` and the legacy evidence-only proposal correlation rose from approximately `-0.011` to `+0.249`; the closest development rule selected 13 actions, 4 safe positives and 1 harmful action with safe recall `0.50` and mean teacher advantage `+0.151`. On certificate it selected 14 actions, 0 safe positives and 7 harmful actions with mean advantage `-0.263`. Balanced Near similarly selected 2 safe positives on development but 0 on certificate.
- Contact action identity remains unresolved. Balanced/Precision certificate candidate safe-positive AUC is approximately `0.570/0.557`, proposal evidence correlation is `-0.205/-0.115`, safe-positive selections are `1/1`, harmful selections are `17/19`, and mean selected teacher advantage is `-0.204/-0.159`.
- v48.33 therefore learned a development-local Near ordering signal but not a transferable action-level physical ordering. Contact remains close to candidate-level random discrimination and still assigns excessive evidence to apparently beneficial but physically unsafe actions.
- All eight uploaded v48.33 ablations are invalid for algorithm comparison. They exited pipeline `RC=30` before identity training because Stage-2 settings and inconsistent defaults were included in the Stage-1 factor-cache identity. No C/D/A/B performance attribution is made from those runs.
- The uploaded development shadow is exploratory and contains only eight paired scenes per variant/regime. Near produces approximately `+0.011 s` TTC-p05 and `+0.015 s` terminal-TTC changes but decreases bounded NUP by approximately `0.014`; Contact produces millimetre-scale clearance/free-space changes and small TTC-recovery changes while also decreasing bounded NUP. These results do not establish submission-level closed-loop superiority.

### Root cause

1. **Soft improvement did not cross executable safety boundaries.** Precision selected epoch 12 for a small improvement in threshold-free soft risk even though valid-safe admissions remained zero and the maximum invalid-admission rate remained one. Balanced training moved from one valid-safe admission at epoch 0 to zero while soft recall increased. The scalar checkpoint objective rewarded probability mass shifts that never produced a deployable action.
2. **Unsafe recovery evidence remained compensatory.** The v48.33 admission residual could still overcome an unfavourable learned safety slack. High raw recoverability evidence therefore remained able to dominate even when one supported physical component predicted boundary violation.
3. **Development-local scene shortcuts dominated action identity.** Near candidate AUC and development correlation improved, but certificate safe hits remained zero. The selector learned opportunity/context correlations concentrated in a few scenes rather than invariant candidate-vs-nominal causal differences.
4. **Contact representation is still underidentified.** Candidate safe-positive AUC remains near random and proposal correlation remains negative. Threshold calibration cannot repair a representation that assigns the wrong sign or relative order to action-level safety.
5. **Legacy diagnostics obscured the exact failure location.** v48.33 reported an evidence-only top-1 diagnostic that ignored eligibility. The Natural gate used the correct eligible-set policy, so RC=20 is valid, but the diagnostic could not distinguish an eligibility-head failure from a ranking-after-filter failure.

### Engineering and protocol corrections

1. **Stage-1 cache boundary is exact.** Factor-cache identity contains only Stage-1 inputs and hyperparameters. Stage-2 prior, boundary and checkpoint settings no longer invalidate a reusable factor checkpoint. Reuse validates source and copied checkpoint hashes and rewrites run-local metadata.
2. **Exact and legacy policy diagnostics coexist.** Calibration emits both evidence-only top-1 and exact `rank top-k -> eligibility -> evidence rerank -> one action or nominal` metrics, plus proposal candidate rows. This separates unsafe-filter errors from eligible-set ranking errors.
3. **Hard-policy checkpoint metadata is authoritative.** Best-checkpoint selection records the complete lexicographic key, actual validation loss and scalar audit metric separately; it no longer stores a tuple element as `best_val_loss`.
4. **No all-abstain preference.** Lexicographic ordering first minimizes regimes with zero valid-safe admissions, then maximizes total valid-safe admissions and cross-scene fold-min safe top-1 recall before minimizing invalid admission and regret.
5. **Cross-scene fold validation is mandatory.** Scene-fold minimum safe top-1 recall participates in checkpoint ordering, reducing selection of epochs that concentrate all apparent success in one or two development scenes.
6. **Exploratory data scope is fail-closed.** Adaptation-dev closed-loop can no longer silently default to held-out test roots. Held-out test inspection requires explicit authorization and writes a permanent contamination declaration. Exact target roots, WOMD sources and target-contract hashes are recorded.
7. **Ablation status is unambiguous.** Every task records both pipeline and certificate exit codes, failure stage and cache identity. An engineering `RC=30` cannot be interpreted as an algorithmic `RC=20`.
8. **Critical-scene videos are auditable.** Selection scores every paired scene and exports both positive and failure examples. Videos cannot be generated from an unpaired or cherry-picked scene list.

### v48.34 unified algorithm

1. **Barrier-gated safety slack.** Let `m_max(a)` be the maximum supported learned candidate-vs-nominal safety margin. A continuous safety gate `g(a)=sigmoid(-m_max/tau)` attenuates both raw benefit and the learnable admission residual, while a softplus barrier penalizes positive safety slack. Unsafe evidence can no longer be fully compensated by a large residual. The same equation is used for Safe, Near and Contact; no regime ID or case routing is introduced.
2. **Eligibility-boundary continuation.** In addition to the eligible-set KL, safe-positive candidates are pushed beyond opportunity, harm and admission boundaries by a registered margin; harmful candidates are pushed below harm/admission boundaries; dead candidates receive a weak nominal preference. This directly optimizes executable transitions rather than only soft probabilities.
3. **Hard-first lexicographic checkpointing.** Checkpoints are selected by valid-safe admissions and cross-scene safe top-1 coverage before invalid-admission rate, safe regret and soft population risk. Soft risk is only a tie-breaker once executable behaviour is comparable.
4. **Two-stage natural-population training.** Stage 1 learns raw benefit and signed physical margins with no replacement. Stage 2 jointly trains benefit/opportunity, supported harm components and admission under barrier-gated eligible-set supervision. Adaptive teacher-gap margin and Stage 3 remain disabled.
5. **Unified top-5 and independent measured veto are retained.** Proposal generation, measured hard veto, exact eligibility, bounded one-action policy, scene-disjoint certificate and sealed test/stress roots are unchanged.

### Exploratory closed-loop and external baselines

- External baseline results uploaded with v48.33 are not directly numerically comparable to OC-RAP because they were not evaluated on the same target scene set; the observed Near/Contact scene overlap with the existing OC-RAP shadow is zero.
- v48.34 provides a same-target paired runner for OC-RAP, scalar control and external baselines. It validates identical scene IDs before paired bootstrap reporting.
- After `RC=20`, adaptation-dev exploratory closed-loop is allowed only with an explicit diagnostic flag and cannot be used to tune thresholds/checkpoints. Held-out test evaluation additionally requires an explicit contamination flag and permanently disqualifies those scenes from future model selection.
- The critical-scene pipeline records render traces and produces side-by-side Control/OC-RAP MP4 and GIF files for both positive and failure cases. These are toy examples, not substitutes for aggregate paired metrics.

### Required next decision

- Run only the v48.34 main experiment first.
- `RC=30`: stop and inspect the structured failure; do not run ablation, shadow, test, stress or exploratory comparison.
- `RC=20`: run the authorized v48.34 ablations and adaptation-dev shadow. Exploratory same-target baseline comparison/video generation is optional and must retain its disclosure. Do not use held-out results for v48.35 design.
- `RC=0`: execute only generated `NEXT_COMMANDS.txt` for formal test/stress. Exploratory tooling may still be used for visualization, but formal and exploratory outputs must remain separate.
- No claim is made in advance that v48.34 will pass. The decisive evidence is whether valid-safe admissions become nonzero in both target regimes, Near certificate safe hits replace development-only gains, and Contact exact-eligible harmful switches/negative advantage fall without regime routing.

## v48.33 — ELIGIBLE-SET-POLICY (2026-08-02)

### v48.32.1 result attribution

- The uploaded v48.32.1 main pipeline completed adaptation and the dedicated certificate controller. Its observed controller result is `RC=20`, with `certificate_executed=true`, `gate_evaluated=true`, `gate_passed=false`, and `test_roots_read=false`. The operational rejection label is `development_rule_fit_rejection`.
- The run is not an RC=30 engineering crash. It provides valid evidence that the frozen v48.32.1 selector did not satisfy the Natural gate under the rule that was actually fitted.
- A protocol audit found that the development-rule command substituted the looser verification thresholds for the preregistered fit thresholds. Therefore the rejection remains real, but the reported fit deficits, proposal-oracle feasibility, and the unconditional `pipeline_valid=true` claim are not formal evidence under the declared preregistration.
- The intended strict fit thresholds are Near `min_selected=10`, precision LCB `>=0.50`, harmful-group UCB `<=0.12`, selected-harm UCB `<=0.22`; Contact `16`, `0.50`, `0.14`, `0.22`. v48.32.1 instead fitted with Near `8`, `0.40`, `0.14`, `0.25` and Contact `10`, `0.40`, `0.16`, `0.25`.
- Under the rule actually used, the best Precision Near adaptation-dev frontier selected 10 actions with 3 safe positives, 1 harmful action, precision LCB90 `0.154`, safe recall `0.375`, mean teacher advantage `+0.191`, and macro share `0.60`. On certificate it selected 8 actions with 0 safe positives, 4 harmful actions, mean advantage `-0.298`, and macro share `0.875`.
- Balanced Near retained strong candidate discrimination (`candidate_safe_positive_auc≈0.831`) but proposal evidence correlation was approximately `-0.014`; certificate selected no action. Precision Near had `candidate_safe_positive_auc≈0.796`, proposal evidence correlation approximately `-0.011`, and also no certificate safe-positive hit.
- Contact remains weaker at the candidate level and worse at proposal identity. Balanced/Precision certificate candidate safe-positive AUC was approximately `0.581/0.553`, proposal evidence correlation approximately `-0.136/-0.167`, selected safe positives were `0/0`, harmful selections were `5/15`, and mean selected teacher advantage was `-0.134/-0.230`.
- The uploaded v48.32 ablations all returned RC=20. Joint detached updates worsened Balanced Contact relative to admission-only; coupling reduced that degradation but suppressed Near activity. The adaptive teacher-gap margin produced no measurable difference from the fixed-margin coupled configuration. Precision A/B/C/D were effectively identical, indicating repeated epoch-0/no-op selection. Stage 3 provided no demonstrated benefit and is removed from the default main path.

### Root cause

1. **Scene opportunity is learned, action identity is not.** Near candidate AUC is useful, yet candidate-vs-nominal evidence ordering inside the frozen proposal is approximately uncorrelated with teacher utility. Contact candidate safety discrimination is close to random and proposal correlation is negative.
2. **Training and deployment selected actions in different order.** v48.32.1 checkpoint metrics selected the largest evidence score inside rank top-k and only then checked opportunity/harm. Calibration and runtime first filter proposal members by opportunity/harm and then rerank the eligible set. A deployable runner-up could therefore receive no checkpoint credit.
3. **Soft early stopping also ignored eligibility.** The threshold-free population risk assigned categorical mass from evidence alone, rewarding high-evidence candidates that the deployed policy would reject.
4. **Top-3 is structurally insufficient for the strict Near fit contract.** Adaptation-dev contains only seven top-3 proposal-contained Near safe opportunities. With strict `min_selected=10`, even an optimistic 7/10 precision has one-sided 90% Wilson LCB below 0.50. Unified top-5 contains all eight Near safe opportunities and makes the strict optimistic support bound feasible. Contact support is already saturated by top-5.
5. **Additional admission-only calibration cannot repair representation.** The former Stage 3 repeatedly selected epoch 0 and did not change certificate outcomes.

### Engineering and protocol corrections

1. **Preregistered fit thresholds are passed exactly.** Verification thresholds are no longer reused as fit thresholds.
2. **Fail closed before certificate access.** The new metric/calibration identity checker validates exact dev group counts, proposal safe-opportunity counts, strict fit thresholds, proposal top-k, evidence-rerank flag, selection order, and strict proposal-oracle feasibility.
3. **Exact hard policy order in checkpoint metrics.** Validation now executes `rank top-k -> opportunity/harm filter -> evidence rerank -> one action or nominal`.
4. **Exact soft policy order in early stopping.** Soft categorical checkpoint mass includes differentiable opportunity/harm eligibility before evidence, matching the new training objective and runtime ordering.
5. **Unified top-5 contract.** Top-k is fixed to five across factor training, identity training, checkpoint metrics, dev rule fitting, certificate verification, runtime policy metadata, cache identity, and audit tools. No regime-specific top-k is introduced.
6. **Selection semantics are explicit metadata.** `SELECTION_SEMANTICS=rank_topk_then_filter_then_evidence_rerank` is checked in `POLICY_CONTRACT.env` and `GATE_SPEC.json`.
7. **Ablations are authorization-gated.** The v48.33 eight-task suite runs only after a valid main pipeline with certificate RC=20. It reuses the exact top-5 Stage-1 factors and never reads test/stress roots.
8. **Ineffective Stage 3 is disabled by default.** The identity checkpoint is copied atomically into the final run with stage-transfer integrity checks.
9. **Known safety contracts are retained.** Natural no-replacement training, exact physical eligibility, support reliability, independent measured hard veto, bounded admission, and test-root sealing remain unchanged.

### v48.33 unified algorithm

1. **Eligible-set policy objective.** Inside the frozen unified top-5 proposal, the student categorical score is admission evidence plus continuous log soft-eligibility from opportunity and harm heads; nominal is an explicit abstention class. The teacher distribution is continuous safe utility. This gives deployable runner-up actions gradient instead of rewarding an ineligible evidence top-1.
2. **Multi-head identity coupling.** Stage 2 jointly updates compact benefit, supported physical-margin harm, and admission calibrators. The eligible-set KL propagates finite gradients through all three heads. No Safe/Near/Contact ID or case-specific strategy is used.
3. **Fixed hardest-negative margin.** The adaptive teacher-gap scale is disabled because C/D ablations were indistinguishable. Hardest-negative supervision remains as a simpler proposal-local separation term.
4. **Two-stage default.** Stage 1 learns raw benefit and signed continuous physical margins on the natural population. Stage 2 learns the exact eligible-set one-action policy. Admission-only Stage 3 is disabled unless a future ablation provides evidence for it.
5. **Strict checkpoint contract.** Loss, soft early stopping, hard validation counters, dev threshold fitting, certificate verification, and runtime now share the same proposal/filter/rerank semantics.

### Required next decision

- Run only the v48.33 main experiment first.
- `RC=30`: stop; no algorithm conclusion, ablation, shadow, test or stress.
- `RC=20`: the corrected strict Natural gate was evaluated; run the authorized v48.33 ablations and adaptation-dev physical shadow only.
- `RC=0`: execute only the generated `NEXT_COMMANDS.txt`.
- No claim is made in advance that v48.33 will pass. The decisive evidence is whether top-5 plus eligible-set training converts Near candidate AUC into certificate safe top-1 hits and whether Contact harmful selections/negative advantage fall materially without regime routing.

## v48.32.1 — RC30-INTEGRITY-HOTFIX (2026-08-02)

### v48.32 result attribution

- The uploaded v48.32 main controller returned a genuine pipeline `RC=30`: Balanced and Precision adaptation both exited 1, `failure_stage=adaptation`, `certificate_exit_code=null`, `gate_evaluated=false`, `pipeline_valid=false`, and `test_roots_read=false`.
- This run is not `development_rule_fit_rejection` and provides no valid evidence about the v48.32 algorithm. No ablation, physical shadow, test or stress conclusion is authorized.
- Both variants completed Stage-1 factor training, then failed during Stage-2 epoch-0 validation with the same `IndexError: too many indices for tensor of dimension 0` at the factorized component-veto call.
- The deterministic root cause is Python variable shadowing. The candidate-level vector `teacher_gap` was overwritten inside the group loop by the scalar adaptive teacher-utility gap. After the first safe-positive group, a later group attempted `teacher_gap[recs]` on a zero-dimensional scalar.

### Engineering hotfixes

1. **Separate tensor/scalar identities.** The population vector is `teacher_gap_vector`; the per-group scalar is `adaptive_teacher_gap`. The algorithm and margin formula are unchanged.
2. **Exact multi-group preflight.** Before index construction or GPU training, a two-group synthetic contract exercises factorized harm, adaptive hardest-negative, forward, backward and finite-gradient checks.
3. **Static group-loop shadowing guard.** The preflight inspects the exact loss function AST and rejects assignments that overwrite outer tensors inside the scene-time group loop.
4. **Strict shape contract.** Main training no longer uses `n=min(sizes)` to silently truncate mismatched model, teacher or metadata tensors. It fails closed and requires exactly one nominal per group.
5. **Deterministic CUDA contract.** All v48.32.1 entry points set `CUBLAS_WORKSPACE_CONFIG=:4096:8`; deterministic CUDA LCVaR avoids the nondeterministic `cumsum` path by using an exact lower-triangular prefix operator.
6. **Stage-aware failures.** Variant failures record the active stage, shell command and return code. The controller adds a parsed exception type, message, Python frame and bounded log tail.
7. **Exact Stage-1 materialization.** Reuse verifies source checkpoint SHA against both completion files, atomically copies the stage, verifies the copied SHA, and rewrites completion metadata to the new checkpoint path.
8. **Variant-specific reuse.** Balanced and Precision may independently reuse their successfully completed v48.32 factor stages after source/index/support/hyperparameter contract verification.
9. **Correct certificate semantics.** `certificate_executed`, `gate_evaluated`, nullable `certificate_exit_code`, and `pipeline_exit_code` are recorded separately. Certificate artifact/protocol RC=30 no longer claims that the Natural gate was evaluated.
10. **Explicit authorization state.** RC=0 requires `NEXT_COMMANDS.txt` plus generated status; RC=20/30 requires blocked status. Manual authorization remains prohibited.
11. **Complete certificate dependency closure.** The v48.32.1 validation/calibration population-identity checker is packaged under the exact name used by the certificate controller and covered by a release dependency audit.

### Runtime effect

- The failed v48.32 run already spent 1175.16 seconds on Balanced Stage-1 and 1147.04 seconds on Precision Stage-1.
- If the original server run still contains both factor checkpoints, v48.32.1 can resume from Stage-2 after exact cache verification, avoiding approximately 38.7 GPU-minutes of repeated factor training and roughly 19.6 minutes of concurrent wall time.
- Teacher indexes may be copied to the new output directory and are reused only after exact dataset/label contract validation.

### Decision rules

- `RC=30`: no algorithm conclusion and no ablation/shadow/test/stress. Inspect the multigroup preflight, pipeline failure, variant stage marker and exception signature.
- `RC=20`: pipeline and certificate are valid; only then is a Natural-gate rejection established.
- `RC=0`: execute only generated `NEXT_COMMANDS.txt`.

### Local validation boundary

- Exact two-group loss preflight: passed with finite loss and non-zero admission gradient.
- `PYTHONPATH="$PWD/src" pytest -q`: 294 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- Shell syntax and new-script dependency closure: passed.
- Real WOMD/Waymax, the server-side factor `.pt` files, and two A30s are unavailable locally; no gate result is claimed in advance.

## v48.32 — OC-TRAC-IDENTITY-UTILITY-BRIDGE (2026-08-02)

### v48.31 result attribution

- The uploaded v48.31 main controller did **not** return a valid Natural-gate result. Balanced adaptation exited 0, Precision exited 31, the controller normalized this to `RC=30`, `failure_stage=adaptation`, `raw_certificate_exit_code=null`, `gate_evaluated=false`, and `pipeline_valid=false`. The main run is therefore an engineering failure before certificate access, not `development_rule_fit_rejection` and not RC=0.
- `NEXT_COMMANDS.txt` was absent because v48.31 emitted it only after a valid certificate RC=0. The controller exited before certificate evaluation. The underlying Precision failure was a false stage-transfer rejection: Stage 3 legally selected its epoch-0 input checkpoint, changed zero allowed and zero disallowed parameters, but the checker treated the no-op fail-safe as corruption.
- The eight-task ablation suite was incomplete. A/B Balanced and Precision copied a no-joint checkpoint without `TRAINING_COMPLETE.json` and `EVIDENCE_CORRECTION_COMPLETE.json`; C/D Precision hit the same epoch-0 false rejection. Only C/D Balanced reached certificate, and both returned a valid `RC=20` with `development_rule_fit_rejection`.
- The v48.31 adaptation-dev physical shadow did not produce physical evidence. Both variants exited 2 because the invalid main pipeline never produced calibration gamma. No v48.31 TTC, clearance, free-space, re-contact, stable-stop or intervention conclusion is claimed.
- Valid C/D Balanced evidence confirms that proposal support is not the blocker: the certificate contains 9 Near and 20 Contact proposal-contained safe-positive groups. The learned selector still chooses no safe positives.
- Near retains strong candidate diagnostics but weak deployable identity. D Balanced development safe-positive AUC is approximately 0.902 and proposal safe-top-1 AUC approximately 0.917, yet the closest rule selects only 1 safe positive in 4 actions, precision LCB90 approximately 0.078 and recall 0.125. On certificate it abstains entirely; C selects 3 actions, 0 safe positives and 2 harmful actions.
- Contact remains substantially below submission maturity. D Balanced certificate safe-positive AUC is approximately 0.580, group top-1 correlation approximately -0.170, safe-top-1 AUC approximately 0.450, and 17 selections contain 0 safe positives and 5 harmful actions with mean teacher advantage approximately -0.141.
- v48.31 support reliability is not a side-effect-free performance gain. It suppresses Near harmful selection by abstaining, but Contact harmful selections increase from 3 to 5 and mean selected advantage becomes more negative. It remains a support-safety contract, not a proven ranking module.

### Confirmed engineering corrections

1. **Correct no-op checkpoint semantics.** Stage-2 or Stage-3 selection of the evaluated initial checkpoint is valid when no frozen parameter changed and no key disappeared. It no longer produces a false RC=30.
2. **Complete no-final metadata.** Disabled-final ablations copy checkpoint, architecture, policy, training-complete and evidence-complete artifacts.
3. **Explicit NEXT state.** RC=0 writes `NEXT_COMMANDS.txt` plus generated status; RC=20/30 writes `NEXT_COMMANDS_BLOCKED.json` plus status. The controller verifies RC/file consistency.
4. **Unambiguous controller status.** `pipeline_exit_code` is separate from nullable `certificate_exit_code`; a pre-certificate failure never masquerades as a certificate return.
5. **Run-local status hygiene.** Every controller start clears stale adaptation, gate, calibration, NEXT and completion markers. Successful variant reruns delete stale failure markers.
6. **Complete task failure materialization.** Ablation root failures and every task-stage failure write structured JSON and log tails instead of disappearing under `set -e`.
7. **Fail-closed shadow preflight.** Waymax is not launched unless pipeline, certificate status, checkpoint, gamma, target, provenance, runtime and paired-scene contracts are valid.
8. **Exact factor-cache identity.** Cache reuse requires matching source-checkpoint SHA, train/dev index SHAs, support-contract SHA, variant and all relevant factor hyperparameters. Paths may differ only when file contents are identical.
9. **Audited teacher-index reuse.** Train and adaptation-dev indexes are reused only after exact dataset/label contract checks; otherwise they are rebuilt automatically.
10. **Correct script dependency closure.** Every referenced v48.32 Python and shell tool exists and is covered by shell-to-tool dependency audit.

### Algorithm diagnosis

- v48.31 Stage 2 trained only the admission calibrator while freezing compact benefit and component-harm calibrators. The deployed safety-slack prior also detached benefit and component logits. Safe-utility, listwise and hardest-negative gradients therefore could not correct the action-identity errors that caused a harmful action to outrank a safe action within the same proposal.
- This explains the observed candidate-AUC/top-1 gap: candidate classification can improve while exact proposal top-1 correlation remains near zero or negative and valid-safe admission collapses to zero.
- Contact additionally has weak candidate representation, so threshold fitting or additional binary admission loss cannot repair it. The next change must couple group-local safe utility to benefit and supported physical margins without regime routing.

### v48.32 unified algorithm

1. **One continuous selector across all regimes.** No Safe/Near/Contact ID or regime-specific branch is added. Reporting strata remain external to the model.
2. **Support-weighted safety slack.** Keep the global nominal-relative coordinates and independent measured hard veto. The deployed utility remains `B(a) - lambda * relu(max_k r_k m_k(a))`.
3. **Proposal-local identity stage.** Stage 2 jointly trains the compact benefit, component-harm and admission calibrators by default.
4. **Deployment-utility gradient bridge.** Stage-2 admission prior can be non-detached, allowing safe-utility/listwise/hard-negative loss to update benefit and supported component margins. This changes gradient flow only, not inference semantics.
5. **Adaptive teacher-gap margin.** Hard-negative separation is `base_margin + scale * clamp(teacher_safe_utility_gap, 0, 0.25)`; no-safe groups receive a continuous no-op-depth margin. The same formula is used across all regimes.
6. **Admission-only final calibration.** Stage 3 adjusts only the bounded admission residual with low learning rate; epoch zero remains a legal fail-safe.
7. **Retain natural population and exact checkpoint contract.** All stages use natural groups without replacement, exact executable eligibility and safe-top-1 checkpoint barriers.

### v48.32 ablations

1. `A_admission_only_detached_fixed_margin`
2. `B_joint_identity_detached_fixed_margin`
3. `C_joint_identity_coupled_fixed_margin`
4. `D_full_identity_utility_bridge`

B>A tests compact joint identity learning; C>B tests deployment-utility gradient coupling; D>C tests the adaptive continuous teacher-gap margin. Support reliability is retained in all groups because it is an already-required unsupported-coordinate safety contract.

### Runtime changes

- v48.31 repeated Stage-1 factor training eight times in ablations, consuming approximately 11,302 seconds (3.14 hours) and about 65.3% of effective ablation training time.
- Standalone v48.32 ablations train only one factor stage per variant, saving approximately 2.38 hours.
- The recommended post-main run reuses the two exact-contract main factor stages, eliminating all additional factor training and saving approximately 3.14 hours relative to v48.31.
- Certificate populations and Waymax rollout horizons are not reduced. Speed is obtained only by exact-contract reuse and index caching.

### Decision rules

- `RC=0`: `NEXT_COMMANDS.txt` and generated status must both exist; execute only those commands.
- `RC=20`: pipeline and certificate are valid but the Natural gate rejected the selector. Do not read test/stress; run v48.32 ablations and adaptation-dev physical shadow.
- `RC=30`: make no algorithm conclusion. Inspect `PIPELINE_FAILED.json`, `NEXT_COMMANDS_BLOCKED.json` and the named contract failure.
- Do not lower the registered gate, expand to top-8, split regimes into separate policies or manually create authorization.

### Local validation boundary

- `PYTHONPATH="$PWD/src" pytest -q`: 285 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for every shell script: passed.
- v48.32 shell-to-tool dependency audit: passed with no missing references.
- Real WOMD/Waymax and two A30s are unavailable in this audit environment; no v48.32 gate or physical result is claimed in advance.

## v48.31 — OC-TRAC-CONTRACT-SLACK-RANK (2026-08-02)

### v48.30 result attribution

- The uploaded v48.30 controller produced a valid `RC=20`: the pipeline completed, the Natural gate was evaluated, test/stress roots were not read, and all four branches were rejected during adaptation-development rule fitting.
- Proposal support is not the blocker. The frozen top-3 proposal contains 9 Near and 20 Contact safe-positive certificate groups, with optimistic oracle precision LCBs of approximately 0.846 and 0.924.
- Learned action identity is the blocker. Near proposal safe-positive AUC is approximately 0.735/0.784 for Balanced/Precision, yet proposal evidence top-1 correlation is approximately +0.021/-0.023 and both certificate branches select zero safe positives. Contact correlation is approximately -0.118/-0.116 and both certificate branches also select zero safe positives.
- Precision Near retains the strongest local signal: the closest development rule selects 3 safe positives in 7 interventions, safe recall 0.375 and mean teacher advantage +0.273. It does not generalize to the certificate, where 3 of 6 selections are harmful and mean advantage is -0.294.
- Contact has not established safe admission. Balanced/Precision select 17/28 certificate actions, zero safe positives, 6/16 harmful actions and negative mean teacher advantage.
- The Safe recovery threshold calibration is valid, but no scene-disjoint Safe policy certificate is registered. Safe therefore still requires non-empty paired non-inferiority evidence.
- The uploaded v48.30 development-shadow controller did not run simulation. It stopped before Waymax because `audit_v48_30_shadow_provenance.py` was missing; two additional referenced v48.30 checker files were also absent. No v48.30 closed-loop physical conclusion is claimed from this package.

### Confirmed engineering corrections

1. **Exact executable validation population.** Training validation now applies the same eligibility contract as calibration: supported macro, feasible candidate, measured hard rule no greater than the registered maximum, and candidate-vs-nominal prefix deviation above the registered minimum.
2. **Correct all-abstain semantics.** When the safe contract is available, abstention is defined by zero valid safe admissions. Harmful or invalid switches can no longer make a checkpoint appear executable.
3. **Safe top-1 checkpoint barrier.** `direct_contract_safe_rank_risk` lexicographically penalizes any reporting stratum with zero proposal-contained safe top-1 hits, safe top-1 recall shortfall, invalid admission, valid-safe abstention and safe top-1 regret.
4. **Natural population in every optimization stage.** Factor training, admission training and joint refinement all use natural, without-replacement scene-time groups. v48.30 applied this repair only to the admission stage while freezing the factor heads learned under replacement sampling.
5. **Metric/calibration population audit.** Before certificate verification, the selected checkpoint's exact-eligible group counts and proposal-contained safe-opportunity counts must exactly match adaptation-dev calibration.
6. **Fail-closed model/inference audit.** Component count, scale, frontier prior, bounded admission, slack temperature/penalty and support reliability must be identical in checkpoint training and inference.
7. **Stage-transfer audit.** Stage 2 must not alter the frozen benefit/harm factors. Stage 3 may change only the three registered evidence calibrators.
8. **Repaired development shadow.** Add v48.31 provenance, runtime, physical-support and regime-target checkers; require non-empty paired scenes and valid metric semantics; all referenced tools are covered by a focused regression test.

### Algorithm corrections

1. **Global support-conditioned continuous slack.** Keep one nominal-relative continuous physical representation across Safe, Near and Contact, with no regime ID or regime-specific policy. Each learned component is shrunk toward its semantic non-harm prior according to global data support.
2. **Do not learn unsupported veto coordinates.** The current training index supports DRS, deployability and gap margins, but `harm_proxy` is constant and the exact-eligible learned hard-rule coordinate has no positive examples. The default support contract is therefore `1,1,1,0,0`. The independent measured hard veto remains active and uncompensated.
3. **Support-weighted factor supervision.** Component BCE and signed-margin regression are weighted by the same global reliability used at inference, eliminating train/runtime disagreement.
4. **Three-stage optimization.** Stage 1 learns raw benefit and supported physical factors; Stage 2 learns bounded admission with proposal-level safe utility, listwise ranking and hardest negatives; Stage 3 performs low-learning-rate joint refinement of benefit, harm and admission calibrators only.
5. **Action identity over scene classification.** Candidate AUC is retained only as a diagnostic. Checkpoint selection is driven by proposal-contained safe top-1 support, safe admission precision/recall, harmful mass and regret.
6. **Preserve the frozen top-3 proposal.** Oracle support remains feasible. Expanding to top-8 would increase calibration burden and harmful exposure without addressing the observed identity failure.
7. **Preserve the unified paper semantics.** Near/Contact are reporting strata only. The appendix's legacy regime-conditioned protective-certificate text should be replaced before submission with the support-conditioned unified margin contract.

### v48.31 ablations

Four waves, at most two concurrent tasks, one Balanced job on GPU0 and one Precision job on GPU1:

1. `A_contract_natural_no_reliability_no_joint`
2. `B_add_support_reliability_no_joint`
3. `C_add_joint_refinement_no_reliability`
4. `D_full_contract_slack_rank`

B>A isolates global support reliability. C>A isolates low-rate joint refinement. D>max(B,C) demonstrates complementarity. Development improvement without certificate improvement indicates nuisance/macro generalization failure; offline improvement without paired physical improvement indicates teacher/closed-loop mismatch.

### Decision rules

- `RC=0`: execute only the generated authorization commands; then run held-out stress and publication checks.
- `RC=20`: do not read test/stress. Run the repaired adaptation-dev shadow and four-wave ablations. First inspect exact-contract audits, safe top-1 hits, harmful selection and macro concentration.
- `RC=30`: make no algorithm conclusion. Inspect protocol, support, model/inference, stage-transfer, metric/calibration and artifact failures.
- Do not lower the registered Natural gate after observing certificate results.

### Local validation boundary

- Full pytest suite: passed after adding the v48.31 focused tests.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for every shell script: passed.
- v48.31 shell-to-tool reference audit: passed.
- Real WOMD/Waymax and two A30s are unavailable in this audit environment; no v48.31 gate or closed-loop result is claimed in advance.

## v48.27 — OC-TRAC-FACTOR-PHYSICS-BRIDGE (2026-08-01)

### v48.26 result attribution

- The v48.26 controller returned a valid `RC=20`: `pipeline_valid=true`, `gate_evaluated=true`, `gate_passed=false`, and `test_roots_read=false`. This is not a pipeline failure.
- Proposal-constrained top-3 oracle support is feasible on both adaptation-dev and the complete certificate population. The certificate contains 9 Near and 20 Contact proposal-contained safe-positive groups; oracle precision LCB is approximately 0.846 and 0.924. The current failure is therefore not candidate support or a mathematically impossible certificate contract.
- All four adaptation-dev threshold-fit jobs failed to produce a rule satisfying the joint selected-count, safe-positive precision/recall and harmful-selection constraints. v48.26 subsequently verified a `diagnostic_fit_rule` with `source_rule_satisfied_dev_constraints=false`. The correct rejection class is `development_rule_fit_rejection`, not a generic learned-gate failure.
- The v48.26 development shadow did not produce physical evidence. Each Near/Contact audit loaded 16 offline targets but scanned only the first 900 `validation_interactive` scenarios, matched zero targets and emitted zero-scene JSON. Paired comparison then failed. Existing shadow zeros/nulls cannot be interpreted as no collision, no re-contact or good intervention behavior.
- Near Precision retains useful local evidence (safe-positive AUC about 0.828, high-opportunity conditional-harm AUC about 0.842, false-switch about 0.085), but selects zero safe-positive certificate actions. Near Balanced selects 12 actions with zero safe positives and nine harmful selections.
- Contact does not establish safe admission. Balanced/Precision select 50/46 actions, only one safe positive each, approximately 44%/46% harmful, recall 0.05 and negative mean selected teacher advantage.
- The A/B ablations contain a small but important viable region: Balanced Near selects 3 safe positives among 5 actions with empirical precision 0.60 and mean teacher advantage +0.251. C/D safe-utility/full objectives destroy that region, supporting an objective-scale and gradient-interference diagnosis rather than a no-signal diagnosis.

### Engineering corrections

1. Remove the fixed development-shadow scan cap. `DEV_SHADOW_RAW_MAX_SCENARIOS=0` now scans the complete raw source for sparse target IDs.
2. Add `DEV_SHADOW_WOMD_SOURCE` so the exact raw WOMD source used by the offline bucket can be supplied explicitly.
3. Canonicalize target and raw scenario IDs by preferring `original_scenario_id` and removing only the operational `__wx########` loader suffix.
4. Set `closed_loop.require_bucket_targets=true`; fail immediately when the target manifest is empty or no target matches after the scan.
5. Mark empty closed-loop aggregates with `metrics_valid=false` and `empty_reason=no_closed_loop_scenes`; unavailable physical metrics are `null`, never evidence-valued zero.
6. Require non-empty paired scenes in the shadow controller and comparator. Add a repair-only script that re-runs existing v48.26 checkpoints with the corrected runner without retraining.
7. Distinguish `development_rule_fit_rejection`, `certificate_verification_rejection`, `structural_support_infeasible`, and engineering/artifact failure.
8. Rename sampler diagnostics semantically: retain legacy `num_safe_positive` for compatibility, add `legacy_root_safe_sample_count` and the gate-relevant `safe_positive_group_count`.
9. Persist and fail-closed check the number of component-harm heads in training and inference checkpoints.

### Algorithm corrections

1. **Five-factor non-compensatory harm representation.** Match the Natural-gate harmful label with five explicit heads: DRS, deployability, oracle-to-deployable gap, hard-rule violation and harm proxy. v48.26 predicted only the first three, so two gate vetoes were unrepresentable.
2. **Separate raw benefit from safe admission.** The opportunity head learns continuous raw benefit. Five harm heads learn veto factors. The admission head alone learns safe utility. Gate positives remain proposal-contained safe-beneficial actions.
3. **Execution-exact objective scale.** Regression, listwise ranking and frontier contrast all use the deployed score `sigmoid(admission_logit)-0.5`; no auxiliary objective compares unbounded logits with a bounded teacher.
4. **Two-stage factor/admission training.** Stage 1 trains only raw-benefit ranking and five harm factors. It disables admission, setwise admission and selective-risk/coverage gradients. Stage 2 freezes all factors and trains only a bounded admission residual with deployment-exact safe utility and categorical nominal-plus-top-k supervision.
5. **Bounded admission after factor repair.** v48.26 development checkpoints had invalid-admission rates of approximately 0.83–0.94. v48.27 returns to an identity-preserving bounded residual after the factor heads have been trained separately.
6. Keep the frozen top-3 proposal and disabled legacy Noisy-OR. Oracle support does not justify expanding proposal width.

### Physical-diagnostic policy

- The Near metric set covers clearance/TTC minima and terminal values, exposure duration/episodes/longest runs, deficit AUC, recovery gain, time to the dangerous point, collision/offroad, intervention bursts, route progression, acceleration, jerk and yaw rate.
- The Contact set covers secondary/re-contact events and episodes, overlap duration/longest run, free-space and clearance-deficit AUC, terminal/peak clearance, sustained escape and time-to-escape, stable-stop quality and time-to-quality-stop, plus offroad/dynamics/intervention burden.
- v48.26 did not execute these metrics on any scene, so their numerical correctness is not inferred from that run. v48.27 adds non-empty/fail-closed checks and focused unit tests. Empirical validation requires the repaired development shadow.
- Existing teacher indices do not contain candidate-level temporal physical labels. Do not fabricate them. If offline safe ranking improves while the repaired shadow does not, the next preregistered version should build candidate-level physical teacher rollouts and a separate temporal-recovery auxiliary head.

### Required v48.27 experiment and ablations

Main experiment: `scripts/run_v48_27_factor_physics_dedicated.sh`, Balanced on GPU0 and Precision on GPU1.

Before retraining, `scripts/repair_v48_26_dev_shadow_with_v48_27.sh` may re-run the existing v48.26 development shadow with complete scanning and canonical IDs.

Four ablation waves, one task per A30:

1. `A_three_factor_joint` — legacy three-factor joint training.
2. `B_five_factor_joint` — add hard-rule and harm-proxy factors.
3. `C_five_factor_two_stage_regression` — five factors, two stages and exact safe-utility regression.
4. `D_full_factor_physics_bridge` — C plus exact listwise/frontier objectives and corrected physical diagnostics.

### Decision rules

- `RC=0`: only the automatically generated `NEXT_COMMANDS.txt` authorizes held-out stress.
- `RC=20`: do not read test/stress. Run non-empty adaptation-dev shadow and the four-wave ablations. First identify development-rule fit versus certificate generalization failure.
- `RC=30`: no algorithm conclusion. Inspect pipeline, model-contract, index, checkpoint and artifact failures.
- B>A identifies missing harm representation; C>B identifies joint-gradient interference; D>C identifies value from exact listwise/frontier supervision.
- Offline improvement without physical-shadow improvement identifies teacher/closed-loop mismatch and motivates preregistered temporal physical supervision.
- Do not lower the registered Natural gate, manually create authorization, or call the repeatedly inspected certificate a final untouched paper certificate.

### Local validation

- `PYTHONPATH="$PWD/src" python -m pytest -q`: 242 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for every shell script: passed.
- Real WOMD/Waymax and two A30s are unavailable in this audit environment; no v48.27 gate or closed-loop result is claimed in advance.

## v48.26 — OC-TRAC-EXECUTION-PHYSICS-BRIDGE (2026-07-31)

### v48.25 result attribution

- The uploaded v48.25 controller returned `RC=30`, but both Balanced and Precision adaptation jobs completed successfully and produced checkpoints. All four Near/Contact certificate workers also loaded and scored their full populations (Near: 2,412 samples / approximately 305 groups; Contact: 6,929 samples / approximately 780 groups). The first actual failure occurred only when `tools/calibrate_policy_risk_v48.py` serialized `vars(args)`: `args.frozen_rule_json` is a `pathlib.PosixPath`, causing `TypeError: Object of type PosixPath is not JSON serializable`. Consequently `gate_evaluated=false`; v48.25 has no Natural-gate result and cannot be called an algorithmic rejection or regression.
- A second silent engineering defect would have invalidated the certificate even after fixing JSON. Training forwarded `direct_recovery_evidence_frontier`, `direct_recovery_evidence_component_prior_logit`, and `direct_recovery_evidence_admission_bounded`, but checkpoint inference omitted all three. The same checkpoint was therefore instantiated as different algorithms in training validation and certificate inference.
- The v48.25 checkpoint metric did not implement the Natural-gate opportunity contract. Its denominator used raw-positive groups in the full candidate set rather than proposal-contained safe-positive groups, and an admitted harmful action could still count as a positive hit. This could select an epoch that improves raw-benefit admission while failing safe admission.
- The safe-utility auxiliary target was also scale-inconsistent: training used `tanh(admission_delta/2)`, whereas runtime executes `sigmoid(admission_delta)-0.5`; the former is exactly twice the latter.
- v48.25 adaptation-dev trajectories do show a limited signal rather than universal abstention. Balanced and Precision reach Contact raw admission rates of approximately 0.034 and 0.048, and Balanced Contact evidence positive top-1 accuracy temporarily rises from approximately 0.321 to 0.500 while regret falls from approximately 0.265 to 0.204. However the flawed checkpoint metric selects epoch 0 for Balanced, Near positive admission remains zero, and the old metric does not require non-harmful selection. These observations are diagnostic only and do not establish safe admission.

### Certificate decision

- The certificate layer is retained. Its purpose is to verify a deterministic, adaptation-dev-frozen selector on an independent scene population before any held-out test/stress access; deleting it would hide unsupported behavior rather than improve the planner.
- v48.26 keeps the external-rule full-verification protocol: thresholds are fit only on `evidence_adapt_dev`, the complete rule and SHA256 are frozen, and the full `certificate_pool` is verification-only. Certificate labels never modify the rule.
- Valid structural-support or learned-selector rejection maps to controller `RC=20`. Empty data, corrupt checkpoint/artifact, split/index/protocol mismatch, inference-contract mismatch, or runtime exception maps to `RC=30`.
- `calibration.exact_split_ids=true` prevents the historical `calibration -> {calibration, certificate_pool}` alias from allowing accidental cross-role reads. Safe calibration, adaptation-dev fitting and certificate verification each consume exactly one registered split ID.
- A diagnostic dev rule may still be emitted when no dev rule satisfies constraints, but provenance records `source_rule_satisfied_dev_constraints=false`; it cannot be represented as a successfully constrained development selector.
- Safe currently has no independent scene-disjoint policy Natural-gate population. v48.26 writes `SAFE_REGIME_STATUS.json` and treats Safe through standard recovery-threshold calibration, nominal-first execution and held-out paired non-inferiority. It does not falsely describe standard calibration as a Safe policy gate.
- Because the current certificate population has been repeatedly inspected, it is development evidence. A final paper claim requires a newly sealed or preregistered certificate population.

### Engineering corrections

1. Recursively JSON-normalize `Path`, NumPy scalars/arrays, tuples, sets and nested structures in the policy-certificate output.
2. Forward and checkpoint the frontier, component-prior and bounded/unbounded-admission fields in both training and inference. Add a fail-closed `MODEL_INFERENCE_CONTRACT.json` preflight before certificate evaluation.
3. Define checkpoint opportunities exactly as proposal-contained actions with `teacher_advantage >= positive_gain` and `component_harmful=false`. Harmful actions cannot count as positive admission.
4. Add safe-positive admission recall, safe admission precision, invalid admission rate, evidence safe top-1 accuracy and evidence safe top-1 regret to checkpoint selection.
5. Train safe utility on the exact runtime score `sigmoid(admission_delta)-0.5`, with teacher targets clamped to the same `[-0.5,0.5]` execution range.
6. Enforce exact split IDs for standard calibration, adaptation-dev rule fitting and certificate verification.
7. Preserve three-way return-code semantics: 0 pass, 20 valid gate rejection, 30 engineering/protocol/artifact failure.
8. Add a repair-only script that re-evaluates existing server-side v48.25 checkpoints using corrected inference and JSON serialization without retraining. Its result is diagnostic and does not validate v48.26 training changes.
9. Fix post-contact bucket classification so `near_contact` cannot be classified as Contact merely because its name contains the substring `contact`.

### Physical execution diagnostics

The existing offline PCD teacher (`DRS * sigmoid(R_dep) * exp(-gap)`) describes deployable recovery headroom but does not directly encode temporal recovery processes. Existing NPZ/teacher indices do not contain candidate-level physical labels, so v48.26 does not fabricate an auxiliary training target. Instead it first makes the development shadow and authorized held-out evaluation sufficiently expressive to diagnose offline/physical transfer.

**Near-contact additions**

- near/critical exposure episode count and longest continuous exposure run;
- time to minimum clearance and minimum TTC;
- terminal clearance/TTC and recovery gain from the most dangerous point;
- clearance/TTC deficit AUC;
- absolute acceleration p95/max, maximum deceleration, jerk max and yaw-rate max;
- paired checks for collision/overlap non-inferiority, intervention bursts, route progression and dynamics.

**Post-contact additions/fixes**

- use an explicit causal post-contact anchor at step 0 for a post-contact target instead of discovering the first overlap inside the rollout;
- secondary/re-contact event, episode and scene rates;
- overlap episode count, duration and longest run;
- normalized free-space AUC and clearance-deficit AUC;
- terminal clearance, clearance gain and time to peak clearance;
- sustained escape rate and time to sustained escape;
- stable-stop quality requiring low speed, no overlap, on-road state, bounded yaw rate and sustained duration, plus time to quality stable stop;
- paired checks for offroad, route progression, jerk, yaw rate and intervention burden.

If safe offline ranking improves but these development-shadow metrics do not, the next version should preregister candidate-level physical teacher rollouts and a separate temporal recovery auxiliary head. Certificate labels must not be changed retrospectively to fit observed shadow outcomes.

### v48.26 algorithm: EXECUTION-PHYSICS-BRIDGE

**EXECUTION-PHYSICS = Exact eXecutable Evidence Contract, Unified safe utility, Checkpoint-safe opportunity, Independent Thresholding, Observation-consistent Nominal policy, and Physical recovery diagnostics.** The deployed model remains unified and receives no Safe/Near/Contact regime ID.

1. Preserve frozen top-3 proposal, semantic low-risk prior, centred frontier admission, categorical nominal-plus-top-k policy, safe-utility supervision and disabled legacy Noisy-OR.
2. Replace raw-positive checkpoint accounting with gate-exact safe-positive accounting.
3. Make safe-utility training mathematically identical to the executed score.
4. Select checkpoints using executable safe recall/precision, invalid admission and safe ranking, rather than soft mass or harmful raw-positive hits.
5. Keep adaptation-dev threshold fitting and independent full-certificate verification separate and hash-addressed.
6. Add rich temporal closed-loop diagnostics before introducing any new physical auxiliary teacher.

### Required v48.26 ablations and two-A30 schedule

1. `A_engineering_contract_only`: JSON, inference parity, exact split and return-code repair only.
2. `B_add_safe_checkpoint_contract`: A plus gate-exact checkpoint opportunity/precision/invalid-admission metrics.
3. `C_add_execution_exact_safe_utility`: B plus exact runtime safe-utility scale.
4. `D_full_execution_physics_bridge`: complete v48.26 algorithm and physical diagnostics.

The launcher runs four waves. In every wave Balanced occupies GPU0 and Precision occupies GPU1; each A30 runs one task at a time and receives four tasks total. Maximum concurrency is two.

### Decision rules

- Run the repair-only certificate first to determine whether the existing v48.25 checkpoint can now be evaluated. It is not a replacement for the full v48.26 experiment.
- `RC=0`: only the automatically generated `NEXT_COMMANDS.txt` may authorize Safe paired and held-out stress/closed-loop.
- `RC=20`: do not read test/stress. Run adaptation-dev shadow and the four-wave ablation suite. Distinguish low safe recall, low precision/high invalid admission, dev-to-certificate generalization failure, and offline-to-physical mismatch.
- `RC=30`: no algorithm conclusion is permitted. Inspect `PIPELINE_FAILED.json`, model-contract files, checkpoint summaries and logs.
- Do not manually create `NEXT_COMMANDS.txt`, lower the registered gate after observing results, call the repeatedly inspected certificate a final untouched paper certificate, or invent physical training targets absent from the current teacher data.

### Local validation

- `PYTHONPATH="$PWD/src" python -m pytest -q`: 233 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for every shell script: passed.
- JSON serialization, training/inference parity, safe-positive checkpoint semantics, exact execution score, exact split IDs, Near/Contact bucket separation and new physical metrics have focused tests.
- Real WOMD/Waymax and two A30s are unavailable in this audit environment; no v48.26 gate or closed-loop result is claimed in advance.

## v48.25 — OC-TRAC-INTEGRITY-BRIDGE (2026-07-31)

### v48.24 result attribution

- The uploaded v48.24 run returned `RC=30`, but this does **not** establish algorithmic regression. Balanced and Precision both completed adaptation and produced checkpoints. Four non-empty Near/Contact certificate artifacts were evaluated; the controller converted each worker's structural-support rejection (`rc=4` in v48.24) into pipeline failure 30. Structural infeasibility is a valid Natural-gate rejection and must map to worker 3 / controller `RC=20`; only missing, empty, corrupt or protocol-inconsistent artifacts map to `RC=30`.
- The v48.24 model was not the intended SUPPORT-BRIDGE implementation. The run scripts set `model.direct_recovery_evidence_frontier=true` and `model.direct_recovery_evidence_component_prior_logit=-2.0`, but `ocrap.cli.train` omitted both keyword arguments when constructing `OCRAPModel`. Runtime therefore used the legacy default `frontier=false`; zero component logits again represented approximately 0.5 harmful probability and admission again used the non-centred `benefit-softplus(harm)` prior. This silent configuration drop is sufficient to produce universal abstention and invalidates v48.24 as a test of the intended semantic prior/frontier design.
- Validation stratification also reused the adaptation-train teacher index for adaptation-dev paths. The log consequently reported all 409 dev groups as `dead_or_mixed`, despite separately computed validation statistics containing positive groups. Checkpoint selection was therefore based on a mislabeled validation sampler.
- Both variants selected checkpoints with `direct_raw_admission_rate_near=0`, `direct_raw_admission_rate_contact=0`, and zero positive admission recall. The v48.24 `direct_frontier_selection_risk` could still improve through soft mass while the executable policy remained all-abstain.
- Expanding the frozen proposal from top-3 to top-8 did not recover additional fit-side safe-positive support. Near fit remains 3 safe-positive groups at k=1/3/5/8. Contact fit increases from 7 at k=1 to 10 at k=3 and remains 10 at k=5/8. Proposal width is therefore not the current bottleneck; top-8 adds computation and ranking ambiguity without adding certificate opportunities.
- Near Balanced preserves raw-benefit AUC (0.855) but learned safe-benefit AUC, conditional harm AUC and regret worsen. Near Precision shows one partial positive signal—conditional harm AUC improves from 0.527 to 0.611 and false-switch falls from 0.221 to 0.153—but harmful-switch rises from 0.492 to 0.622, top-1 regret rises from 0.091 to 0.202, and coverage stays zero. This is not safe admission.
- Contact remains unresolved. Precision Contact learned benefit/safe-benefit/harm AUC all decline, conditional harm AUC falls to 0.443, correlation remains negative, and top-1 regret rises to 0.264. Balanced Contact remains below-random for safe/harm ordering and has negative correlation. No v48.24 closed-loop shadow result exists, so none of the physical Near/Contact publication targets is supported.

### Certificate decision

- The certificate concept is retained. It is the statistical authorization layer that correctly prevents an unsafe or unsupported learned selector from being evaluated on held-out test/stress. Removing it would hide failure rather than solve it.
- The old internal 50/50 certificate fit/verify split is no longer suitable for this sparse safe-positive population. Across the complete Near certificate there are 9 proposal-contained safe-positive groups, but the old split separately required approximately 8 fit positives and 5 verify positives; that contract can be impossible even when the full independent population has enough support. Contact has 20 total safe-positive groups, but its old fit half still misses the precision-LCB requirement.
- v48.25 fits all opportunity/harm/score/rank thresholds on `evidence_adapt_dev`, freezes the rule and SHA256 provenance, then evaluates the **entire** scene-disjoint certificate population in verification-only mode. Certificate labels never alter thresholds. The numerical verification requirements are not relaxed.
- This is a new protocol and must not be used to reinterpret v48.24 retrospectively. Because the current certificate population has already been inspected during algorithm development, results on it are development evidence. A final CCF-A paper claim requires a newly sealed/preregistered certificate population, even if the fixed dataset is retained for the next diagnostic round.

### Engineering corrections

1. Forward `direct_recovery_evidence_frontier`, `direct_recovery_evidence_component_prior_logit`, and `direct_recovery_evidence_admission_bounded` from the CLI config into `OCRAPModel`.
2. Build a separate exact teacher-PCD index for adaptation-dev and pass it through `training.validation_group_index_path`; otherwise validation stratification is disabled rather than silently using a train-only index.
3. Rename the file-name heuristic `safe_positive_fraction` to `legacy_safe_root_positive_fraction`; exact safe-positive prevalence is reported only by the teacher index.
4. Map valid structural/learned certificate rejection to `RC=20`. Reserve `RC=30` for empty data, corrupt artifacts, protocol/index mismatch, training/checkpoint failure or runtime exceptions.
5. Keep strict deployment authorization, while allowing an adaptation-dev-only shadow diagnostic to load the dev-frozen selector after independent certificate rejection.
6. Add `check_v48_25_regime_targets.py` and restore complete Near/Contact physical target checking in the shadow workflow.

### v48.25 algorithm: INTEGRITY-BRIDGE

**INTEGRITY = Identity-preserving Non-regime Evidence with Gate-True Risk, Independent dev labels, Threshold freezing and Yielding admission.** It remains one unified model and does not expose Near/Contact regime IDs at inference.

1. **Correct semantic frontier execution.** The low-risk component prior and centred identity-preserving admission path are now actually instantiated, not merely written to the shell command.
2. **Executable-admission checkpoint barrier.** `direct_integrity_selection_risk` adds hard Near/Contact positive-recall shortfall and an explicit all-abstain penalty to the existing frontier risk. A checkpoint cannot win only by improving soft mass while selecting no action.
3. **Unbounded, zero-initialised admission residual.** The primary model removes the `tanh` ceiling from the admission residual. Zero initialisation still preserves the transferred prior exactly, but the residual can cross the nominal-vs-recovery boundary when the source prior is conservatively negative. Global gradient clipping remains active.
4. **Top-3 restored.** Since k=8 adds no safe-positive support, the default returns to frozen top-3. This avoids extra harmful/ambiguous candidates and makes the ablation isolate algorithm integrity rather than proposal width.
5. **Safe-utility remains the deployed target.** Continuous safe-utility regression/listwise supervision and categorical nominal-plus-top-k learning remain active; legacy Noisy-OR remains disabled.
6. **No retrospective label relaxation.** The current component-veto teacher is kept for the clean v48.25 main run. A future protective-Pareto label ablation may separate hard collision/offroad vetoes from soft DRS/deployability/gap trade-offs, but it must be versioned and evaluated on fresh sealed evidence.

### Non-repeated ablations and two-A30 schedule

- `A_wiring_fix_bounded`: only repair the missing semantic frontier/prior wiring; retain bounded admission and the previous checkpoint metric.
- `B_add_integrity_checkpoint`: A plus executable-admission checkpoint barrier.
- `C_add_unbounded_admission`: B plus the unbounded zero-initialised admission residual.
- `D_full_integrity_bridge`: C plus continuous benefit listwise and stronger safe-vs-harmful frontier contrast.

The launcher runs four waves. In every wave Balanced occupies GPU0 and Precision occupies GPU1; each A30 runs one task at a time and receives four tasks total. Maximum concurrency is two.

### Return-code interpretation

- `RC=0`: both Near and Contact pass the dev-frozen, full-certificate verification gate; only the authorization-checked stress script may read held-out test/stress.
- `RC=20`: pipeline and certificate data are valid, but structural support or the learned selector fails. Do not read held-out test/stress. Run adaptation-dev shadow and the A/B/C/D ablations.
- `RC=30`: engineering/protocol/index/training/checkpoint/empty-artifact failure. Inspect `PIPELINE_FAILED.json` and logs before drawing any algorithm conclusion.

### Local validation

- `PYTHONPATH="$PWD/src" python -m pytest -q`: 225 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for every shell script: passed.
- `check_v48_25_regime_targets.py --help`: passed.
- Real WOMD/Waymax and two A30s are unavailable in this audit environment; no v48.25 gate or closed-loop result is claimed in advance.

## v48.24 — OC-TRAC-SUPPORT-BRIDGE (2026-07-31)

### v48.23 result attribution

- The uploaded v48.23 controller is a valid Natural-gate evaluation: both variants trained, certificate fit/verify folds are non-empty and scene-disjoint, held-out test/stress roots were not read, and the controller returned `RC=20`.
- The decisive new finding is that the v48.23 proposal-constrained oracle does **not** pass the fit fold. Under frozen top-3 and the current component-veto label, Near fit contains only 3 safe-positive groups; its optimistic oracle can select 10 but obtains only 3 positives and precision LCB 0.1538. Contact fit contains 10 safe-positive groups; selecting 16 yields precision LCB 0.4652, below the 0.5 fit requirement. Verify is feasible in both regimes. Therefore this round is not only a calibrator/representation failure: the fit-side proposal/label/gate support contract is itself insufficient.
- The earlier statement that proposal recall is approximately 0.97--1.00 remains true only for **raw-benefit** opportunity. It must not be interpreted as safe-positive proposal sufficiency. The distinction is now reported explicitly.
- Near Balanced retains high raw-benefit AUC but has below-random harm ordering and negative learned top-k correlation. Near Precision retains part of broad risk recognition, but harmful-switch remains about 0.49 and verify coverage remains zero. The benefit signal has not become safe admission.
- Contact Precision retains broad harm AUC near 0.65, but conditional harm AUC falls to about 0.50, learned benefit AUC remains about 0.42, correlation is negative, and regret is unchanged. Balanced Contact is materially worse than v48.22. The claimed Contact improvement therefore did not materialize.
- All eight v48.23 ablations complete and reject. A recovers part of broad harm AUC; B does not establish continuous ranking; C gives only small frontier changes; D does not dominate B or C. Additional epochs on the same objective are not a justified next step.

### Engineering defects fixed

1. **RC=20 dev-shadow was impossible to launch.** `run_v48_23_dev_shadow_closed_loop.sh` called the strict deployment entry, while `run_ocrap_v48_trac_sr.sh` rejected every certificate with `valid_for_deployment=false`. This is an engineering defect, not a user command error. A dedicated `DEV_SHADOW_DIAGNOSTIC=1` path now consumes only a fit-derived diagnostic selector and remains forbidden from test/stress.
2. **Runtime did not execute the certified policy.** Training and calibration used frozen top-k plus Evidence reranking, but the closed-loop loader read only score/opportunity/harm thresholds. Runtime silently fell back to `proposal_top_k=1` and `evidence_rerank_top_k=false`. It now loads the complete selector contract, including rank margin, top-k, rerank and conditional-ranking flags.
3. **Categorical and Noisy-OR objectives were both active.** v48.23 introduced a one-action categorical objective but left the legacy group-opportunity/Noisy-OR term at weight 1.25. The two targets are incompatible when only one action can execute. SUPPORT-BRIDGE disables Noisy-OR by default.

### v48.24 algorithm: SUPPORT-BRIDGE

**SUPPORT = Safe-Utility Proposal-Policy Ordering with Runtime-True Transfer.** It preserves one unified model and does not expose Near/Contact IDs to inference.

1. **Safe-positive support width.** The frozen proposal is widened from top-3 to top-8 for the new preregistered version. This is not unrestricted proposal retraining; it tests whether safe recovery variants exist below the raw-benefit top-3.
2. **Support curve audit.** Every certificate reports optimistic proposal-constrained oracle feasibility for k=1,3,5,8 and the active k, separately for fit and verify. Structural support failure is no longer confused with learned-gate failure.
3. **Safe-benefit opportunity semantics.** Gate training and calibration use continuous positive PCD only when component harm is false. Raw-beneficial but harmful actions are not counted as admission opportunities.
4. **Direct deployed safe-utility target.** The exact final admission logit receives continuous regression and listwise supervision. A safe action target is its signed PCD advantage; a component-harmful action receives a strictly negative target `-max(|delta|, positive_gain)`. This removes the requirement that an indirect benefit head and sparse risk head happen to cancel correctly.
5. **One-action-only group learning.** The categorical nominal-plus-top-k policy remains active, while legacy Noisy-OR group opportunity is disabled.
6. **Safe-positive group sampling.** Group batching explicitly preserves safe-positive groups and balances hard-negative/harmful groups without regenerating the dataset.
7. **Light frontier contrast.** Pairwise safe-versus-harmful contrast is retained only as a small auxiliary term; it no longer carries the main safety-transfer burden.
8. **Runtime-true certificate contract.** Deployment and diagnostic execution consume the same top-k/rerank/rank-margin contract written by the adaptation and certificate stages.

### New non-repeated ablations and two-A30 schedule

- `A_top3_safe_label_baseline`: safe-benefit labels with top-3; isolates label semantics from support width.
- `B_top8_support_only`: A plus top-8; isolates proposal support width.
- `C_top8_safe_utility`: B plus direct continuous safe-utility regression/listwise learning.
- `D_full_support_bridge`: C plus light high-benefit frontier contrast.

The launcher runs four waves. Each wave starts Balanced on GPU0 and Precision on GPU1, so only one task occupies each A30 at a time. Each GPU receives exactly four tasks; maximum concurrency is two rather than four processes per card.

### Decision rules

- `RC=0`: the preregistered certificate passed; run only the authorization-checked held-out stress/closed-loop script.
- `RC=20`: top-8 proposal support is feasible but the learned policy still fails. Do not read test/stress. Run fixed adaptation-dev shadow and A/B/C/D to distinguish safe-utility learning from physical transfer.
- `RC=30` with certificate support diagnostics: the new top-8 safe-positive oracle still cannot satisfy the gate or an engineering/protocol stage failed. No amount of calibrator retraining can make the current contract pass; inspect `PIPELINE_FAILED.json`, `proposal_support_curve`, and logs.
- Do not claim RC=0 in advance. It is theoretically plausible only if top-8 recovers enough fit-side safe-positive support and the learned safe-utility ordering reaches the finite-sample gate.

### Local validation

- `PYTHONPATH="$PWD/src" pytest -q`: 220 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for all shell scripts: passed.
- Real WOMD/Waymax and two A30s are unavailable in this audit environment, so no v48.24 gate or physical closed-loop result is claimed.

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

## v48.28 — PROVENANCE-MARGIN-BRIDGE

### Motivation

v48.27 returned a valid `RC=20`, but two independent defects prevented a clean interpretation:

1. adaptation-dev shadow targets were built from standard WOMD `validation`, while the closed-loop audit defaulted to `validation_interactive`; all 16 targets missed after scanning 43,479 raw scenarios;
2. the stage-1 factor checkpoint metric was constant across epochs and selected epoch 0 for both Balanced and Precision, so the five harm-factor heads were frozen at their semantic prior.

The component-harm parameterization also used `prior=-2, scale=2`, which bounded each candidate component logit to at most zero and therefore could not represent `p(harm)>0.5` for strong veto violations.

### Engineering changes

- Added an official WOMD `scenario/id` preserving Waymax loader, following the custom-loader contract used by Waymax when string IDs are required.
- Persisted `official_scenario_id`, `legacy_scenario_id`, source scenario index, source role, source pattern, and `max_num_objects` into RawScenario metadata, sample NPZ files, and manifests.
- Changed adaptation-dev shadow default from `validation_interactive` to the same standard `validation` TFRecord family used by the calibration-regime builder.
- Added fail-closed target/source-role provenance audit before shadow execution.
- Restricted legacy source-order matching to the same declared source role. It is migration-only; official `scenario/id` is the primary identity.
- Added `repair_v48_27_dev_shadow_with_v48_28.sh` so existing v48.27 checkpoints can be re-evaluated without retraining.
- Added model-contract validation for component count, prior, frontier mode, bounded admission, and component scale.
- Added factor-transfer integrity validation: stage-1 and stage-2 checkpoints must be post-epoch-0, factor heads must be nonzero and frozen during admission training, and the admission head must be trained.
- Added a structured `GATE_FAILURE_DECOMPOSITION.json` separating proposal infeasibility, development-rule fitting failure, certificate generalization failure, and pass.

### Algorithm changes

- Replaced the stage-1 checkpoint metric `direct_factor_selection_risk` with `direct_factor_supervised_risk`, which includes the actual supervised factor loss and therefore changes when the benefit and component-risk heads learn.
- Disabled initial-checkpoint eligibility in both factor and admission stages.
- Increased the default component-harm residual scale from 2.0 to 6.0 while retaining the semantic prior of -2.0. The representable logit range changes from approximately `[-4, 0]` to `[-8, 4]`.
- Kept five non-compensatory factors: DRS, deployability, oracle-to-deployable gap, hard rule, and harm proxy.
- Kept top-3 frozen proposals, categorical one-action admission, bounded identity-preserving admission, and legacy Noisy-OR disabled.
- The v48.28 main model uses two-stage factor→admission training with deployment-exact safe-utility regression only. Listwise/frontier terms are no longer in the main objective and remain an ablation because v48.27 showed no consistent benefit.

### Ablations

The four groups are:

1. `A_three_factor_wide_range` — three factors, scale 6;
2. `B_five_factor_old_range` — five factors, scale 2;
3. `C_five_factor_wide_range_regression` — five factors, scale 6, regression-only main design;
4. `D_add_listwise_frontier` — C plus listwise/frontier terms.

All eight Balanced/Precision jobs are launched concurrently: four jobs per 24 GB A30. Per-task data workers and host threads are limited to one to reduce TFRecord and CPU contention.

### Protocol decisions

- The certificate concept is retained. Complete certificate oracle support is feasible for Near and Contact in v48.27, so the dominant failure is learned development-rule fitting, not mathematical gate impossibility.
- Gate thresholds are not reduced post hoc.
- Safe remains nominal-first with held-out non-inferiority checks; Near and Contact use the registered Natural gate.
- Existing v48.27 shadow outputs contain no valid physical rollouts and must not be interpreted as zero collision/exposure/intervention.

### Non-claims

The local environment does not contain the user's WOMD/Waymax runtime or two A30 GPUs. v48.28 has passed static and unit tests, but no claim is made that it already obtains `RC=0`, passes the Natural gate, or reaches the Near/Contact closed-loop publication targets.

## v48.29 — VETO-RANK-PHYSICS-BRIDGE

### Motivation

v48.28 returned a valid `RC=20`. The proposal-constrained oracle remained feasible on the complete Near and Contact certificates, while all Balanced/Precision branches failed during adaptation-dev rule fitting. The nearest rules had low safe-positive precision/recall and excessive harmful selection, so the certificate was correctly rejecting an unsafe learned selector.

The v48.28 shadow matched eight paired scenes per branch, but audit found a runtime alias defect: dataset buckets were named `evidence_adapt_dev_near_contact` and `evidence_adapt_dev_contact`, whereas selector overrides, calibrated `gamma_rec`, and Contact physics recognized only bare `near_contact`/`contact`. Consequently every scene ran with `gamma_rec=0`; Contact was not marked as a post-contact target and its anchor/free-space/escape/re-contact metrics were missing or misleading. The matched shadow therefore established provenance only, not physical efficacy.

Runtime timing also showed that online `selected_topk` OC-MERO audit labels consumed 98.48%–98.57% of scene wall time. Model selection and Waymax step metrics were a small fraction of total cost.

### Engineering changes

- Added a shared canonical regime parser for dataset provenance prefixes. `evidence_adapt_dev_*`, `certificate_pool_*`, calibration, validation and test names now resolve to `safe`, `near_contact` or `contact` without misclassifying Near as post-contact.
- Applied the same alias contract to all selection `*_by_bucket`/`*_by_regime` overrides and to `gamma_rec_by_bucket`.
- Added explicit `canonical_regime`, `bucket_aliases`, `post_contact_target` and runtime-contract metadata to closed-loop results.
- Contact physics now recognizes provenance-prefixed Contact buckets, creates a finite causal contact anchor, and enables re-contact, overlap, post-contact free-space, escape and stable-stop metrics.
- Added `check_v48_29_shadow_runtime_contract.py`. Shadow execution fails closed unless Near/Contact regimes are correct, every scene has a finite positive calibrated gamma, Contact anchors are finite, post-contact semantics are active, and metrics are valid.
- Fixed the runtime-contract auditor itself to serialize invalid/non-finite values as JSON `null` rather than crashing while reporting an invalid legacy result.
- Added `repair_v48_28_dev_shadow_with_v48_29.sh` so v48.28 checkpoints can be re-evaluated without retraining.
- Changed physical dev-shadow default to `label_mode=fast`, zero online audit labels. Policy execution and Waymax physical metrics are unchanged. A separate suffix directory can run a sparse `selected_topk` teacher audit when needed.
- Kept official WOMD `scenario/id`, source-role provenance and legacy source-index migration checks from v48.28.
- Added fail-closed checkpoint/inference validation for admission prior mode, bounded admission, frontier, component count, component scale and semantic risk prior.
- Added eight v48.29-specific tests. Full regression result: 259 passed, 5 warnings.

### Algorithm changes

1. **Independent five-factor veto.** DRS, deployability, oracle-to-deployable gap, hard rule and harm proxy remain separately supervised non-compensatory risk factors with semantic prior -2 and scale 6.
2. **Benefit-only admission prior.** Added `direct_recovery_evidence_admission_prior_mode=benefit_only`. Admission inherits detached raw-benefit evidence but no longer subtracts the same maximum risk a second time. The five factors remain an independent calibrated hard veto, and harmful actions still receive negative safe-utility targets.
3. **Hardest-negative safe ranking.** For every proposal group with a safe-positive opportunity, the teacher-best safe action must outrank nominal and the hardest non-safe proposal by a registered margin. Groups with no safe opportunity push every recovery score below nominal.
4. **Two-stage training retained.** Stage 1 learns dense raw-benefit ordering and five harm factors only. Stage 2 freezes them and learns bounded admission with deployment-exact safe-utility regression, categorical one-action supervision and hardest-negative ranking.
5. **Listwise/frontier removed from the default main model.** v48.28 did not show stable incremental benefit. A small frontier term remains only in the D ablation.
6. **Top-3 frozen proposal retained.** Complete certificate oracle support remains feasible, so proposal expansion is not justified.
7. **Legacy Noisy-OR remains disabled.** Deployment selects exactly one recovery action or nominal.

### v48.29 ablations

1. `A_risk_centered_reference` — old risk-centered admission prior;
2. `B_veto_decoupled` — independent veto plus benefit-only admission;
3. `C_add_safe_hard_negative` — B plus hardest-negative safe ranking; v48.29 main design;
4. `D_add_frontier_to_hard_negative` — C plus a light frontier term.

All eight Balanced/Precision jobs launch concurrently: four tasks per 24 GB A30. Each task is limited to one DataLoader worker and one OMP/MKL/OpenBLAS thread to control CPU and filesystem contention.

### Protocol and decision rules

- The Natural certificate is retained. Oracle feasibility means the current gate is not mathematically impossible; thresholds are not reduced post hoc.
- `RC=20` means a valid algorithmic rejection. `RC=30` is reserved for engineering, provenance, checkpoint, index, artifact or runtime-contract failure.
- A physical shadow result is interpretable only when `SHADOW_RUNTIME_CONTRACT.json` is valid.
- If v48.29 improves offline precision/risk but valid physical shadow does not improve, the next change must be a preregistered candidate-level temporal physical teacher, not threshold tuning on certificate or held-out stress.
- No claim is made locally that v48.29 already passes the gate or meets CCF-A Near/Contact targets; WOMD/Waymax execution on the user's two A30s is required.

## v48.30 — SLACK-RANK-BRIDGE

### Motivation

v48.29 returned a valid `RC=20`. All four Balanced/Precision Near/Contact branches still failed at `development_rule_fit`, while the complete top-3 proposal-constrained certificate oracle remained feasible. The failure was therefore not a mathematically impossible certificate or missing proposal support.

The hardest-negative objective produced a real local gain on adaptation-dev—Near safe-opportunity recall increased—but certificate selection became substantially more aggressive and harmful. Joint audit found a population-prior contract error in the admission stage: only 52 of 1,167 training groups (4.46%) were safe-beneficial, while stage 2 forced 50% safe-positive groups with replacement and applied no importance correction. The model learned a recovery-heavy resampled prior that was incompatible with the natural development/certificate population and the low-intervention Natural gate.

v48.29 paired shadow execution was technically valid, but it did not establish publication-level physical gains. Near produced only small TTC changes with lower NUP and nontrivial intervention. The eight Contact targets were floor/ceiling saturated for overlap, re-contact, escape and stable-stop events, so those event metrics were not informative; continuous clearance/free-space changes were negligible or adverse.

### Unified algorithm change

SLACK-RANK-BRIDGE uses one regime-agnostic physical semantic for Safe, Near and Contact. It does not expose a regime identifier to the Evidence model and does not dispatch to separate policies.

For each recovery candidate relative to nominal, the model predicts five signed non-degradation margins:

1. DRS margin;
2. deployability margin;
3. oracle-to-deployable-gap margin;
4. hard-rule margin;
5. harm-proxy margin.

Each target margin already includes its preregistered tolerance. A value at or below zero is inside the allowed envelope; a positive value crosses an uncompensated safety boundary. The unified safety slack is

```text
s(a) = max_k m_k(a)
```

and the admission prior is

```text
U_safe(a) = B(a) - lambda * relu(s(a))
```

where `B(a)` is detached raw-benefit evidence. The independent component veto remains fail-closed. The continuous hinge supplies stable ordering close to the boundary, while the hard veto prevents benefit from compensating for a true violation.

This semantic protects Safe because unnecessary non-nominal actions lack positive benefit or violate at least one non-degradation margin; it permits Near recovery only when benefit is obtained inside the common physical envelope; and it permits Contact escape/stabilization only when deployability, recovery gap, hard-rule and harm-proxy coordinates do not deteriorate beyond the same registered tolerances.

### Training changes

1. **Natural-population stage 2.** Admission training now uses every scene-time group at most once per epoch:

   ```text
   GROUP_BATCH_STRATIFIED=false
   GROUP_BATCHING_REPLACEMENT=false
   ```

   Positive weighting remains inside the loss; it no longer alters deployment prevalence through replacement sampling.

2. **Signed component-margin regression.** Binary component targets are retained, and stage 1 additionally regresses the continuous distance to each veto boundary:

   ```text
   predicted_margin_k = factor_temperature * component_logit_k
   L_margin = SmoothL1(predicted_margin_k, teacher_margin_k)
   ```

3. **Population-aware checkpoint metric.** Added `direct_population_safe_rank_risk`, evaluated on the natural adaptation-dev population. It combines safe top-1 regret, harmful recovery mass, false-admission mass, safe-recall shortfall and safe-mass shortfall. Near and Contact are used as worst-stratum reports only; no regime ID enters the model.

4. **Hardest-negative retained under the corrected prior.** Best-safe-vs-nominal-and-hardest-non-safe supervision remains in the full design, but it is now trained on natural groups rather than an 11x positive-oversampled population.

5. **Default objective simplification.** Safe-utility listwise and frontier contrast remain disabled in the main model because v48.29 C/D ablations showed no stable incremental benefit. Legacy Noisy-OR, unbounded admission and top-8 proposal remain disabled.

### Engineering and attribution safeguards

- Added checkpoint/inference persistence for slack temperature and slack penalty.
- Added `TRAINING_CONTRACT.json`, which fails closed unless stage 2 is natural and without replacement, the population checkpoint metric is finite and varies across epochs, signed margin regression is enabled, factor transfer is valid, five factors are present, no regime routing is used and legacy Noisy-OR is disabled.
- Main runner explicitly pins five factors, natural stage-2 defaults, zero listwise/frontier weights, separate factor/admission epoch budgets and the safety-slack model contract. This prevents ambient environment variables from silently changing the registered main algorithm.
- Added `PHYSICAL_TARGET_SUPPORT.json`. It warns when Contact event targets are floor/ceiling saturated; continuous physical deltas remain reportable, but event non-improvement cannot be interpreted as success.
- Added structured `GATE_FAILURE_DECOMPOSITION.json` for proposal infeasibility, development fitting, certificate generalization and engineering failures.
- Retained exact train/dev index separation, official WOMD provenance, canonical regime aliases, positive calibrated gamma checks and Contact anchor checks from v48.28/v48.29.
- Corrected the v48.30 controller event name and pinned factor/admission epoch variables in the main runner.

### Ablations

Eight jobs are launched concurrently, four per 24 GB A30:

1. `A_natural_population_reference` — natural population, benefit-only admission;
2. `B_add_signed_component_margin` — A plus continuous five-factor margin regression;
3. `C_add_safety_slack_projection` — B plus unified safety-slack prior;
4. `D_full_slack_rank` — C plus hardest-negative, the v48.30 main design.

Per-task workers and host threads remain limited to one during eight-way execution. If filesystem/CPU contention dominates, reduce `TASKS_PER_GPU` to two rather than increasing DataLoader workers.

### Protocol decisions

- The Natural certificate and registered gate thresholds are unchanged. Complete oracle feasibility means post-hoc gate relaxation is not justified.
- `RC=0` alone authorizes held-out stress. `RC=20` remains a valid algorithmic rejection; `RC=30` remains engineering/protocol failure.
- If natural-population training improves precision but reduces recall, future changes must use loss weighting or better representations without changing the sampling prior.
- If development passes but certificate fails, focus on scene-level generalization and slack calibration.
- If offline safe ranking improves but valid physical shadow does not, the next step is a preregistered candidate-level temporal physical teacher, not certificate threshold tuning.

### Validation and non-claims

Local validation:

```text
265 passed, 5 warnings
compileall PASS
all shell bash -n PASS
```

The local environment does not contain the user's WOMD/Waymax runtime or two A30 GPUs. No claim is made that v48.30 already obtains `RC=0`, passes the Natural gate or reaches the Near/Contact CCF-A closed-loop targets.

## v48.35 — CONTINUOUS-FRONTIER

### Motivation and audited failure

v48.34 produced a valid algorithmic rejection (`pipeline_valid=true`, `certificate_executed=true`, `gate_evaluated=true`, `certificate_exit_code=20`, `test_roots_read=false`). The top-5 proposal still contained safe recovery opportunities, but the learned selector did not transfer them into a stable scene-disjoint certificate policy. Near retained useful candidate discrimination but had only one clean certificate hit; Contact selected no safe-positive action and many harmful actions. The v48.34 barrier/boundary ablations did not solve this failure.

The audit found four attribution errors that had to be removed before another algorithm claim:

1. Near and Contact were fitted with separate frozen threshold rules, creating a deployment policy fork despite the network not receiving a regime ID.
2. `proposal_exact_eligible_*` used fixed diagnostic thresholds rather than the frozen deployed rule, so “exact” diagnostics could disagree with actual selection.
3. the hard-boundary training continuation used semantic thresholds that were not constrained to the final fitted rule domain, making the corresponding ablation nearly inert;
4. post-gate commands referenced stale/missing scripts and could turn an algorithmic RC=3/20 into an engineering RC=30.

### Unified algorithm

v48.35 keeps one continuous mechanism for Safe, Near and Contact. Regime labels are not model inputs and are not used to choose a rule. Near and Contact names appear only as certificate audit strata for worst-stratum constraints.

For each candidate, the compact trainable evidence bridge now receives executable prefix physics relative to the nominal candidate:

- prefix parameters;
- macro identity;
- prefix state trajectory;
- control sequence.

It excludes absolute ego state, utility/hard/harm/feasibility audit scalars, nominal/time flags, agents, map and BEV suffixes. This restores action identity without exposing scene or regime shortcuts.

The five signed component logits define a continuous worst-component safety frontier. The free admission logit is capped by the safety frontier using a differentiable smooth minimum:

```text
free(a) = benefit(a) + residual(a)
cap(a)  = - max_k component_logit_k(a)
admission(a) = smooth_min(free(a), cap(a))
```

Therefore benefit or a large learned residual cannot compensate for a predicted component violation. The shared rule fitter is additionally restricted to the semantic domain:

```text
opportunity_threshold >= 0.5
harm_threshold        <= 0.5
score_threshold       >= 0.0
```

This is required for the cap to remain non-compensatory at deployment, not only during training.

### One shared deployment rule

`calibrate_shared_continuous_rule_v48_35.py` pools adaptation-dev proposal rows and fits exactly one four-threshold rule. Audit strata are used only to require that the same rule satisfies every stratum's minimum support, precision LCB, harmful-exposure UCB and macro-concentration constraints. Both certificate workers consume the byte-identical frozen JSON and record its SHA256.

A failed shared development fit exits with RC=3 and is preserved as an algorithmic rejection. The controller maps only missing/corrupt/protocol-inconsistent artifacts to RC=30. Held-out test commands are generated only after certificate RC=0.

### Engineering and protocol fixes

- real deployed-rule diagnostics are emitted as `proposal_deployed_rule_*`;
- legacy `proposal_exact_eligible_*` fields remain only as explicitly deprecated aliases;
- duplicate argparse registration in the mixed source snapshot is removed;
- model, training, metric/calibration and continuous-frontier fail-closed contract checks are added;
- factor-cache identity includes the context source and verifies source/copy checkpoint hashes;
- training metadata no longer claims that the train-time semantic boundary equals the final fitted threshold;
- Safe and stress wrappers verify `V48_35_COMPLETE.json`, gate authorization and candidate identity;
- stress execution verifies that Near and Contact certificates reference the same shared frozen-rule SHA and expose identical selector overrides;
- generated command dependencies are regression-tested;
- repository-local pytest import configuration is added.

### Preregistered ablation

The v48.35 ablation is a 2x2 design with one shared rule in every task:

1. legacy relative context + compensatory safety slack;
2. executable physical-relative context + compensatory safety slack;
3. legacy relative context + non-compensatory frontier cap;
4. executable physical-relative context + non-compensatory frontier cap (main).

The design isolates representation and admission geometry without creating Near/Contact-specific policies. Compatible Stage-1 factor caches are reused only after exact semantic-contract validation.

### Decision rules and non-claims

- `RC=0`: shared development rule and independent certificate pass; only then Safe paired non-inferiority and held-out stress are authorized.
- `RC=20`: valid algorithmic rejection; inspect shared-rule deficits and certificate rows, then improve representation/losses without reading test.
- `RC=30`: engineering, provenance, cache, checkpoint, script, artifact or protocol failure; no algorithm comparison is valid.

Local CPU validation checks code and contracts only. WOMD/Waymax and the user's A30 environment are unavailable locally, so v48.35 is not claimed to obtain `RC=0` or publication-ready closed-loop gains before the registered experiments are run.

## v48.35.1 — RC30-TRAINING-CONTRACT-HOTFIX

### Scope and attribution

This release is an engineering-only hotfix for the uploaded v48.35 run. It does **not** change the OC-RAP model, candidate set, training objective, checkpoint-selection metric, shared-rule fitter, certificate thresholds, gate, datasets, or Safe/Near/Contact semantics. The algorithm remains one network, one continuous physical representation, one non-compensatory frontier, and one shared deployment rule; Near and Contact remain audit strata only.

The uploaded run stopped with:

```text
failure_stage=training_contract
raw_exit_code=4
normalized_exit_code=30
balanced_adaptation_rc=0
precision_adaptation_rc=0
certificate_executed=false
gate_evaluated=false
```

The sole failed check was `exact_eligibility_all_stages`. The trainer did enable
`POLICY_METRIC_EXACT_ELIGIBILITY=true`, which was persisted as
`cfg.training.direct_policy_metric_exact_eligibility=true` in every checkpoint, but
`adapt_ocrap_v48_35_continuous_frontier_single_stage.sh` wrote only the older
`semantic_frontier_eligibility_metric` metadata key. The training-contract checker
looked only for the never-written `exact_deployment_eligibility_metric` key. This
metadata/checker mismatch converted a completed adaptation into RC=30 and prevented
all certificate execution.

### Engineering changes

1. New stage metadata writes both:
   - `semantic_frontier_eligibility_metric=true`;
   - `exact_deployment_eligibility_metric=true`, with checkpoint-config provenance.
2. `check_v48_35_training_contract.py` now verifies the actual exact-eligibility bit
   in the factor, identity, and final checkpoint configs. A legacy v48.35 stage file
   is accepted only when its semantic metadata is present **and** every trusted
   checkpoint independently proves exact eligibility. Removing the check or trusting
   metadata alone is not allowed.
3. Added `check_v48_35_resume_contract.py`. No-retraining reuse is authorized only
   for the exact known signature: training-contract raw RC=4 normalized to 30, both
   adaptations RC=0, no certificate/gate/test access, unchanged source/protocol,
   matching checkpoint/support hashes, valid stage transfer, unified physical
   semantics, and exact eligibility in every checkpoint config.
4. Added `RESUME_AFTER_ADAPTATION=1` to the v48.35 controller. Authorization occurs
   before stale status cleanup. A valid resume skips both GPU adaptations, refuses
   index rebuilds, reruns protocol/index/model/training contracts, and then executes
   the original shared-rule certificate path.
5. Added `repair_v48_35_rc30_training_contract_with_v48_35_1.sh` as the operator-facing
   no-retraining repair wrapper.
6. Completion metadata records `adaptation_reused_without_retraining` and the resume
   contract path. Different RC=30 signatures, algorithmic RC=20, changed data/index
   contracts, changed checkpoint bytes, or prior certificate artifacts are rejected.
7. Added regression tests for new metadata, checkpoint-proven legacy repair,
   rejection when one checkpoint disables exact eligibility, resume ordering before
   cleanup, no-retraining behavior, and index fail-closed behavior.
8. Tightened legacy compatibility: an absent new metadata key may be repaired from
   checkpoint evidence, but an explicitly false new key is treated as a contradiction
   and is rejected even when the old semantic key is true.
9. The resume contract additionally checks the final checkpoint hash against the
   controller-level completion record and refuses any pre-existing calibration/GATE
   artifacts that could indicate prior certificate access.
10. The repair wrapper now implements `--help`, rejects positional arguments, and
    cannot accidentally interpret a help request as an experiment launch.

### Interpretation rule

The uploaded v48.35 result is not an algorithm result because the certificate never
ran. Training diagnostics may be used only as debugging signals; they cannot establish
Near/Contact gate effectiveness. The correct next action is to reuse the byte-identical
adaptation checkpoints through the v48.35.1 resume path. Only a resulting valid RC=0
or RC=20 may be used for algorithm attribution. No additional algorithm modification
is introduced before that certificate evidence exists.

### Local validation for v48.35.1

- 17 focused v48.35/v48.35.1 tests passed.
- 174 supported release-matrix tests passed with 6 non-fatal PyTorch warnings.
- 57 shell scripts passed `bash -n`.
- `python -m compileall -q src tools tests` passed.
- The continuous-frontier contract preflight passed all finite-gradient, physical-relative input-isolation, non-compensation, no-regime-ID, and one-shared-rule checks.
- The uploaded result archive does not contain checkpoint `.pt` bytes. Therefore the no-retraining authorization cannot be executed against the archive alone; it is intentionally rechecked on the original experiment machine, where the registered checkpoints must still exist and match their stored SHA256 values.

