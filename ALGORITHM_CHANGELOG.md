# OC-RAP Algorithm Update Log

This file records algorithm changes, hypotheses, observed evidence, failures, and next actions. Do not repeat a previously failed change without new evidence.

## Experiment targets used during development

| Regime | Development gate | Publication-scale target |
|---|---|---|
| Safe | intervention = 0, bounded NUP >= 0.999 | 3 seeds; intervention <= 0.5%; NUP >= 0.999; no degradation in physical metrics |
| Near-contact | direct path used; NUP >= 0.995; no no-op action; paired PCD/regret gain >= 0.005 | >=15% relative FRA/miss reduction or about +0.007~0.010 PCD on the current candidate set, with hundreds of audited decisions |
| Contact | direct path used; NUP >= 0.985; no no-op action; paired PCD/regret gain >= 0.005 | >=20% relative secondary-failure/FRA reduction or about +0.009~0.012 PCD on the current candidate set, with stable-stop/yaw/rejoin evidence |

## Version history

### v40 — OC-UVRA
- **Change:** Added a direct recovery-value head while freezing OC-MERO certificate modules.
- **Hypothesis:** A separate value preference could improve Near/Contact without damaging Safe calibration.
- **Observed:** Direct-value selection reason was 0; Near closed-loop physics did not improve; legacy brake-tail produced tiny-deviation/zero-speed interventions.
- **Decision:** Keep certificate/preference separation, but do not reuse CLS-only absolute-value learning.

### v41 — OC-CAVA
- **Change:** Candidate token pooling, group-centred advantage loss, additive conformal correction, and minimum trajectory-deviation actionability.
- **Observed:** No-op brake was filtered, but challenge rate and positive-sign recall were 0; the direct path still never entered final decisions.
- **Decision:** Actionability is retained. Max-residual additive conformal gating is not reused as the only deployment gate.

### v42 — OCSAVA
- **Change:** Unbounded candidate score, pairwise/top-rank losses, raw candidate/action adapter, top-1 selection-conditional additive conformal calibration.
- **Observed on 2026-07-21:**
  - Balanced: Near q=0.5223, Contact q=0.4983, both challenge_rate=0.
  - Hard: Near q=0.4842, Contact q=0.4495, both challenge_rate=0.
  - Top-1 opportunity capture was non-random (70.5%-84.4%), but selected mean teacher advantage stayed negative.
  - Offline Near used nominal in 276/276 groups; Contact intervened only 1/449 and not through the direct-value reason.
  - Removing the raw adapter improved top-1 capture (Near 87.5%, Contact 88.6%) with similar calibration error. Its 48.7-minute run was faster than balanced (81.2 minutes) but slower than hard (25.4 minutes); therefore speed alone is inconclusive, while the raw adapter still has no accuracy justification and is rejected.
- **Root causes:** additive q remained larger than all deployable score advantages; direct value was preference-only while the base admission set usually contained only nominal; raw flattened trajectories added noise; group sampler boosted absolute r_dep rather than positive advantage versus nominal.

### v43 — OC-RSC (completed; failed development gate)
- **Name:** Observation-Consistent Risk-controlled Selective Certificate.
- **Change:**
  1. Replace the zero-coverage additive LCB deployment rule with a deterministic top-1 selective threshold calibrated on scene-time groups.
  2. Calibrate and verify the threshold on disjoint deterministic folds; report precision and an upper confidence bound on bad challenge exposure.
  3. Permit a threshold-passing, actionable top-1 candidate to augment admission in Near/Contact only; Safe remains locked and the certificate is disabled there.
  4. Default retraining removes the raw adapter and oversamples groups with positive r_dep gain relative to nominal rather than merely high absolute r_dep.
  5. Use a 3-rollout selected-only probe before 6- and 12-rollout top-k audits; reuse/resume outputs and defer Safe closed-loop.
- **Falsification rule:** If no candidate obtains a finite verified threshold with nonzero held-out selections, do not run Waymax. If offline direct reasons remain zero, do not run Waymax. If the 3-rollout probe has no direct selection or violates actionability/NUP, do not expand.

## v43 implementation status (2026-07-21)

- Added `tools/calibrate_direct_value_risk_v43.py`: deterministic top-1 scene-time selection, hash-disjoint fit/verification folds, selective threshold search, held-out precision, and a one-sided 90% Wilson upper bound on harmful challenge group exposure.
- Added `selection.direct_value_risk_controlled_admission`. It is disabled by default and explicitly disabled for Safe. When enabled in Near/Contact, only a feasible, hard/harm-bounded, allowed-macro, actionable deterministic top-1 candidate above the verified threshold can augment admission.
- Added positive-advantage group sampling based on `r_dep(candidate)-r_dep(nominal)`, replacing v42's absolute-`r_dep` hard-group boost in the v43 training scripts.
- Rejected `candidate_concat_raw` as the v43 default. The default is `candidate_concat`.
- Added a reuse-first pipeline: recalibrate existing v42 checkpoints before retraining. Retraining starts only if calibration or offline direct-use gates fail.
- Added 3 -> 6 -> 12 rollout gates. The 3-rollout probe uses selected-only labels, eight candidates, six recovery options, and two-step replanning. Safe closed-loop is deferred until stress gates pass.
- Added tests for stress-only admission augmentation, v42 preference-only abstention, and selective threshold fitting.
- Validation completed locally: `77 passed, 1 warning`; compile and shell syntax checks passed.

### Statistical validity note

The hash-disjoint folds prevent threshold fitting and verification from sharing scene-time groups, but a publication claim still requires the verification/calibration data to be independent of checkpoint selection. The scripts accept `RSC_CAL_NEAR_DATA` and `RSC_CAL_CONTACT_DATA`; paper-scale runs should point these to dedicated calibration roots not used for training or early stopping. The default validation roots are intended only for development screening.

## v43 observed result and decision (2026-07-21, completed)

- **Training:** `selective_balanced` stopped at epoch 1 (`val direct loss=0.4022`, 25.2 min); `selective_precision` stopped at epoch 2 (`0.4817`, 9.4 min).
- **Calibration:** neither checkpoint produced a finite selective rule in Near or Contact. All four held-out evaluations had `num_selected=0`, `challenge_precision=null`, and `valid_for_deployment=false`.
- **Ranking diagnostics:**
  - Balanced Near/Contact pair MAE: `0.2820 / 0.2639`.
  - Precision Near/Contact pair MAE: `0.2839 / 0.2692`.
  - Top-ranked score was negatively associated with teacher advantage; merge was systematically over-scored while many selected merge candidates had non-positive teacher advantage.
- **Correct pipeline behavior:** `v43: no checkpoint passed calibration + offline-use gates; do not run Waymax.` The falsification rule prevented an expensive closed-loop run and should be retained.
- **Primary implementation defect discovered:** the group sampler and direct-value loss grouped candidates by `(scene_hash,time)` but omitted the dataset bucket. The same WOMD scene-time appears in Safe/Near/Contact roots. The logged maximum group size was `25`, exactly compatible with concatenating the per-regime candidate sets, and multiple regime-specific nominals/teachers were compared inside one ranking group.
- **Additional causes:**
  1. The direct head received no regime/state indicator, so cross-regime pressure-future labels were contradictory for observationally identical candidates.
  2. The positive-group sampler used `r_dep` gain while the direct target was PCD advantage, creating objective/sampling mismatch.
  3. Safe samples contributed no direct-value gradient but consumed batches.
  4. A forced top-1 score had no explicit probability that the current group should intervene; rare positive groups were mixed with predominantly negative/tied groups.
  5. Relaxing the v43 risk threshold is rejected because the score-teacher relationship was directionally wrong, not merely under-covered.
- **Do not repeat:** no more additive-q tuning, threshold-only relaxation, raw flattened action adapter, or bucket-agnostic scene-time grouping without new evidence.

## External baseline snapshot available with the v43 analysis

- **Safe offline:** OC-RAP's nominal-preserving behavior (`NUP=1`, `FRA=0`, `PCD=0.6163`, intervention `0`) matches log replay, nominal replay, and Wayformer-BC-lite. GameFormer-lite increased PCD only to `0.6189` but reduced NUP to `0.9468` with `39.2%` intervention. BeTopNet-lite intervened `17.6%` without PCD gain.
- **Near-contact offline:** the existing OC-RAP nominal/old-certificate result (`NUP=1`, `FRA=0.0761`, `PCD=0.5735`) is stronger than the completed uploaded lite baselines; the strongest external PCD was predictive-safety-filter at `0.5466` with `44.6%` intervention. This does **not** validate the new value module because its direct path was unused.
- **Contact offline:** the existing OC-RAP result (`NUP=1`, `FRA=0.0780`, `PCD=0.5723`) exceeds the uploaded post-impact lite baselines in PCD/NUP. Severity minimization achieved lower FRA (`0.0557`) but at `NUP=0.7212`, `PCD=0.4159`, and `80.0%` intervention.
- **Comparability warning:** uploaded methods are implementation-level `*_lite` baselines, not official reproduced SOTA checkpoints. Near MARC closed-loop lacks a matched bucket (`bucket_dataset=null`); Contact post-impact closed-loop is incomplete (`18/50`). Use them as diagnostic baselines until matched, completed runs exist.

### v44 — OC-RAVA (current)

- **Name:** Observation-Consistent Risk-Aware Value Abstention.
- **Design objective:** one deployable shared planner should preserve Safe behavior and selectively intervene in Near/Contact without receiving the evaluation regime label as a neural input.
- **Changes:**
  1. Group sampler key and direct-loss key are now `(bucket_id, scene_hash, time)`, preventing multiple regime-specific nominals and teachers from entering one ranking group.
  2. Safe is removed from direct-head training. The frozen v39 OC-MERO backbone remains the Safe policy and is checked independently.
  3. The default neural value branch is observation-only (`direct_recovery_value_regime_conditioning=false`). Optional bucket embedding code is retained only for an explicitly labelled leakage/upper-bound ablation and is not used by the main method.
  4. Add a candidate opportunity head estimating whether a recovery candidate has positive PCD advantage over nominal. Deployment first applies opportunity abstention, then selects deterministic top-1 score advantage, then applies a held-out score-risk threshold.
  5. Default direct macros are `merge` and `stabilize`; brake/yield remain under the established certificate paths until the learned head demonstrates reliable macro-specific evidence.
  6. Risk-controlled direct admission now guarantees that a verified challenge outranks nominal; v43 could admit a negative-score-threshold candidate yet give it zero score bonus, producing another zero-use path.
  7. Closed-loop policy regime can be inferred online from observable current clearance/TTC/contact thresholds (`selection.auto_regime_from_observation=true`) rather than from the targeted evaluation bucket. Per-decision `active_regime_counts` are saved for audit.
  8. Closed-loop development is reduced to `2 -> 4 -> 8` rollouts: selected-only execution proof, paired mechanism gate, then confirmation plus compact Safe. Teacher labels and candidate counts are minimized at early gates.
- **Falsification rules:**
  - No finite opportunity+score rule in both Near and Contact: stop before offline evaluation.
  - Offline direct reason is zero, Safe intervention is non-zero, or NUP fails: stop before Waymax.
  - Two-rollout probe has no direct execution, a deviation below `0.002`, or unacceptable NUP/intervention: stop.
  - Four-rollout paired test degrades PCD/regret by more than `0.01`: stop.
  - Eight-rollout confirmation must improve paired PCD or paper regret by at least `0.005` in both stress regimes before ablations or paper-scale seeds.
- **Implementation validation:** `81 passed, 2 warnings`; all Python modules compile and all new shell scripts pass `bash -n`. Warnings are the existing PyTorch nested-tensor notice.

### Updated development targets after observing the candidate-set teacher frontier

The older `+0.02/+0.03 PCD` targets exceed the empirical selector-only headroom of the current candidate set and are retained only as longer-term goals after candidate-generation improvements.

| Regime | Immediate development target | Publication-scale target on current candidate set |
|---|---|---|
| Safe | intervention `0`; NUP `>=0.999`; no physical regression | 3 seeds; intervention `<=0.5%`; statistically non-inferior to nominal/log replay and competitive with complete learned planners |
| Near-contact | direct path non-zero; no no-op; paired PCD/regret gain `>=0.005` | FRA relative reduction about `>=15%` or PCD `+0.007~0.010`, hundreds of paired audited decisions |
| Contact | direct path non-zero; NUP `>=0.985`; paired gain `>=0.005` | secondary-failure/FRA relative reduction about `>=20%` or PCD `+0.009~0.012`, plus stable-stop/yaw/rejoin evidence |

### Statistical validity note for v44

The main neural head is observation-only, but threshold verification data must still be independent of training and checkpoint selection for a paper claim. Development scripts use `val_near_contact` and `val_contact`; publication runs must set dedicated `RAVA_CAL_NEAR_DATA` and `RAVA_CAL_CONTACT_DATA` roots that were not used for training, early stopping, or architecture selection.

## v44 observed result and decision (2026-07-21, completed; failed before offline)

- **Pipeline outcome:** both `rava_balanced` and `rava_precision` failed the opportunity+score calibration gate; offline evaluation and Waymax were correctly skipped.
- **Training:** both variants selected epoch 3. Balanced validation direct loss was `5.3875`; precision was `9.5148`. The lower balanced loss did not translate into a deployable selective rule.
- **Calibration evidence:**
  - Balanced Near/Contact pair MAE: `0.3124 / 0.3078`.
  - Precision Near/Contact pair MAE: `0.3291 / 0.3289`.
  - Near had `246` eligible scene-time groups and `30` positive-opportunity groups across fit+verify; Contact had `357` groups and `43` opportunities.
  - In all four calibrations, `num_top1_after_opportunity_gate=0`, fitted thresholds were infinite, and held-out selections were zero.
- **Calibration implementation defect:** v44 imposed a fixed minimum opportunity probability of `0.05`. Every learned candidate opportunity was below this floor, so all candidates were removed before the score-risk rule was fitted. A fixed probability floor is rejected; future calibration must search the observed score support and report distributions even when no rule is valid.
- **Model/target defect:** v44 predicted an absolute per-candidate opportunity logit but supervised the relative event `PCD(candidate)-PCD(nominal) >= positive_gain`. The nominal quality changes by group, so an absolute candidate probability is not the deployment quantity. Future opportunity supervision and inference must use candidate-minus-nominal logit differences.
- **Remaining task conflict:** fixing `(bucket, scene, time)` grouping prevented within-group cross-regime comparisons, but one unconditional Near/Contact head still received different pressure-future targets for observationally similar scene-prefixes. A single head is rejected until an observation-derived task representation or lightweight regime experts are tested.
- **Sampling decision:** do not restore proxy `r_dep` positive-group oversampling. It is misaligned with the PCD target. Use exhaustive non-replacement group epochs and target-aligned positive/negative loss weights.
- **Do not repeat:** fixed `0.05` opportunity floors, absolute opportunity logits for relative events, threshold-only relaxation after zero pre-gate coverage, or `r_dep`-proxy oversampling.

## Dataset diagnostic contract discovered with v44 results

- **Train/validation semantic separation is mostly usable for development:**
  - `train_near_contact` post-contact fraction `1.88%`; `val_near_contact` `1.70%`.
  - `train_contact` and `val_contact` are `100%` post-contact.
- **Current test roots are not valid for the paper three-regime comparison:**
  - `test_near_contact` contains `1270/2058 = 61.71%` post-contact samples, so it is primarily a Contact set despite its name.
  - Near `|test-val r_dep mean shift| = 1.474` and hard-violation mean shift `0.409`.
  - Contact `|test-val r_dep mean shift| = 1.582` and hard-violation mean shift `0.371`.
  - Safe validation/test diagnostics contain only `22/28` scenes, below a paper-scale target.
- **Decision:** development calibration/training may use current train/val roots, but paper claims and final baseline comparisons are blocked until clean disjoint Safe/Near/Contact test roots are rebuilt. `tools/check_regime_dataset_contract.py` enforces this contract.

### v45 — OC-RAVE (current)

- **Name:** Observation-Consistent Regime-Expert Value Abstention.
- **Root-level changes:**
  1. Keep one frozen shared OC-MERO scene encoder, but replace the contradictory unconditional value branch with two lightweight task experts: Near-contact and Contact. Safe never invokes the direct branch.
  2. Train and deploy opportunity as a **relative-to-nominal logit difference**. The candidate opportunity probability is `sigmoid(logit(candidate)-logit(nominal))`, matching the deployment event.
  3. Remove the fixed `0.05` probability floor. Calibration searches the empirical opportunity support from `0.0` upward and writes diagnostic top-1 rows even when no deployment rule passes.
  4. Broaden the learned candidate audit to `brake,yield,merge,pull_over,stabilize` so calibration can identify which macros contain real PCD opportunities instead of hard-coding the v43 merge bias.
  5. Use exhaustive group epochs (`group_batching_replacement=false`) and PCD-target loss weighting. Proxy `r_dep` oversampling is disabled.
  6. Preserve physical actionability, held-out risk verification, Safe lock, and the `2 -> 4 -> 8` closed-loop cost ladder.
  7. Add a dataset contract preflight. Because uploaded test diagnostics fail, Stage 2 uses validation roots only for development zero-use screening and explicitly forbids treating that result as held-out paper evidence.
- **Regime routing:** training selects the Near/Contact expert by the clean dataset task. Closed-loop selects it from current observable clearance/TTC/contact using `auto_regime_from_observation`; no teacher outcome is supplied to the neural input. This is a hard task router, not three full models and not an oracle regime embedding.
- **Smooth switching decision:** soft routing / continuous risk-token attention is postponed until both experts show positive held-out score-teacher correlation and nonzero verified use. Adding a soft mixture before basic expert learnability is established would confound target failure with routing failure. A later version should compare hard routing, hysteresis, and soft risk-token mixing on boundary scenes.
- **Falsification rules:**
  - If either expert has non-positive or unstable held-out score-teacher relation and no finite opportunity+score rule, stop before offline/Waymax.
  - If calibration succeeds but offline direct reasons remain zero, stop before Waymax.
  - If the 2-rollout execution probe has no real direct action or violates actionability/NUP, do not expand.
  - Do not use contaminated `test_near_contact` for final claims, regardless of model result.
- **Implementation validation:** `87 passed, 2 warnings`; all Python modules compile and v45 shell scripts pass syntax checks. Warnings are existing PyTorch nested-tensor notices.

## v45 observed result and decision (2026-07-23, completed)

- **Natural gate outcome:** both archived checkpoints failed the joint Near+Contact contract because Near produced no finite opportunity+score rule and `verify.num_selected=0`. The pipeline correctly stopped before offline evaluation and Waymax.
- **Balanced checkpoint:** best epoch 1, validation direct loss `2.5666`; Near correlation/MAE `0.1570/0.2863`, predicted advantage median `-0.2495`, no verified selections. Contact correlation/MAE `0.2726/0.2761`, 7 verified selections, 5 positive and 2 harmful.
- **Precision checkpoint:** best epoch 1, validation direct loss `4.2477`; Near correlation/MAE `0.1533/0.3052`, predicted advantage median `-0.3832`, no verified selections. Contact correlation/MAE `0.2939/0.2794`, 9 verified selections, 6 positive and 3 harmful.
- **Primary target defect:** teacher advantage median and q75 are exactly zero in both Near and Contact. The v45 loss used `t_delta <= 0` as a hard negative mask, so tied candidates were forced below a negative margin. This explains the systematic negative predicted-advantage shift.
- **Expert identifiability defect:** despite the v45 design note describing task experts, the implementation trained only the soft mixture plus a balance regularizer. No expert-specific Near/Contact loss was present, so the two heads could collapse to exchangeable/duplicated solutions.
- **Risk-denominator defect:** Contact was marked valid by controlling harmful selections divided by all groups. The actually executed conditional harmful rates were `2/7=28.6%` and `3/9=33.3%`; group-exposure UCBs hid this because coverage was low.
- **Optimization evidence:** both variants deteriorated after epoch 1. The head-only LR/direct-loss scale was too aggressive for the sparse/tied target.
- **Contract ambiguity:** archived JSON uses the development constraints (`required_min_scenes=20`, harmful group UCB cap `0.12`) even though the supplied launch command states `FINAL_RUN=1`. Treat the artifacts as development evidence only.
- **Do not repeat:** do not relax the Near threshold, count ties as negative, accept group-exposure-only safety, or call a val-derived rule final.

## v46 — OC-RACE (implemented, awaiting GPU/Waymax results)

- **Name:** Observation-Consistent Regime-Adaptive Calibrated Experts.
- **Role in the paper:** a selective residual-admission extension of OC-RAP/OC-MERO, not a replacement for observation-consistent recoverability.
- **Learning changes:**
  1. Use a positive / dead-zone tie / meaningful-negative target with separate `positive_gain` and `negative_gain`.
  2. Restrict listwise ranking to nominal plus deployment-eligible recovery macros.
  3. Directly supervise expert 0 on Near and expert 1 on Contact; retain a small loss on the deployable soft mixture.
  4. Route from candidate-independent shared observation features. Macro, prefix parameters, states, and controls are excluded from the router input, so all candidates in one scene-time share expert weights.
  5. Select checkpoints by worst-regime validation loss rather than mixed average loss.
  6. Reduce LR/direct loss scale and add gradient clipping.
- **Calibration changes:**
  1. Select supported macro families only from the fit fold, then freeze them before verification.
  2. Use scene-disjoint folds by default.
  3. Report and enforce both harmful/all-group exposure UCB and harmful/selected-action conditional UCB.
  4. Enforce positive pred/teacher advantage correlation in both development and final gates.
  5. Stamp `contract_mode`, `valid_for_development`, `valid_for_deployment`, and `valid_for_active_contract` in every artifact.
  6. Final mode requires dedicated calibration roots and forbids the default validation roots.
- **Engineering fixes:**
  1. Fix `FREEZE_PREFIXES=""`: `${VAR-default}` now preserves an explicit empty value for full clean-base unfreezing.
  2. Stage-0 reads `train_summary.json` and writes the clean marker only when `freeze_param_prefixes=[]`.
  3. Fully frozen module subtrees remain in eval mode so dropout does not create train/calibration feature shift.
- **Cost control:** retain calibration -> offline-use -> 2 -> 4 -> 8 Waymax gates; no checkpoint enters Waymax unless both regimes pass the active calibration contract.
- **Validation:** full local test suite `100 passed, 3 warnings`; warnings are existing PyTorch nested-tensor notices. Python compilation and shell syntax are rechecked at packaging time.
- **Falsification rules:**
  - If Near or Contact has no finite rule, stop.
  - If correlation is below the active threshold, stop.
  - If conditional harmful-selection UCB exceeds budget, stop even when harmful/all-group exposure is small.
  - If offline direct reasons are zero, stop.
  - If the oracle candidate frontier has too few positive opportunities, modify candidate generation rather than the admission threshold.
- **Required ablations:** v45; +tie dead zone; +expert supervision; +shared-observation router; +deployable listwise set; +fit-supported macros; +dual-risk contract; full v46. Also compare hard bucket, uniform, candidate-conditioned, and shared-observation routing.
- **Publication blockers still open:** independent calibration roots; paper-scale Contact scenes; actual/validated post-contact states rather than only counterfactual labels; complete official/matched baseline runs; multi-seed closed-loop confidence intervals; alignment of the paper's five regimes with implemented evidence.

## v46 observed result and decision (2026-07-23, completed)

- **Natural gate:** both `ocrap_v46_race_balanced` and `ocrap_v46_race_precision` failed the joint Near+Contact contract. The pipeline correctly blocked learned-policy offline evaluation and Waymax.
- **Near-contact evidence:** candidate-level positive AUC remained informative (`0.6725/0.6652` for balanced/precision), but unconstrained per-group predicted top-1 teacher-advantage correlation was negative (`-0.0792/-0.0701`). Mean teacher advantage of the predicted top-1 was `-0.1618/-0.1559`, with harmful rates `31.2%/30.4%`. No fit-fold rule existed and verify selected zero actions.
- **Contact evidence:** balanced degraded from fit `20 selections, 55% precision, +0.0915 mean advantage` to verify `25 selections, 48% precision, 44% harmful, -0.1333 mean advantage`. Precision degraded from fit `6/6 positive, +0.3489` to verify `12 selections, 66.7% precision, 33.3% harmful, -0.0465 mean advantage`.
- **Checkpoint provenance defect:** both archives initialized from `runs/ocrap_v39_ocrac_balanced/model_v39_ocrac/best.pt` and froze the shared encoder/certificate stack. The intended current clean-base refresh was not used.
- **Sampler defect:** logs recorded `replacement=false`, `positive_advantage_boost=0`, and `positive_advantage_groups=0`. Rare recovery-improving groups were not emphasized, and the historical proxy target was not the deployed PCD objective.
- **Router result:** validation router accuracy stayed around `0.52-0.56`, insufficient to identify a hidden Near/Contact task and conceptually misaligned with observation-indistinguishable future handling.
- **Closed-loop control-flow defect:** calibration failure exited before any learned-policy rollouts, and the nominal all-regime reference was not separated from certificate gating.
- **Bucket split defect discovered in follow-up:** development passed `val_*` roots while `run_ocrap_v46/v47` hard-coded `closed_loop.bucket_split=test`; strict target filtering could produce zero Safe/Near/Contact bucket targets. Development and held-out split propagation must be explicit.
- **Dataset rebuild concurrency defect:** Safe and Near `wait_pair` calls were commented while PIDs were overwritten by later launches. Up to six Waymax/JAX workers could contend, and the script only waited for Contact. This can leave nominal/stress roots incomplete or silently failed.
- **Raw-source mismatch:** the synchronized bucket rebuild intentionally uses standard WOMD validation for Near/Contact, while the closed-loop audit default scanned `validation_interactive`. Scene ids can therefore have zero matches even after fixing the split. v47 introduces `WOMD_STRESS`, defaulting to the standard validation source used by the rebuild.
- **Decision:** do not relax opportunity/score thresholds. The failure is policy-level setwise ranking and false-admission control, not a small coverage deficit.

## v47 — OC-TRAC (implemented; GPU/Waymax results pending)

- **Name:** Observation-Consistent Tri-state Risk-Calibrated Recovery Admission Certificate.
- **Paper role:** selective recovery admission on top of OC-MERO and CRISP. The novelty claim remains observation-consistent recovery affordance and the oracle-to-deployable gap; MoE itself is not claimed as novel.
- **Learning changes:**
  1. Train positive-gain, dead-zone/tie, and harmful-switch states explicitly rather than reducing every non-positive delta to a hard negative.
  2. Add a dedicated harmful-admission head and propagate it through inference, calibration, evaluation, and the selector.
  3. Add nominal-relative setwise admission/abstention loss over deployment-eligible macros; nominal is the correct class when no positive recovery exists.
  4. Scan groups using the exact teacher PCD composite used by calibration and enable weighted replacement sampling for rare positive PCD groups.
  5. Replace hidden regime experts with two all-stress risk-attitude experts: recovery-seeking and harm-averse. Both see Near and Contact; asymmetric loss weights create complementary hypotheses without oracle bucket routing.
  6. Aggregate experts with candidate-invariant uniform weights and an uncertainty certificate: mean-minus-disagreement for benefit/opportunity, mean-plus-disagreement for harm.
  7. Retain structured `candidate_concat` as the default. Raw flattened state/control features remain an ablation only because previous versions did not justify them.
  8. Select checkpoints with a worst-regime direct objective and keep frozen subtrees in eval mode.
  9. Add a configurable stress macro schedule and per-macro variant numbering. Near/Contact rebuilds front-load distinct merge/brake/stabilize/yield variants before the 8/9-candidate quality cap; repeated merge/stabilize/yield variants are no longer parameter duplicates.
- **Calibration changes:**
  1. Search a joint opportunity, predicted-harm, and score rule on fit scenes and freeze it before verification.
  2. Require positive policy-level selected teacher advantage, precision/precision-LCB, selection support, harmful/all-group UCB, and harmful/selected-action UCB.
  3. Report candidate AUC separately from unconstrained group-top1 correlation so pointwise success cannot hide setwise failure.
  4. Separate network `predicted_harm` from physical/data `harm_proxy`.
  5. Keep global pair correlation diagnostic by default; it can be a final-contract requirement but is not allowed to replace policy-level verification.
- **Closed-loop changes:**
  1. Add explicit `SAFE_BUCKET_SPLIT`, `NEAR_BUCKET_SPLIT`, and `CONTACT_BUCKET_SPLIT`; development uses `val`, held-out uses `test`.
  2. Add a certificate-independent nominal reference runner for Safe/Near/Contact when all learned checkpoints fail. These outputs are reference physics only and must not be reported as learned OC-TRAC results.
  2a. Use `WOMD_STRESS` to bind Near/Contact closed-loop replay to the same raw WOMD source used during bucket construction; default to standard validation for the synchronized rebuild.
  3. Add all-regime collision/offroad scene and step rates, minimum clearance/TTC, path length, net displacement, progress efficiency, acceleration, hard braking, jerk, and yaw-rate metrics.
  4. Keep FRA/DRS/ODG for regimes whose stress data supports observation-consistent recovery labels. Safe is compared with physical, comfort, progress, intervention, and NUP metrics.
  5. Add metadata/WOMD preflight and report actual matched/missing scene-time targets at runtime.
- **Engineering guards:**
  - separate `TRAIN_OCRAP_ROOT` from `EVAL_OCRAP_ROOT`, because the synchronized rebuild creates val/test roots only;
  - enforce full-unfreeze clean-base marker;
  - enforce v47 initialization from that exact clean checkpoint;
  - block test feedback until the optional one-shot held-out stage;
  - fail closed on invalid calibration artifacts;
  - keep learned-policy Waymax behind the natural gate while preserving nominal reference evaluation.
- **Validation:** `110 passed, 4 warnings`; warnings are existing PyTorch nested-tensor notices. Python compilation and shell syntax are verified at packaging time.
- **Falsification rules:**
  - If exact-PCD sampler still reports zero positive groups, inspect teacher PCD construction or the candidate frontier; do not tune thresholds.
  - If candidate AUC is positive but group-top1 correlation remains non-positive, improve setwise candidate representation/generation rather than lowering calibration gates.
  - If Contact verify harmful UCB remains high, collect more independent calibration scenes and/or improve harm features; do not hide it with all-group exposure.
  - If the learned certificate fails, report nominal references separately and do not call them OC-TRAC results.
- **Required ablations:** v46; +exact-PCD sampler; +tri-state loss; +setwise abstention; +harm head/veto; +asymmetric all-stress experts; +disagreement certificate; full v47. Also compare single head, hard bucket router, shared-observation router, and uniform robust aggregation.
- **Publication blockers:** independent calibration roots; larger Contact calibration/test sets; at least three seeds; paired scene bootstrap; completed matched external baselines; no repeated tuning on held-out test.
