# OC-RAP Algorithm Update Log

This file records algorithm changes, hypotheses, observed evidence, failures, and next actions. Do not repeat a previously failed change without new evidence.

## Experiment targets used during development

| Regime | Development gate | Publication-scale target |
|---|---|---|
| Safe | intervention = 0, bounded NUP >= 0.999 | 3 seeds; intervention <= 0.5%; NUP >= 0.999; no degradation in physical metrics |
| Near-contact | direct path used; NUP >= 0.995; no no-op action; PCD/regret improves | >=15% relative reduction in FRA/miss/escalation or about +0.02 PCD, with hundreds of audited decisions |
| Contact | direct path used; NUP >= 0.985; no no-op action; PCD/regret improves | >=15-25% relative reduction in secondary-collision failures or about +0.03 PCD, with stable-stop/yaw/rejoin evidence |

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

### v43 — OC-RSC (current)
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
