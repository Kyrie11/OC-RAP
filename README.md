> **Current stage — V48.124.10.7.5 final-baseline contract.** Internal mechanism search remains closed and the OC-RAP planner is unchanged. This engineering/evaluation-only revision keeps the V48.124.10.7.4 observation-legal target lock, fixes full-cohort propagation so `MAX_SCENARIOS=0` cannot silently fall back to 50 baseline scenes, rejects completed artifacts whose journals do not exactly match the frozen target lock, repairs native validation/checkpoint selection for Diffusion Planner and Flow Planner, keeps GameFormer-lite in FP32 after the historical AMP gradient failure, and requires isolated single-process/single-GPU latency for both OC-RAP and external baselines. V48.125 remains unauthorized; the historical deployment-acceptance Near STOP is unchanged.

## Final characterization entrypoint

```bash
cd /home/senzeyu2/code/OC-RAP
OCRAP_CONSTRAINT_AUDIT_MODE=final_characterization \
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_constraint_native_orientation_audit.sh
```

Build the method-independent target locks first, then OC-RAP characterization, external-baseline accuracy/isolated-latency, and OC-RAP isolated latency may run independently. Historical external closed-loop summaries are debug provenance only; final comparison uses the clean `external_baselines_v48_124_final_v2` root. See `CLEAN_COMMANDS.md` for exact retraining, checkpoint-reuse, latency, and table commands.

> **Current audit stage — V48.124.10.7.2 terminal internal closure (no new GPU experiment).** V48.124.10.7.1 passed reliability/scientific attribution and returned `ONE_SHOT_ACTION_REALIZATION_MIXED`: the preregistered strongest seed action is locally positive in one scene and worsens Near physical endpoints in the other even when executed exactly once from the untouched nominal trajectory. This removes the repeated-oracle trajectory confound as the explanation for the sign split, closes the current absolute-admission-repair hypothesis, and freezes further internal mechanism/threshold/capacity iteration. The deployed Main itself is unchanged and therefore **not frozen**: V48.124 still requires Coverage + Determinism + Safe + Near + Contact all GO. The stable command now performs only an offline closure adjudication over the authoritative 10.7.1 result bundle; historical GPU diagnostics remain explicit reproduction modes.

## Current terminal closure entrypoint

```bash
cd /home/senzeyu2/code/OC-RAP
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_constraint_native_orientation_audit.sh
```

Required evidence: `$BASE_OUT/OC-RAP-v48.124.10.7.1-ONE-SHOT-ACTION-REALIZATION-results.zip`. Expected output: `$BASE_OUT/OC-RAP-v48.124.10.7.2-TERMINAL-INTERNAL-CLOSURE.json`. This stage launches no GPU job. To reproduce 10.7.1 explicitly, set `OCRAP_CONSTRAINT_AUDIT_MODE=one_shot_action_realization`. The scientific model remains `v48.124-OC-FMSA`; no selector/model/checkpoint/candidate/recovery/threshold mechanism is changed.

# OC-RAP — cleaned audit/deployment workspace

**Historical V48.124 fixed-Main workspace.** The scientific model/recovery mechanism remains the frozen `v48.124-OC-FMSA`. V48.124.9 closed Reliability + Scientific Attribution and left exactly one failed deployed-system gate: Near. V48.124.10 does not train or recalibrate anything; it diagnoses that Near failure by completing frozen relative role-isolation only after the historical absolute-admission selector would already intervene.

The stable command is unchanged and now runs this licensed Near-only diagnostic by default. It replays only the 35 V48.124.9 Near scenes that actually contained an intervention, while proving the other 215 zero-intervention scenes remain action/state-identical. GPU0 owns balanced and GPU1 owns precision; each GPU runs the sign-only arm followed by the nested-evidence arm. A promoted selector must still pass a fresh full 250-scene Near confirmation before deployed-Main freeze or paper Main-table use. Set `OCRAP_CONSTRAINT_AUDIT_MODE=full` only to reproduce the historical V48.124.9 full five-gate audit.

## What is active

- `scripts/run_constraint_native_orientation_audit.sh`: the stable semantic launcher. It now runs **V48.124 OC-FMSA**, not a new recovery representation. The authoritative V48.123 scientific result was `ZERO_BOUNDARY_VIABILITY_STATE_TRANSITION_STOP`, which freezes the recovery-set mechanism family and licenses only fixed-Main system adjudication.
- V48.124 evaluates the existing frozen L80 Main (`balanced` and `precision` robustness variants) against exact same-target nominal replay on Safe / Near-Contact / Contact. It performs no planner/source/root training, no recalibration, no carrier integration, no regime router, and no capacity/source/horizon/threshold search.
- `scripts/run_ocrap_evaluation.sh`: direct Safe / Near-Contact / Contact closed-loop evaluation of an existing frozen OC-RAP model run.
- `scripts/run_external_baselines.sh`: external-baseline train/registration/calibration/test entry. Final submission comparison is authorized only after V48.124 GO.
- `scripts/run_submission_ablations.sh`: frozen-module submission ablations.
- `scripts/build_regime_visualizations.sh`: selected-scene trace generation and MP4 rendering.
- `tools/select_recovery_toy_examples.py`: dataset-only toy-example selection.
- `scripts/build_ocrap_datasets.sh`: unversioned dataset-construction dispatcher.

## V48.124 freeze criterion

The Main is external-baseline-ready only if exact bucket coverage/source identity, deterministic sentinel replay, Safe non-interference, Near closed-loop validity, and Contact recovery validity all pass. Balanced/precision are robustness variants, not independent population replications. A V48.124 STOP does **not** reopen recovery-set representation search; only the failed provenance/stability/closed-loop axis may be diagnosed.

## WOMD publication contract

Training data use WOMD `training`. All publication validation/test/calibration replay in this codebase uses standard WOMD `validation`; `validation_interactive` is not a publication/test source. V48.124 explicitly requests `validation` and fails closed if bucket provenance disagrees. Any paper text claiming `validation-interactive` for these test sets is incorrect and must be changed to standard `validation`; the code must not be changed to match that paper typo.

## Historical compatibility versus code dependency

The cleaned code has no historical version-named Python filenames or imports. Version strings in frozen model/result paths are provenance identifiers, not old-code imports. `ALGORITHM_CHANGELOG.md` is the scientific history and preregistration record.

Use `scripts/analyze_dataset_properties.sh` for read-only dataset/property/construction provenance.
See `CLEAN_COMMANDS.md` for operator commands.

## V48.124 fixed-Main adjudication note

The historical full-audit engineering build is `v48.124.9-OC-FMSA-RUNTIME-SNAPSHOT-ENGFIX`; the current diagnostic engineering layer is `v48.124.10-OC-FMSA-NEAR-RIFA-SYSTEM-AXIS-AUDIT`. The scientific model/recovery mechanism remains `v48.124-OC-FMSA`.


### Candidate-quality adjudicator hotfix (V48.124.10.3.1)

The candidate-quality GPU rollout remains V48.124.10.3. The offline adjudicator now (a) treats matching undefined `NaN` diagnostic fields as replay-equivalent, and (b) uses execution-consistent teacher PCD delta as the primary nominal-relative diagnostic utility; exact historical training-target identity is not assumed without the original checkpoint training configuration. Existing 10.3 outputs can be re-adjudicated without rerunning GPU rollouts.
