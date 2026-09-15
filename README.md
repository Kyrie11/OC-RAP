## Current closed-loop adjudication entrypoint

For the frozen V48.124 scientific Main, the current engineering/evaluation successor is **V48.124.10.2**. It fixes result provenance and enforces an observation-only planner-route contract without changing model weights or selector equations. Use the stable command:

```bash
GPU0=0 GPU1=1 BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
  bash scripts/run_constraint_native_orientation_audit.sh
```

The default path first runs a fresh 250-scene Near base with WOMD v1.3.1 `sdc_paths` connectivity geometry, then performs reduced delta/nested diagnosis, and if an arm is promoted automatically runs one fresh 250-scene confirmation. Results are packaged even on fail-closed exits.

# OC-RAP — cleaned audit/deployment workspace

**Current engineering build: `v48.124.10-OC-FMSA-NEAR-RIFA-SYSTEM-AXIS-AUDIT`.** The scientific model/recovery mechanism remains the frozen `v48.124-OC-FMSA`. V48.124.9 closed Reliability + Scientific Attribution and left exactly one failed deployed-system gate: Near. V48.124.10 does not train or recalibrate anything; it diagnoses that Near failure by completing frozen relative role-isolation only after the historical absolute-admission selector would already intervene.

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
