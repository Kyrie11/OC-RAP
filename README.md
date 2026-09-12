# OC-RAP — cleaned audit/deployment workspace

**V48.124.4 RIFA conformance repair.** Publication closed-loop runs now enforce the already-stated RIFA nesting invariant: when the absolute admission set is empty, no non-nominal recovery fallback may execute. This is a theory-to-code repair, not a new recovery mechanism or threshold sweep.

This repository is the cleaned engineering workspace for the frozen OC-RAP mechanism line and the current **V48.124 OC-FMSA fixed-Main stability/non-interference adjudication**. The current engineering hotfix is **v48.124.4-OC-FMSA**; the scientific version remains `v48.124-OC-FMSA`.

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
