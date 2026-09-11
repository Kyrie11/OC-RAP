# OC-RAP — cleaned audit/deployment workspace

This repository is the cleaned engineering workspace for the current V48.116 OC-WRCF audit plus the
publication-facing evaluation utilities used by the deployed OC-RAP stack.

## What is active

- `scripts/run_constraint_native_orientation_audit.sh`: the stable semantic scientific launcher.  It currently runs V48.116 OC-WRCF.
  V48.116 is an **audit-only** equal-capacity closed-form nominal OC-MERO weak-root cotangent recovery-set flow experiment. It evaluates
  same-option candidate-minus-nominal signed constraint response over the existing actuator-projected recovery continuation;
  it does not train or mutate the frozen OC-RAP planner.
- `scripts/run_ocrap_evaluation.sh`: direct Safe / Near-Contact / Contact closed-loop evaluation of
  an existing frozen OC-RAP model run.
- `scripts/run_external_baselines.sh`: direct external-baseline train/registration/calibration/test
  entry. By default it reuses learned checkpoints only when their configured epoch budget is
  complete; `--retrain` and `--recalibrate` force the optional expensive stages.
- `scripts/run_submission_ablations.sh`: frozen-module submission ablations.
- `scripts/build_regime_visualizations.sh`: direct selected-scene trace generation and MP4 rendering.
- `tools/select_recovery_toy_examples.py`: dataset-only toy-example selection.
- `scripts/build_ocrap_datasets.sh`: unversioned dataset-construction dispatcher.

## WOMD publication contract

Training data use WOMD `training`. Primary validation, test, calibration, external-baseline replay,
and visualization use standard WOMD `validation`. `validation_interactive` remains available only as
an explicit diagnostic compatibility role; it is not the default publication path.

## Historical compatibility versus code dependency

The cleaned code has no historical `vXX` Python filenames or imports. Some strings such as
`model_v48_trac_sr`, historical checkpoint implementation tags, and frozen result-directory names are
kept deliberately so existing frozen model assets can be loaded and audited. They are data/checkpoint
provenance identifiers, not imports of old code.

`ALGORITHM_CHANGELOG.md` is retained as a historical scientific record only. It may mention removed
historical launchers; none of those names are executable dependencies of the cleaned workspace.

The current launcher consumes versioned frozen scientific assets by design, including the authoritative V48.112 STOP result plus V48.80/V48.93/V48.96 checkpoint/evidence assets.  These are provenance inputs, not old-code dependencies.  Source/tool/test filenames remain semantic and unversioned.

Use `scripts/analyze_dataset_properties.sh` for a read-only 12-bucket dataset/property/construction-provenance export.

See `CLEAN_COMMANDS.md` for operator commands and `ALGORITHM_CHANGELOG.md` for the scientific audit history and preregistered branches.
