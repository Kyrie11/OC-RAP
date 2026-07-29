# OC-RAP v48.15 PRISM-CC Change Manifest

## Engineering correctness

- Fixed `calibrate_v48_14_certificate_pool.sh` local-variable expansion under `set -u`.
- Added `CALIBRATION_FAILED.json`/exit 30 to distinguish artifact failure from a genuine Natural-gate rejection (`GATE_FAILED.json`/exit 20).
- Added per-task `VARIANTS` filtering to certificate calibration.
- Added no-retraining v48.14 certificate recovery script.
- Prevented Safe closed loop from silently falling back to arbitrary WOMD scenes when a bucket dataset matches zero targets.
- Removed forced `bucket_split=test` for `calibration_safe` and defaulted Safe paired runs to fresh, non-resumed outputs.
- Added scene-level jerk p95 and yaw-rate p95 metrics.

## Algorithm

- Added two regime-specific, zero-initialized evidence residual calibrators.
- Each calibrator consumes frozen evidence center/width plus frozen policy score/gap.
- Total new calibrator state size: 132 parameters.
- Residuals are bounded and initial predictions exactly equal the source checkpoint.
- v48.15 adaptation freezes the proposal and full source evidence experts; only the tiny calibrators train.
- Reduced hard-harm amplification and increased missed-opportunity importance for checkpoint selection without changing Natural-gate thresholds.

## Experiment tooling

- Added `scripts/adapt_ocrap_v48_15_prism_cc_variant.sh`.
- Added `scripts/run_v48_15_prism_cc_dedicated.sh`.
- Added `scripts/run_v48_15_parallel_ablations.sh` with four simultaneous tasks per variant wave.
- Added `scripts/recover_v48_14_certificate_pool.sh`.
- Added `tools/check_v48_15_learning_gates.py`.
- Added v48.15 regression tests.
