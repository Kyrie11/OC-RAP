# v48.16 Change Manifest

## Protocol and certificate correctness

- `src/ocrap/models/data.py`: semantic dedicated split roles and alias matching.
- `src/ocrap/cli/calibrate.py`: configurable allowed calibration roles and optional validation fallback.
- `tools/calibrate_policy_risk_v48.py`: explicit `--allowed-splits`, split accounting, empty-data artifact exit.
- `scripts/calibrate_v48_14_certificate_pool.sh`: strict `certificate_pool` use, non-empty data/fold validation, correct 0/20/30 semantics.
- `scripts/calibrate_v48_16_certificate_pool.sh`: corrected compatibility wrapper.
- `tools/audit_dedicated_protocol_v48_16.py`: role and scene-leakage preflight.
- `scripts/recover_v48_15_certificate_pool_v48_16.sh`: no-retraining recovery of existing checkpoints.

## ANCHOR algorithm

- `src/ocrap/models/losses.py`: class-balanced ordinal proposal loss, beneficial/harmful probability margins, residual anchoring.
- `src/ocrap/cli/train.py`: new loss/config plumbing and calibrator residual input.
- `scripts/train_ocrap_v48_trac_sr.sh`: new ANCHOR hyperparameter overrides.
- `scripts/adapt_ocrap_v48_16_anchor_variant.sh`: low-capacity anchored target-domain correction.
- `scripts/run_v48_16_anchor_dedicated.sh`: robust main experiment controller.
- `scripts/run_v48_16_parallel_ablations.sh`: four concurrent ablations per variant wave.
- `tools/check_v48_16_learning_gates.py`: data-validity-aware layered diagnostics.

## Safe and closed-loop correctness

- `tools/validate_womd_shards_v48_16.py`: validates TensorFlow `prefix@150` source and all shard files.
- `scripts/run_ocrap_v48_trac_sr.sh`: avoids duplicate shard suffix, defaults full scan for sparse Safe targets.
- `scripts/run_v48_7_safe_noninferiority.sh`: correct validation `@150` default and full-scan setting.
- `src/ocrap/simulation/closed_loop_runner.py`: hard failure when required targets match zero raw scenarios.
- `scripts/run_v48_16_stress_if_authorized.sh`: stress execution only after `NEXT_COMMANDS.txt` authorization.

## Documentation and tests

- Root `ALGORITHM_CHANGELOG.md` updated with v48.16 evidence, non-repetition rule, and required ablation.
- Added `tests/test_v48_16_anchor.py`.
- Added Chinese audit/design report and complete run commands.
