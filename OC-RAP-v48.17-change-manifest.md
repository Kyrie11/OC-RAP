# OC-RAP v48.17 BRIDGE change manifest

## Algorithm and training

- `src/ocrap/models/ocrap.py` — identity-preserving tri-simplex Evidence residual; optional frozen candidate-vs-nominal context.
- `src/ocrap/models/losses.py` — minibatch/regime-level harmful/dead/beneficial balancing and batch-scope bipolar margins.
- `src/ocrap/cli/train.py` — exact teacher-PCD Evidence strata, stratified scene-time batches, recall-shortfall checkpoint metric, config/checkpoint plumbing.
- `src/ocrap/models/inference.py` — load/save support for the new calibrator mode and context contract.
- `configs/default.yaml` — backward-compatible defaults for BRIDGE features.
- `scripts/train_ocrap_v48_trac_sr.sh` — command-line/environment plumbing for sampler, loss, calibrator, and checkpoint settings.

## Controllers and experiment attribution

- `scripts/adapt_ocrap_v48_17_bridge_variant.sh` — low-capacity frozen-source BRIDGE adaptation.
- `scripts/run_v48_17_bridge_dedicated.sh` — two-A30 balanced/precision main controller and valid 0/20/30 semantics.
- `scripts/run_v48_17_parallel_ablations.sh` — non-redundant A/B/C component ablations.
- `scripts/run_v48_17_stress_if_authorized.sh` — Natural-gate authorization wrapper.
- `scripts/calibrate_v48_14_certificate_pool.sh` — correct `teacher_advantage_mean` use when choosing between gate-passing variants.

## Safe and reporting

- `src/ocrap/simulation/closed_loop_runner.py` — fixed-route signed progression at scene level, including explicit route source.
- `tools/analyze_safe_paired_noninferiority_v48_8.py` — 5% jerk/yaw-rate margins and strict all-metric paper-ready gate.
- `tools/check_v48_16_learning_gates.py` — corrected certificate field names.
- `tools/summarize_v48_14_ablations.py` — repaired dedicated paths, metric keys, version metadata and error handling.

## Tests and documentation

- `tests/test_v48_17_bridge.py` — simplex identity/bounds, parameter/context path, stratified sampler, route progression.
- `ALGORITHM_CHANGELOG.md` — v48.17 diagnosis, algorithm, non-repetition contract and stopping rules.
- `OC-RAP-v48.16-audit-and-v48.17-BRIDGE-plan-ZH.md` — full paper/code/result/dataset audit.
- `OC-RAP-v48.17-run-commands-ZH.txt` — exact two-A30 execution sequence.
- `OC-RAP-v48.16-results-audit-summary.json` — structured uploaded-result audit.
- `OC-RAP-v48.16-safe-reanalysis-with-v48.17-code.json` — old Safe result rechecked with corrected margins.
- `OC-RAP-v48.17-validation-status.txt` — local validation evidence and unexecuted-runtime boundary.
