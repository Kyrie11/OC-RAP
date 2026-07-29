# v48.14 PRISM change manifest

## Algorithm

- Added calibration-domain dynamic false-safe hard-harm and missed-benefit weighting to the ordered three-state evidence loss.
- Added scene-disjoint dedicated calibration partitioning into evidence adaptation train, adaptation dev, and independent certificate pool.
- Added frozen-proposal, evidence-adapter-only target-domain adaptation.
- Added dedicated certificate-pool calibration with atomic standard calibration, gamma, policy fit/verify, candidate selection, and Natural-gate completion markers.

## Engineering correctness

- Safe nominal-only paired evaluation no longer requires Near/Contact gamma or calibration artifacts.
- Fixed the v48.13 Bash `GROUPS` special-variable ablation bug.
- Fixed v48.13 ordered-NLL effective-weight propagation.
- Added source-checkpoint dedicated-only recalibration without modifying the proxy run.
- Added required-file checks, SHA256 checkpoint provenance, policy-contract propagation, and no-test-root protocol markers.

## Experiment automation

- `tools/partition_dedicated_calibration_v48_14.py`
- `scripts/adapt_ocrap_v48_14_prism_variant.sh`
- `scripts/calibrate_v48_14_certificate_pool.sh`
- `scripts/run_v48_14_prism_dedicated.sh`
- `scripts/recalibrate_source_v48_14_dedicated.sh`
- `scripts/run_v48_14_parallel_ablations.sh`
- `tools/summarize_v48_14_ablations.py`
- `OC-RAP-v48.14-run-commands-ZH.txt`

## Validation

- Added `tests/test_v48_14_prism.py` for hard-harm weighting, scene-disjoint partitioning, Safe nominal-only dependencies, and the Bash special-variable regression.
