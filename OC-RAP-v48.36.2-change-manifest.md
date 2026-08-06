# OC-RAP v48.36.2 Change Manifest

- Base algorithm: `v48.36-OCAF`
- Implementation: `v48.36.2-STAGE-TRANSFER-HOTFIX`
- Scope: engineering-only stage-transfer contract hotfix
- Algorithm/dataset/gate/certificate-threshold changes: none

## Added (8)

- `OC-RAP-v48.36.1-results-audit-and-v48.36.2-hotfix-ZH.md`
- `OC-RAP-v48.36.2-run-commands-ZH.txt`
- `scripts/repair_v48_36_1_stage_transfer_with_v48_36_2.sh`
- `tests/test_v48_36_2_stage_transfer_hotfix.py`
- `tools/check_v48_36_stage_transfer.py`
- `tools/extract_v48_36_failure_signature.py`
- `tools/finalize_v48_36_adaptation_variant.py`
- `tools/repair_v48_36_1_stage_transfer_failure.py`

## Modified (5)

- `ALGORITHM_CHANGELOG.md`
- `scripts/adapt_ocrap_v48_36_ocaf_variant.sh`
- `scripts/run_v48_36_ocaf_dedicated.sh`
- `tools/check_v48_36_ocaf_training_contract.py`
- `tools/check_v48_36_resume_contract.py`

## Deleted (0)

- None

## Behavioral contract

- A registered OCAF interaction bridge may change during identity training only when the controller prefix set and stage architecture agree exactly.
- Any encoder/source-expert/proposal/frozen-policy drift remains RC=31.
- No-retraining recovery is authorized only for the exact uploaded v48.36.1 failure signature and byte-identical checkpoints.
- Unknown RC=30 states, prior calibration/certificate artifacts, source/protocol changes, or checkpoint hash mismatches fail closed.
