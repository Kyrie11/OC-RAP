# OC-RAP v48.35.1 Change Manifest

## Version

`v48.35.1-RC30-TRAINING-CONTRACT-HOTFIX`

## Scope

Engineering-only repair for the known v48.35 stale training-contract metadata-key failure. No algorithm, model, loss, dataset, candidate set, rule fitter, certificate threshold or gate change.

## Modified files

1. `scripts/adapt_ocrap_v48_35_continuous_frontier_single_stage.sh`
   - writes exact eligibility metadata and checkpoint-config provenance.
2. `tools/check_v48_35_training_contract.py`
   - verifies exact eligibility from all trusted stage checkpoints;
   - permits legacy metadata only when the new key is absent;
   - rejects explicit false/legacy conflicts.
3. `scripts/run_v48_35_continuous_frontier_dedicated.sh`
   - adds fail-closed `RESUME_AFTER_ADAPTATION=1` path;
   - authorizes before cleanup, skips retraining, refuses index rebuild;
   - records no-retraining provenance in completion metadata.
4. `ALGORITHM_CHANGELOG.md`
   - records root cause, repair, interpretation and validation.

## New files

1. `tools/check_v48_35_resume_contract.py`
2. `scripts/repair_v48_35_rc30_training_contract_with_v48_35_1.sh`
3. `tests/test_v48_35_1_rc30_training_contract_hotfix.py`
4. `OC-RAP-v48.35-RC30-audit-and-v48.35.1-hotfix-ZH.md`
5. `OC-RAP-v48.35.1-run-commands-ZH.txt`
6. `OC-RAP-v48.35.1-engineering-error-audit.csv`
7. `OC-RAP-v48.35.1-result-debug-signals.csv`
8. `OC-RAP-v48.35.1-dataset-report-index.csv`
9. `OC-RAP-v48.35.1-validation-status.txt`
10. `OC-RAP-v48.35.1-CONTINUOUS_FRONTIER_CONTRACT.json`
11. `OC-RAP-v48.35.1-release-audit.json`
12. `OC-RAP-v48.35.1-VERSION-CONTRACT.json`
13. `OC-RAP-v48.35-to-v48.35.1-RC30-TRAINING-CONTRACT-HOTFIX.patch`

## Provenance invariants

- One network and one shared deployment rule.
- No regime ID exposed to the evidence model.
- Candidate-relative physical context only.
- Non-compensatory frontier cap retained.
- Near/Contact remain audit strata.
- Test roots remain sealed until authorization.
- RC=20 remains algorithm rejection; RC=30 remains engineering/protocol failure.

