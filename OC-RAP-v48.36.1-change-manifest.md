# OC-RAP v48.36.1 RC30 CUDA Index Hotfix — Change Manifest

## Algorithm scope

No algorithm or dataset change. The package retains v48.36 OCAF, one network,
one continuous physical representation, one shared deployment rule, five signed
safety components and the non-compensatory frontier cap.

## Runtime code

- `src/ocrap/models/ocrap.py`
  - replaces implicit CUDA advanced-index broadcast with explicit
    `index_select/index_copy_`;
  - hardens the other group-wise row gather/scatter paths;
  - makes the zero-action RMS derivative finite while preserving exact-zero output.
- `scripts/run_v48_36_ocaf_dedicated.sh`
  - executes the exact group-broadcast forward/backward contract on GPU0 and GPU1
    before adaptation.
- `tools/check_v48_36_cuda_group_broadcast_contract.py`
  - new attempt-independent CUDA/CPU runtime contract for batch 96, group 8,
    141-D action and 529-D nominal observation.

## Tests

- `tests/test_v48_36_ocaf.py`
  - real batch-geometry CPU regression;
  - optional CUDA regression;
  - finite backward at exact-zero nominal action;
  - runtime-contract subprocess test;
  - two-GPU main-runner wiring test.

## Analysis and execution artifacts

- `OC-RAP-v48.36-RC30-audit-and-v48.36.1-hotfix-ZH.md`
- `OC-RAP-v48.36-RC30-failure-audit.json`
- `OC-RAP-v48.36.1-run-commands-ZH.txt`
- `OC-RAP-v48.36.1-engineering-error-audit.csv`
- `OC-RAP-v48.36.1-VERSION-CONTRACT.json`
- `OC-RAP-v48.36.1-validation-status.txt`
- `OC-RAP-v48.36.1-full-factor-geometry-smoke.json`
- `OC-RAP-v48.36.1-script-dependency-audit.json`
- `OC-RAP-v48.36.1-release-audit.json`
- `OC-RAP-v48.36-to-v48.36.1-RC30-CUDA-INDEX-HOTFIX.patch`
- `ALGORITHM_CHANGELOG.md`

## Rerun requirement

Use `RESUME_AFTER_ADAPTATION=0`. The failed run produced no factor-stage checkpoint,
so only contract-verified teacher indexes may be reused; old stage outputs and
terminal markers must not be copied into the new output directory.
