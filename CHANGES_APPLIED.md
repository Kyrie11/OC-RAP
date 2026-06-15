# OC-RAP patch summary

This patch aligns the implementation more closely with the paper and adds build-time profiling tools.

## Main changes

- Replaced the previous global-scene-output neural model with a root-query decoder:
  - learned recovery-root queries
  - cross-attention from root queries to scene tokens
  - root self-attention
  - option-conditioned recovery margin prediction
  - root-specific post-prefix observation embeddings and compatibility matrix
  - optional recovery-signature and future-signature prediction heads
- Exposed token-level outputs from `StructuredTokenEncoder` for root-query decoding.
- Fixed candidate feature construction to use semantic macro type, not candidate index.
- Added training losses for root recovery signatures and future signatures when labels exist.
- Fixed CUDA requirement handling for `training.device=auto`.
- Added per-sample build profiling CSV at `build_profile.csv` when `profiling.enabled=true`.
- Added `tools/watch_build.py` to monitor dataset building progress and bottleneck timings.

## Validation performed

- `PYTHONPATH=src pytest -q` passed.
- `python -m compileall -q src tools` passed.
- A synthetic profiled dataset build completed and produced `build_profile.csv`.
- `tools/watch_build.py --once` successfully summarized the build and identified the teacher-margin stage as the bottleneck on the synthetic smoke run.

## Remaining scope notes

The repository was validated without external WOMD/Waymax data. Full benchmark fidelity still depends on running the patched code in the target Waymax/WOMD environment and comparing the resulting metrics against the paper tables.
