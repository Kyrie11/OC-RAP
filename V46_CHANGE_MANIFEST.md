# v46 OC-RACE Change Manifest

## Modified existing files

- `ALGORITHM_CHANGELOG.md`: appended v45 observed evidence, v46 design, falsification rules, ablations, and publication blockers.
- `scripts/train_ocrap_v39_ocrac.sh`: preserve explicit empty `FREEZE_PREFIXES` for full unfreeze.
- `src/ocrap/cli/train.py`: frozen-subtree eval mode, expert-specific direct losses, supervised router, per-regime/worst metrics.
- `src/ocrap/config/defaults.py`: router supervision loss and router-pooling compatibility defaults.
- `src/ocrap/models/inference.py`: persist/load router pooling configuration.
- `src/ocrap/models/losses.py`: tie dead zone and deployable candidate listwise objective.
- `src/ocrap/models/ocrap.py`: candidate-invariant shared-observation router option and expert diagnostics.

## New files

- `run_v46_two_gpu_fast_commands.txt`
- `scripts/train_ocrap_v46_race.sh`
- `scripts/calibrate_ocrap_v46_race.sh`
- `scripts/run_ocrap_v46_race.sh`
- `tools/calibrate_direct_value_risk_v46.py`
- `tools/select_v46_candidate.py`
- `tools/check_v46_quick_gate.py`
- `tests/test_v46_race.py`
- `V45_GATE_FAILURE_AND_V46_OC_RACE_ZH.md`
- `V46_RUN_INSTRUCTIONS_ZH.md`
- `V45_METRICS_SUMMARY.json`
- `V46_CHANGE_MANIFEST.md`

## Explicitly not modified

- `post-collision.tex`: analyzed but not edited in this delivery.
- OC-MERO core equations/implementation, observation kernel, root construction, teacher margins, and Safe selector semantics.
- Existing v40–v45 scripts and historical result artifacts.
- Dataset contents, split membership, Waymax/WOMD files, and external baseline outputs.
- Experimental facts, figures, tables, citations, or paper layout.

## Verification

- `pytest`: 100 passed, 3 pre-existing nested-tensor warnings.
- `compileall`: passed.
- `bash -n`: passed for all v46 entry scripts and the modified v39 training script.
- CLI `--help`: passed for all new v46 tools.
