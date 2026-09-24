# Contact recovery re-characterization v134

## Why this exists
The frozen 8-anchor Contact result executes OC-RAP nominally at every decision (`intervention_rate=0`). The publication launcher hard-coded `selection.require_absolute_admission_for_intervention=true`, so 297/320 decisions with no absolute-admitted action abstained to nominal. Re-running the same target/configuration is deterministic and will reproduce the same result.

v134 keeps historical behavior by default but makes the absolute-admission deployment gate explicit. It adds a held-out validation sweep before any new test evaluation.

## Stage A — validation-only selector sweep

```bash
BASE_OUT=runs \
OCRAP_MODEL_RUN="$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main" \
CUDA_DEVICES=0,1 \
bash scripts/run_contact_recovery_validation_sweep.sh
```

This mines exact-a0 observed-contact anchors from `val_contact` only and evaluates three already-existing selector semantics:

- `strict_abs`: historical publication launcher (`lcb_constrained`, absolute intervention admission required)
- `guarded_fallback`: `lcb_constrained` with the selector's existing recovery-guarded fallback enabled
- `calibrated_guarded`: existing `calibrated_constrained` selector, no hard absolute abstention

The freeze rule uses validation only. A candidate must cause real interventions, must not materially worsen off-road/re-contact, and must improve controlled recovery success or overlap duration. If none passes, the script exits nonzero and explicitly advises against test tuning.

Outputs:

```
runs/contact_recovery_validation_v134/
  anchors/
  profiles/{strict_abs,guarded_fallback,calibrated_guarded}/
  CONTACT_RECOVERY_PROFILE.json
```

## Stage B — larger test re-characterization
Run only if Stage A produces `"valid": true`.

```bash
BASE_OUT=runs \
OCRAP_MODEL_RUN="$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main" \
CUDA_DEVICES=0,1 \
JOBS_PER_GPU=1 MAX_PARALLEL=2 USE_DYNAMIC_SCHEDULER=auto \
bash scripts/run_contact_recovery_test_recharacterization.sh
```

Default test cohort requires 30 post-contact steps (3.0 s), which is a stronger quantitative compromise than the current 4.0 s / 8-anchor cohort while substantially increasing sample count. All methods share the same new exact-a0 scene-disjoint cohort. Baselines are not retrained; only their closed-loop evaluation is rerun.

Outputs:

```
runs/contact_recovery_recharacterization_v134/
  test_30step_anchor/
  ocrap/contact/
  external_contact/
  RECHARACTERIZATION_SUMMARY.json
```

## Audit metric
`tools/audit_contact_recovery_execution.py` defines controlled recovery as:

- no off-road event,
- no re-contact,
- sustained post-contact escape,
- terminal clearance >= 0.5 m.

It also reports terminal clearance capped at 5 m and a runaway rate, so large off-road distances cannot improve the audit score.

## What not to do
Do not select a Contact selector using `test_contact`; do not lower off-road/re-contact gates to fill a visualization quota; and do not retrain external non-learning baselines. If all existing selector profiles fail held-out validation, the next scientifically defensible step is a dedicated observed-contact recovery training/calibration pipeline built from train/val/calibration splits only, followed by one frozen test evaluation.
