# OC-RAP visualization WOMD replay source fix v53

## Root cause

The physical filesystem layout was not the main problem. The repository had different hard-coded collection defaults:

- OC-RAP three-regime launcher: Safe=validation, Near=validation_interactive, Contact=validation_interactive.
- External Near launcher: validation.
- External Contact launcher: validation_interactive.

Therefore a dataset bucket could be replayed against a different raw collection solely because a launcher default changed.

## New contract

Set one physical root:

`WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example`

The root may contain both:

- `validation/validation_tfexample.tfrecord-xxxxx-of-00150`
- `validation_interactive/validation_interactive_tfexample.tfrecord-xxxxx-of-00150`

`tools/resolve_womd_replay_source.py` reads the OC-RAP bucket's stored `womd_source_role`, maps it to the corresponding subdirectory, verifies all shards, and returns an exact `...tfrecord@150` spec.

All current Safe/Near/Contact external launchers and the OC-RAP three-regime launcher now default to dataset-owned `auto` replay. Explicit manual WOMD specs remain supported, but an explicit role that contradicts bucket provenance is rejected by the normal closed-loop dataset-support preflight.

The submission visualization input contract also records `canonical_replay` for all three regimes. Historical full-metric artifacts are still required to match the bucket-owned role: the fix does not suppress a scientifically real mismatch. If an old result was generated against the wrong collection, rerun only that side before selecting videos.

Near CPSF calibration is also dataset-owned now (`CALIB_WOMD=auto`) so a repaired closed-loop replay cannot silently reuse/refit against the wrong raw collection.
