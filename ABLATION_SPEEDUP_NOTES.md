# OC-RAP submission ablation safe-speedup notes

## 1. What the launcher actually evaluates

For each deployed variant (`balanced`, `precision` by default), `ABLATION_SET=main`
queues a fresh native-certified Full reference on Safe/Near/Contact plus these
functional frozen-checkpoint knockouts:

- `no_obs_consistency`: Near + Contact
- `mean_tail`: Near + Contact
- `no_actuator_projection`: Near + Contact
- `no_persistent_reentry`: Contact
- `no_rifa_absolute_admission`: Safe + Near + Contact
- `no_nominal_abstention`: Safe + Near + Contact
- `no_route_alignment`: Near + Contact

That is 18 accuracy regime jobs per deployed variant, 36 for the default two
variants.  If `PROFILE_ISOLATED_LATENCY=true`, the same publication-facing jobs
are subsequently executed under the isolated single-process/single-GPU latency
contract.  That second pass is intentional and is not used as an accuracy
replacement.

Every arm uses the same frozen checkpoint and per-bucket `gamma_rec`; the arm
changes inference-time semantics only.  Jobs use the same frozen target-key
locks and Contact anchor contract.

## 2. Main runtime costs

The per-replan hot path remains scientifically intact:

1. reconstruct `SceneHistory`;
2. generate 24 candidate prefix rollouts;
3. batch model inference and the native semantic recovery certificate;
4. execute Waymax step and publication geometry metrics.

For the semantic witness, each candidate can evaluate 12 deterministic recovery
options over the configured recovery horizon (default 4 s at 10 Hz, about 40
recovery steps).  With 24 candidates this is the expected dominant computation
for native-certified OC-RAP.  It is not removed because it is part of the
scientific intervention.

Cross-arm cache sharing is intentionally not implemented: two ablations can
select different actions at the first replan, after which their closed-loop
histories diverge.  Reusing candidate/model/certificate results across arms
would invalidate the causal ablation.

## 3. Safe optimizations in this revision

### Resume-aware WOMD selected replay

The scene JSONL journal is loaded first.  For canonical target locks, completed
`target_key`s are removed from the sparse Waymax source-index replay request.
Only unfinished target WOMD records are materialized.  Resumed scene records
remain in aggregation, cohort coverage and publication validation.

### Frozen target-lock auto reuse

`BUILD_TARGET_LOCKS` now defaults to `auto`:

- existing Safe/Near/Contact locks + Contact anchor => reuse, then validate;
- missing input => build;
- `BUILD_TARGET_LOCKS=true` => force rebuild;
- `BUILD_TARGET_LOCKS=false` => force reuse/fail if missing.

The original user command needs no change.

### Preflight cache reuse

Per-regime preflight reports now record the SHA256 of the target-lock file.
A later launcher invocation reuses a report only when dataset path, split, WOMD
pattern, source role, target-lock path/hash and validity all match.  Older reports
without the hash are rescanned once.

### Complete jobs are not queued

Publication-valid completed jobs are detected before entering the dynamic work
queue.  Worker-side validation remains as a race-safe fallback.

### No automatic moving/deleting of previous results

The old `incompatible_pre_publication_contract_*` archive/move path is removed.
Compatible partials resume in place.  Incompatible legacy Contact partials or
completed artifacts are left exactly where they are and the launcher fails
closed with `REFUSE-IN-PLACE`.

### Less partial-snapshot overhead without weaker resume

The aggregate `.partial` snapshot default is changed from 64 to 128 scenes.
The authoritative `.scenes.jsonl` journal still appends every completed scene,
so scene-granular resume safety is unchanged.  `partial_write_every_scenes` is
an operational setting excluded from the scientific fingerprint.

### Bottleneck report can inspect currently running jobs

`tools/analyze_ablation_bottlenecks.py` now reads `.scenes.jsonl` when a final
artifact does not yet exist, reports partial completion, and computes measured
per-component totals and deployed-planner seconds/decision.

## 4. Resume behavior

Keep the same `OUT_ROOT`, checkpoint, configs, target locks, `MAX_STEPS`,
`NUM_CANDIDATES`, `NUM_RECOVERY_OPTIONS`, WOMD source and other scientific
settings.  The runner compares its result-affecting fingerprint and loads prior
scenes from final/partial/journal sources.  Complete scenes are not rerun.

The changes in this revision do not alter the scientific fingerprint.  The
partial snapshot frequency and serialization controls are explicitly excluded
from that fingerprint.

## 5. Original compatible command

```bash
GPU0=0 GPU1=1 CUDA_DEVICES=0,1 \
BASE_OUT=/home/senzeyu2/code/OC-RAP/runs \
OUT_ROOT=/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_124_final_ablations \
MODEL_RUN=/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main \
FULL_RUN_ROOT=/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_124_final_characterization/ocrap \
VARIANTS=balanced,precision \
MAX_SCENARIOS=0 MAX_STEPS=40 \
NUM_CANDIDATES=24 NUM_RECOVERY_OPTIONS=12 \
WOMD_ROLE=validation \
ABLATION_SET=main \
bash scripts/run_submission_ablations.sh
```

## 6. Inspect bottlenecks while the run is still in progress

```bash
PYTHONPATH=src python tools/analyze_ablation_bottlenecks.py \
  --root /home/senzeyu2/code/OC-RAP/runs/ocrap_v48_124_final_ablations \
  --full-root /home/senzeyu2/code/OC-RAP/runs/ocrap_v48_124_final_ablations/_native_full_reference \
  --latency-root /home/senzeyu2/code/OC-RAP/runs/ocrap_v48_124_final_ablations/latency_isolated \
  --output /home/senzeyu2/code/OC-RAP/runs/ocrap_v48_124_final_ablations/ablation_bottlenecks.csv
```

## 7. Optional staged workflow

If the immediate goal is to finish all accuracy ablations first, temporarily set
`PROFILE_ISOLATED_LATENCY=false BUILD_TABLES=false`.  Later rerun the original
command.  Completed accuracy artifacts will be skipped and the isolated latency
phase can be completed separately.  Do not parallelize publication latency jobs
if the table is meant to retain the uncontended single-process/single-GPU
latency contract.

## 8. Validation

The modified tree passes the complete repository test suite: `328 passed`.
