# Contact qualitative supplement v132

This supplement is qualitative-only and does not modify the frozen publication Contact cohort or quantitative tables.

## Main fixes in v132

- Fixes slowed all-method montage rendering (`metric_name`/layout initialization).
- Renders videos atomically through a temporary file so a failed frame cannot leave a corrupt final MP4.
- Samples the clip endpoint explicitly, so the last rendered state matches the selected clip terminal state.
- Uses gentler default short-clip playback: target 3.5 s, maximum slowdown 1.4x. The video visibly labels the playback factor.
- Adds `tools/prune_stale_regime_visualization_media.py` to remove stale main `rank_*` media while preserving every `supplement_*` directory.

## Resume supplement_01

```bash
BASE_OUT=runs \
OCRAP_MODEL_RUN="$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main" \
MAIN_CHARACTERIZATION_ROOT="$BASE_OUT/ocrap_v48_124_final_characterization" \
MAIN_VIS_ROOT="$BASE_OUT/regime_visualization_v48_124_final_optimized" \
CONTACT_SUPPLEMENT_NAME=supplement_01 \
CUDA_DEVICES=0,1 \
JOBS_PER_GPU=1 \
MAX_PARALLEL=2 \
USE_DYNAMIC_SCHEDULER=auto \
bash scripts/build_contact_qualitative_supplement.sh
```

Completed closed-loop artifacts are resumed/reused.  Figure/video rendering is forced and therefore replaces the previously rendered 0.62x pair with the corrected endpoint-inclusive, gentler-slowdown version.

Final media:

- `runs/regime_visualization_v48_124_final_optimized/videos/contact/supplement_01/`
- `runs/regime_visualization_v48_124_final_optimized/paper_figures/contact/supplement_01/`

Working state and audit:

- `runs/contact_qualitative_supplements/supplement_01/`

## Prune stale main media

```bash
python tools/prune_stale_regime_visualization_media.py \
  --root runs/regime_visualization_v48_124_final_optimized
```

The pruner never touches `supplement_*` directories.
