#!/usr/bin/env bash
# Re-aggregate publication Near-Contact TTC p05 from existing scene journals.
# No GPU/planner rerun is required because TTC p05 is a scene-level aggregate of
# metric_summary.ttc_s_min already persisted by closed-loop evaluation.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

BASE_OUT="${BASE_OUT:-runs}"
BASELINE_RUN="${BASELINE_RUN:-$BASE_OUT/external_baselines_v48_124_final_v2/near}"
OCRAP_RUN="${OCRAP_RUN:-$BASE_OUT/ocrap_v48_124_final_characterization}"
OUTPUT="${OUTPUT:-$BASELINE_RUN/near_ttc_p05_reaggregation.json}"
EXPECTED_SCENES="${EXPECTED_SCENES:-250}"

python tools/recompute_near_ttc_p05.py \
  --baseline-run "$BASELINE_RUN" \
  --ocrap-run "$OCRAP_RUN" \
  --expected-scenes "$EXPECTED_SCENES" \
  --output "$OUTPUT"
