#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

OUT="${OUT:?OUT must point to the completed v48.26 dedicated run}"
[[ -f "$OUT/V48_26_COMPLETE.json" || -f "$OUT/V48_27_COMPLETE.json" ]] || {
  echo "missing V48_26_COMPLETE.json/V48_27_COMPLETE.json under $OUT" >&2; exit 30;
}
for variant in balanced precision; do
  [[ -f "$OUT/candidates/$variant/model_v48_trac_sr/best.pt" ]] || {
    echo "missing $variant checkpoint under $OUT" >&2; exit 30;
  }
done

old="$OUT/dev_shadow_closed_loop"
if [[ -e "$old" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  mv "$old" "$OUT/dev_shadow_closed_loop.v48_26_failed.$stamp"
fi

# The old run is not retrained. This command only re-executes the adaptation-dev
# physical diagnostic with canonical WOMD IDs, complete raw scanning, strict
# target matching and fail-closed empty-result checks from v48.27.
OUT="$OUT" \
DEV_SHADOW_RAW_MAX_SCENARIOS="${DEV_SHADOW_RAW_MAX_SCENARIOS:-0}" \
CL_RESUME=0 \
  bash scripts/run_v48_27_dev_shadow_closed_loop.sh
