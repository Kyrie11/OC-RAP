#!/usr/bin/env bash
set -Eeuo pipefail
# v48.56 TCSA: audit the existing v48.55 teacher/component semantics before any
# centering, root-logit recalibration, or new performance training.  No test root.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
V4855_RUN="${V4855_RUN:-$BASE_OUT/ocrap_v48_55_dcp_drfc_bcde_tcbc_main}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-/data0/senzeyu2/dataset/OCRAP/calibration_v48_14_prism_4814}"
TRAIN_INDEX="${TRAIN_INDEX:-$V4855_RUN/evidence_adapt_teacher_pcd_index.jsonl}"
DEV_INDEX="${DEV_INDEX:-$V4855_RUN/evidence_adapt_dev_teacher_pcd_index.jsonl}"
OUTPUT="${OUTPUT:-$BASE_OUT/OC-RAP-v48.56-DCP-DRFC-BCDE-TCSA-semantic-audit.json}"
MAX_SOURCE_SAMPLES="${V4856_SOURCE_MAX_SAMPLES:-0}"

[[ -s "$TRAIN_INDEX" ]] || { echo "missing train teacher index: $TRAIN_INDEX" >&2; exit 30; }
[[ -s "$DEV_INDEX" ]] || { echo "missing dev teacher index: $DEV_INDEX" >&2; exit 30; }

DATASETS=(
  "$PROTOCOL_ROOT/evidence_adapt_train_near_contact"
  "$PROTOCOL_ROOT/evidence_adapt_train_contact"
  "$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"
  "$PROTOCOL_ROOT/evidence_adapt_dev_contact"
)
source_args=()
all_source=1
for d in "${DATASETS[@]}"; do
  case "$(basename "$d")" in test|test_*|*_test) echo "refuse test-role input: $d" >&2; exit 30;; esac
  [[ -d "$d" && -s "$d/manifest.csv" ]] || all_source=0
done
if [[ "$all_source" == 1 ]]; then
  joined="$(IFS=,; echo "${DATASETS[*]}")"
  source_args+=(--dataset "$joined" --max-source-samples "$MAX_SOURCE_SAMPLES")
else
  echo "[v48.56] canonical adaptation roots unavailable; running index-only semantic audit." >&2
  echo "[v48.56] source-label freshness will remain unresolved until dataset roots are supplied." >&2
fi

mkdir -p "$(dirname "$OUTPUT")"
"$PYTHON_BIN" tools/audit_v48_56_teacher_component_semantics.py \
  --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" \
  "${source_args[@]}" --output "$OUTPUT"
printf '[v48.56] semantic audit written: %s\n' "$OUTPUT"
