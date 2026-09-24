#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
usage(){ cat <<'TXT'
Usage:
  scripts/build_ocrap_datasets.sh --role train [--role validation-test] [--role calibration]
  scripts/build_ocrap_datasets.sh --all

Primary source contract:
  train                -> WOMD training
  validation/test      -> WOMD standard validation
  calibration          -> WOMD standard validation
validation_interactive is deliberately excluded from the publication build.

Environment:
  OCRAP_ROOT             final dataset root (default /data0/senzeyu2/dataset/OCRAP)
  WOMD_ROOT              WOMD tf_example root
  GPU0/GPU1              build GPUs (default 0/1 where used)
  CALIBRATION_BUILD_ROOT temporary calibration build root (default $OCRAP_ROOT/.calibration_build)
  CALIBRATION_OVERWRITE  1 only when intentionally replacing final calibration_* roots
TXT
}
roles=()
while (($#)); do
  case "$1" in
    --role) roles+=("${2:?missing value for --role}"); shift 2 ;;
    --all) roles=(train validation-test calibration); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
done
((${#roles[@]})) || { usage >&2; exit 2; }

export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
export GPU0="${GPU0:-0}" GPU1="${GPU1:-1}"

for r in "${roles[@]}"; do
  case "$r" in
    train)
      bash scripts/build_training_datasets.sh
      ;;
    validation-test)
      bash scripts/build_validation_test_datasets.sh
      ;;
    calibration)
      CALIBRATION_BUILD_ROOT="${CALIBRATION_BUILD_ROOT:-$OCRAP_ROOT/.calibration_build}"
      OUTPUT_ROOT="$CALIBRATION_BUILD_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
        bash scripts/build_calibration_datasets.sh
      BUILD_ROOT="$CALIBRATION_BUILD_ROOT" TARGET_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
        OVERWRITE="${CALIBRATION_OVERWRITE:-0}" \
        bash scripts/merge_calibration_datasets.sh
      ;;
    *) echo "invalid role $r" >&2; exit 2 ;;
  esac
done
