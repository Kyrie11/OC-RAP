#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=scripts/lib/runtime.sh
source scripts/lib/runtime.sh

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
OUT="${OCRAP_FINAL_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
WOMD_NUM_SHARDS="${WOMD_NUM_SHARDS:-150}"
BUCKET_SPLIT="${BUCKET_SPLIT:-test}"
WOMD_ROLE="${WOMD_ROLE:-validation}"
SAFE_BUCKET="${SAFE_BUCKET:-$OCRAP_ROOT/test_safe}"
NEAR_BUCKET="${NEAR_BUCKET:-$OCRAP_ROOT/test_near_contact}"
CONTACT_BUCKET="${CONTACT_BUCKET:-$OCRAP_ROOT/test_contact}"
mkdir -p "$OUT/target_keys"

SAFE_WOMD="$(runtime_resolve_bucket_womd_spec "$SAFE_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "$WOMD_ROLE")"
NEAR_WOMD="$(runtime_resolve_bucket_womd_spec "$NEAR_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "$WOMD_ROLE")"
CONTACT_WOMD="$(runtime_resolve_bucket_womd_spec "$CONTACT_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "$WOMD_ROLE")"

build_one() {
  local regime="$1" bucket="$2" womd="$3"
  python tools/build_observation_legal_target_lock.py \
    --bucket-dataset "$bucket" --bucket-split "$BUCKET_SPLIT" \
    --max-targets-per-scene 1 --womd-pattern "$womd" \
    --expected-womd-role "$WOMD_ROLE" \
    --output "$OUT/target_keys/$regime.json"
}
build_one safe "$SAFE_BUCKET" "$SAFE_WOMD"
build_one near "$NEAR_BUCKET" "$NEAR_WOMD"
build_one contact "$CONTACT_BUCKET" "$CONTACT_WOMD"

echo "Final observation-legal target locks are ready under $OUT/target_keys"
