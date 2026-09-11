#!/usr/bin/env bash
# Current constraint-native orientation audit entrypoint.
# V48.114 OC-CCW: common-option full-horizon constraint-work audit after
# authoritative V48.113 OC-ECJ STOP. Audit only: no planner/source training.
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RUN_ID="${OCRAP_ORIENTATION_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}"
export OCRAP_ORIENTATION_RUN_ID="$RUN_ID"

REFERENCE_A="${OCRAP_ORIENTATION_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${OCRAP_ORIENTATION_MODEL_RUN:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${OCRAP_ORIENTATION_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
CERT_INDEX="${OCRAP_ORIENTATION_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"
DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"

# V48.113 is an immutable versioned scientific input, not a source-code dependency.
V113_PIPELINE="${OCRAP_ORIENTATION_V113_PIPELINE:-$BASE_OUT/OC-RAP-v48.113-PIPELINE_COMPLETE.json}"
V113_COMPARE="${OCRAP_ORIENTATION_V113_COMPARE:-$BASE_OUT/OC-RAP-v48.113-DCP-DRFC-BCDE-RIFA-OC-ECJ-comparison.json}"
V113_BALANCED="${OCRAP_ORIENTATION_V113_BALANCED:-$BASE_OUT/OC-RAP-v48.113-ECJ-balanced.json}"
V113_PRECISION="${OCRAP_ORIENTATION_V113_PRECISION:-$BASE_OUT/OC-RAP-v48.113-ECJ-precision.json}"

CACHE="${OCRAP_ORIENTATION_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_114_ccw_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.114-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.114-CCW-balanced.json"
POUT="$BASE_OUT/OC-RAP-v48.114-CCW-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.114-CCW-balanced.pt"
PSTATE="$BASE_OUT/OC-RAP-v48.114-CCW-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.114-DCP-DRFC-BCDE-RIFA-OC-CCW-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.114-PIPELINE_COMPLETE.json"
BUNDLE_MANIFEST="$BASE_OUT/OC-RAP-v48.114-OC-CCW-result-bundle-manifest.json"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.114-OC-CCW-results.zip"

mkdir -p "$BASE_OUT" "$CACHE"
rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$BUNDLE_MANIFEST" "$RESULTS_ZIP"

# Fail before GPU work if the current checkout/import path does not satisfy the
# V48.114 scientific contract.
python tools/check_constraint_native_orientation_contract.py \
  --repo "$REPO" --run-id "$RUN_ID" --output "$RUNTIME"

# Authoritative V48.113 STOP is the only branch that licenses this audit.
python - "$V113_PIPELINE" "$V113_COMPARE" "$V113_BALANCED" "$V113_PRECISION" <<'PY'
import hashlib, json, pathlib, sys
p, c, b, q = map(pathlib.Path, sys.argv[1:])
want = {
    p: '141164bc3881f734ec64963cf32c1f1b07ac0180e4835cc27b078b02b923930a',
    c: '6ca24f95aaca4198eb565547abdcdd4d7d37aebc2bdf9758b6b003653411cc00',
    b: '544ff65a97ccb54cb90e081e437dfc165f6fb3cda08932c67656e7671efa7baf',
    q: 'ee709ea955731e05277709f0622a9d8cf6bfab58962f01372eb392c3073e4048',
}
for path, digest in want.items():
    if not path.is_file():
        raise SystemExit(f'missing V48.113 prerequisite {path}')
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise SystemExit(f'authoritative V48.113 SHA mismatch {path.name}: {got}')
pd = json.loads(p.read_text())
cd = json.loads(c.read_text())
d = cd.get('preregistered_decision') or {}
if not (pd.get('valid') and pd.get('attribution_ready') and pd.get('preregistered_status') == 'EXECUTABLE_CONSTRAINT_JACOBIAN_STOP'):
    raise SystemExit('authoritative V48.113 STOP pipeline prerequisite missing')
if not (cd.get('valid') and cd.get('attribution_ready') and d.get('status') == 'EXECUTABLE_CONSTRAINT_JACOBIAN_STOP'):
    raise SystemExit('authoritative V48.113 comparison STOP prerequisite missing')
if d.get('next_branch') != 'close_selected_option_first_order_jacobian_then_preregister_common_option_constraint_work_audit_no_capacity_or_regime_sweep':
    raise SystemExit('V48.113 did not authorize common-option constraint-work branch')
PY

for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT"; do
  [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }
done

run_one() {
  local variant="$1"
  local gpu="$2"
  local out="$3"
  local state="$4"
  local ckpt="$L80_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing frozen L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_constraint_native_recovery_orientation_audit.py \
    --checkpoint "$ckpt" \
    --train-index "$TRAIN_INDEX" \
    --dev-index "$DEV_INDEX" \
    --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" \
    --cache-dir "$CACHE/$variant" \
    --device cuda \
    --variant "$variant" \
    --run-id "$RUN_ID" \
    --output "$out" \
    --state-output "$state"
}

set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" & p1=$!
wait "$p0"; r0=$?
wait "$p1"; r1=$?
set -e
[[ $r0 == 0 && $r1 == 0 ]] || {
  echo "V48.114 CCW run failure balanced=$r0 precision=$r1" >&2
  exit 30
}

python tools/compare_constraint_native_recovery_orientation.py \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --v113-pipeline "$V113_PIPELINE" \
  --v113-comparison "$V113_COMPARE" \
  --v113-balanced "$V113_BALANCED" \
  --v113-precision "$V113_PRECISION" \
  --run-id "$RUN_ID" \
  --output "$COMPARE"

python tools/check_constraint_native_orientation_pipeline.py \
  --runtime "$RUNTIME" \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --balanced-state "$BSTATE" \
  --precision-state "$PSTATE" \
  --comparison "$COMPARE" \
  --v48-113-pipeline "$V113_PIPELINE" \
  --v48-113-comparison "$V113_COMPARE" \
  --run-id "$RUN_ID" \
  --output "$COMPLETE"

python tools/package_constraint_native_orientation_results.py \
  --pipeline "$COMPLETE" \
  --base-out "$BASE_OUT" \
  --run-id "$RUN_ID" \
  --manifest "$BUNDLE_MANIFEST" \
  --output "$RESULTS_ZIP"

printf 'V48.114 OC-CCW result bundle ready. Upload ONLY this file:\n%s\nrun_instance_id=%s\n' "$RESULTS_ZIP" "$RUN_ID"
