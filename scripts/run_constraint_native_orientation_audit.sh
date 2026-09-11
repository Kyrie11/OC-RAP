#!/usr/bin/env bash
# Stable constraint-native orientation audit entrypoint.
# V48.115 OC-RSCF: selector-free recovery-set constraint-flow audit after
# authoritative V48.114 OC-CCW STOP. Audit only: no planner/source training.
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

# V48.114 is an immutable versioned scientific input, not a source-code dependency.
V114_PIPELINE="${OCRAP_ORIENTATION_V114_PIPELINE:-$BASE_OUT/OC-RAP-v48.114-PIPELINE_COMPLETE.json}"
V114_COMPARE="${OCRAP_ORIENTATION_V114_COMPARE:-$BASE_OUT/OC-RAP-v48.114-DCP-DRFC-BCDE-RIFA-OC-CCW-comparison.json}"
V114_BALANCED="${OCRAP_ORIENTATION_V114_BALANCED:-$BASE_OUT/OC-RAP-v48.114-CCW-balanced.json}"
V114_PRECISION="${OCRAP_ORIENTATION_V114_PRECISION:-$BASE_OUT/OC-RAP-v48.114-CCW-precision.json}"

CACHE="${OCRAP_ORIENTATION_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_115_rscf_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.115-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.115-RSCF-balanced.json"
POUT="$BASE_OUT/OC-RAP-v48.115-RSCF-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.115-RSCF-balanced.pt"
PSTATE="$BASE_OUT/OC-RAP-v48.115-RSCF-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.115-DCP-DRFC-BCDE-RIFA-OC-RSCF-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.115-PIPELINE_COMPLETE.json"
BUNDLE_MANIFEST="$BASE_OUT/OC-RAP-v48.115-OC-RSCF-result-bundle-manifest.json"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.115-OC-RSCF-results.zip"

mkdir -p "$BASE_OUT" "$CACHE"
rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$BUNDLE_MANIFEST" "$RESULTS_ZIP"

# Fail before GPU work if checkout/import path does not satisfy V48.115.
python tools/check_constraint_native_orientation_contract.py \
  --repo "$REPO" --run-id "$RUN_ID" --output "$RUNTIME"

# Authoritative V48.114 STOP is the only branch that licenses this audit.
python - "$V114_PIPELINE" "$V114_COMPARE" "$V114_BALANCED" "$V114_PRECISION" <<'PY'
import hashlib, json, pathlib, sys
p, c, b, q = map(pathlib.Path, sys.argv[1:])
want = {
    p: '4887300ad1af2232c36ce4d8101ca3526ce7eeae056be846b114f91b14e43ece',
    c: '7ce7d09809da348fb3229c0d89338858fa05fe461301088a9a0e6392bf8c0319',
    b: '432dce3a32a5c4a52ad10f77ca4ff823a9d2ded14502efff9c6eb689dc5e2b21',
    q: '7b6915c09502485d06ddc6b08f6f30e0876ead70f3a6ad85fffd020a7e7039ce',
}
for path, digest in want.items():
    if not path.is_file():
        raise SystemExit(f'missing V48.114 prerequisite {path}')
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise SystemExit(f'authoritative V48.114 SHA mismatch {path.name}: {got}')
pd = json.loads(p.read_text())
cd = json.loads(c.read_text())
d = cd.get('preregistered_decision') or {}
if not (pd.get('valid') and pd.get('attribution_ready') and pd.get('preregistered_status') == 'COMMON_OPTION_CONSTRAINT_WORK_STOP'):
    raise SystemExit('authoritative V48.114 STOP pipeline prerequisite missing')
if not (cd.get('valid') and cd.get('attribution_ready') and d.get('status') == 'COMMON_OPTION_CONSTRAINT_WORK_STOP'):
    raise SystemExit('authoritative V48.114 comparison STOP prerequisite missing')
if d.get('next_branch') != 'close_selected_option_fixed_bin_work_family_then_preregister_selector_free_recovery_set_constraint_flow_audit_no_capacity_or_regime_sweep':
    raise SystemExit('V48.114 did not authorize selector-free recovery-set flow branch')
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
  echo "V48.115 RSCF run failure balanced=$r0 precision=$r1" >&2
  exit 30
}

python tools/compare_constraint_native_recovery_orientation.py \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --v114-pipeline "$V114_PIPELINE" \
  --v114-comparison "$V114_COMPARE" \
  --v114-balanced "$V114_BALANCED" \
  --v114-precision "$V114_PRECISION" \
  --run-id "$RUN_ID" \
  --output "$COMPARE"

python tools/check_constraint_native_orientation_pipeline.py \
  --runtime "$RUNTIME" \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --balanced-state "$BSTATE" \
  --precision-state "$PSTATE" \
  --comparison "$COMPARE" \
  --v48-114-pipeline "$V114_PIPELINE" \
  --v48-114-comparison "$V114_COMPARE" \
  --run-id "$RUN_ID" \
  --output "$COMPLETE"

python tools/package_constraint_native_orientation_results.py \
  --pipeline "$COMPLETE" \
  --base-out "$BASE_OUT" \
  --run-id "$RUN_ID" \
  --manifest "$BUNDLE_MANIFEST" \
  --output "$RESULTS_ZIP"

printf 'V48.115 OC-RSCF result bundle ready. Upload ONLY this file:\n%s\nrun_instance_id=%s\n' "$RESULTS_ZIP" "$RUN_ID"
