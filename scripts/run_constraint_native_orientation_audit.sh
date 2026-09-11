#!/usr/bin/env bash
# Stable constraint-native orientation audit entrypoint.
# V48.116 OC-WRCF: nominal OC-MERO weak-root cotangent weighted recovery-set
# constraint-flow audit after authoritative V48.115 RSCF STOP.
# Audit only: frozen root/margin heads are read-only; no planner/source training.
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

# V48.115 is an immutable versioned scientific input, not a source-code dependency.
V115_PIPELINE="${OCRAP_ORIENTATION_V115_PIPELINE:-$BASE_OUT/OC-RAP-v48.115-PIPELINE_COMPLETE.json}"
V115_COMPARE="${OCRAP_ORIENTATION_V115_COMPARE:-$BASE_OUT/OC-RAP-v48.115-DCP-DRFC-BCDE-RIFA-OC-RSCF-comparison.json}"
V115_BALANCED="${OCRAP_ORIENTATION_V115_BALANCED:-$BASE_OUT/OC-RAP-v48.115-RSCF-balanced.json}"
V115_PRECISION="${OCRAP_ORIENTATION_V115_PRECISION:-$BASE_OUT/OC-RAP-v48.115-RSCF-precision.json}"

CACHE="${OCRAP_ORIENTATION_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_116_wrcf_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.116-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.116-WRCF-balanced.json"
POUT="$BASE_OUT/OC-RAP-v48.116-WRCF-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.116-WRCF-balanced.pt"
PSTATE="$BASE_OUT/OC-RAP-v48.116-WRCF-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.116-DCP-DRFC-BCDE-RIFA-OC-WRCF-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.116-PIPELINE_COMPLETE.json"
BUNDLE_MANIFEST="$BASE_OUT/OC-RAP-v48.116-OC-WRCF-result-bundle-manifest.json"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.116-OC-WRCF-results.zip"

mkdir -p "$BASE_OUT" "$CACHE"
rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$BUNDLE_MANIFEST" "$RESULTS_ZIP"

# Fail before GPU work if checkout/import path does not satisfy V48.116.
python tools/check_constraint_native_orientation_contract.py \
  --repo "$REPO" --run-id "$RUN_ID" --output "$RUNTIME"

# Authoritative V48.115 STOP is the only branch that licenses this audit.
python - "$V115_PIPELINE" "$V115_COMPARE" "$V115_BALANCED" "$V115_PRECISION" <<'PY'
import hashlib, json, pathlib, sys
p, c, b, q = map(pathlib.Path, sys.argv[1:])
want = {
    p: 'eaf196f55c8b5d9ab32111e7c2ea27eacd7a01ce123fa50237d51562da15c4e2',
    c: '70d5fe0ed97ad08f1d571ba73add12a152e0b1115312b420ff481a233e037b42',
    b: '9be23e261c8ba0f5e494eb136f663c3e4e960e2400d15af10b78e3ea41e50248',
    q: '4a0a53eea966e8e9c2791b75e12ab520045d9a5b0d033a648833a8bd1c361118',
}
for path, digest in want.items():
    if not path.is_file():
        raise SystemExit(f'missing V48.115 prerequisite {path}')
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise SystemExit(f'authoritative V48.115 SHA mismatch {path.name}: {got}')
pd = json.loads(p.read_text())
cd = json.loads(c.read_text())
d = cd.get('preregistered_decision') or {}
if not (pd.get('valid') and pd.get('attribution_ready') and pd.get('preregistered_status') == 'RECOVERY_SET_CONSTRAINT_FLOW_STOP'):
    raise SystemExit('authoritative V48.115 STOP pipeline prerequisite missing')
if not (cd.get('valid') and cd.get('attribution_ready') and d.get('status') == 'RECOVERY_SET_CONSTRAINT_FLOW_STOP'):
    raise SystemExit('authoritative V48.115 comparison STOP prerequisite missing')
if d.get('next_branch') != 'close_observation_only_option_set_mean_flow_then_preregister_ocmero_weak_root_conditioned_recovery_set_flow_audit_frozen_roots_no_training_or_capacity_sweep':
    raise SystemExit('V48.115 did not authorize weak-root-conditioned recovery-set flow branch')
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
  echo "V48.116 WRCF run failure balanced=$r0 precision=$r1" >&2
  exit 30
}

python tools/compare_constraint_native_recovery_orientation.py \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --v115-pipeline "$V115_PIPELINE" \
  --v115-comparison "$V115_COMPARE" \
  --v115-balanced "$V115_BALANCED" \
  --v115-precision "$V115_PRECISION" \
  --run-id "$RUN_ID" \
  --output "$COMPARE"

python tools/check_constraint_native_orientation_pipeline.py \
  --runtime "$RUNTIME" \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --balanced-state "$BSTATE" \
  --precision-state "$PSTATE" \
  --comparison "$COMPARE" \
  --v48-115-pipeline "$V115_PIPELINE" \
  --v48-115-comparison "$V115_COMPARE" \
  --run-id "$RUN_ID" \
  --output "$COMPLETE"

python tools/package_constraint_native_orientation_results.py \
  --pipeline "$COMPLETE" \
  --base-out "$BASE_OUT" \
  --run-id "$RUN_ID" \
  --manifest "$BUNDLE_MANIFEST" \
  --output "$RESULTS_ZIP"

printf 'V48.116 OC-WRCF result bundle ready. Upload ONLY this file:\n%s\nrun_instance_id=%s\n' "$RESULTS_ZIP" "$RUN_ID"
