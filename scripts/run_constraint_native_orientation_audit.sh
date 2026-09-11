#!/usr/bin/env bash
# Stable constraint-native orientation audit entrypoint.
# V48.117 OC-TBCF: weak-root-exposed zero-boundary ownership and
# first-violation / persistent-reentry crossing-flow audit after authoritative
# V48.116 WRCF STOP.
# Audit only: frozen root/margin heads are read-only; no planner/source/root training.
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

# V48.116 is an immutable versioned scientific input, not a source-code dependency.
V116_PIPELINE="${OCRAP_ORIENTATION_V116_PIPELINE:-$BASE_OUT/OC-RAP-v48.116-PIPELINE_COMPLETE.json}"
V116_COMPARE="${OCRAP_ORIENTATION_V116_COMPARE:-$BASE_OUT/OC-RAP-v48.116-DCP-DRFC-BCDE-RIFA-OC-WRCF-comparison.json}"
V116_BALANCED="${OCRAP_ORIENTATION_V116_BALANCED:-$BASE_OUT/OC-RAP-v48.116-WRCF-balanced.json}"
V116_PRECISION="${OCRAP_ORIENTATION_V116_PRECISION:-$BASE_OUT/OC-RAP-v48.116-WRCF-precision.json}"

CACHE="${OCRAP_ORIENTATION_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_117_tbcf_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.117-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.117-TBCF-balanced.json"
POUT="$BASE_OUT/OC-RAP-v48.117-TBCF-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.117-TBCF-balanced.pt"
PSTATE="$BASE_OUT/OC-RAP-v48.117-TBCF-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.117-DCP-DRFC-BCDE-RIFA-OC-TBCF-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.117-PIPELINE_COMPLETE.json"
BUNDLE_MANIFEST="$BASE_OUT/OC-RAP-v48.117-OC-TBCF-result-bundle-manifest.json"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.117-OC-TBCF-results.zip"

mkdir -p "$BASE_OUT" "$CACHE"
rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$BUNDLE_MANIFEST" "$RESULTS_ZIP"

# Fail before GPU work if checkout/import path does not satisfy V48.117.
python tools/check_constraint_native_orientation_contract.py \
  --repo "$REPO" --run-id "$RUN_ID" --output "$RUNTIME"

# Authoritative V48.116 STOP is the only branch that licenses this audit.
python - "$V116_PIPELINE" "$V116_COMPARE" "$V116_BALANCED" "$V116_PRECISION" <<'PY'
import hashlib, json, pathlib, sys
p, c, b, q = map(pathlib.Path, sys.argv[1:])
want = {
    p: '105d6e47cb046f5dac106bf2930004a89f032ab92faf5b3c988d1ee11b500e9e',
    c: 'cca845f43b2d3e0c7a774f87ab26faad79de739a96217ef1a72242acf94d8f0f',
    b: 'c35ecae10f6ed9729314e19f8e46203e0c763eaad902a2cd3470f4d8c01746c3',
    q: '87b9e08e11b4fb58def4c58c0577f82117e23f0b6137972ebb967cd06f77bede',
}
for path, digest in want.items():
    if not path.is_file():
        raise SystemExit(f'missing V48.116 prerequisite {path}')
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise SystemExit(f'authoritative V48.116 SHA mismatch {path.name}: {got}')
pd = json.loads(p.read_text())
cd = json.loads(c.read_text())
d = cd.get('preregistered_decision') or {}
if not (pd.get('valid') and pd.get('attribution_ready') and pd.get('preregistered_status') == 'WEAK_ROOT_RECOVERY_SET_FLOW_STOP'):
    raise SystemExit('authoritative V48.116 STOP pipeline prerequisite missing')
if not (cd.get('valid') and cd.get('attribution_ready') and d.get('status') == 'WEAK_ROOT_RECOVERY_SET_FLOW_STOP'):
    raise SystemExit('authoritative V48.116 comparison STOP prerequisite missing')
if d.get('next_branch') != 'close_first_order_nominal_ocmero_cotangent_option_pushforward_then_preregister_tail_boundary_crossing_flow_audit_no_training_capacity_regime_or_source_sweep':
    raise SystemExit('V48.116 did not authorize tail-boundary crossing-flow branch')
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
  echo "V48.117 TBCF run failure balanced=$r0 precision=$r1" >&2
  exit 30
}

python tools/compare_constraint_native_recovery_orientation.py \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --v116-pipeline "$V116_PIPELINE" \
  --v116-comparison "$V116_COMPARE" \
  --v116-balanced "$V116_BALANCED" \
  --v116-precision "$V116_PRECISION" \
  --run-id "$RUN_ID" \
  --output "$COMPARE"

python tools/check_constraint_native_orientation_pipeline.py \
  --runtime "$RUNTIME" \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --balanced-state "$BSTATE" \
  --precision-state "$PSTATE" \
  --comparison "$COMPARE" \
  --v48-116-pipeline "$V116_PIPELINE" \
  --v48-116-comparison "$V116_COMPARE" \
  --run-id "$RUN_ID" \
  --output "$COMPLETE"

python tools/package_constraint_native_orientation_results.py \
  --pipeline "$COMPLETE" \
  --base-out "$BASE_OUT" \
  --run-id "$RUN_ID" \
  --manifest "$BUNDLE_MANIFEST" \
  --output "$RESULTS_ZIP"

printf 'V48.117 OC-TBCF result bundle ready. Upload ONLY this file:\n%s\nrun_instance_id=%s\n' "$RESULTS_ZIP" "$RUN_ID"
