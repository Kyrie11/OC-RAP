#!/usr/bin/env bash
# Stable constraint-native orientation audit entrypoint.
# V48.118 OC-VSE: recovery-set joint viability survival-envelope audit after
# authoritative V48.117 OC-TBCF STOP. The primary family is a set-level
# max-min envelope over all common valid recovery options; no Main integration.
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

# V48.117 is an immutable versioned scientific input, not a source-code dependency.
V117_PIPELINE="${OCRAP_ORIENTATION_V117_PIPELINE:-$BASE_OUT/OC-RAP-v48.117-PIPELINE_COMPLETE.json}"
V117_COMPARE="${OCRAP_ORIENTATION_V117_COMPARE:-$BASE_OUT/OC-RAP-v48.117-DCP-DRFC-BCDE-RIFA-OC-TBCF-comparison.json}"
V117_BALANCED="${OCRAP_ORIENTATION_V117_BALANCED:-$BASE_OUT/OC-RAP-v48.117-TBCF-balanced.json}"
V117_PRECISION="${OCRAP_ORIENTATION_V117_PRECISION:-$BASE_OUT/OC-RAP-v48.117-TBCF-precision.json}"

CACHE="${OCRAP_ORIENTATION_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_118_vse_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.118-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.118-VSE-balanced.json"
POUT="$BASE_OUT/OC-RAP-v48.118-VSE-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.118-VSE-balanced.pt"
PSTATE="$BASE_OUT/OC-RAP-v48.118-VSE-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.118-DCP-DRFC-BCDE-RIFA-OC-VSE-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.118-PIPELINE_COMPLETE.json"
BUNDLE_MANIFEST="$BASE_OUT/OC-RAP-v48.118-OC-VSE-result-bundle-manifest.json"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.118-OC-VSE-results.zip"

mkdir -p "$BASE_OUT" "$CACHE"
rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$BUNDLE_MANIFEST" "$RESULTS_ZIP"

# Fail before GPU work if checkout/import path does not satisfy V48.118.
python tools/check_constraint_native_orientation_contract.py \
  --repo "$REPO" --run-id "$RUN_ID" --output "$RUNTIME"

# Authoritative V48.117 STOP is the only branch that licenses this audit.
python - "$V117_PIPELINE" "$V117_COMPARE" "$V117_BALANCED" "$V117_PRECISION" <<'PY'
import hashlib, json, pathlib, sys
p, c, b, q = map(pathlib.Path, sys.argv[1:])
want = {
    p: '674902c61b68c05387798f011d5e0b9639efd1eff5f3e1b08b3222d819f7c399',
    c: '8a4e54a71dfc77821842041a053002ef8b4b10c9b6a778270384b7da219fc99b',
    b: '9574b2d30a772c6952f3c5ffd601b2bc15ae5703892c51d8162ced9fc08a8d24',
    q: '836d45e7cf3f8f32692847bde8aefdd985523fe63bee333a16a031d92bee9ea8',
}
for path, digest in want.items():
    if not path.is_file():
        raise SystemExit(f'missing V48.117 prerequisite {path}')
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise SystemExit(f'authoritative V48.117 SHA mismatch {path.name}: {got}')
pd = json.loads(p.read_text())
cd = json.loads(c.read_text())
d = cd.get('preregistered_decision') or {}
if not (pd.get('valid') and pd.get('attribution_ready') and pd.get('preregistered_status') == 'TAIL_BOUNDARY_CROSSING_FLOW_STOP'):
    raise SystemExit('authoritative V48.117 STOP pipeline prerequisite missing')
if not (cd.get('valid') and cd.get('attribution_ready') and d.get('status') == 'TAIL_BOUNDARY_CROSSING_FLOW_STOP'):
    raise SystemExit('authoritative V48.117 comparison STOP prerequisite missing')
if d.get('next_branch') != 'close_static_boundary_witness_hitting_flow_then_preregister_recovery_set_viability_survival_envelope_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep':
    raise SystemExit('V48.117 did not authorize recovery-set viability survival-envelope branch')
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
  echo "V48.118 VSE run failure balanced=$r0 precision=$r1" >&2
  exit 30
}

python tools/compare_constraint_native_recovery_orientation.py \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --v117-pipeline "$V117_PIPELINE" \
  --v117-comparison "$V117_COMPARE" \
  --v117-balanced "$V117_BALANCED" \
  --v117-precision "$V117_PRECISION" \
  --run-id "$RUN_ID" \
  --output "$COMPARE"

python tools/check_constraint_native_orientation_pipeline.py \
  --runtime "$RUNTIME" \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --balanced-state "$BSTATE" \
  --precision-state "$PSTATE" \
  --comparison "$COMPARE" \
  --v48-117-pipeline "$V117_PIPELINE" \
  --v48-117-comparison "$V117_COMPARE" \
  --run-id "$RUN_ID" \
  --output "$COMPLETE"

python tools/package_constraint_native_orientation_results.py \
  --pipeline "$COMPLETE" \
  --base-out "$BASE_OUT" \
  --run-id "$RUN_ID" \
  --manifest "$BUNDLE_MANIFEST" \
  --output "$RESULTS_ZIP"

printf 'V48.118 OC-VSE result bundle ready. Upload ONLY this file:\n%s\nrun_instance_id=%s\n' "$RESULTS_ZIP" "$RUN_ID"
