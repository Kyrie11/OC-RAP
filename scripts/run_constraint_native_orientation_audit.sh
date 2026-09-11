#!/usr/bin/env bash
# Current constraint-native orientation audit entrypoint.
# V48.113 OC-ECJ: candidate x recovery-option executable constraint-Jacobian
# audit after authoritative V48.112 OC-HCNC STOP. Audit only: no planner/source training.
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

# V48.112 is an immutable versioned scientific input, not a source-code dependency.
V112_PIPELINE="${OCRAP_ORIENTATION_V112_PIPELINE:-$BASE_OUT/OC-RAP-v48.112-PIPELINE_COMPLETE.json}"
V112_COMPARE="${OCRAP_ORIENTATION_V112_COMPARE:-$BASE_OUT/OC-RAP-v48.112-DCP-DRFC-BCDE-RIFA-OC-HCNC-comparison.json}"
V112_BALANCED="${OCRAP_ORIENTATION_V112_BALANCED:-$BASE_OUT/OC-RAP-v48.112-HCNC-balanced.json}"
V112_PRECISION="${OCRAP_ORIENTATION_V112_PRECISION:-$BASE_OUT/OC-RAP-v48.112-HCNC-precision.json}"

CACHE="${OCRAP_ORIENTATION_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_113_ecj_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.113-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.113-ECJ-balanced.json"
POUT="$BASE_OUT/OC-RAP-v48.113-ECJ-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.113-ECJ-balanced.pt"
PSTATE="$BASE_OUT/OC-RAP-v48.113-ECJ-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.113-DCP-DRFC-BCDE-RIFA-OC-ECJ-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.113-PIPELINE_COMPLETE.json"
BUNDLE_MANIFEST="$BASE_OUT/OC-RAP-v48.113-OC-ECJ-result-bundle-manifest.json"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.113-OC-ECJ-results.zip"

mkdir -p "$BASE_OUT" "$CACHE"
rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$BUNDLE_MANIFEST" "$RESULTS_ZIP"

python tools/check_constraint_native_orientation_contract.py \
  --repo "$REPO" --run-id "$RUN_ID" --output "$RUNTIME"

python - "$V112_PIPELINE" "$V112_COMPARE" "$V112_BALANCED" "$V112_PRECISION" <<'PY'
import hashlib, json, pathlib, sys
p, c, b, q = map(pathlib.Path, sys.argv[1:])
want = {
    p: '426329407549b0408c4d6a225fa40239c13115f49b05d834a6c631f1e460e6ea',
    c: '0f07aed5ad1d52572a91b3491dbb63d21f1e2c231c1321bb52a66e5cbe351d64',
    b: '20bc2c049f4793ca8ff74fe428ffeaf1284fad74ce03036e1f45ff20bf212a1b',
    q: '16129135e0509566e31604e6a80c5a5a82788d2452eca1d00c19cd0459953995',
}
for path, digest in want.items():
    if not path.is_file():
        raise SystemExit(f'missing V48.112 prerequisite {path}')
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise SystemExit(f'authoritative V48.112 SHA mismatch {path.name}: {got}')
pd = json.loads(p.read_text())
cd = json.loads(c.read_text())
d = cd.get('preregistered_decision') or {}
if not (pd.get('valid') and pd.get('attribution_ready') and pd.get('preregistered_status') == 'HETEROGENEOUS_CONSTRAINT_NORMAL_CONE_STOP'):
    raise SystemExit('authoritative V48.112 STOP pipeline prerequisite missing')
if not (cd.get('valid') and cd.get('attribution_ready') and d.get('status') == 'HETEROGENEOUS_CONSTRAINT_NORMAL_CONE_STOP'):
    raise SystemExit('authoritative V48.112 comparison STOP prerequisite missing')
if d.get('next_branch') != 'close_prefix_level_first_order_constraint_cone_then_preregister_candidate_option_executable_constraint_jacobian_audit_no_training_or_source_sweep':
    raise SystemExit('V48.112 did not authorize candidate-option executable constraint-Jacobian branch')
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
  echo "V48.113 ECJ run failure balanced=$r0 precision=$r1" >&2
  exit 30
}

python tools/compare_constraint_native_recovery_orientation.py \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --v112-pipeline "$V112_PIPELINE" \
  --v112-comparison "$V112_COMPARE" \
  --v112-balanced "$V112_BALANCED" \
  --v112-precision "$V112_PRECISION" \
  --run-id "$RUN_ID" \
  --output "$COMPARE"

python tools/check_constraint_native_orientation_pipeline.py \
  --runtime "$RUNTIME" \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --balanced-state "$BSTATE" \
  --precision-state "$PSTATE" \
  --comparison "$COMPARE" \
  --v48-112-pipeline "$V112_PIPELINE" \
  --v48-112-comparison "$V112_COMPARE" \
  --run-id "$RUN_ID" \
  --output "$COMPLETE"

python tools/package_constraint_native_orientation_results.py \
  --pipeline "$COMPLETE" \
  --base-out "$BASE_OUT" \
  --run-id "$RUN_ID" \
  --manifest "$BUNDLE_MANIFEST" \
  --output "$RESULTS_ZIP"

printf 'V48.113 OC-ECJ result bundle ready. Upload ONLY this file:\n%s\nrun_instance_id=%s\n' "$RESULTS_ZIP" "$RUN_ID"
