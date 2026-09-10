#!/usr/bin/env bash
# Current constraint-native orientation audit entrypoint.
# V48.112 OC-HCNC: heterogeneous active-constraint normal-cone audit after
# authoritative V48.111 OC-CNRO STOP.  Audit only: no planner/source training.
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

# V48.111 is a versioned immutable scientific input, not a code dependency.
V111_PIPELINE="${OCRAP_ORIENTATION_V111_PIPELINE:-$BASE_OUT/OC-RAP-v48.111-PIPELINE_COMPLETE.json}"
V111_COMPARE="${OCRAP_ORIENTATION_V111_COMPARE:-$BASE_OUT/OC-RAP-v48.111-DCP-DRFC-BCDE-RIFA-OC-CNRO-comparison.json}"
V111_BALANCED="${OCRAP_ORIENTATION_V111_BALANCED:-$BASE_OUT/OC-RAP-v48.111-CNRO-balanced.json}"
V111_PRECISION="${OCRAP_ORIENTATION_V111_PRECISION:-$BASE_OUT/OC-RAP-v48.111-CNRO-precision.json}"

CACHE="${OCRAP_ORIENTATION_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_112_hcnc_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.112-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.112-HCNC-balanced.json"
POUT="$BASE_OUT/OC-RAP-v48.112-HCNC-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.112-HCNC-balanced.pt"
PSTATE="$BASE_OUT/OC-RAP-v48.112-HCNC-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.112-DCP-DRFC-BCDE-RIFA-OC-HCNC-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.112-PIPELINE_COMPLETE.json"
BUNDLE_MANIFEST="$BASE_OUT/OC-RAP-v48.112-OC-HCNC-result-bundle-manifest.json"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.112-OC-HCNC-results.zip"

mkdir -p "$BASE_OUT" "$CACHE"
rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$BUNDLE_MANIFEST" "$RESULTS_ZIP"

python tools/check_constraint_native_orientation_contract.py \
  --repo "$REPO" --run-id "$RUN_ID" --output "$RUNTIME"

python - "$V111_PIPELINE" "$V111_COMPARE" "$V111_BALANCED" "$V111_PRECISION" <<'PY'
import hashlib, json, pathlib, sys
p, c, b, q = map(pathlib.Path, sys.argv[1:])
want = {
    p: 'c155ac8277b2fe690be030eaaf4031e75873e1e22d2521ce56d10aabebb56187',
    c: 'ee2a3f13f2793dd8d0a4a1bdf73192a188d21bfac31c1549b3d6d0ae63cb8373',
    b: 'df5dd1a3a590774255f09e92de3e5d4a9762272fb871a3e64abfe6c3d8eb7e19',
    q: 'ec431e27086a2fe2f30f278a1c3fbfbafca5e4d62094a87dc838e9b52e627b66',
}
for path, digest in want.items():
    if not path.is_file():
        raise SystemExit(f'missing V48.112 prerequisite {path}')
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise SystemExit(f'authoritative V48.111 SHA mismatch {path.name}: {got}')
pd = json.loads(p.read_text())
cd = json.loads(c.read_text())
d = cd.get('preregistered_decision') or {}
if not (pd.get('valid') and pd.get('attribution_ready') and pd.get('preregistered_status') == 'CONSTRAINT_NATIVE_ACTIVE_GEOMETRY_STOP'):
    raise SystemExit('authoritative V48.111 STOP pipeline prerequisite missing')
if not (cd.get('valid') and cd.get('attribution_ready') and d.get('status') == 'CONSTRAINT_NATIVE_ACTIVE_GEOMETRY_STOP'):
    raise SystemExit('authoritative V48.111 comparison STOP prerequisite missing')
if d.get('next_branch') != 'close_fixed_cv_circle_agent_geometry_then_preregister_heterogeneous_active_constraint_normal_cone_audit_no_training_or_source_sweep':
    raise SystemExit('V48.111 did not authorize heterogeneous normal-cone branch')
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
  echo "V48.112 HCNC run failure balanced=$r0 precision=$r1" >&2
  exit 30
}

python tools/compare_constraint_native_recovery_orientation.py \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --v111-pipeline "$V111_PIPELINE" \
  --v111-comparison "$V111_COMPARE" \
  --v111-balanced "$V111_BALANCED" \
  --v111-precision "$V111_PRECISION" \
  --run-id "$RUN_ID" \
  --output "$COMPARE"

python tools/check_constraint_native_orientation_pipeline.py \
  --runtime "$RUNTIME" \
  --balanced "$BOUT" \
  --precision "$POUT" \
  --balanced-state "$BSTATE" \
  --precision-state "$PSTATE" \
  --comparison "$COMPARE" \
  --v48-111-pipeline "$V111_PIPELINE" \
  --v48-111-comparison "$V111_COMPARE" \
  --run-id "$RUN_ID" \
  --output "$COMPLETE"

python tools/package_constraint_native_orientation_results.py \
  --pipeline "$COMPLETE" \
  --base-out "$BASE_OUT" \
  --run-id "$RUN_ID" \
  --manifest "$BUNDLE_MANIFEST" \
  --output "$RESULTS_ZIP"

printf 'V48.112 HCNC result bundle ready. Upload ONLY this file:\n%s\nrun_instance_id=%s\n' "$RESULTS_ZIP" "$RUN_ID"
