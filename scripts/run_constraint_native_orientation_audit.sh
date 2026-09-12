#!/usr/bin/env bash
# Stable constraint-native orientation audit entrypoint.
# V48.123 OC-ZBST: zero-boundary viability state-transition audit after
# authoritative V48.122 OC-SVRT STOP. Candidate-independent nominal rank and
# same-option causal correspondence are retained; the candidate displacement is
# exactly decomposed into safe-side reserve transition and debt repayment.
# Audit only: no planner/source/root training, regime routing, boundary transport,
# capacity/rank-cut/transition-window/horizon/option-count sweep, or Main integration.
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

# V48.122 is an immutable versioned scientific input, not a source-code dependency.
V122_PIPELINE="${OCRAP_ORIENTATION_V122_PIPELINE:-$BASE_OUT/OC-RAP-v48.122-PIPELINE_COMPLETE.json}"
V122_COMPARE="${OCRAP_ORIENTATION_V122_COMPARE:-$BASE_OUT/OC-RAP-v48.122-DCP-DRFC-BCDE-RIFA-OC-SVRT-comparison.json}"
V122_BALANCED="${OCRAP_ORIENTATION_V122_BALANCED:-$BASE_OUT/OC-RAP-v48.122-SVRT-balanced.json}"
V122_PRECISION="${OCRAP_ORIENTATION_V122_PRECISION:-$BASE_OUT/OC-RAP-v48.122-SVRT-precision.json}"

CACHE="${OCRAP_ORIENTATION_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_123_zbst_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.123-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.123-ZBST-balanced.json"
POUT="$BASE_OUT/OC-RAP-v48.123-ZBST-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.123-ZBST-balanced.pt"
PSTATE="$BASE_OUT/OC-RAP-v48.123-ZBST-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.123-DCP-DRFC-BCDE-RIFA-OC-ZBST-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.123-PIPELINE_COMPLETE.json"
BUNDLE_MANIFEST="$BASE_OUT/OC-RAP-v48.123-OC-ZBST-result-bundle-manifest.json"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.123-OC-ZBST-results.zip"

mkdir -p "$BASE_OUT" "$CACHE"
rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$BUNDLE_MANIFEST" "$RESULTS_ZIP"

# Fail before GPU work if checkout/import path does not satisfy V48.123.
python tools/check_constraint_native_orientation_contract.py \
  --repo "$REPO" --run-id "$RUN_ID" --output "$RUNTIME"

# Authoritative V48.122 STOP is the only branch that licenses this audit.
python - "$V122_PIPELINE" "$V122_COMPARE" "$V122_BALANCED" "$V122_PRECISION" <<'PY'
import hashlib, json, pathlib, sys
p, c, b, q = map(pathlib.Path, sys.argv[1:])
want = {
    p: '5432df0f0937969d01616f96f8a27e992932d9c297813b6866dd42831b111e8e',
    c: '6d32da4ace62ed2f01dc76bcda55a80623e339195b28267ac93c2dde92914fe2',
    b: '4ed2c1c5ddd2bc2fe7647664bb053cd4066e2997362a9b1db3a55814583b660f',
    q: 'd03c8ba6b34287f61df1345a56372d5601bd9dd0b58cf509fe71ac09f7b1911d',
}
for path, digest in want.items():
    if not path.is_file(): raise SystemExit(f'missing V48.122 prerequisite {path}')
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest: raise SystemExit(f'authoritative V48.122 SHA mismatch {path.name}: {got}')
pd=json.loads(p.read_text()); cd=json.loads(c.read_text()); d=cd.get('preregistered_decision') or {}
if not (pd.get('valid') and pd.get('attribution_ready') and pd.get('preregistered_status') == 'SIGNED_VIABILITY_RANK_STATE_TRANSPORT_STOP'):
    raise SystemExit('authoritative V48.122 STOP pipeline prerequisite missing')
if not (cd.get('valid') and cd.get('attribution_ready') and d.get('status') == 'SIGNED_VIABILITY_RANK_STATE_TRANSPORT_STOP'):
    raise SystemExit('authoritative V48.122 STOP comparison prerequisite missing')
if d.get('next_branch') != 'close_signed_viability_rank_state_transport_then_preregister_zero_boundary_viability_state_transition_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep':
    raise SystemExit('V48.122 did not authorize zero-boundary viability state-transition branch')
PY

for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT"; do
  [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }
done

run_one() {
  local variant="$1" gpu="$2" out="$3" state="$4"
  local ckpt="$L80_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing frozen L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_constraint_native_recovery_orientation_audit.py \
    --checkpoint "$ckpt" --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" \
    --certificate-index "$CERT_INDEX" --v93-audit "$V93_AUDIT" \
    --cache-dir "$CACHE/$variant" --device cuda --variant "$variant" \
    --run-id "$RUN_ID" --output "$out" --state-output "$state"
}

set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" & p1=$!
wait "$p0"; r0=$?
wait "$p1"; r1=$?
set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.123 ZBST run failure balanced=$r0 precision=$r1" >&2; exit 30; }

python tools/compare_constraint_native_recovery_orientation.py \
  --balanced "$BOUT" --precision "$POUT" \
  --v122-pipeline "$V122_PIPELINE" --v122-comparison "$V122_COMPARE" \
  --v122-balanced "$V122_BALANCED" --v122-precision "$V122_PRECISION" \
  --run-id "$RUN_ID" --output "$COMPARE"

python tools/check_constraint_native_orientation_pipeline.py \
  --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" \
  --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" \
  --v48-122-pipeline "$V122_PIPELINE" --v48-122-comparison "$V122_COMPARE" \
  --run-id "$RUN_ID" --output "$COMPLETE"

python tools/package_constraint_native_orientation_results.py \
  --pipeline "$COMPLETE" --base-out "$BASE_OUT" --run-id "$RUN_ID" \
  --manifest "$BUNDLE_MANIFEST" --output "$RESULTS_ZIP"

printf 'V48.123 OC-ZBST result bundle ready. Upload ONLY this file:\n%s\nrun_instance_id=%s\n' "$RESULTS_ZIP" "$RUN_ID"
