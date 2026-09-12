#!/usr/bin/env bash
# Stable constraint-native orientation audit entrypoint.
# V48.122 OC-SVRT: signed nominal viability rank-state transport audit after
# authoritative V48.121 OC-VRPC STOP. Candidate-independent nominal rank and
# same-option candidate causal correspondence are retained; the only new primitive
# is the absolute signed nominal viability level relative to the physical zero boundary.
# Audit only: no planner/source/root training, regime routing, boundary transport,
# capacity/rank-cut/state-threshold/horizon/option-count sweep, or Main integration.
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

# V48.121 is an immutable versioned scientific input, not a source-code dependency.
V121_PIPELINE="${OCRAP_ORIENTATION_V121_PIPELINE:-$BASE_OUT/OC-RAP-v48.121-PIPELINE_COMPLETE.json}"
V121_COMPARE="${OCRAP_ORIENTATION_V121_COMPARE:-$BASE_OUT/OC-RAP-v48.121-DCP-DRFC-BCDE-RIFA-OC-VRPC-comparison.json}"
V121_BALANCED="${OCRAP_ORIENTATION_V121_BALANCED:-$BASE_OUT/OC-RAP-v48.121-VRPC-balanced.json}"
V121_PRECISION="${OCRAP_ORIENTATION_V121_PRECISION:-$BASE_OUT/OC-RAP-v48.121-VRPC-precision.json}"

CACHE="${OCRAP_ORIENTATION_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_122_svrt_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.122-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.122-SVRT-balanced.json"
POUT="$BASE_OUT/OC-RAP-v48.122-SVRT-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.122-SVRT-balanced.pt"
PSTATE="$BASE_OUT/OC-RAP-v48.122-SVRT-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.122-DCP-DRFC-BCDE-RIFA-OC-SVRT-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.122-PIPELINE_COMPLETE.json"
BUNDLE_MANIFEST="$BASE_OUT/OC-RAP-v48.122-OC-SVRT-result-bundle-manifest.json"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.122-OC-SVRT-results.zip"

mkdir -p "$BASE_OUT" "$CACHE"
rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$BUNDLE_MANIFEST" "$RESULTS_ZIP"

# Fail before GPU work if checkout/import path does not satisfy V48.122.
python tools/check_constraint_native_orientation_contract.py \
  --repo "$REPO" --run-id "$RUN_ID" --output "$RUNTIME"

# Authoritative V48.121 STOP is the only branch that licenses this audit.
python - "$V121_PIPELINE" "$V121_COMPARE" "$V121_BALANCED" "$V121_PRECISION" <<'PY'
import hashlib, json, pathlib, sys
p, c, b, q = map(pathlib.Path, sys.argv[1:])
want = {
    p: 'b2e08fbe1b1b1e2b676a3d9bd0e9b84082fd35844b8fa6dd0cb7fa193d22e6e6',
    c: '7557a98d23f79d146c620c51deabe41ade063295f906fec02c07836e8af1513f',
    b: '2b5cdc822071a4e1d2258e080e8ee3413e347099a11e40e77f1cd0803c7012fe',
    q: 'b51eb9bfcf353a6df6337fcfe83fd303e69716241ac274af6e0d73471e829d24',
}
for path, digest in want.items():
    if not path.is_file():
        raise SystemExit(f'missing V48.121 prerequisite {path}')
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise SystemExit(f'authoritative V48.121 SHA mismatch {path.name}: {got}')
pd = json.loads(p.read_text())
cd = json.loads(c.read_text())
d = cd.get('preregistered_decision') or {}
if not (pd.get('valid') and pd.get('attribution_ready') and pd.get('preregistered_status') == 'VIABILITY_RANK_PERSISTENCE_COUPLING_STOP'):
    raise SystemExit('authoritative V48.121 STOP pipeline prerequisite missing')
if not (cd.get('valid') and cd.get('attribution_ready') and d.get('status') == 'VIABILITY_RANK_PERSISTENCE_COUPLING_STOP'):
    raise SystemExit('authoritative V48.121 STOP comparison prerequisite missing')
if d.get('next_branch') != 'close_nominal_rank_persistence_coupling_then_preregister_signed_viability_rank_state_transport_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep':
    raise SystemExit('V48.121 did not authorize signed viability rank-state transport branch')
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
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.122 SVRT run failure balanced=$r0 precision=$r1" >&2; exit 30; }

python tools/compare_constraint_native_recovery_orientation.py \
  --balanced "$BOUT" --precision "$POUT" \
  --v121-pipeline "$V121_PIPELINE" --v121-comparison "$V121_COMPARE" \
  --v121-balanced "$V121_BALANCED" --v121-precision "$V121_PRECISION" \
  --run-id "$RUN_ID" --output "$COMPARE"

python tools/check_constraint_native_orientation_pipeline.py \
  --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" \
  --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" \
  --v48-121-pipeline "$V121_PIPELINE" --v48-121-comparison "$V121_COMPARE" \
  --run-id "$RUN_ID" --output "$COMPLETE"

python tools/package_constraint_native_orientation_results.py \
  --pipeline "$COMPLETE" --base-out "$BASE_OUT" --run-id "$RUN_ID" \
  --manifest "$BUNDLE_MANIFEST" --output "$RESULTS_ZIP"

printf 'V48.122 OC-SVRT result bundle ready. Upload ONLY this file:\n%s\nrun_instance_id=%s\n' "$RESULTS_ZIP" "$RUN_ID"
