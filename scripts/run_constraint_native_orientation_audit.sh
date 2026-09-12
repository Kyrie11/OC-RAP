#!/usr/bin/env bash
# Stable constraint-native orientation audit entrypoint.
# V48.121 OC-VRPC: same-option nominal-rank persistence coupling audit after
# authoritative V48.120 OC-VRT STOP. Candidate-independent nominal rank and same-option
# candidate causal correspondence are retained; the only new mechanism couples signed
# viability displacement to directional persistence of each option's nominal rank path.
# Audit only: no planner/source/root training, regime routing, boundary transport,
# capacity/rank-cut/persistence-window/horizon/threshold sweep, or Main integration.
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

# V48.120 is an immutable versioned scientific input, not a source-code dependency.
V120_PIPELINE="${OCRAP_ORIENTATION_V120_PIPELINE:-$BASE_OUT/OC-RAP-v48.120-PIPELINE_COMPLETE.json}"
V120_COMPARE="${OCRAP_ORIENTATION_V120_COMPARE:-$BASE_OUT/OC-RAP-v48.120-DCP-DRFC-BCDE-RIFA-OC-VRT-comparison.json}"
V120_BALANCED="${OCRAP_ORIENTATION_V120_BALANCED:-$BASE_OUT/OC-RAP-v48.120-VRT-balanced.json}"
V120_PRECISION="${OCRAP_ORIENTATION_V120_PRECISION:-$BASE_OUT/OC-RAP-v48.120-VRT-precision.json}"

CACHE="${OCRAP_ORIENTATION_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_121_vrpc_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.121-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.121-VRPC-balanced.json"
POUT="$BASE_OUT/OC-RAP-v48.121-VRPC-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.121-VRPC-balanced.pt"
PSTATE="$BASE_OUT/OC-RAP-v48.121-VRPC-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.121-DCP-DRFC-BCDE-RIFA-OC-VRPC-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.121-PIPELINE_COMPLETE.json"
BUNDLE_MANIFEST="$BASE_OUT/OC-RAP-v48.121-OC-VRPC-result-bundle-manifest.json"
RESULTS_ZIP="$BASE_OUT/OC-RAP-v48.121-OC-VRPC-results.zip"

mkdir -p "$BASE_OUT" "$CACHE"
rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$BUNDLE_MANIFEST" "$RESULTS_ZIP"

# Fail before GPU work if checkout/import path does not satisfy V48.121.
python tools/check_constraint_native_orientation_contract.py \
  --repo "$REPO" --run-id "$RUN_ID" --output "$RUNTIME"

# Authoritative V48.120 STOP is the only branch that licenses this audit.
python - "$V120_PIPELINE" "$V120_COMPARE" "$V120_BALANCED" "$V120_PRECISION" <<'PY'
import hashlib, json, pathlib, sys
p, c, b, q = map(pathlib.Path, sys.argv[1:])
want = {
    p: '2839be06885f1b05cf6064b934fbe6ed55eda9ff0db6ffe46bf50e89c21cb471',
    c: '9a7827d769b3abe4cda0802fa4aa95b879f38c0ffe8392df65c21bc017cae3c4',
    b: 'a507bedd5f61cdfc6086dce7e1a1b12955fa62b1525d1e865d2fc574e993a263',
    q: '57aafc304b0ec29d04b902d6dd5cec7680a380f69350614f3546fc0c9fc7f647',
}
for path, digest in want.items():
    if not path.is_file():
        raise SystemExit(f'missing V48.120 prerequisite {path}')
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise SystemExit(f'authoritative V48.120 SHA mismatch {path.name}: {got}')
pd = json.loads(p.read_text())
cd = json.loads(c.read_text())
d = cd.get('preregistered_decision') or {}
if not (pd.get('valid') and pd.get('attribution_ready') and pd.get('preregistered_status') == 'VIABILITY_RANK_TRANSPORT_STOP'):
    raise SystemExit('authoritative V48.120 STOP pipeline prerequisite missing')
if not (cd.get('valid') and cd.get('attribution_ready') and d.get('status') == 'VIABILITY_RANK_TRANSPORT_STOP'):
    raise SystemExit('authoritative V48.120 STOP comparison prerequisite missing')
if d.get('next_branch') != 'close_nominal_rank_viability_transport_then_preregister_recovery_set_rank_persistence_coupling_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep':
    raise SystemExit('V48.120 did not authorize recovery-set rank-persistence coupling branch')
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
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.121 VRPC run failure balanced=$r0 precision=$r1" >&2; exit 30; }

python tools/compare_constraint_native_recovery_orientation.py \
  --balanced "$BOUT" --precision "$POUT" \
  --v120-pipeline "$V120_PIPELINE" --v120-comparison "$V120_COMPARE" \
  --v120-balanced "$V120_BALANCED" --v120-precision "$V120_PRECISION" \
  --run-id "$RUN_ID" --output "$COMPARE"

python tools/check_constraint_native_orientation_pipeline.py \
  --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" \
  --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" \
  --v48-120-pipeline "$V120_PIPELINE" --v48-120-comparison "$V120_COMPARE" \
  --run-id "$RUN_ID" --output "$COMPLETE"

python tools/package_constraint_native_orientation_results.py \
  --pipeline "$COMPLETE" --base-out "$BASE_OUT" --run-id "$RUN_ID" \
  --manifest "$BUNDLE_MANIFEST" --output "$RESULTS_ZIP"

printf 'V48.121 OC-VRPC result bundle ready. Upload ONLY this file:\n%s\nrun_instance_id=%s\n' "$RESULTS_ZIP" "$RUN_ID"
