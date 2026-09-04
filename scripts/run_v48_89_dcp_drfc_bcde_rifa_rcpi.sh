#!/usr/bin/env bash
# V48.89 OC-RCPI: Observation-Consistent Root-Correspondence Physical Identifiability.
# Audit-only preregistered branch after V48.88 OC-QTRR STOP.
# No planner training, no new model parameters, no dataset reconstruction, no regime router,
# no boundary transport, no relative-ranker modification, and no capacity sweep.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
L80_RUN="${V4889_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V88_COMPARE="${V4889_V88_COMPARE:-$BASE_OUT/OC-RAP-v48.88-DCP-DRFC-BCDE-RIFA-OC-QTRR-comparison.json}"

DEV_NEAR="${DEV_NEAR:-$PROTOCOL_ROOT/evidence_adapt_dev_near_contact}"
DEV_CONTACT="${DEV_CONTACT:-$PROTOCOL_ROOT/evidence_adapt_dev_contact}"
CERT_NEAR="${CERT_NEAR:-$PROTOCOL_ROOT/certificate_pool_near_contact}"
CERT_CONTACT="${CERT_CONTACT:-$PROTOCOL_ROOT/certificate_pool_contact}"

RUNTIME="$BASE_OUT/OC-RAP-v48.89-runtime-code-contract.json"
INDEX="$BASE_OUT/OC-RAP-v48.89-root-correspondence-audit.jsonl"
SUMMARY="$BASE_OUT/OC-RAP-v48.89-root-correspondence-audit-summary.json"
COMPARE="$BASE_OUT/OC-RAP-v48.89-DCP-DRFC-BCDE-RIFA-OC-RCPI-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.89-PIPELINE_COMPLETE.json"
AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.89-OC-RCPI-audits.zip"

mkdir -p "$BASE_OUT"
rm -f "$RUNTIME" "$INDEX" "$SUMMARY" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_89_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"

python - "$V88_COMPARE" "$L80_RUN" "$DEV_NEAR" "$DEV_CONTACT" "$CERT_NEAR" "$CERT_CONTACT" <<'PY'
import json, pathlib, sys
v88=pathlib.Path(sys.argv[1]); l80=pathlib.Path(sys.argv[2])
if not v88.is_file(): raise SystemExit(f'missing V48.88 comparison: {v88}')
d=json.loads(v88.read_text()); p=d.get('preregistered_decision') or {}
if not(d.get('valid') and p.get('status')=='QUOTIENT_TAIL_RESPONSE_STOP' and not p.get('quotient_tail_identifiability_go') and 'root_correspondence' in str(p.get('next_branch',''))):
    raise SystemExit('V48.88 root-correspondence audit prerequisite missing')
if not l80.is_dir(): raise SystemExit(f'missing historical L80 proposal run: {l80}')
for raw in sys.argv[3:]:
    q=pathlib.Path(raw)
    if not q.is_dir() or not (q/'manifest.csv').is_file(): raise SystemExit(f'missing dataset root/manifest: {q}')
PY

python tools/build_v48_89_root_correspondence_audit.py \
  --root "dev_near=$DEV_NEAR" \
  --root "dev_contact=$DEV_CONTACT" \
  --root "certificate_near=$CERT_NEAR" \
  --root "certificate_contact=$CERT_CONTACT" \
  --proposal-run "$L80_RUN" \
  --workers "${V4889_WORKERS:-8}" \
  --output "$INDEX" \
  --summary "$SUMMARY"

python tools/compare_v48_89_rcpi.py \
  --audit-summary "$SUMMARY" \
  --v48-88-comparison "$V88_COMPARE" \
  --output "$COMPARE"

python tools/check_v48_89_pipeline_complete.py \
  --runtime "$RUNTIME" \
  --audit-index "$INDEX" \
  --audit-summary "$SUMMARY" \
  --comparison "$COMPARE" \
  --v48-88-comparison "$V88_COMPARE" \
  --output "$COMPLETE"

zip -qj "$AUDITS_ZIP" "$RUNTIME" "$SUMMARY" "$COMPARE" "$COMPLETE"
printf 'V48.89 complete. Upload:\n%s\n%s\n' "$AUDITS_ZIP" "$INDEX"
