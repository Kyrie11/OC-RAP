#!/usr/bin/env bash
# V48.90 OC-CEPT: Observation-Consistent Counterfactual Equivalence-Partition Transport.
# Audit-only preregistered branch after V48.89.1 root-correspondence STOP.
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
L80_RUN="${V4890_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V89_COMPARE="${V4890_V89_COMPARE:-$BASE_OUT/OC-RAP-v48.89-DCP-DRFC-BCDE-RIFA-OC-RCPI-comparison.json}"

DEV_NEAR="${DEV_NEAR:-$PROTOCOL_ROOT/evidence_adapt_dev_near_contact}"
DEV_CONTACT="${DEV_CONTACT:-$PROTOCOL_ROOT/evidence_adapt_dev_contact}"
CERT_NEAR="${CERT_NEAR:-$PROTOCOL_ROOT/certificate_pool_near_contact}"
CERT_CONTACT="${CERT_CONTACT:-$PROTOCOL_ROOT/certificate_pool_contact}"

RUNTIME="$BASE_OUT/OC-RAP-v48.90-runtime-code-contract.json"
INDEX="$BASE_OUT/OC-RAP-v48.90-partition-transport-audit.jsonl"
SUMMARY="$BASE_OUT/OC-RAP-v48.90-partition-transport-audit-summary.json"
COMPARE="$BASE_OUT/OC-RAP-v48.90-DCP-DRFC-BCDE-RIFA-OC-CEPT-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.90-PIPELINE_COMPLETE.json"
AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.90-OC-CEPT-audits.zip"
mkdir -p "$BASE_OUT"
rm -f "$RUNTIME" "$INDEX" "$SUMMARY" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_90_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V89_COMPARE" "$L80_RUN" "$DEV_NEAR" "$DEV_CONTACT" "$CERT_NEAR" "$CERT_CONTACT" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); l80=pathlib.Path(sys.argv[2])
if not p.is_file(): raise SystemExit(f'missing V48.89 comparison: {p}')
d=json.loads(p.read_text()); q=d.get('preregistered_decision') or {}
if not(d.get('valid') and q.get('status')=='COUNTERFACTUAL_ROOT_CORRESPONDENCE_STOP' and not q.get('root_correspondence_go') and 'partition_stability' in str(q.get('next_branch',''))):
    raise SystemExit('V48.89 partition-stability prerequisite missing')
if not l80.is_dir(): raise SystemExit(f'missing historical L80 proposal run: {l80}')
for raw in sys.argv[3:]:
    x=pathlib.Path(raw)
    if not x.is_dir() or not (x/'manifest.csv').is_file(): raise SystemExit(f'missing dataset root/manifest: {x}')
PY

python tools/build_v48_90_partition_transport_audit.py \
  --root "dev_near=$DEV_NEAR" \
  --root "dev_contact=$DEV_CONTACT" \
  --root "certificate_near=$CERT_NEAR" \
  --root "certificate_contact=$CERT_CONTACT" \
  --proposal-run "$L80_RUN" \
  --output "$INDEX" --summary "$SUMMARY"

python tools/compare_v48_90_cept.py --audit-summary "$SUMMARY" --v48-89-comparison "$V89_COMPARE" --output "$COMPARE"
python tools/check_v48_90_pipeline_complete.py \
  --runtime "$RUNTIME" --audit-index "$INDEX" --audit-summary "$SUMMARY" \
  --comparison "$COMPARE" --v48-89-comparison "$V89_COMPARE" --output "$COMPLETE"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$SUMMARY" "$COMPARE" "$COMPLETE"
printf 'V48.90 complete. Upload:\n%s\n%s\n' "$AUDITS_ZIP" "$INDEX"
