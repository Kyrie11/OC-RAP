#!/usr/bin/env bash
# V48.92 OC-FRAD: Observation-Consistent Factorized Recovery-Advantage Decomposition.
# Audit-only follow-up to the preregistered V48.91 physical-response STOP.
# Reuses the exact V48.91 future sidecar; DOES NOT replay WOMD/Waymax and trains zero planner parameters.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
L80_RUN="${V4892_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V91_AUDIT="${V4892_V91_AUDIT:-$BASE_OUT/OC-RAP-v48.91-common-exogenous-physical-response-audit.jsonl}"
V91_SIDECAR="${V4892_V91_SIDECAR:-$BASE_OUT/OC-RAP-v48.91-common-exogenous-future-physical-sidecar.jsonl.gz}"
V91_SIDECAR_SUMMARY="${V4892_V91_SIDECAR_SUMMARY:-$BASE_OUT/OC-RAP-v48.91-common-exogenous-future-physical-sidecar-summary.json}"
V91_COMPARE="${V4892_V91_COMPARE:-$BASE_OUT/OC-RAP-v48.91-DCP-DRFC-BCDE-RIFA-OC-CEPMI-comparison.json}"
V91_COMPLETE="${V4892_V91_COMPLETE:-$BASE_OUT/OC-RAP-v48.91-PIPELINE_COMPLETE.json}"
RUNTIME="$BASE_OUT/OC-RAP-v48.92-runtime-code-contract.json"
AUDIT="$BASE_OUT/OC-RAP-v48.92-factorized-recovery-advantage-audit.jsonl"
SUMMARY="$BASE_OUT/OC-RAP-v48.92-factorized-recovery-advantage-audit-summary.json"
COMPARE="$BASE_OUT/OC-RAP-v48.92-DCP-DRFC-BCDE-RIFA-OC-FRAD-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.92-PIPELINE_COMPLETE.json"
AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.92-OC-FRAD-audits.zip"
mkdir -p "$BASE_OUT"
rm -f "$RUNTIME" "$AUDIT" "$SUMMARY" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_92_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V91_COMPLETE" "$V91_COMPARE" "$V91_AUDIT" "$V91_SIDECAR" "$V91_SIDECAR_SUMMARY" "$L80_RUN" <<'PY'
import json,pathlib,sys
pc,cmp,aud,side,ss,l80=map(pathlib.Path,sys.argv[1:])
for p in (pc,cmp,aud,side,ss):
    if not p.is_file(): raise SystemExit(f'missing V48.92 prerequisite: {p}')
if not l80.is_dir(): raise SystemExit(f'missing historical L80 proposal run: {l80}')
d=json.loads(pc.read_text()); c=json.loads(cmp.read_text()); q=c.get('preregistered_decision') or {}
if not(d.get('valid') and d.get('attribution_ready') and d.get('preregistered_status')=='COMMON_EXOGENOUS_PHYSICAL_RESPONSE_STOP'):
    raise SystemExit('V48.91 completed scientific STOP prerequisite missing')
if not(c.get('valid') and q.get('status')=='COMMON_EXOGENOUS_PHYSICAL_RESPONSE_STOP' and not q.get('source_training_authorized')):
    raise SystemExit('V48.91 comparison STOP/source-authorization contract missing')
PY

echo "[v48.92-perf] reusing exact V48.91 sidecar; raw WOMD/Waymax replay is intentionally skipped" >&2
python tools/build_v48_92_factorized_recovery_advantage_audit.py \
  --l80-run "$L80_RUN" \
  --v48-91-audit "$V91_AUDIT" \
  --v48-91-sidecar "$V91_SIDECAR" \
  --v48-91-sidecar-summary "$V91_SIDECAR_SUMMARY" \
  --output "$AUDIT" --summary "$SUMMARY"
python tools/compare_v48_92_frad.py --summary "$SUMMARY" --v48-91-comparison "$V91_COMPARE" --output "$COMPARE"
python tools/check_v48_92_pipeline_complete.py \
  --runtime "$RUNTIME" --audit "$AUDIT" --audit-summary "$SUMMARY" --comparison "$COMPARE" \
  --v48-91-pipeline "$V91_COMPLETE" --v48-91-comparison "$V91_COMPARE" --output "$COMPLETE"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$SUMMARY" "$COMPARE" "$COMPLETE"
printf 'V48.92 complete. Upload:\n%s\n%s\n' "$AUDITS_ZIP" "$AUDIT"
