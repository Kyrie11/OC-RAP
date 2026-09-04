#!/usr/bin/env bash
# V48.91 OC-CEPMI: Common-Exogenous Physical-Margin Identifiability.
# Audit-only preregistered branch after V48.90 transport GO / physical-response STOP.
# Replays teacher construction on the exact same labeled cohort to expose future-level
# pre-structural physical margins. Canonical datasets are never rewritten/reselected.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
V90_INDEX="${V4891_V90_INDEX:-$BASE_OUT/OC-RAP-v48.90-partition-transport-audit.jsonl}"
V90_SUMMARY="${V4891_V90_SUMMARY:-$BASE_OUT/OC-RAP-v48.90-partition-transport-audit-summary.json}"
V90_COMPARE="${V4891_V90_COMPARE:-$BASE_OUT/OC-RAP-v48.90-DCP-DRFC-BCDE-RIFA-OC-CEPT-comparison.json}"
REPLAY_CONFIG="${V4891_REPLAY_CONFIG:-}"
WOMD_SOURCE_PATTERN="${V4891_WOMD_SOURCE:-${WOMD_VAL:-}}"
RUNTIME="$BASE_OUT/OC-RAP-v48.91-runtime-code-contract.json"
SIDECAR="$BASE_OUT/OC-RAP-v48.91-common-exogenous-future-physical-sidecar.jsonl.gz"
SIDECAR_SUMMARY="$BASE_OUT/OC-RAP-v48.91-common-exogenous-future-physical-sidecar-summary.json"
AUDIT="$BASE_OUT/OC-RAP-v48.91-common-exogenous-physical-response-audit.jsonl"
SUMMARY="$BASE_OUT/OC-RAP-v48.91-common-exogenous-physical-response-audit-summary.json"
COMPARE="$BASE_OUT/OC-RAP-v48.91-DCP-DRFC-BCDE-RIFA-OC-CEPMI-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.91-PIPELINE_COMPLETE.json"
AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.91-OC-CEPMI-audits.zip"
mkdir -p "$BASE_OUT"; rm -f "$RUNTIME" "$SIDECAR" "$SIDECAR_SUMMARY" "$AUDIT" "$SUMMARY" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"
python tools/check_v48_91_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V90_INDEX" "$V90_SUMMARY" "$V90_COMPARE" <<'PY'
import json,pathlib,sys
for p in map(pathlib.Path,sys.argv[1:]):
    if not p.is_file(): raise SystemExit(f'missing V48.90 prerequisite: {p}')
c=json.loads(pathlib.Path(sys.argv[3]).read_text()); q=c.get('preregistered_decision') or {}
if not(c.get('valid') and q.get('exogenous_partition_transport_go') and q.get('partition_stability_directional_relevance_go') and not q.get('transport_physical_response_identifiability_go')):
    raise SystemExit('V48.90 transport-GO / physical-response-STOP prerequisite missing')
PY
REPLAY_ARGS=()
if [[ -n "$REPLAY_CONFIG" ]]; then REPLAY_ARGS+=(--replay-config "$REPLAY_CONFIG"); fi
if [[ -n "$WOMD_SOURCE_PATTERN" ]]; then REPLAY_ARGS+=(--womd-source-pattern "$WOMD_SOURCE_PATTERN"); fi
python tools/build_v48_91_common_exogenous_physical_sidecar.py \
  --v48-90-audit "$V90_INDEX" --output "$SIDECAR" --summary "$SIDECAR_SUMMARY" "${REPLAY_ARGS[@]}"
python tools/build_v48_91_common_exogenous_physical_response_audit.py \
  --v48-90-audit "$V90_INDEX" --sidecar "$SIDECAR" --sidecar-summary "$SIDECAR_SUMMARY" \
  --output "$AUDIT" --summary "$SUMMARY"
python tools/compare_v48_91_cepmi.py --summary "$SUMMARY" --v48-90-summary "$V90_SUMMARY" --v48-90-comparison "$V90_COMPARE" --output "$COMPARE"
python tools/check_v48_91_pipeline_complete.py \
  --runtime "$RUNTIME" --sidecar "$SIDECAR" --sidecar-summary "$SIDECAR_SUMMARY" --audit "$AUDIT" --audit-summary "$SUMMARY" \
  --comparison "$COMPARE" --v48-90-comparison "$V90_COMPARE" --output "$COMPLETE"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$SIDECAR_SUMMARY" "$SUMMARY" "$COMPARE" "$COMPLETE"
printf 'V48.91 complete. Upload:\n%s\n%s\n%s\n' "$AUDITS_ZIP" "$AUDIT" "$SIDECAR_SUMMARY"
