#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
BASE_OUT="${BASE_OUT:-$REPO/runs}"
V94_MAIN="${V4895_V94_MAIN:-$BASE_OUT/ocrap_v48_94_dcp_drfc_bcde_rifa_srca_main}"
V93_AUDIT="${V4895_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V94_COMPARISON="${V4895_V94_COMPARISON:-$BASE_OUT/OC-RAP-v48.94-DCP-DRFC-BCDE-RIFA-OC-SRCA-comparison.json}"
V94_PIPELINE="${V4895_V94_PIPELINE:-$BASE_OUT/OC-RAP-v48.94-PIPELINE_COMPLETE.json}"
RUNTIME="$BASE_OUT/OC-RAP-v48.95-runtime-code-contract.json"
AUDIT="$BASE_OUT/OC-RAP-v48.95-native-recovery-observability-audit.json"
CMP="$BASE_OUT/OC-RAP-v48.95-DCP-DRFC-BCDE-RIFA-OC-NROA-comparison.json"
PIPE="$BASE_OUT/OC-RAP-v48.95-PIPELINE_COMPLETE.json"
mkdir -p "$BASE_OUT"
python tools/check_v48_95_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python tools/audit_v48_95_native_recovery_observability.py \
  --v94-main "$V94_MAIN" --v93-audit "$V93_AUDIT" --v94-comparison "$V94_COMPARISON" --v94-pipeline "$V94_PIPELINE" --output "$AUDIT"
python tools/compare_v48_95_nroa.py --audit "$AUDIT" --output "$CMP"
python tools/check_v48_95_pipeline_complete.py --runtime "$RUNTIME" --audit "$AUDIT" --comparison "$CMP" --output "$PIPE"
python - "$BASE_OUT/OC-RAP-v48.95-OC-NROA-audits.zip" "$RUNTIME" "$AUDIT" "$CMP" "$PIPE" <<'PY'
import sys,zipfile
out,*paths=sys.argv[1:]
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in paths: z.write(p,arcname=p.rsplit('/',1)[-1])
print(out)
PY
