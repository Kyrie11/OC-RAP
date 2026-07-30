#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_15_prism_cc_dedicated_4815}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
python tools/audit_dedicated_protocol_v48_16.py --protocol-root "$PROTOCOL_ROOT" --output "$OUTPUTDIR/dedicated_protocol_audit_recovered.json"
rm -f "$OUTPUTDIR/GATE_FAILED.json" "$OUTPUTDIR/CALIBRATION_FAILED.json" "$OUTPUTDIR/NEXT_COMMANDS.txt" "$OUTPUTDIR/chosen_base_run_dedicated.txt"
set +e
OUTPUTDIR="$OUTPUTDIR" CAL_SAFE="$CAL_SAFE" \
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact" \
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact" \
GPU0="${GPU0:-0}" GPU1="${GPU1:-1}" \
  bash scripts/calibrate_v48_14_certificate_pool.sh
rc=$?
set -e
printf '%s\n' "$rc" > "$OUTPUTDIR/CERTIFICATE_RECOVERY_EXIT_CODE.txt"
if [[ "$rc" == 30 ]]; then echo "certificate pipeline failed; inspect CALIBRATION_FAILED.json/logs" >&2; fi
exit "$rc"
