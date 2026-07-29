#!/usr/bin/env bash
set -euo pipefail
# Re-run only the failed v48.14 certificate stage; no retraining.
OUTPUTDIR="${OUTPUTDIR:?set OUTPUTDIR to the existing v48.14 run}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
rm -f "$OUTPUTDIR/GATE_FAILED.json" "$OUTPUTDIR/CALIBRATION_FAILED.json" "$OUTPUTDIR/NEXT_COMMANDS.txt"
set +e
OUTPUTDIR="$OUTPUTDIR" CAL_SAFE="$CAL_SAFE" \
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact" \
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact" \
GPU0="${GPU0:-0}" GPU1="${GPU1:-1}" \
  bash scripts/calibrate_v48_14_certificate_pool.sh \
  >"$OUTPUTDIR/logs/certificate_controller.recovered.log" 2>&1
rc=$?
set -e
echo "certificate recovery rc=$rc (0=gate passed, 20=valid calibration but gate rejected, 30=artifact failure)"
exit "$rc"
