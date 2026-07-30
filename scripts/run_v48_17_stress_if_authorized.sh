#!/usr/bin/env bash
set -euo pipefail
OUT="${OUT:-runs/ocrap_v48_17_bridge_dedicated_4817}"
export OUT
[[ -f "$OUT/NEXT_COMMANDS.txt" ]] || { echo "Natural gate has not authorized stress closed loop: $OUT/NEXT_COMMANDS.txt missing" >&2; exit 20; }
exec bash scripts/run_v48_16_stress_if_authorized.sh
