#!/usr/bin/env bash
set -euo pipefail
OUT="${OUT:-runs/ocrap_v48_18_duet_dedicated_4818}"
export OUT
[[ -f "$OUT/NEXT_COMMANDS.txt" ]] || {
  echo "Natural gate has not authorized stress closed loop: $OUT/NEXT_COMMANDS.txt missing" >&2
  echo "Inspect $OUT/learning_gates_v48_18.json; do not bypass the held-out gate." >&2
  exit 20
}
exec bash scripts/run_v48_16_stress_if_authorized.sh
