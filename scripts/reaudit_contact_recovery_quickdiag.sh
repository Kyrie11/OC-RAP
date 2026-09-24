#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
: "${BASE_OUT:=runs}"
: "${OUT:=$BASE_OUT/contact_recovery_quickdiag_v135}"
KEYS="$OUT/anchors/contact_anchor_target_keys.json"
[[ -s "$KEYS" ]] || { echo "missing diagnostic target lock: $KEYS" >&2; exit 30; }
for name in strict_abs guarded_fallback; do
  root="$OUT/profiles/$name/contact"
  [[ -s "$root/closed_loop_ocrap.json" && -s "$root/closed_loop_ocrap.json.scenes.jsonl" ]] || { echo "missing completed profile artifacts for $name" >&2; exit 30; }
  python tools/audit_contact_recovery_execution.py --result "$root/closed_loop_ocrap.json" --target-keys-file "$KEYS" --output "$root/CONTACT_RECOVERY_AUDIT.json" | tee "$root/audit_v137.log"
done
CAL=()
if [[ -s "$OUT/profiles/calibrated_guarded/contact/closed_loop_ocrap.json" ]]; then
  root="$OUT/profiles/calibrated_guarded/contact"
  python tools/audit_contact_recovery_execution.py --result "$root/closed_loop_ocrap.json" --target-keys-file "$KEYS" --output "$root/CONTACT_RECOVERY_AUDIT.json" | tee "$root/audit_v137.log"
  CAL=(--calibrated "$root/CONTACT_RECOVERY_AUDIT.json")
fi
python tools/summarize_contact_recovery_quickdiag.py --strict "$OUT/profiles/strict_abs/contact/CONTACT_RECOVERY_AUDIT.json" --guarded "$OUT/profiles/guarded_fallback/contact/CONTACT_RECOVERY_AUDIT.json" --target-keys-file "$KEYS" "${CAL[@]}" --output "$OUT/QUICK_DIAGNOSTIC_SUMMARY_v137.json" | tee "$OUT/summary_v137.log"
echo "[CONTACT-QUICKDIAG][REAUDIT-DONE] $OUT/QUICK_DIAGNOSTIC_SUMMARY_v137.json"
