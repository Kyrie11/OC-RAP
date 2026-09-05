#!/usr/bin/env bash
# V48.93 OC-FMCA: Observation-Consistent Factor-Mediation Complementarity Adjudication.
# Audit-only tie adjudication after V48.92 produced multiple marginal mediator winners.
# Reuses V48.92 audit rows; no raw WOMD/Waymax replay and zero planner parameters.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
V92_AUDIT="${V4893_V92_AUDIT:-$BASE_OUT/OC-RAP-v48.92-factorized-recovery-advantage-audit.jsonl}"
V92_SUMMARY="${V4893_V92_SUMMARY:-$BASE_OUT/OC-RAP-v48.92-factorized-recovery-advantage-audit-summary.json}"
V92_COMPARE="${V4893_V92_COMPARE:-$BASE_OUT/OC-RAP-v48.92-DCP-DRFC-BCDE-RIFA-OC-FRAD-comparison.json}"
V92_COMPLETE="${V4893_V92_COMPLETE:-$BASE_OUT/OC-RAP-v48.92-PIPELINE_COMPLETE.json}"
RUNTIME="$BASE_OUT/OC-RAP-v48.93-runtime-code-contract.json"
AUDIT="$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl"
SUMMARY="$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit-summary.json"
COMPARE="$BASE_OUT/OC-RAP-v48.93-DCP-DRFC-BCDE-RIFA-OC-FMCA-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.93-PIPELINE_COMPLETE.json"
AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.93-OC-FMCA-audits.zip"
mkdir -p "$BASE_OUT"
rm -f "$RUNTIME" "$AUDIT" "$SUMMARY" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_93_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V92_COMPLETE" "$V92_COMPARE" "$V92_AUDIT" "$V92_SUMMARY" <<'PY'
import json,pathlib,sys
pc,cmp,aud,summary=map(pathlib.Path,sys.argv[1:])
for p in (pc,cmp,aud,summary):
    if not p.is_file(): raise SystemExit(f'missing V48.93 prerequisite: {p}')
p=json.loads(pc.read_text()); c=json.loads(cmp.read_text()); q=c.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and p.get('preregistered_status')=='SHARED_RECOVERY_ADVANTAGE_MEDIATOR_GO'):
    raise SystemExit('V48.92 completed screening GO prerequisite missing')
if not(c.get('valid') and q.get('status')=='SHARED_RECOVERY_ADVANTAGE_MEDIATOR_GO'):
    raise SystemExit('V48.92 comparison screening GO missing')
if len(q.get('shared_mediator_winners') or []) < 2:
    raise SystemExit('V48.93 is only the multi-winner tie-adjudication branch')
PY

echo "[v48.93-perf] reusing V48.92 audit; no raw replay, no model training" >&2
python tools/build_v48_93_factor_mediation_audit.py \
  --v48-92-audit "$V92_AUDIT" --v48-92-summary "$V92_SUMMARY" --v48-92-comparison "$V92_COMPARE" \
  --output "$AUDIT" --summary "$SUMMARY"
python tools/compare_v48_93_fmca.py --summary "$SUMMARY" --v48-92-comparison "$V92_COMPARE" --output "$COMPARE"
python tools/check_v48_93_pipeline_complete.py \
  --runtime "$RUNTIME" --audit "$AUDIT" --audit-summary "$SUMMARY" --comparison "$COMPARE" \
  --v48-92-pipeline "$V92_COMPLETE" --v48-92-comparison "$V92_COMPARE" --output "$COMPLETE"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$SUMMARY" "$COMPARE" "$COMPLETE"
printf 'V48.93 complete. Upload:\n%s\n%s\n' "$AUDITS_ZIP" "$AUDIT"
