#!/usr/bin/env bash
# V48.98 OC-ERTA: centered rank-2 Stage-I executable-recovery tangent alignment.
# Representation-learning only. V48.97 state chart frozen; no source/admission training.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
REFERENCE_A="${V4898_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V4898_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V4898_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V97_COMPLETE="${V4898_V97_COMPLETE:-$BASE_OUT/OC-RAP-v48.97-PIPELINE_COMPLETE.json}"
V97_COMPARE="${V4898_V97_COMPARE:-$BASE_OUT/OC-RAP-v48.97-DCP-DRFC-BCDE-RIFA-OC-ERSS-comparison.json}"
V97_BOUT="${V4898_V97_BALANCED:-$BASE_OUT/OC-RAP-v48.97-ERSS-balanced.json}"
V97_POUT="${V4898_V97_PRECISION:-$BASE_OUT/OC-RAP-v48.97-ERSS-precision.json}"
V97_BSTATE="${V4898_V97_BALANCED_STATE:-$BASE_OUT/OC-RAP-v48.97-ERSS-balanced.pt}"
V97_PSTATE="${V4898_V97_PRECISION_STATE:-$BASE_OUT/OC-RAP-v48.97-ERSS-precision.pt}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_INDEX="${V4898_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
CACHE="${V4898_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_98_erta_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.98-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.98-ERTA-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.98-ERTA-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.98-ERTA-balanced.pt"; PSTATE="$BASE_OUT/OC-RAP-v48.98-ERTA-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.98-DCP-DRFC-BCDE-RIFA-OC-ERTA-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.98-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.98-OC-ERTA-audits.zip"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_98_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V97_COMPLETE" "$V97_COMPARE" "$V97_BOUT" "$V97_POUT" "$V97_BSTATE" "$V97_PSTATE" <<'PY'
import json,pathlib,sys
pc,cc,br,pr,bs,ps=map(pathlib.Path,sys.argv[1:])
for p in (pc,cc,br,pr,bs,ps):
    if not p.is_file(): raise SystemExit(f'missing V48.98 prerequisite {p}')
p=json.loads(pc.read_text()); c=json.loads(cc.read_text()); d=c.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and p.get('engineering_version')=='v48.97.2-OC-ERSS-STRATAFIX' and p.get('preregistered_status')=='EXECUTABLE_RECOVERY_SUFFICIENT_STATE_STOP'):
    raise SystemExit('V48.97.2 STOP prerequisite missing')
if not(c.get('valid') and c.get('attribution_ready') and d.get('state_representation_go') is True and d.get('support_action_representation_go') is False and d.get('reserve_debt_representation_go') is False):
    raise SystemExit('V48.97.2 branch-shape prerequisite missing')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT"; do [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }; done

run_one(){
  local v="$1"; local gpu="$2"; local out="$3"; local state="$4"; local erss="$5"
  local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_98_executable_recovery_tangent.py \
    --checkpoint "$ckpt" --erss-state "$erss" \
    --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out" --state-output "$state"
}
set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" "$V97_BSTATE" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" "$V97_PSTATE" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.98 tangent run failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_98_erta.py --balanced "$BOUT" --precision "$POUT" \
  --v48-97-balanced "$V97_BOUT" --v48-97-precision "$V97_POUT" --v48-97-comparison "$V97_COMPARE" --output "$COMPARE"
python tools/check_v48_98_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" \
  --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" --v48-97-pipeline "$V97_COMPLETE" --output "$COMPLETE"
cd "$BASE_OUT"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE"
printf 'V48.98 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
