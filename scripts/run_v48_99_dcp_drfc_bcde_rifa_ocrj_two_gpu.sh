#!/usr/bin/env bash
# V48.99 OC-RJCA: observation-conditioned control-affine recovery Jacobian on decoded roots.
# Representation-learning only. V48.97 base state frozen; V48.98 STOP prerequisite; no source/admission training.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
REFERENCE_A="${V4899_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V4899_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V4899_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V97_BSTATE="${V4899_V97_BALANCED_STATE:-$BASE_OUT/OC-RAP-v48.97-ERSS-balanced.pt}"
V97_PSTATE="${V4899_V97_PRECISION_STATE:-$BASE_OUT/OC-RAP-v48.97-ERSS-precision.pt}"
V98_COMPLETE="${V4899_V98_COMPLETE:-$BASE_OUT/OC-RAP-v48.98-PIPELINE_COMPLETE.json}"
V98_COMPARE="${V4899_V98_COMPARE:-$BASE_OUT/OC-RAP-v48.98-DCP-DRFC-BCDE-RIFA-OC-ERTA-comparison.json}"
V98_BOUT="${V4899_V98_BALANCED:-$BASE_OUT/OC-RAP-v48.98-ERTA-balanced.json}"
V98_POUT="${V4899_V98_PRECISION:-$BASE_OUT/OC-RAP-v48.98-ERTA-precision.json}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_INDEX="${V4899_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
CACHE="${V4899_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_99_ocrj_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.99-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.99-OCRJ-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.99-OCRJ-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.99-OCRJ-balanced.pt"; PSTATE="$BASE_OUT/OC-RAP-v48.99-OCRJ-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.99-DCP-DRFC-BCDE-RIFA-OC-RJCA-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.99-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.99-OC-RJCA-audits.zip"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_99_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V98_COMPLETE" "$V98_COMPARE" "$V98_BOUT" "$V98_POUT" "$V97_BSTATE" "$V97_PSTATE" <<'PY'
import json,pathlib,sys
pc,cc,br,pr,bs,ps=map(pathlib.Path,sys.argv[1:])
for p in (pc,cc,br,pr,bs,ps):
    if not p.is_file(): raise SystemExit(f'missing V48.99 prerequisite {p}')
p=json.loads(pc.read_text()); c=json.loads(cc.read_text()); d=c.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and p.get('engineering_version')=='v48.98.0-OC-ERTA' and p.get('preregistered_status')=='EXECUTABLE_RECOVERY_TANGENT_ALIGNMENT_STOP'):
    raise SystemExit('V48.98 STOP prerequisite missing')
if not(c.get('valid') and c.get('attribution_ready') and d.get('state_chart_preserved') is True and d.get('support_tangent_go') is False and d.get('reserve_debt_tangent_go') is False):
    raise SystemExit('V48.98 branch-shape prerequisite missing')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT"; do [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }; done

run_one(){
  local v="$1"; local gpu="$2"; local out="$3"; local state="$4"; local erss="$5"
  local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_99_recovery_jacobian.py \
    --checkpoint "$ckpt" --erss-state "$erss" \
    --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out" --state-output "$state"
}
set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" "$V97_BSTATE" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" "$V97_PSTATE" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.99 recovery-Jacobian run failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_99_ocrj.py --balanced "$BOUT" --precision "$POUT" \
  --v48-98-balanced "$V98_BOUT" --v48-98-precision "$V98_POUT" --v48-98-comparison "$V98_COMPARE" --output "$COMPARE"
python tools/check_v48_99_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" \
  --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" --v48-98-pipeline "$V98_COMPLETE" --output "$COMPLETE"
cd "$BASE_OUT"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE"
printf 'V48.99 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
