#!/usr/bin/env bash
# V48.106 OC-PEAO: preregistered one-block-earlier pre-encoder action-orientation audit after V48.105 STOP.
# Audit only: no planner/source/Stage-I/root-decoder parameters are trained.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
REFERENCE_A="${V48106_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V48106_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V48106_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V102_COMPARE="${V48106_V102_COMPARE:-$BASE_OUT/OC-RAP-v48.102-DCP-DRFC-BCDE-RIFA-OC-AITS-comparison.json}"
V102_BALANCED="${V48106_V102_BALANCED:-$BASE_OUT/OC-RAP-v48.102-AITS-balanced.json}"
V102_PRECISION="${V48106_V102_PRECISION:-$BASE_OUT/OC-RAP-v48.102-AITS-precision.json}"
V105_PIPELINE="${V48106_V105_PIPELINE:-$BASE_OUT/OC-RAP-v48.105-PIPELINE_COMPLETE.json}"
V105_COMPARE="${V48106_V105_COMPARE:-$BASE_OUT/OC-RAP-v48.105-DCP-DRFC-BCDE-RIFA-OC-PAEL-comparison.json}"
V105_BALANCED="${V48106_V105_BALANCED:-$BASE_OUT/OC-RAP-v48.105-PAEL-balanced.json}"
V105_PRECISION="${V48106_V105_PRECISION:-$BASE_OUT/OC-RAP-v48.105-PAEL-precision.json}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_INDEX="${V48106_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
CACHE="${V48106_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_106_peao_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.106-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.106-PEAO-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.106-PEAO-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.106-PEAO-balanced.pt"; PSTATE="$BASE_OUT/OC-RAP-v48.106-PEAO-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.106-DCP-DRFC-BCDE-RIFA-OC-PEAO-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.106-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.106-OC-PEAO-audits.zip"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_106_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V105_PIPELINE" "$V105_COMPARE" "$V105_BALANCED" "$V105_PRECISION" "$V102_COMPARE" "$V102_BALANCED" "$V102_PRECISION" <<'PY'
import json,pathlib,sys
p105,c105,b105,q105,c102,b102,q102=map(pathlib.Path,sys.argv[1:])
for p in (p105,c105,b105,q105,c102,b102,q102):
    if not p.is_file(): raise SystemExit(f'missing V48.106 prerequisite {p}')
p=json.loads(p105.read_text()); c=json.loads(c105.read_text()); d=c.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and p.get('engineering_version')=='v48.105.0-OC-PAEL' and p.get('preregistered_status')=='PRELAST_ACTION_EQUIVARIANCE_LOCALIZATION_STOP'):
    raise SystemExit('V48.105 STOP pipeline prerequisite missing')
if not(c.get('valid') and c.get('attribution_ready') and d.get('status')=='PRELAST_ACTION_EQUIVARIANCE_LOCALIZATION_STOP' and d.get('next_branch')=='prelast_action_equivariance_insufficient_then_preregister_one_block_earlier_action_interaction_audit_no_training_or_source_sweep'):
    raise SystemExit('V48.105 one-block-earlier audit branch prerequisite missing')
for f,v in ((b105,'balanced'),(q105,'precision')):
    r=json.loads(f.read_text())
    if not(r.get('valid') and r.get('engineering_version')=='v48.105.0-OC-PAEL' and r.get('variant')==v and r.get('prelast_only')):
        raise SystemExit(f'V48.105 result contract mismatch {f}')
x=json.loads(c102.read_text()); dx=x.get('preregistered_decision') or {}
if not(x.get('valid') and x.get('attribution_ready') and x.get('engineering_version')=='v48.102.0-OC-AITS' and dx.get('status')=='STAGE_I_ACTION_INFORMATION_SUFFICIENCY_STOP'):
    raise SystemExit('V48.102 final Stage-I audit reference missing')
for f,v in ((b102,'balanced'),(q102,'precision')):
    r=json.loads(f.read_text())
    if not(r.get('valid') and r.get('engineering_version')=='v48.102.0-OC-AITS' and r.get('variant')==v):
        raise SystemExit(f'V48.102 result contract mismatch {f}')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT"; do [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }; done

run_one(){
  local v="$1"; local gpu="$2"; local out="$3"; local state="$4"
  local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_106_preencoder_action_orientation_audit.py \
    --checkpoint "$ckpt" --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out" --state-output "$state"
}
set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.106 PEAO run failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_106_peao.py --balanced "$BOUT" --precision "$POUT" \
  --v105-balanced "$V105_BALANCED" --v105-precision "$V105_PRECISION" --v105-comparison "$V105_COMPARE" --v105-pipeline "$V105_PIPELINE" \
  --v102-balanced "$V102_BALANCED" --v102-precision "$V102_PRECISION" --v102-comparison "$V102_COMPARE" --output "$COMPARE"
python tools/check_v48_106_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" \
  --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" \
  --v48-105-pipeline "$V105_PIPELINE" --v48-105-comparison "$V105_COMPARE" --v48-102-comparison "$V102_COMPARE" --output "$COMPLETE"
cd "$BASE_OUT"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE"
printf 'V48.106 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
