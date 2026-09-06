#!/usr/bin/env bash
# V48.105 OC-PAEL: preregistered pre-last Stage-I action-equivariance/localization audit after V48.104 STOP.
# Audit only: no planner/source/Stage-I/root-decoder parameters are trained.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
REFERENCE_A="${V48105_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V48105_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V48105_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V102_COMPARE="${V48105_V102_COMPARE:-$BASE_OUT/OC-RAP-v48.102-DCP-DRFC-BCDE-RIFA-OC-AITS-comparison.json}"
V102_BALANCED="${V48105_V102_BALANCED:-$BASE_OUT/OC-RAP-v48.102-AITS-balanced.json}"
V102_PRECISION="${V48105_V102_PRECISION:-$BASE_OUT/OC-RAP-v48.102-AITS-precision.json}"
V104_PIPELINE="${V48105_V104_PIPELINE:-$BASE_OUT/OC-RAP-v48.104-PIPELINE_COMPLETE.json}"
V104_COMPARE="${V48105_V104_COMPARE:-$BASE_OUT/OC-RAP-v48.104-DCP-DRFC-BCDE-RIFA-OC-NICR-comparison.json}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_INDEX="${V48105_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
CACHE="${V48105_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_105_pael_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.105-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.105-PAEL-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.105-PAEL-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.105-PAEL-balanced.pt"; PSTATE="$BASE_OUT/OC-RAP-v48.105-PAEL-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.105-DCP-DRFC-BCDE-RIFA-OC-PAEL-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.105-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.105-OC-PAEL-audits.zip"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_105_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V104_PIPELINE" "$V104_COMPARE" "$V102_COMPARE" "$V102_BALANCED" "$V102_PRECISION" <<'PY'
import json,pathlib,sys
p104,c104,c102,b102,p102=map(pathlib.Path,sys.argv[1:])
for p in (p104,c104,c102,b102,p102):
    if not p.is_file(): raise SystemExit(f'missing V48.105 prerequisite {p}')
p=json.loads(p104.read_text()); c=json.loads(c104.read_text()); d=c.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and p.get('engineering_version')=='v48.104.0-OC-NICR' and p.get('preregistered_status')=='NOMINAL_INVARIANT_CONTROL_REFINEMENT_STOP'):
    raise SystemExit('V48.104 STOP pipeline prerequisite missing')
if not(c.get('valid') and c.get('attribution_ready') and d.get('state_go') is True and d.get('support_go') is False and d.get('reserve_go') is False and d.get('next_branch')=='close_last_stage_i_block_refinement_then_preregister_pre_last_token_action_equivariance_audit_no_broad_encoder_or_source_sweep'):
    raise SystemExit('V48.104 pre-last audit branch prerequisite missing')
x=json.loads(c102.read_text()); dx=x.get('preregistered_decision') or {}
if not(x.get('valid') and x.get('attribution_ready') and x.get('engineering_version')=='v48.102.0-OC-AITS' and dx.get('status')=='STAGE_I_ACTION_INFORMATION_SUFFICIENCY_STOP'):
    raise SystemExit('V48.102 final Stage-I audit reference missing')
for f,v in ((b102,'balanced'),(p102,'precision')):
    r=json.loads(f.read_text())
    if not(r.get('valid') and r.get('engineering_version')=='v48.102.0-OC-AITS' and r.get('variant')==v):
        raise SystemExit(f'V48.102 result contract mismatch {f}')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT"; do [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }; done

run_one(){
  local v="$1"; local gpu="$2"; local out="$3"; local state="$4"
  local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_105_prelast_action_equivariance_localization_audit.py \
    --checkpoint "$ckpt" --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out" --state-output "$state"
}
set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.105 PAEL run failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_105_pael.py --balanced "$BOUT" --precision "$POUT" \
  --v102-balanced "$V102_BALANCED" --v102-precision "$V102_PRECISION" --v102-comparison "$V102_COMPARE" \
  --v104-comparison "$V104_COMPARE" --v104-pipeline "$V104_PIPELINE" --output "$COMPARE"
python tools/check_v48_105_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" \
  --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" \
  --v48-104-pipeline "$V104_PIPELINE" --v48-104-comparison "$V104_COMPARE" --v48-102-comparison "$V102_COMPARE" --output "$COMPLETE"
cd "$BASE_OUT"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE"
printf 'V48.105 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
