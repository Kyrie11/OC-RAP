#!/usr/bin/env bash
# V48.102 OC-AITS: Stage-I action-information transport sufficiency audit after V48.101 STOP.
# Audit only: no planner/source/Stage-I/root-decoder parameters are trained.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
REFERENCE_A="${V48102_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V48102_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V48102_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V101_PIPELINE="${V48102_V101_PIPELINE:-$BASE_OUT/OC-RAP-v48.101-PIPELINE_COMPLETE.json}"
V101_COMPARE="${V48102_V101_COMPARE:-$BASE_OUT/OC-RAP-v48.101-DCP-DRFC-BCDE-RIFA-OC-RCSA-comparison.json}"
V101_BALANCED="${V48102_V101_BALANCED:-$BASE_OUT/OC-RAP-v48.101-RCSA-balanced.json}"
V101_PRECISION="${V48102_V101_PRECISION:-$BASE_OUT/OC-RAP-v48.101-RCSA-precision.json}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_INDEX="${V48102_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
CACHE="${V48102_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_102_aits_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.102-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.102-AITS-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.102-AITS-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.102-AITS-balanced.pt"; PSTATE="$BASE_OUT/OC-RAP-v48.102-AITS-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.102-DCP-DRFC-BCDE-RIFA-OC-AITS-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.102-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.102-OC-AITS-audits.zip"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_102_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V101_PIPELINE" "$V101_COMPARE" "$V101_BALANCED" "$V101_PRECISION" <<'PY'
import json,pathlib,sys
pp,cc,br,pr=map(pathlib.Path,sys.argv[1:])
for p in (pp,cc,br,pr):
    if not p.is_file(): raise SystemExit(f'missing V48.102 prerequisite {p}')
p=json.loads(pp.read_text()); c=json.loads(cc.read_text()); d=c.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and p.get('engineering_version')=='v48.101.0-OC-RCSA' and p.get('preregistered_status')=='ROOT_CROSS_ATTENTION_SEMANTIC_ALIGNMENT_STOP'):
    raise SystemExit('V48.101 STOP pipeline prerequisite missing')
if not(c.get('valid') and c.get('attribution_ready') and d.get('state_representation_go') is True and d.get('support_action_representation_go') is False and d.get('reserve_debt_representation_go') is False and d.get('next_branch')=='close_root_decoder_semantic_family_then_preregister_stage_i_action_information_transport_audit_no_capacity_or_source_sweep'):
    raise SystemExit('V48.101 branch-shape prerequisite missing')
for f,v in ((br,'balanced'),(pr,'precision')):
    r=json.loads(f.read_text())
    if not(r.get('valid') and r.get('engineering_version')=='v48.101.0-OC-RCSA' and r.get('variant')==v):
        raise SystemExit(f'V48.101 result contract mismatch {f}')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT"; do [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }; done

run_one(){
  local v="$1"; local gpu="$2"; local out="$3"; local state="$4"
  local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_102_stage_i_action_information_transport_audit.py \
    --checkpoint "$ckpt" --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out" --state-output "$state"
}
set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.102 AITS run failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_102_aits.py --balanced "$BOUT" --precision "$POUT" \
  --v101-balanced "$V101_BALANCED" --v101-precision "$V101_PRECISION" --v101-comparison "$V101_COMPARE" --output "$COMPARE"
python tools/check_v48_102_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" \
  --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" \
  --v48-101-pipeline "$V101_PIPELINE" --v48-101-comparison "$V101_COMPARE" --output "$COMPLETE"
cd "$BASE_OUT"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE"
printf 'V48.102 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
