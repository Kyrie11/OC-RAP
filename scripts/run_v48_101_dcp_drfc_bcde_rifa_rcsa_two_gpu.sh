#!/usr/bin/env bash
# V48.101 OC-RCSA: root cross-attention semantic alignment after V48.100 STOP.
# Opens existing root cross-attention only. V48.100 query/chart are frozen and must reproduce the V48.100 baseline exactly.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
REFERENCE_A="${V48101_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V48101_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V48101_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V100_COMPLETE="${V48101_V100_COMPLETE:-$BASE_OUT/OC-RAP-v48.100-PIPELINE_COMPLETE.json}"
V100_COMPARE="${V48101_V100_COMPARE:-$BASE_OUT/OC-RAP-v48.100-DCP-DRFC-BCDE-RIFA-OC-JRSD-comparison.json}"
V100_BOUT="${V48101_V100_BALANCED:-$BASE_OUT/OC-RAP-v48.100-JRSD-balanced.json}"
V100_POUT="${V48101_V100_PRECISION:-$BASE_OUT/OC-RAP-v48.100-JRSD-precision.json}"
V100_BSTATE="${V48101_V100_BALANCED_STATE:-$BASE_OUT/OC-RAP-v48.100-JRSD-balanced.pt}"
V100_PSTATE="${V48101_V100_PRECISION_STATE:-$BASE_OUT/OC-RAP-v48.100-JRSD-precision.pt}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_INDEX="${V48101_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
CACHE="${V48101_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_101_rcsa_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.101-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.101-RCSA-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.101-RCSA-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.101-RCSA-balanced.pt"; PSTATE="$BASE_OUT/OC-RAP-v48.101-RCSA-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.101-DCP-DRFC-BCDE-RIFA-OC-RCSA-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.101-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.101-OC-RCSA-audits.zip"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_101_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V100_COMPLETE" "$V100_COMPARE" "$V100_BOUT" "$V100_POUT" "$V100_BSTATE" "$V100_PSTATE" <<'PY'
import json,pathlib,sys,torch
pc,cc,br,pr,bs,ps=map(pathlib.Path,sys.argv[1:])
for p in (pc,cc,br,pr,bs,ps):
    if not p.is_file(): raise SystemExit(f'missing V48.101 prerequisite {p}')
p=json.loads(pc.read_text()); c=json.loads(cc.read_text()); d=c.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and p.get('engineering_version')=='v48.100.0-OC-JRSD' and p.get('preregistered_status')=='JOINT_ROOT_SEMANTIC_DECODER_STOP'):
    raise SystemExit('V48.100 STOP prerequisite missing')
if not(c.get('valid') and c.get('attribution_ready') and d.get('state_representation_go') is True and d.get('support_action_representation_go') is False and d.get('reserve_debt_representation_go') is False and d.get('next_branch')=='close_root_query_plus_chart_family_then_preregister_root_cross_attention_semantic_objective_no_source_sweep'):
    raise SystemExit('V48.100 branch-shape prerequisite missing')
for f,v in ((bs,'balanced'),(ps,'precision')):
    s=torch.load(f,map_location='cpu',weights_only=False)
    if s.get('engineering_version')!='v48.100.0-OC-JRSD' or s.get('variant')!=v or int(s.get('joint_representation_parameter_count',-1))!=2306:
        raise SystemExit(f'V48.100 state contract mismatch {f}')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT"; do [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }; done

run_one(){
  local v="$1"; local gpu="$2"; local out="$3"; local state="$4"; local v100_state="$5"; local v100_result="$6"
  local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_101_root_cross_attention_semantic_alignment.py \
    --checkpoint "$ckpt" --v100-state "$v100_state" --v100-result "$v100_result" \
    --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out" --state-output "$state"
}
set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" "$V100_BSTATE" "$V100_BOUT" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" "$V100_PSTATE" "$V100_POUT" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.101 RCSA run failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_101_rcsa.py --balanced "$BOUT" --precision "$POUT" \
  --v100-balanced "$V100_BOUT" --v100-precision "$V100_POUT" --v100-comparison "$V100_COMPARE" --output "$COMPARE"
python tools/check_v48_101_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" \
  --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" \
  --v48-100-pipeline "$V100_COMPLETE" --v48-100-comparison "$V100_COMPARE" --output "$COMPLETE"
cd "$BASE_OUT"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE"
printf 'V48.101 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
