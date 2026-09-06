#!/usr/bin/env bash
# V48.100 OC-JRSD: joint root-query + recovery-chart semantic representation learning.
# Representation-only. V48.99 STOP prerequisite. No source/admission/planner training.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
REFERENCE_A="${V48100_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V48100_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V48100_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V97_BSTATE="${V48100_V97_BALANCED_STATE:-$BASE_OUT/OC-RAP-v48.97-ERSS-balanced.pt}"
V97_PSTATE="${V48100_V97_PRECISION_STATE:-$BASE_OUT/OC-RAP-v48.97-ERSS-precision.pt}"
V99_COMPLETE="${V48100_V99_COMPLETE:-$BASE_OUT/OC-RAP-v48.99-PIPELINE_COMPLETE.json}"
V99_COMPARE="${V48100_V99_COMPARE:-$BASE_OUT/OC-RAP-v48.99-DCP-DRFC-BCDE-RIFA-OC-RJCA-comparison.json}"
V99_BOUT="${V48100_V99_BALANCED:-$BASE_OUT/OC-RAP-v48.99-OCRJ-balanced.json}"
V99_POUT="${V48100_V99_PRECISION:-$BASE_OUT/OC-RAP-v48.99-OCRJ-precision.json}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_INDEX="${V48100_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
CACHE="${V48100_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_100_jrsd_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.100-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.100-JRSD-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.100-JRSD-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.100-JRSD-balanced.pt"; PSTATE="$BASE_OUT/OC-RAP-v48.100-JRSD-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.100-DCP-DRFC-BCDE-RIFA-OC-JRSD-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.100-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.100-OC-JRSD-audits.zip"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_100_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V99_COMPLETE" "$V99_COMPARE" "$V99_BOUT" "$V99_POUT" "$V97_BSTATE" "$V97_PSTATE" <<'PY'
import json,pathlib,sys
pc,cc,br,pr,bs,ps=map(pathlib.Path,sys.argv[1:])
for p in (pc,cc,br,pr,bs,ps):
    if not p.is_file(): raise SystemExit(f'missing V48.100 prerequisite {p}')
p=json.loads(pc.read_text()); c=json.loads(cc.read_text()); d=c.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and p.get('engineering_version')=='v48.99.0-OC-RJCA' and p.get('preregistered_status')=='RECOVERY_JACOBIAN_ALIGNMENT_STOP'):
    raise SystemExit('V48.99 STOP prerequisite missing')
if not(c.get('valid') and c.get('attribution_ready') and d.get('state_chart_preserved') is True and d.get('support_jacobian_go') is False and d.get('reserve_debt_jacobian_go') is False):
    raise SystemExit('V48.99 branch-shape prerequisite missing')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT"; do [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }; done

run_one(){
  local v="$1"; local gpu="$2"; local out="$3"; local state="$4"; local erss="$5"
  local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_100_joint_root_semantic_decoder.py \
    --checkpoint "$ckpt" --erss-state "$erss" \
    --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out" --state-output "$state"
}
set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" "$V97_BSTATE" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" "$V97_PSTATE" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.100 JRSD run failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_100_jrsd.py --balanced "$BOUT" --precision "$POUT" \
  --v48-99-balanced "$V99_BOUT" --v48-99-precision "$V99_POUT" --v48-99-comparison "$V99_COMPARE" --output "$COMPARE"
python tools/check_v48_100_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" \
  --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" --v48-99-pipeline "$V99_COMPLETE" --output "$COMPLETE"
cd "$BASE_OUT"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE"
printf 'V48.100 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
