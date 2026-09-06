#!/usr/bin/env bash
# V48.103 OC-FCSS: minimal Stage-I factorized control-sufficient representation after V48.102 all-STOP.
# Representation experiment only: historical planner/Stage-I/root decoder/source remain frozen.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
REFERENCE_A="${V48103_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V48103_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V48103_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V100_BSTATE="${V48103_V100_BSTATE:-$BASE_OUT/OC-RAP-v48.100-JRSD-balanced.pt}"
V100_PSTATE="${V48103_V100_PSTATE:-$BASE_OUT/OC-RAP-v48.100-JRSD-precision.pt}"
V101_BALANCED="${V48103_V101_BALANCED:-$BASE_OUT/OC-RAP-v48.101-RCSA-balanced.json}"
V101_PRECISION="${V48103_V101_PRECISION:-$BASE_OUT/OC-RAP-v48.101-RCSA-precision.json}"
V102_PIPELINE="${V48103_V102_PIPELINE:-$BASE_OUT/OC-RAP-v48.102-PIPELINE_COMPLETE.json}"
V102_COMPARE="${V48103_V102_COMPARE:-$BASE_OUT/OC-RAP-v48.102-DCP-DRFC-BCDE-RIFA-OC-AITS-comparison.json}"
V102_BALANCED="${V48103_V102_BALANCED:-$BASE_OUT/OC-RAP-v48.102-AITS-balanced.json}"
V102_PRECISION="${V48103_V102_PRECISION:-$BASE_OUT/OC-RAP-v48.102-AITS-precision.json}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_INDEX="${V48103_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
CACHE="${V48103_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_103_fcss_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.103-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.103-FCSS-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.103-FCSS-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.103-FCSS-balanced.pt"; PSTATE="$BASE_OUT/OC-RAP-v48.103-FCSS-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.103-DCP-DRFC-BCDE-RIFA-OC-FCSS-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.103-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.103-OC-FCSS-audits.zip"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_103_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V102_PIPELINE" "$V102_COMPARE" "$V102_BALANCED" "$V102_PRECISION" <<'PY'
import json,pathlib,sys
pp,cc,br,pr=map(pathlib.Path,sys.argv[1:])
for p in (pp,cc,br,pr):
    if not p.is_file(): raise SystemExit(f'missing V48.103 prerequisite {p}')
p=json.loads(pp.read_text()); c=json.loads(cc.read_text()); d=c.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and p.get('engineering_version')=='v48.102.0-OC-AITS' and p.get('preregistered_status')=='STAGE_I_ACTION_INFORMATION_SUFFICIENCY_STOP'):
    raise SystemExit('V48.102 STOP pipeline prerequisite missing')
if not(c.get('valid') and c.get('attribution_ready') and d.get('stage_i_state_observability_go') is False and d.get('stage_i_support_action_observability_go') is False and d.get('stage_i_reserve_action_observability_go') is False and d.get('next_branch')=='stage_i_action_information_insufficient_then_preregister_minimal_stage_i_recovery_representation_objective_no_source_or_broad_encoder_sweep'):
    raise SystemExit('V48.102 all-STOP branch-shape prerequisite missing')
for f,v in ((br,'balanced'),(pr,'precision')):
    r=json.loads(f.read_text())
    if not(r.get('valid') and r.get('engineering_version')=='v48.102.0-OC-AITS' and r.get('variant')==v and r.get('audit_only') is True):
        raise SystemExit(f'V48.102 result contract mismatch {f}')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT" "$V100_BSTATE" "$V100_PSTATE" "$V101_BALANCED" "$V101_PRECISION"; do [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }; done

run_one(){
  local v="$1"; local gpu="$2"; local out="$3"; local state="$4"; local v100_state="$5"
  local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_103_factorized_control_sufficient_state.py \
    --checkpoint "$ckpt" --v100-state "$v100_state" \
    --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out" --state-output "$state"
}
set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" "$V100_BSTATE" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" "$V100_PSTATE" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.103 FCSS run failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_103_fcss.py --balanced "$BOUT" --precision "$POUT" \
  --v102-balanced "$V102_BALANCED" --v102-precision "$V102_PRECISION" --v102-comparison "$V102_COMPARE" \
  --v101-balanced "$V101_BALANCED" --v101-precision "$V101_PRECISION" --output "$COMPARE"
python tools/check_v48_103_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" \
  --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" \
  --v48-102-pipeline "$V102_PIPELINE" --v48-102-comparison "$V102_COMPARE" --output "$COMPLETE"
cd "$BASE_OUT"
zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE"
printf 'V48.103 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
