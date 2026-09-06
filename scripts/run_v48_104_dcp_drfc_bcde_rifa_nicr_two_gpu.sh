#!/usr/bin/env bash
# V48.104 OC-NICR: preregistered last-Stage-I-block control-sufficient representation objective after V48.103 STOP.
# Only the historical last Stage-I Transformer block is adapted through a nominal-invariant counterfactual residual.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
REFERENCE_A="${V48104_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V48104_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V48104_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V103_PIPELINE="${V48104_V103_PIPELINE:-$BASE_OUT/OC-RAP-v48.103-PIPELINE_COMPLETE.json}"
V103_COMPARE="${V48104_V103_COMPARE:-$BASE_OUT/OC-RAP-v48.103-DCP-DRFC-BCDE-RIFA-OC-FCSS-comparison.json}"
V103_BALANCED="${V48104_V103_BALANCED:-$BASE_OUT/OC-RAP-v48.103-FCSS-balanced.json}"
V103_PRECISION="${V48104_V103_PRECISION:-$BASE_OUT/OC-RAP-v48.103-FCSS-precision.json}"
V103_BSTATE="${V48104_V103_BSTATE:-$BASE_OUT/OC-RAP-v48.103-FCSS-balanced.pt}"
V103_PSTATE="${V48104_V103_PSTATE:-$BASE_OUT/OC-RAP-v48.103-FCSS-precision.pt}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_INDEX="${V48104_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
CACHE="${V48104_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_104_nicr_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.104-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.104-NICR-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.104-NICR-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.104-NICR-balanced.pt"; PSTATE="$BASE_OUT/OC-RAP-v48.104-NICR-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.104-DCP-DRFC-BCDE-RIFA-OC-NICR-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.104-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.104-OC-NICR-audits.zip"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_104_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V103_PIPELINE" "$V103_COMPARE" <<'PY'
import json,pathlib,sys
p,c=map(pathlib.Path,sys.argv[1:]);
for x in (p,c):
    if not x.is_file(): raise SystemExit(f'missing V48.104 prerequisite {x}')
pd=json.loads(p.read_text()); cd=json.loads(c.read_text()); d=cd.get('preregistered_decision') or {}
if not(pd.get('valid') and pd.get('attribution_ready') and pd.get('engineering_version')=='v48.103.0-OC-FCSS' and pd.get('preregistered_status')=='FACTORIZED_CONTROL_SUFFICIENT_STATE_STOP'):
    raise SystemExit('V48.103 STOP pipeline prerequisite missing')
if not(cd.get('valid') and cd.get('attribution_ready') and d.get('factorized_state_representation_go') is True and d.get('factorized_support_action_go') is False and d.get('factorized_reserve_debt_go') is False and d.get('next_branch')=='close_frozen_stage_i_readout_family_then_preregister_last_stage_i_block_control_sufficient_representation_objective_no_broad_encoder_or_source_sweep'):
    raise SystemExit('V48.103 last-block branch prerequisite missing')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT" "$V103_BALANCED" "$V103_PRECISION" "$V103_BSTATE" "$V103_PSTATE"; do [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }; done

run_one(){
  local v="$1"; local gpu="$2"; local out="$3"; local state="$4"; local v103_state="$5"; local v103_result="$6"
  local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_104_nominal_invariant_control_refinement.py \
    --checkpoint "$ckpt" --v103-state "$v103_state" --v103-result "$v103_result" \
    --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out" --state-output "$state"
}
set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" "$V103_BSTATE" "$V103_BALANCED" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" "$V103_PSTATE" "$V103_PRECISION" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.104 NICR run failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_104_nicr.py --balanced "$BOUT" --precision "$POUT" --v103-balanced "$V103_BALANCED" --v103-precision "$V103_PRECISION" --v103-comparison "$V103_COMPARE" --output "$COMPARE"
python tools/check_v48_104_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" --v48-103-pipeline "$V103_PIPELINE" --v48-103-comparison "$V103_COMPARE" --output "$COMPLETE"
cd "$BASE_OUT"; zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE"
printf 'V48.104 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
