#!/usr/bin/env bash
# V48.94 OC-SRCA: Observation-Consistent Support-Reserve Complementarity Admission.
# Fixed zero-parameter absolute-source experiment after V48.93 complementarity GO.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"; export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
L80_RUN="${V4894_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V4894_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V93_COMPARE="${V4894_V93_COMPARE:-$BASE_OUT/OC-RAP-v48.93-DCP-DRFC-BCDE-RIFA-OC-FMCA-comparison.json}"
V93_COMPLETE="${V4894_V93_COMPLETE:-$BASE_OUT/OC-RAP-v48.93-PIPELINE_COMPLETE.json}"
MAIN_RUN="$BASE_OUT/ocrap_v48_94_dcp_drfc_bcde_rifa_srca_main"
RUNTIME="$BASE_OUT/OC-RAP-v48.94-runtime-code-contract.json"; AUDIT="$BASE_OUT/OC-RAP-v48.94-support-reserve-admission-audit.json"; COMPARE="$BASE_OUT/OC-RAP-v48.94-DCP-DRFC-BCDE-RIFA-OC-SRCA-comparison.json"; COMPLETE="$BASE_OUT/OC-RAP-v48.94-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.94-OC-SRCA-audits.zip"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"; CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
mkdir -p "$BASE_OUT"; rm -rf "$MAIN_RUN"; rm -f "$RUNTIME" "$AUDIT" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"
python tools/check_v48_94_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V93_COMPLETE" "$V93_COMPARE" "$V93_AUDIT" "$L80_RUN" <<'PY'
import json,pathlib,sys
pc,cmp,aud,l80=map(pathlib.Path,sys.argv[1:]);
for p in (pc,cmp,aud):
 if not p.is_file(): raise SystemExit(f'missing V48.94 prerequisite: {p}')
if not l80.is_dir(): raise SystemExit(f'missing L80 run: {l80}')
p=json.loads(pc.read_text());c=json.loads(cmp.read_text());d=c.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and p.get('preregistered_status')=='PCD_FACTOR_COMPLEMENTARITY_GO'): raise SystemExit('V48.93 completed complementarity GO prerequisite missing')
if not(c.get('valid') and d.get('status')=='PCD_FACTOR_COMPLEMENTARITY_GO'): raise SystemExit('V48.93 comparison complementarity GO missing')
PY

eval_variant(){
 local v="$1"
 local gpu="$2"
 local base="$L80_RUN/candidates/$v"
 local ckpt="$base/model_v48_trac_sr/best.pt"
 local out="$MAIN_RUN/candidates/$v/evaluation"
 [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
 [[ -f "$base/POLICY_CONTRACT.env" ]] || { echo "missing L80 policy contract" >&2; return 30; }
 mkdir -p "$out" "$MAIN_RUN/logs"
 set -a
 source "$base/POLICY_CONTRACT.env"
 set +a
 local common=(--checkpoint "$ckpt" --method-version=v48_94_oc_srca --risk-source="${RISK_SOURCE:-ordinal_evidence}" --option-execution-semantics=observation_class --conditional-recovery-ranking --proposal-top-k 5 --evidence-rerank-top-k --absolute-feasibility-mode=support_reserve --absolute-feasibility-threshold=0.5 --positive-gain=0.015 --negative-gain=0.010 --harm-label-mode=component_veto --opportunity-label-mode=raw_benefit --gate-positive-mode=safe_benefit --required-min-groups=1 --required-min-scenes=1 --min-fit-selected=1 --min-fit-precision-lcb=0 --max-fit-harmful-group-ucb=1 --max-fit-harmful-selected-ucb=1)
 CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py --dataset "$DEV_NEAR" --allowed-splits=evidence_adapt_dev --bucket near "${common[@]}" --development-fit-only --output "$out/dev_diagnostic_near_v48.json" --proposal-rows-output "$out/dev_diagnostic_near_v48.proposal_rows.jsonl" >"$MAIN_RUN/logs/${v}_dev_near.log" 2>&1
 CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py --dataset "$DEV_CONTACT" --allowed-splits=evidence_adapt_dev --bucket contact "${common[@]}" --development-fit-only --output "$out/dev_diagnostic_contact_v48.json" --proposal-rows-output "$out/dev_diagnostic_contact_v48.proposal_rows.jsonl" >"$MAIN_RUN/logs/${v}_dev_contact.log" 2>&1
 CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py --dataset "$CERT_NEAR" --allowed-splits=certificate_pool --bucket near "${common[@]}" --development-fit-only --output "$out/direct_value_risk_near_v48.json" --proposal-rows-output "$out/direct_value_risk_near_v48.proposal_rows.jsonl" >"$MAIN_RUN/logs/${v}_cert_near.log" 2>&1
 CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py --dataset "$CERT_CONTACT" --allowed-splits=certificate_pool --bucket contact "${common[@]}" --development-fit-only --output "$out/direct_value_risk_contact_v48.json" --proposal-rows-output "$out/direct_value_risk_contact_v48.proposal_rows.jsonl" >"$MAIN_RUN/logs/${v}_cert_contact.log" 2>&1
}
set +e; eval_variant balanced "$GPU0" & p0=$!; eval_variant precision "$GPU1" & p1=$!; wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e; [[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.94 evaluation failed balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/audit_v48_94_srca.py --l80-run "$L80_RUN" --v93-audit "$V93_AUDIT" --main-run "$MAIN_RUN" --output "$AUDIT"
python tools/compare_v48_94_srca.py --audit "$AUDIT" --v93-comparison "$V93_COMPARE" --output "$COMPARE"
python tools/check_v48_94_pipeline_complete.py --runtime "$RUNTIME" --audit "$AUDIT" --comparison "$COMPARE" --v48-93-pipeline "$V93_COMPLETE" --v48-93-comparison "$V93_COMPARE" --output "$COMPLETE"
cd "$BASE_OUT"; zip -qr "$(basename "$MAIN_RUN").zip" "$(basename "$MAIN_RUN")"; zip -qj "$AUDITS_ZIP" "$RUNTIME" "$AUDIT" "$COMPARE" "$COMPLETE"
printf 'V48.94 complete. Upload:\n%s\n%s\n' "$BASE_OUT/$(basename "$MAIN_RUN").zip" "$AUDITS_ZIP"
