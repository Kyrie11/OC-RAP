#!/usr/bin/env bash
# V48.96 OC-SRROA: target-specific frozen Stage-I/root support-reserve observability audit.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"; PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
REFERENCE_A="${V4896_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V4896_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V4896_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V95_COMPLETE="${V4896_V95_COMPLETE:-$BASE_OUT/OC-RAP-v48.95-PIPELINE_COMPLETE.json}"
RUNTIME="$BASE_OUT/OC-RAP-v48.96-runtime-code-contract.json"; BOUT="$BASE_OUT/OC-RAP-v48.96-SRROA-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.96-SRROA-precision.json"; COMPARE="$BASE_OUT/OC-RAP-v48.96-DCP-DRFC-BCDE-RIFA-OC-SRROA-comparison.json"; COMPLETE="$BASE_OUT/OC-RAP-v48.96-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.96-OC-SRROA-audits.zip"
CERT_INDEX="${V4896_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"; CERT_SUM="${V4896_CERT_SUMMARY:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index-summary.json}"; CACHE="${V4896_FEATURE_CACHE:-$BASE_OUT/.ocrap_v48_96_root_probe_cache}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"
python tools/check_v48_96_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V95_COMPLETE" "$V93_AUDIT" "$REFERENCE_A" "$L80_RUN" <<'PY'
import json,pathlib,sys
p95,v93,ref,l80=map(pathlib.Path,sys.argv[1:])
for p in (p95,v93):
 if not p.is_file():raise SystemExit(f'missing V48.96 prerequisite {p}')
if not ref.is_dir() or not l80.is_dir():raise SystemExit('missing reference/L80 run')
p=json.loads(p95.read_text())
if not(p.get('valid') and p.get('attribution_ready') and p.get('preregistered_status')=='FROZEN_NATIVE_RECOVERY_OBSERVABILITY_STOP'):
 raise SystemExit('V48.95 STOP prerequisite missing')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX";do [[ -s "$f" ]]||{ echo "missing teacher index $f" >&2;exit 30;};done
if [[ ! -s "$CERT_INDEX" || ! -s "$CERT_SUM" ]]; then
 rm -f "$CERT_INDEX" "$CERT_SUM"
 python tools/build_teacher_pcd_index_v48.py --dataset "$CERT_NEAR,$CERT_CONTACT" --output "$CERT_INDEX" --summary-output "$CERT_SUM" --alpha 0.2 --beta 0.2 --top-m 8 --option-execution-semantics observation_class --positive-gain 0.015 --deployable-macro-ids 2,3,5,6,7 --quality-mode off --workers "${V4896_LABEL_WORKERS:-8}"
fi
run_probe(){
 local v="$1"; local gpu="$2"; local out="$3"; local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
 [[ -f "$ckpt" ]]||{ echo "missing L80 checkpoint $ckpt" >&2;return 30;}
 CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_96_support_reserve_root_observability_probe.py --checkpoint "$ckpt" --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out"
}
set +e; run_probe balanced "$GPU0" "$BOUT" & p0=$!; run_probe precision "$GPU1" "$POUT" & p1=$!; wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.96 probe failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_96_srroa.py --balanced "$BOUT" --precision "$POUT" --output "$COMPARE"
python tools/check_v48_96_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" --comparison "$COMPARE" --v48-95-pipeline "$V95_COMPLETE" --output "$COMPLETE"
cd "$BASE_OUT"; zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$CERT_SUM" "$COMPARE" "$COMPLETE"
printf 'V48.96 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
