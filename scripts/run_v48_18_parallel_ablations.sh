#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONUNBUFFERED=1
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_18_duet_ablations}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"
python tools/audit_dedicated_protocol_v48_16.py --protocol-root "$PROTOCOL_ROOT" --output "$ROOT/dedicated_protocol_audit.json"
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
GROUP_INDEX="$ROOT/evidence_adapt_teacher_pcd_index.jsonl"; GROUP_SUMMARY="$ROOT/evidence_adapt_teacher_pcd_index_summary.json"
if [[ ! -f "$GROUP_INDEX" || "${REBUILD_ADAPT_INDEX:-0}" == 1 ]]; then
  python tools/build_teacher_pcd_index_v48.py --dataset "$TRAIN_NEAR,$TRAIN_CONTACT" --output "$GROUP_INDEX" \
    --summary-output "$GROUP_SUMMARY" --positive-gain="${POSITIVE_GAIN:-0.015}" --quality-mode=warn \
    >"$ROOT/logs/build_teacher_index.log" 2>&1
fi
run_task(){
  local group="$1" variant="$2" gpu="$3"; local out="$ROOT/tasks/${group}_${variant}"
  local source="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  local context=false context_source=relative stratified=false batch=false replace=false metric=direct_certificate_risk_fold_robust
  case "$group" in
    A_dual_scalar) ;;
    B_dual_tournament) context=true; context_source=tournament ;;
    C_dual_tournament_balanced) context=true; context_source=tournament; stratified=true; batch=true; replace=true ;;
    D_full_duet) context=true; context_source=tournament; stratified=true; batch=true; replace=true; metric=direct_duet_selection_risk ;;
    *) echo "unknown group $group" >&2; return 30 ;;
  esac
  [[ -f "$source" ]] || { echo "missing source $source" >&2; return 30; }
  rm -rf "$out"; mkdir -p "$out/logs"
  set +e
  RUN="$out/candidates/$variant" INIT_CKPT="$source" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
  NUM_WORKERS="${NUM_WORKERS:-1}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" BATCH_SIZE="${BATCH_SIZE:-64}" \
  EVIDENCE_ADAPT_EPOCHS="${EVIDENCE_ADAPT_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${EVIDENCE_ADAPT_PATIENCE:-5}" \
  EVIDENCE_ADAPT_LR="${EVIDENCE_ADAPT_LR:-0.00020}" EVIDENCE_CALIBRATOR_MODE=dual_tail_context \
  EVIDENCE_CALIBRATOR_CONTEXT="$context" EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="$context_source" \
  GROUP_BATCH_STRATIFIED="$stratified" ORDINAL_EVIDENCE_BATCH_BALANCED="$batch" \
  ORDINAL_EVIDENCE_BALANCED_REPLACES_ERM="$replace" ORDINAL_EVIDENCE_INDEPENDENT_TAILS=true \
  POSITIVE_GROUP_BOOST="${POSITIVE_GROUP_BOOST:-1.0}" BEST_METRIC="$metric" \
    bash scripts/adapt_ocrap_v48_18_duet_variant.sh >"$out/logs/adapt.log" 2>&1
  adapt_rc=$?; set -e
  if [[ "$adapt_rc" != 0 ]]; then printf '{"complete":false,"stage":"adaptation","exit_code":%s}\n' "$adapt_rc" > "$out/TASK_FAILED.json"; return 30; fi
  set +e
  OUTPUTDIR="$out" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" GPU0="$gpu" GPU1="$gpu" VARIANTS="$variant" \
    bash scripts/calibrate_v48_16_certificate_pool.sh >"$out/logs/calibrate.log" 2>&1
  cert_rc=$?; set -e
  if [[ "$cert_rc" != 0 && "$cert_rc" != 20 ]]; then printf '{"complete":false,"stage":"certificate","exit_code":%s}\n' "$cert_rc" > "$out/TASK_FAILED.json"; return 30; fi
  python - "$out" "$group" "$variant" "$cert_rc" <<'PY'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); group=sys.argv[2]; variant=sys.argv[3]; rc=int(sys.argv[4]); base=out/'candidates'/variant
ckpt=base/'model_v48_trac_sr'/'best.pt'; cal=base/'calibration'
required=[ckpt,cal/'CERTIFICATE_CALIBRATION_COMPLETE.json',cal/'gamma_rec_by_bucket_v48.json',cal/'direct_value_risk_near_v48.json',cal/'direct_value_risk_contact_v48.json']
missing=[str(p) for p in required if not p.is_file()]
if missing: raise SystemExit('incomplete: '+','.join(missing))
for regime in ('near','contact'):
 d=json.load(open(cal/f'direct_value_risk_{regime}_v48.json'))
 if int(d.get('num_groups',0) or 0)<=0 or int(d.get('num_scenes',0) or 0)<=0: raise SystemExit(f'empty {regime}')
doc={'complete':True,'version':'v48.18-DUET-BRIDGE','group':group,'variant':variant,'certificate_exit':rc,'gate_passed':rc==0,'created_unix':time.time(),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest()}
(out/'TASK_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
}
groups=(A_dual_scalar B_dual_tournament C_dual_tournament_balanced D_full_duet)
pids=(); labels=(); idx=0
for variant in balanced precision; do
  for group in "${groups[@]}"; do
    gpu="$GPU0"; (( idx % 2 == 1 )) && gpu="$GPU1"
    run_task "$group" "$variant" "$gpu" & pids+=("$!"); labels+=("${group}_${variant}"); idx=$((idx+1))
  done
done
failures=0; set +e
for i in "${!pids[@]}"; do wait "${pids[$i]}"; rc=$?; if [[ "$rc" != 0 ]]; then echo "[failed] ${labels[$i]} rc=$rc" >&2; failures=$((failures+1)); fi; done
set -e
python tools/summarize_v48_14_ablations.py --root "$ROOT" --output "$ROOT/ablation_summary_v48_18.json" --version v48.18-DUET-BRIDGE || true
python - "$ROOT" "$failures" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); failures=int(sys.argv[2]); groups=['A_dual_scalar','B_dual_tournament','C_dual_tournament_balanced','D_full_duet']
expected=[f'{g}_{v}' for v in ('balanced','precision') for g in groups]; missing=[x for x in expected if not (root/'tasks'/x/'TASK_COMPLETE.json').is_file()]
doc={'complete':not missing and failures==0,'version':'v48.18-DUET-BRIDGE','max_concurrent_tasks':8,'gpu_assignment':'round_robin; four tasks per A30','workers_per_task':1,'expected_tasks':expected,'missing_tasks':missing,'failed_waits':failures,'created_unix':time.time()}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,indent=2)+'\n')
if doc['complete']: (root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
else: raise SystemExit('incomplete: '+','.join(missing))
PY
