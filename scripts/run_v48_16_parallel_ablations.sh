#!/usr/bin/env bash
set -euo pipefail
# Four ablations run concurrently per variant. A/C share GPU0; B/D share GPU1.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_16_ablations}"
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
if [[ ! -f "$GROUP_INDEX" ]]; then
  python tools/build_teacher_pcd_index_v48.py --dataset "$TRAIN_NEAR,$TRAIN_CONTACT" --output "$GROUP_INDEX" --summary-output "$GROUP_SUMMARY" --quality-mode=warn >"$ROOT/logs/build_teacher_index.log" 2>&1
fi
run_adapt(){
  local out="$1" variant="$2" gpu="$3" source="$4" mode="$5"
  local cb=0 bm=0 hm=0 anchor=0 hh=0 hb=0 hidden=4 scale=0.20
  case "$mode" in
    old_tiny) hh=1.0; hb=1.0; hidden=8; scale=0.30;;
    balanced_margin) cb=1.0; bm=0.75; hm=1.0; hh=0.5; hb=0.75;;
    full_anchor) cb=1.0; bm=0.75; hm=1.0; anchor=0.10; hh=0.5; hb=0.75;;
    *) echo "bad mode $mode" >&2; return 30;;
  esac
  RUN="$out/candidates/$variant" INIT_CKPT="$source/model_v48_trac_sr/best.pt" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" BATCH_SIZE="${BATCH_SIZE:-72}" NUM_WORKERS="${NUM_WORKERS:-3}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" \
  EVIDENCE_ADAPT_EPOCHS="${EVIDENCE_ADAPT_EPOCHS:-12}" EVIDENCE_ADAPT_PATIENCE="${EVIDENCE_ADAPT_PATIENCE:-4}" \
  EVIDENCE_CALIBRATOR_HIDDEN="$hidden" EVIDENCE_CALIBRATOR_SCALE="$scale" \
  ORDINAL_EVIDENCE_CLASS_BALANCED_WEIGHT="$cb" ORDINAL_EVIDENCE_BENEFIT_MARGIN_WEIGHT="$bm" ORDINAL_EVIDENCE_HARM_MARGIN_WEIGHT="$hm" \
  EVIDENCE_CALIBRATOR_ANCHOR_WEIGHT="$anchor" ORDINAL_EVIDENCE_HARD_HARM_WEIGHT="$hh" ORDINAL_EVIDENCE_HARD_BENEFIT_WEIGHT="$hb" \
    bash scripts/adapt_ocrap_v48_16_anchor_variant.sh >"$out/logs/adapt.log" 2>&1
}
run_task(){
  local group="$1" variant="$2" gpu="$3" mode="$4"
  local out="$ROOT/tasks/${group}_${variant}" source="$SOURCE_RUN/candidates/$variant"
  rm -rf "$out"; mkdir -p "$out/candidates/$variant" "$out/logs"
  [[ -f "$source/model_v48_trac_sr/best.pt" ]] || { echo "missing source $source" >&2; return 30; }
  if [[ "$mode" == source ]]; then
    local dst="$out/candidates/$variant"
    ln -s "$(realpath --relative-to="$dst" "$source/model_v48_trac_sr")" "$dst/model_v48_trac_sr"
    ln -s "$(realpath --relative-to="$dst" "$source/POLICY_CONTRACT.env")" "$dst/POLICY_CONTRACT.env"
  else
    run_adapt "$out" "$variant" "$gpu" "$source" "$mode"
  fi
  set +e
  OUTPUTDIR="$out" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" GPU0="$gpu" GPU1="$gpu" VARIANTS="$variant" \
    bash scripts/calibrate_v48_14_certificate_pool.sh >"$out/logs/calibrate.log" 2>&1
  local rc=$?; set -e
  if [[ "$rc" != 0 && "$rc" != 20 ]]; then printf '{"complete":false,"controller_exit":%s}\n' "$rc" > "$out/TASK_FAILED.json"; return "$rc"; fi
  python - "$out" "$group" "$variant" "$rc" <<'PY'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); group=sys.argv[2]; variant=sys.argv[3]; rc=int(sys.argv[4]); base=out/'candidates'/variant
ck=base/'model_v48_trac_sr'/'best.pt'; cal=base/'calibration'
required=[ck,cal/'CERTIFICATE_CALIBRATION_COMPLETE.json',cal/'gamma_rec_by_bucket_v48.json',cal/'direct_value_risk_near_v48.json',cal/'direct_value_risk_contact_v48.json']
missing=[str(p) for p in required if not p.is_file()]
if missing: raise SystemExit('incomplete task: '+','.join(missing))
for b in ('near','contact'):
 d=json.load(open(cal/f'direct_value_risk_{b}_v48.json'))
 if int(d.get('num_groups',0) or 0)<=0 or int(d.get('num_scenes',0) or 0)<=0: raise SystemExit(f'empty {b} certificate')
doc={'complete':True,'group':group,'variant':variant,'certificate_exit':rc,'gate_passed':rc==0,'created_unix':time.time(),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}
(out/'TASK_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
}
SPECS=("A_source source 0" "B_old_tiny old_tiny 1" "C_balanced_margin balanced_margin 0" "D_full_anchor full_anchor 1")
failures=0
for variant in balanced precision; do
  pids=(); labels=()
  for spec in "${SPECS[@]}"; do
    read -r group mode slot <<<"$spec"; gpu="$GPU0"; [[ "$slot" == 1 ]] && gpu="$GPU1"
    run_task "$group" "$variant" "$gpu" "$mode" & pids+=("$!"); labels+=("${group}_${variant}")
  done
  set +e
  for i in "${!pids[@]}"; do wait "${pids[$i]}"; rc=$?; [[ "$rc" == 0 ]] || { echo "[failed] ${labels[$i]} rc=$rc" >&2; failures=$((failures+1)); }; done
  set -e
done
python tools/summarize_v48_14_ablations.py --root "$ROOT" --output "$ROOT/ablation_summary_v48_16.json" || true
python - "$ROOT" "$failures" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); failures=int(sys.argv[2]); groups=['A_source','B_old_tiny','C_balanced_margin','D_full_anchor']
expected=[f'{g}_{v}' for v in ('balanced','precision') for g in groups]; missing=[x for x in expected if not (root/'tasks'/x/'TASK_COMPLETE.json').is_file()]
doc={'complete':not missing and failures==0,'version':'v48.16-ANCHOR','max_concurrent_tasks':4,'tasks_per_gpu':2,'expected_tasks':expected,'missing_tasks':missing,'failed_waits':failures,'created_unix':time.time()}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,indent=2)+'\n')
if doc['complete']:(root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
else: raise SystemExit('ablation suite incomplete: '+','.join(missing))
PY
