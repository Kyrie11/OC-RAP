#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
BASE_OUT="${BASE_OUT:-runs}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OMP_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"; export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-1}"; export PREFETCH_FACTOR="${ABLATION_PREFETCH_FACTOR:-2}"
mkdir -p "$BASE_OUT"

# One shared deterministic calibration protocol is prepared/sealed before any arm
# starts. This prevents four identical RC=30 failures and guarantees every arm
# sees byte-identical role manifests. The bootstrap reads calibration roots only.
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}"
export CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
if [[ "${V4845_TEST_BYPASS_PROTOCOL_PREFLIGHT:-0}" == 1 ]]; then
  # Unit-test harness only: production/operator scripts never set this flag.
  echo "V48.45 protocol preflight bypassed by explicit test harness"
else
  if [[ "${V4845_SKIP_SHARED_PROTOCOL_PREPARE:-0}" != 1 ]]; then
    bash scripts/prepare_v48_45_protocol.sh
  fi
  [[ -s "$PROTOCOL_ROOT/V48_45_PROTOCOL_SEAL.json" ]] || {
    echo "v48.45 shared protocol seal missing after bootstrap: $PROTOCOL_ROOT/V48_45_PROTOCOL_SEAL.json" >&2
    exit 4
  }
  python - "$PROTOCOL_ROOT/V48_45_PROTOCOL_SEAL.json" <<'PY_PROTOCOL_SEAL'
import json,sys
d=json.load(open(sys.argv[1]))
assert d.get('valid') is True, d
assert d.get('test_roots_read') is False, d
print('V48.45 SHARED PROTOCOL SEAL PASS')
PY_PROTOCOL_SEAL
fi
export V4845_SKIP_PROTOCOL_PREPARE=1

MAX_PARALLEL_ARMS="${MAX_PARALLEL_ARMS:-1}"
if ! [[ "$MAX_PARALLEL_ARMS" =~ ^[1-4]$ ]]; then echo "MAX_PARALLEL_ARMS must be 1..4 (1 recommended: each arm already uses GPU0/GPU1 in parallel)" >&2; exit 2; fi
arms=(A B C D)
arm_output() {
  case "$1" in
    D) printf '%s/ocrap_v48_45_sowr_main' "$BASE_OUT" ;;
    A|B|C) printf '%s/ocrap_v48_45_sowr_ablation_%s' "$BASE_OUT" "$1" ;;
    *) return 2 ;;
  esac
}
run_arm() {
  local arm="$1" out log
  out="$(arm_output "$arm")"
  mkdir -p "$out/logs"
  log="$out/logs/v48_45_launcher.log"
  BASE_OUT="$BASE_OUT" GPU0="$GPU0" GPU1="$GPU1" \
    bash scripts/run_v48_45_sowr_ablation_arm.sh "$arm" >"$log" 2>&1
}
engineering_failed=0
gate_failed=0
pids=(); names=()
wait_arm() {
  local pid="$1" name="$2" arc out
  set +e
  wait "$pid"
  arc=$?
  set -e
  out="$(arm_output "$name")"
  mkdir -p "$out/logs"
  printf '%s\n' "$arc" >"$out/logs/v48_45_launcher.rc"
  case "$arc" in
    0)
      echo "arm $name completed: RC=0 gate passed"
      ;;
    20)
      # RC=20 is an authoritative, pipeline-valid Natural-gate miss.  It is a
      # valid ablation result and must not be reported as an engineering crash.
      echo "arm $name completed: RC=20 valid Natural-gate failure"
      gate_failed=1
      ;;
    *)
      echo "arm $name ENGINEERING FAILED: RC=$arc; inspect $out/AUTHORITATIVE_RUN_STATUS.json and $out/logs/v48_45_launcher.log" >&2
      engineering_failed=1
      ;;
  esac
}
for arm in "${arms[@]}"; do
  while (( ${#pids[@]} >= MAX_PARALLEL_ARMS )); do
    pid="${pids[0]}"; name="${names[0]}"
    wait_arm "$pid" "$name"
    pids=("${pids[@]:1}"); names=("${names[@]:1}")
  done
  run_arm "$arm" & pids+=("$!"); names+=("$arm")
done
for i in "${!pids[@]}"; do
  wait_arm "${pids[$i]}" "${names[$i]}"
done
python - "$BASE_OUT" "$engineering_failed" "$gate_failed" <<'PY_STATUS'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1])
rows={}
for arm,name in [('A','ocrap_v48_45_sowr_ablation_A'),('B','ocrap_v48_45_sowr_ablation_B'),('C','ocrap_v48_45_sowr_ablation_C'),('D','ocrap_v48_45_sowr_main')]:
    p=root/name/'logs'/'v48_45_launcher.rc'
    try: rc=int(p.read_text().strip())
    except Exception: rc=None
    rows[arm]={'run':str(root/name),'launcher_exit_code':rc,
               'classification':('gate_passed' if rc==0 else 'valid_natural_gate_failure' if rc==20 else 'engineering_failure')}
doc={'event':'v48_45_sowr_parallel_complete','created_unix':time.time(),
     'engineering_failed':sys.argv[2]=='1','any_natural_gate_failure':sys.argv[3]=='1',
     'arms':rows,'test_roots_read':False}
(root/'ocrap_v48_45_sowr_parallel_status.json').write_text(json.dumps(doc,indent=2)+'\n')
PY_STATUS
# The launcher succeeds when every arm is an authoritative RC=0/20 result.
# Only engineering/protocol failures make the launcher non-zero.
if [[ "$engineering_failed" == 1 ]]; then exit 1; fi
exit 0
