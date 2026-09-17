#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=scripts/lib/runtime.sh
source scripts/lib/runtime.sh

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
OUT="${OCRAP_FINAL_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
WOMD_NUM_SHARDS="${WOMD_NUM_SHARDS:-150}"
BUCKET_SPLIT="${BUCKET_SPLIT:-test}"
WOMD_ROLE="${WOMD_ROLE:-validation}"
SAFE_BUCKET="${SAFE_BUCKET:-$OCRAP_ROOT/test_safe}"
NEAR_BUCKET="${NEAR_BUCKET:-$OCRAP_ROOT/test_near_contact}"
CONTACT_BUCKET="${CONTACT_BUCKET:-$OCRAP_ROOT/test_contact}"
FINAL_MAX_STEPS="${FINAL_MAX_STEPS:-${MAX_STEPS:-40}}"
CONTACT_ANCHOR_GPU="${CONTACT_ANCHOR_GPU:-0}"
CONTACT_ANCHOR_PRELUDE_MAX_STEPS="${CONTACT_ANCHOR_PRELUDE_MAX_STEPS:-60}"
CONTACT_ANCHOR_POOL_MAX_TARGETS_PER_SCENE="${CONTACT_ANCHOR_POOL_MAX_TARGETS_PER_SCENE:-100000}"
FORCE_REBUILD_CONTACT_ANCHOR="${FORCE_REBUILD_CONTACT_ANCHOR:-false}"
mkdir -p "$OUT/target_keys"

SAFE_WOMD="$(runtime_resolve_bucket_womd_spec "$SAFE_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "$WOMD_ROLE")"
NEAR_WOMD="$(runtime_resolve_bucket_womd_spec "$NEAR_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "$WOMD_ROLE")"
CONTACT_WOMD="$(runtime_resolve_bucket_womd_spec "$CONTACT_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "$WOMD_ROLE")"

build_one() {
  local regime="$1" bucket="$2" womd="$3" max_per_scene="${4:-1}" output="${5:-$OUT/target_keys/$1.json}"
  python tools/build_observation_legal_target_lock.py \
    --bucket-dataset "$bucket" --bucket-split "$BUCKET_SPLIT" \
    --max-targets-per-scene "$max_per_scene" --womd-pattern "$womd" \
    --expected-womd-role "$WOMD_ROLE" \
    --output "$output"
}

# Safe/Near use exactly one observation-legal target per scene.
build_one safe "$SAFE_BUCKET" "$SAFE_WOMD" 1
build_one near "$NEAR_BUCKET" "$NEAR_WOMD" 1

# Contact post-impact controllers require a real pre-treatment impact boundary.
# First form the complete observation-legal Contact pool, then select one
# treatment-independent exact-a0 observed-contact anchor per scene.  This is the
# V48.124.8 scientific construct-validity contract and uses no evaluated policy.
CONTACT_POOL="$OUT/target_keys/contact_observation_legal_pool.json"
CONTACT_ANCHOR_OUT="$OUT/contact_anchor"
CONTACT_ANCHOR_MANIFEST="$CONTACT_ANCHOR_OUT/contact_anchor_manifest.json"
build_one contact "$CONTACT_BUCKET" "$CONTACT_WOMD" "$CONTACT_ANCHOR_POOL_MAX_TARGETS_PER_SCENE" "$CONTACT_POOL"

needs_anchor_rebuild=true
if ! runtime_bool_true "$FORCE_REBUILD_CONTACT_ANCHOR" && [[ -s "$CONTACT_ANCHOR_MANIFEST" ]]; then
  if python - "$CONTACT_POOL" "$CONTACT_ANCHOR_MANIFEST" "$FINAL_MAX_STEPS" <<'PY'
import hashlib,json,pathlib,sys
pool=pathlib.Path(sys.argv[1]); manifest=pathlib.Path(sys.argv[2]); min_post=int(sys.argv[3])
try:
    m=json.loads(manifest.read_text(encoding='utf-8'))
except Exception:
    raise SystemExit(1)
want=hashlib.sha256(pool.read_bytes()).hexdigest()
ok=(m.get('schema')=='ocrap-contact-anchor-manifest-v1' and m.get('valid') is True
    and m.get('source_target_keys_sha256')==want
    and int(m.get('min_post_steps') or -1)==min_post
    and int(m.get('num_selected_anchors') or 0)>0)
raise SystemExit(0 if ok else 1)
PY
  then
    needs_anchor_rebuild=false
    echo "[REUSE] Contact anchor manifest matches current observation-legal pool and ${FINAL_MAX_STEPS}-step treatment horizon: $CONTACT_ANCHOR_MANIFEST"
  fi
fi

if [[ "$needs_anchor_rebuild" == true ]]; then
  env OUT="$CONTACT_ANCHOR_OUT" GPU="$CONTACT_ANCHOR_GPU" \
    OCRAP_ROOT="$OCRAP_ROOT" CONTACT_BUCKET="$CONTACT_BUCKET" BUCKET_SPLIT="$BUCKET_SPLIT" \
    WOMD_ROOT="$WOMD_ROOT" WOMD_NUM_SHARDS="$WOMD_NUM_SHARDS" CONTACT_WOMD="$CONTACT_WOMD" WOMD_ROLE="$WOMD_ROLE" \
    SOURCE_TARGET_KEYS_FILE="$CONTACT_POOL" MAX_TARGETS_PER_SCENE="$CONTACT_ANCHOR_POOL_MAX_TARGETS_PER_SCENE" \
    PRELUDE_MAX_STEPS="$CONTACT_ANCHOR_PRELUDE_MAX_STEPS" MIN_POST_STEPS="$FINAL_MAX_STEPS" MAX_SCENARIOS=0 \
    bash scripts/build_contact_anchor_cohort.sh
fi

python - "$CONTACT_POOL" "$CONTACT_ANCHOR_MANIFEST" "$OUT/target_keys/contact.json" <<'PY'
import hashlib,json,pathlib,sys
pool_p,manifest_p,out_p=map(pathlib.Path,sys.argv[1:])
pool=json.loads(pool_p.read_text(encoding='utf-8'))
manifest=json.loads(manifest_p.read_text(encoding='utf-8'))
if manifest.get('schema')!='ocrap-contact-anchor-manifest-v1' or manifest.get('valid') is not True:
    raise SystemExit('invalid Contact anchor manifest')
selected=list(manifest.get('target_keys') or [])
if not selected:
    raise SystemExit('Contact anchor manifest selected no targets')
pool_keys=set(pool.get('target_keys') or [])
missing=sorted(set(selected)-pool_keys)
if missing:
    raise SystemExit(f'Contact anchor keys are not a subset of observation-legal pool: {missing[:10]}')
doc=dict(pool)
doc.update({
    'schema':'ocrap-observation-legal-contact-anchor-target-lock-v1',
    'status':'OBSERVATION_LEGAL_CONTACT_ANCHOR_TARGET_LOCK_COMPLETE',
    'pre_anchor_observation_legal_target_count':len(pool_keys),
    'num_target_keys':len(selected),
    'target_keys':selected,
    'contact_anchor_contract':{
        'protocol':'exact_a0_pretreatment_prelude_v1',
        'manifest_file':str(manifest_p.resolve()),
        'manifest_sha256':hashlib.sha256(manifest_p.read_bytes()).hexdigest(),
        'source_pool_file':str(pool_p.resolve()),
        'source_pool_sha256':hashlib.sha256(pool_p.read_bytes()).hexdigest(),
        'min_post_steps':int(manifest.get('min_post_steps') or 0),
        'scene_disjoint':int(manifest.get('num_selected_anchors') or 0)==int(manifest.get('num_selected_scenes') or -1),
        'selection_policy':manifest.get('selection_policy'),
        'pre_treatment_policy':manifest.get('pre_treatment_policy'),
        'state_fingerprint_required':True,
    },
})
out_p.parent.mkdir(parents=True,exist_ok=True)
out_p.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps({'event':'contact_anchor_target_lock','output':str(out_p),'pool_targets':len(pool_keys),'anchored_targets':len(selected),'manifest':str(manifest_p)}))
PY

for regime in safe near contact; do
  python tools/check_target_key_lock.py --expected "$OUT/target_keys/$regime.json" --observed "$OUT/target_keys/$regime.json" >/dev/null
done

echo "Final fair target locks are ready under $OUT/target_keys"
echo "Contact anchor manifest: $CONTACT_ANCHOR_MANIFEST"
