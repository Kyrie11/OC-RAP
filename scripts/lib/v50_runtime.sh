#!/usr/bin/env bash
# Shared shell helpers for v50 full-regime execution.

v50_normalize_womd_spec() {
  local raw="${1:-}" shards="${2:-150}"
  python - "$raw" "$shards" <<'PY'
import sys
from ocrap.data.womd.sharded_path import ensure_sharded_spec
print(ensure_sharded_spec(sys.argv[1], int(sys.argv[2])))
PY
}

v50_validate_womd_spec() {
  local spec="$1" output="${2:-}"
  local args=(python tools/validate_womd_spec.py --spec "$spec")
  [[ -n "$output" ]] && args+=(--output "$output")
  "${args[@]}"
}

# Resolve a replay collection from the OC-RAP bucket provenance.  The caller
# passes the WOMD tf_example root (the directory containing validation/ and
# validation_interactive/); the dataset's stored womd_source_role owns the
# choice.  This prevents launcher defaults from silently drifting across
# collections.
v50_resolve_bucket_womd_spec() {
  local dataset="$1" split="${2:-test}" womd_root="$3" shards="${4:-150}" role="${5:-auto}"
  python tools/resolve_womd_replay_source.py \
    --dataset "$dataset" --split "$split" --womd-root "$womd_root" --shards "$shards" --role "$role"
}

v50_bool_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

v50_iso_now() {
  date -u +'%Y-%m-%dT%H:%M:%SZ'
}
