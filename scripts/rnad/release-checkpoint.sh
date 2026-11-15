#!/usr/bin/env bash
# RNAD-001 immutable GitHub Release checkpoint transport.
# Intended ONLY for GitHub-hosted runner job on main after Stage0 gates.
set -euo pipefail
mode="${1:?mode: download|upload}"
tag="${2:?immutable research release tag required}"
folder="${3:?checkpoint folder required}"
[[ "$tag" =~ ^rnad-001-[0-9a-f]{16}-s[0-9]{1,10}$ ]] ||
  { echo "Invalid RNAD research tag" >&2; exit 2; }
command -v gh >/dev/null
command -v jq >/dev/null
: "${GH_REPO:?Set fixed GitHub repository}"
: "${GH_TOKEN:?Set Actions scoped token}"
[[ "$GH_REPO" == 'cueikusuta/p-RL1024' ]] ||
  { echo "Wrong checkpoint repository" >&2; exit 2; }
[[ -d "$folder" ]] || mkdir -p "$folder"
case "$mode" in
download)
  if ! gh release view "$tag" --repo "$GH_REPO" >/dev/null 2>&1; then
    # No prior checkpoint is a valid first batch.
    echo "RNAD_RELEASE_EMPTY=1" >> "${GITHUB_ENV:-/dev/null}"
    exit 0
  fi
  asset="$(gh api "repos/$GH_REPO/releases/tags/$tag" --jq '.id' |
    xargs -I{} gh api --paginate "repos/$GH_REPO/releases/{}/assets" \
      --jq '.[].name' |
    grep -E '^rnad-001-u[0-9]{9}-[0-9a-f]{12}\.tar\.gz$' |
    LC_ALL=C sort | tail -1 || true)"
  if [[ -z "$asset" ]]; then
    echo "RNAD_RELEASE_EMPTY=1" >> "${GITHUB_ENV:-/dev/null}"
    exit 0
  fi
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  gh release download "$tag" --repo "$GH_REPO" --pattern "$asset" --dir "$tmp"
  members="$(tar -tzf "$tmp/$asset")"
  [[ "$(printf '%s\n' "$members" | wc -l)" -eq 2 ]] ||
    { echo "Unexpected checkpoint archive members" >&2; exit 2; }
  if printf '%s\n' "$members" | grep -Ev '^latest\.json$|^checkpoint-u[0-9]+-[0-9a-f]{64}\.pt$' >/dev/null; then
    echo "Unsafe checkpoint archive member path" >&2
    exit 2
  fi
  if [[ "$(printf '%s\n' "$members" | grep -Ec '^latest\.json$')" -ne 1 ||
        "$(printf '%s\n' "$members" | grep -Ec '^checkpoint-u[0-9]+-[0-9a-f]{64}\.pt$')" -ne 1 ]]; then
    echo "Missing or duplicated archive members" >&2
    exit 2
  fi
  mkdir -p "$tmp/staged"
  tar -xzf "$tmp/$asset" -C "$tmp/staged" --no-same-owner --no-same-permissions
  # Never trust archive paths, index JSON or a release tag without
  # independent on-disk SHA and provenance validation before publishing.
  checkpoint_name="$(python - "$tmp/staged" "$tag" "$asset" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
stage,tag,asset=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
files=list(stage.iterdir())
if len(files)!=2 or any(not f.is_file() or f.is_symlink() for f in files):
    raise SystemExit("Invalid checkpoint archive files")
index=json.loads((stage/"latest.json").read_text())
metadata=index.get("metadata",{})
name=index.get("checkpoint_file","")
sha=index.get("checkpoint_sha256","")
updates=metadata.get("updates")
seed=metadata.get("initial_seed")
source=metadata.get("source_sha","")
if (index.get("schema")!="rnad-pytorch-checkpoint-v1"
    or not isinstance(name,str)
    or re.fullmatch(r"checkpoint-u[0-9]+-[a-f0-9]{64}\.pt",name) is None
    or re.fullmatch(r"[a-f0-9]{64}",sha) is None
    or type(updates) is not int or not 1<=updates<=2048
    or type(seed) is not int or not 1<=seed<=0xffffffff
    or re.fullmatch(r"[a-f0-9]{40}",source) is None
    or metadata.get("model_id")!="RNAD-001"
    or metadata.get("observation_schema_version")!="rnad-battle-features-v1"
    or metadata.get("completed_matches")!=updates
    or tag!=f"rnad-001-{source[:16]}-s{seed}"
    or asset!=f"rnad-001-u{updates:09d}-{sha[:12]}.tar.gz"
    or name!=f"checkpoint-u{updates}-{sha}.pt"):
    raise SystemExit("Checkpoint tag/index/provenance mismatch")
checkpoint=stage/name
if (not checkpoint.is_file() or checkpoint.is_symlink()
    or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=sha):
    raise SystemExit("Checkpoint SHA mismatch")
print(name)
PY
  )"
  cp -- "$tmp/staged/$checkpoint_name" "$folder/$checkpoint_name"
  cp -- "$tmp/staged/latest.json" "$folder/latest.json"
  echo "RNAD_RELEASE_EMPTY=0" >> "${GITHUB_ENV:-/dev/null}"
  echo "Downloaded immutable verified checkpoint $asset"
  ;;
upload)
  [[ -f "$folder/latest.json" ]] ||
    { echo "No validated completed-Match checkpoint" >&2; exit 2; }
  info="$(python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))' "$folder/latest.json")"
  name="$(jq -r '.checkpoint_file' <<< "$info")"
  sha="$(jq -r '.checkpoint_sha256' <<< "$info")"
  updates="$(jq -r '.metadata.updates' <<< "$info")"
  [[ "$name" =~ ^checkpoint-u[0-9]+-[0-9a-f]{64}\.pt$ &&
     "$sha" =~ ^[0-9a-f]{64}$ && "$updates" =~ ^[0-9]+$ &&
     -f "$folder/$name" ]] || { echo "Invalid checkpoint manifest" >&2; exit 2; }
  test "$(sha256sum "$folder/$name" | cut -d' ' -f1)" = "$sha"
  [[ "$updates" -le 2048 ]] || { echo "Unbounded RNAD scale attempt" >&2; exit 2; }
  printf -v padded '%09d' "$updates"
  asset="rnad-001-u${padded}-${sha:0:12}.tar.gz"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  tar -czf "$tmp/$asset" -C "$folder" latest.json "$name"
  if ! gh release view "$tag" --repo "$GH_REPO" >/dev/null 2>&1; then
    : "${RNAD_SOURCE_SHA:?Pinned source SHA required}"
    gh release create "$tag" --repo "$GH_REPO" --target "$RNAD_SOURCE_SHA" \
      --prerelease --title "RNAD-001 research checkpoints $tag" \
      --notes "Immutable unpromoted research checkpoints. NOT a Champion."
  fi
  # Never --clobber immutable assets. Concurrency is serialized at workflow level.
  gh release upload "$tag" "$tmp/$asset" --repo "$GH_REPO"
  echo "Published immutable completed-Match bundle $asset"
  ;;
*)
  echo "Unknown checkpoint operation: $mode" >&2
  exit 2
  ;;
esac
