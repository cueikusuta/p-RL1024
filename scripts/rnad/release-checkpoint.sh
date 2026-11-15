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
  tar -tzf "$tmp/$asset" | grep -Eq '^\./latest\.json$|^latest\.json$'
  tar -xzf "$tmp/$asset" -C "$folder" --no-same-owner
  # Python checkpoint.restore_checkpoint independently verifies SHA256.
  echo "RNAD_RELEASE_EMPTY=0" >> "${GITHUB_ENV:-/dev/null}"
  echo "Downloaded immutable accepted checkpoint $asset"
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
