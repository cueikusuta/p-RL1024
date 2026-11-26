#!/usr/bin/env bash
set -euo pipefail
umask 077
probe='league-bulk-dummy-v1.bin'
want='f347eb1e19182ef702c48022974147f29d4243fdebb00161707ad1938d6e11ec'
pin="${1:?Pinned synthetic-only upstream commit required}"
[[ "$pin" =~ ^[0-9a-f]{40}$ ]]
[[ -n "${LEAGUE_BULK_READ_SSH_KEY:-}" ]]
scratch="$(mktemp -d "$RUNNER_TEMP/league-bulk.XXXXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT
mkdir -m700 "$scratch/git" "$RUNNER_TEMP/league-synthetic-return"
printf '%s\n' "$LEAGUE_BULK_READ_SSH_KEY" > "$scratch/read_only"
unset LEAGUE_BULK_READ_SSH_KEY
printf '%s\n' 'github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl' > "$scratch/known_hosts"
export GIT_SSH_COMMAND="ssh -F /dev/null -i '$scratch/read_only' -o IdentityAgent=none -o IdentitiesOnly=yes -o UserKnownHostsFile='$scratch/known_hosts' -o GlobalKnownHostsFile=/dev/null -o StrictHostKeyChecking=yes -o HostKeyAlgorithms=ssh-ed25519 -o BatchMode=yes -o ConnectTimeout=12"
export GIT_TERMINAL_PROMPT=0
out="$RUNNER_TEMP/league-synthetic-return/$probe"
start="$(date -u +%s%3N)"
git -C "$scratch/git" init -q
git -C "$scratch/git" -c core.hooksPath=/dev/null fetch --quiet --no-tags --depth=1 git@github.com:cueikusuta/p-rl1024-handoff-private.git "$pin"
arrive="$(date -u +%s%3N)"
rm -f -- "$scratch/read_only"
unset GIT_SSH_COMMAND
[[ "$(git -C "$scratch/git" rev-parse FETCH_HEAD)" == "$pin" ]]
parts=()
for k in {0..9}; do parts+=("part-$k.bin"); done
[[ "$(git -C "$scratch/git" ls-tree --name-only "$pin:dummy-parts")" == "$(printf '%s\n' "${parts[@]}")" ]]
{
  for p in "${parts[@]}"; do
    [[ "$(git -C "$scratch/git" cat-file -s "$pin:dummy-parts/$p")" == 2097152 ]]
    git -C "$scratch/git" cat-file blob "$pin:dummy-parts/$p"
  done
} > "$out"
[[ "$(stat -c '%s' "$out")" == 20971520 ]]
[[ "$(sha256sum "$out" | cut -d' ' -f1)" == "$want" ]]
end="$(date -u +%s%3N)"
printf '{"fabricated_only":true,"verified_bytes":20971520,"sha256":"%s","fetch_start_ms":%s,"fetch_received_ms":%s,"verified_ms":%s}\n' "$want" "$start" "$arrive" "$end" | tee "$RUNNER_TEMP/league-synthetic-return/clock.json"
