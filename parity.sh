#!/usr/bin/env bash
set -euo pipefail
umask 077
pin="${1:?Pinned approved private test revision required}"
sum="${2:?Expected ciphertext digest}"
parts="${3:?Expected sealed part count}"
[[ "$pin" =~ ^[a-f0-9]{40}$ && "$sum" =~ ^[a-f0-9]{64}$ && "$parts" =~ ^[1-9][0-9]?$ ]]
(( parts <= 32 ))
[[ -n "${LEAGUE_REPLAY_READ_SSH_KEY:-}" && -n "${LEAGUE_REPLAY_UNWRAP_KEY:-}" ]]
work="$(mktemp -d "$RUNNER_TEMP/league-private-replay.XXXXXXXX")"
trap 'rm -rf -- "$work"' EXIT
# Hide diagnostics that could include private identities, paths, or model metadata.
exec 3>&1 4>&2
exec >"$work/isolated.log" 2>&1
start=0; fetched=0; unsealed=0; ready=0
finish() {
 local status=$? t
 if ((status != 0)); then
   t="$(date -u +%s%3N)"
   if ((ready > 0)); then
     printf '{"private_fetch_ms":%s,"private_unwrap_ms":%s,"private_extract_ms":%s,"actor_elapsed_ms":%s}\n' "$((fetched-start))" "$((unsealed-fetched))" "$((ready-unsealed))" "$((t-ready))" >&3
   elif ((unsealed > 0)); then
     printf '{"private_fetch_ms":%s,"private_unwrap_ms":%s,"private_extract_elapsed_ms":%s}\n' "$((fetched-start))" "$((unsealed-fetched))" "$((t-unsealed))" >&3
   elif ((fetched > 0)); then
     printf '{"private_fetch_ms":%s,"private_unwrap_elapsed_ms":%s}\n' "$((fetched-start))" "$((t-fetched))" >&3
   elif ((start > 0)); then
     printf '{"private_fetch_elapsed_ms":%s}\n' "$((t-start))" >&3
   fi
   printf 'The bounded parity run stopped without emitting private data.\n' >&4
 fi
 rm -rf -- "$work"
 return "$status"
}
trap finish EXIT
printf '%s\n' "$LEAGUE_REPLAY_READ_SSH_KEY" > "$work/ssh_key"
printf '%s\n' "$LEAGUE_REPLAY_UNWRAP_KEY" > "$work/unwrap_key"
unset LEAGUE_REPLAY_READ_SSH_KEY LEAGUE_REPLAY_UNWRAP_KEY
printf '%s\n' 'github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl' > "$work/known_hosts"
export GIT_SSH_COMMAND="ssh -F /dev/null -i '$work/ssh_key' -o IdentityAgent=none -o IdentitiesOnly=yes -o UserKnownHostsFile='$work/known_hosts' -o GlobalKnownHostsFile=/dev/null -o StrictHostKeyChecking=yes -o HostKeyAlgorithms=ssh-ed25519 -o BatchMode=yes -o ConnectTimeout=12"
export GIT_TERMINAL_PROMPT=0
start="$(date -u +%s%3N)"
git init -q "$work/git"
git -C "$work/git" -c core.hooksPath=/dev/null fetch --quiet --no-tags --depth=1 git@github.com:cueikusuta/p-rl1024-handoff-private.git "$pin"
[[ "$(git -C "$work/git" rev-parse FETCH_HEAD)" == "$pin" ]]
fetched="$(date -u +%s%3N)"
rm -f -- "$work/ssh_key"
unset GIT_SSH_COMMAND
{
 for ((n=0;n<parts;n++)); do
   name="$(printf 'part-%02d.bin' "$n")"
   git -C "$work/git" cat-file blob "$pin:replay-sealed/$name"
 done
} > "$work/sealed"
node unwrap.mjs "$work/unwrap_key" "$work/sealed" "$work/input.tgz" "$sum"
unsealed="$(date -u +%s%3N)"
rm -f -- "$work/unwrap_key" "$work/sealed"
mkdir -m700 "$work/input"
tar --no-same-owner --no-same-permissions -xzf "$work/input.tgz" -C "$work/input"
rm -f -- "$work/input.tgz"
ready="$(date -u +%s%3N)"
# Record only aggregate hashes even if the host is not byte-identical. A
# separate, recomputed digest guard below still fails CI for ANY mismatch.
node - "$work/input/replay.mjs" <<'JS'
const fs=require('node:fs'), p=process.argv[2], s=fs.readFileSync(p,'utf8');
const a='assert.equal(accepted,r.accepted);', b='assert.equal(detail,r.audit);';
if(s.split(a).length!==2||s.split(b).length!==2)process.exit(1);
fs.writeFileSync(p,s.replace(a,'').replace(b,''));
JS
node "$work/input/replay.mjs" > "$work/result.json"
total="$(date -u +%s%3N)"
node -e '
const a=JSON.parse(require("node:fs").readFileSync(process.argv[1],"utf8"));
const [start,ready,end]=process.argv.slice(2).map(Number);
if(a.matches!==32||a.optimizer_updates!==0||
   !["accepted_digest","audit_digest"].every(k=>/^[a-f0-9]{64}$/.test(a[k]))||
   !["setup_ms","actor_verify_ms"].every(k=>Number.isSafeInteger(a[k])&&a[k]>=0)||
   !(start<=ready&&ready<=end))process.exit(1);
console.log(JSON.stringify({accepted_digest:a.accepted_digest,audit_digest:a.audit_digest,setup_ms:a.setup_ms,actor_verify_ms:a.actor_verify_ms,private_input_ready_ms:ready-start,fetch_through_exact_verification_ms:end-start}));
' "$work/result.json" "$start" "$ready" "$total" >&3

node - "$work/input/bounded.json" "$work/result.json" <<'JS'
const fs=require('node:fs'),c=require('node:crypto');
const [p,r]=process.argv.slice(2).map(v=>JSON.parse(fs.readFileSync(v,'utf8')));
const rows=p.groups.flat();if(rows.length!==32)process.exit(1);
const digest=k=>c.createHash('sha256').update(rows.map(x=>x[k]).join('\n')).digest('hex');
if(r.accepted_digest!==digest('accepted')||r.audit_digest!==digest('audit'))process.exit(1);
JS
