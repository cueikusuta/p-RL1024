#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
safe="$(mktemp -d "${RUNNER_TEMP:?}/bench21.XXXXXXXX")"
phase=setup
trap 'code=$?; if (( code != 0 )); then printf "Hosted league failed during %s. No private data emitted.\n" "$phase" >&4; fi; rm -rf -- "$safe"' EXIT
token="${HANDOFF_TOKEN:?}"; unset HANDOFF_TOKEN
exec 3>&1 4>&2
trap 'printf "Hosted stage %s failed at script line %s.\n" "$phase" "$LINENO" >&4' ERR
exec >"$safe/private.log" 2>&1
here="$PWD/ci"
cat > "$safe/askpass" <<'EOF'
#!/bin/sh
case "$1" in
  *Username*) echo x-access-token;;
  *Password*) printf '%s\n' "${HANDOFF_HTTP_TOKEN:?}";;
  *) exit 1;;
esac
EOF
chmod 700 "$safe/askpass"
remote=https://github.com/cueikusuta/p-rl1024-handoff-private.git
gs() {
  GIT_ASKPASS="$safe/askpass" GIT_TERMINAL_PROMPT=0 HANDOFF_HTTP_TOKEN="$token" \
   git -c credential.helper= -c http.followRedirects=false "$@"
}
phase=private-input
source "$here/prod-boot.sh" "$1"
actor="${BENCH20_ACTOR:?}"; [[ "$actor" =~ ^([0-9]|1[0-9])$ ]]
suffix="$(printf 'a%02d' "$actor")"
phase=actor
git clone --quiet --no-local --depth=1 "file://$safe/work/engine" "$safe/work/vm-portable-engine"
git -C "$safe/work/vm-portable-engine" apply --check "$safe/work/actor-native-overlay.patch"
git -C "$safe/work/vm-portable-engine" apply "$safe/work/actor-native-overlay.patch"
cd "$safe/work/engine"
time_left=$((deadline-$(date +%s)))
test "$time_left" -gt 10
timeout --kill-after=2s "${time_left}s" python3 "$here/prod_actor.py" "$safe/run/plan.json" "$safe/actor-out" 0 "$actor"
phase=actor-save
git init -q "$safe/send"
cd "$safe/send";mkdir return
(cd "$safe/actor-out/packets"; tar -cf "$safe/wave.tar" * -C .. work-result.json)
split -b 640K -d -a 3 "$safe/wave.tar" return/part-
python3 - "$safe/wave.tar" "$safe/run/plan.json" "$actor" > return/ready.json <<'RECEIPT'
import hashlib,json,sys,pathlib
p,plan=map(pathlib.Path,sys.argv[1:3]);a=json.loads(plan.read_bytes());i=int(sys.argv[3])
with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
size=p.stat().st_size;assert 0<size<=100*1024**2
print(json.dumps(dict(schema='bench20-private-actor-v1',run=a['runId'],generation=a['generation'],actor=i,manifest_sha256=a['manifestSHA'],size=size,sha256=h,num_parts=(size+655359)//655360)))
RECEIPT
git add -- return
git -c user.name=league-runner -c user.email=league-runner@users.noreply.github.com commit -qm 'Store verified-scope Actor packets privately'
test "$(date +%s)" -lt "$deadline"
gs push -q "$remote" "HEAD:refs/heads/$branch/$suffix"
printf 'One Actor completed and saved privately.\n' >&3
