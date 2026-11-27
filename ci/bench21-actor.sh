#!/usr/bin/env bash
set -euo pipefail
umask 077
tmp="$(mktemp -d "${RUNNER_TEMP:?}/leaguebench.XXXXXXXX")"
trap 'rm -rf -- "$tmp"' EXIT
handoff_token="${BENCH20_INSTALLATION_TOKEN:?}"; unset BENCH20_INSTALLATION_TOKEN
cat > "$tmp/askpass" <<'EOF'
#!/usr/bin/env bash
set -eu
case "$1" in
  *Username*) printf '%s\n' x-access-token;;
  *Password*) printf '%s\n' "${BENCH20_GIT_HTTP_TOKEN:?}";;
  *) exit 1;;
esac
EOF
chmod 700 "$tmp/askpass"
remote='https://github.com/cueikusuta/p-rl1024-handoff-private.git'
handoff_git() {
  GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="$tmp/askpass" BENCH20_GIT_HTTP_TOKEN="$handoff_token" \
    git -c credential.helper= -c credential.useHttpPath=true -c http.followRedirects=false "$@"
}
actor="${BENCH20_ACTOR:?}"; [[ "$actor" =~ ^([0-9]|1[0-9])$ ]]
read -r session workid < <(python3 - "$1" <<'PY'
import json,re,sys,time
p=json.load(open(sys.argv[1]))
assert set(p)=={'session'} and re.fullmatch(r'[0-9]{10}',p['session'])
s=p['session']; print(s,f'g424-bench20-{s}')
PY
)
branch="bench21/$session"
pin=8703a0fa1c07c941f6c49a2153dbe3d4a336f0dd
sha=d6734bbb5c5afbc6d65c22b67d9a24aca6c8d8553551ceb71b5766d8eab69835
suffix="$(printf 'a%02d' "$actor")"
(
  echo 'private input' > "$tmp/phase"
  git init -q "$tmp/stage"
  handoff_git -C "$tmp/stage" fetch -q --depth=1 "$remote" "refs/heads/bench20/1795784311/input"
  test "$(git -C "$tmp/stage" rev-parse FETCH_HEAD)" = "$pin"
  git -C "$tmp/stage" checkout -q --detach FETCH_HEAD
  mkdir "$tmp/work"
  (cd "$tmp/stage" && cat input/payload.part-* > "$tmp/base.tar.gz")
  printf '%s  %s\n' "$sha" "$tmp/base.tar.gz" |sha256sum --quiet -c -
  tar -xzf "$tmp/base.tar.gz" -C "$tmp/work" --no-same-owner --no-same-permissions
  cd "$tmp/work/engine"
  test "$(git rev-parse HEAD)" = 52db2cf0367579ee7f9b097a404649cdad47e903
  git clone --quiet --no-local --depth=1 "file://$PWD" ../vm-portable-engine
  git -C ../vm-portable-engine apply --check ../actor-native-overlay.patch
  git -C ../vm-portable-engine apply ../actor-native-overlay.patch
  python3 - "$tmp/work/plan.json" "$workid" <<'PLAN'
import json,sys
from pathlib import Path
f=Path(sys.argv[1]); p=json.loads(f.read_text()); assert p["runId"]=="g424-bench20-1795784311"
p["runId"]=sys.argv[2]; f.write_text(json.dumps(p))
PLAN
  echo 'exact Actor calculation' > "$tmp/phase"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
  timeout --kill-after=2s 350s python3 ../bench20_actor_worker.py ../plan.json ../actor-out 0 "$actor"
  echo 'private upload' > "$tmp/phase"
  cd "$tmp/stage"
  mkdir return
  (cd "$tmp/work/actor-out/packets"; tar -cf "$tmp/wave.tar" * -C .. work-result.json)
  split -b 640K -d -a 3 "$tmp/wave.tar" return/part-
  python3 - "$tmp/wave.tar" "$tmp/work/plan.json" "$actor" "$workid" > return/ready.json <<'PY'
import hashlib,json,sys,pathlib
p,pin=map(pathlib.Path,sys.argv[1:3]);a=json.loads(pin.read_bytes());i=int(sys.argv[3]);assert a['runId']==sys.argv[4]
with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
size=p.stat().st_size; assert 0<size<=100*1024**2
print(json.dumps(dict(schema='bench20-private-actor-v1',run=a['runId'],generation=424,actor=i,manifest_sha256=a['manifestSHA'],size=size,sha256=h,num_parts=(size+655359)//655360)))
PY
  git add return/
  git -c user.name=league-benchmark -c user.email=league-benchmark@users.noreply.github.com commit -qm 'Return one verified-scope Actor bundle'
  handoff_git push -q "$remote" "HEAD:refs/heads/$branch/$suffix"
) > "$tmp/private.log" 2>&1 || { echo "Actor failed during $(cat "$tmp/phase" || echo setup); private log not printed." >&2; exit 1; }
echo 'Actor completed; result returned to private GitHub storage.'
