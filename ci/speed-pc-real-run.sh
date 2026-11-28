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
relay_pid=''
sim_pid=''
clean() {
  for pid in "$relay_pid" "$sim_pid"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
}
trap 'code=$?; clean; if (( code != 0 )); then printf "Isolated league failed during %s. No private data emitted.\n" "$phase" >&4; fi; rm -rf -- "$safe"' EXIT
started_ms="$(date +%s%3N)"
await_both() {
  local finished code=0
  wait -n -p finished "$relay_pid" "$sim_pid" || code=$?
  if (( code == 0 )); then
    if [[ "$finished" == "$sim_pid" ]]; then wait "$relay_pid" || code=$?; else wait "$sim_pid" || code=$?; fi
  fi
  if (( code != 0 )); then grep -E '^Relay diagnostic: [A-Za-z]+ HTTP [0-9]+ line [0-9]+$' "$safe/private.log" >&4 || true; return "$code"; fi
  relay_pid=''; sim_pid=''
}
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
phase=input
read -r session g pin source started deadline initial used < <(python - "$1" <<'PY'
import json,re,sys,time
a=json.load(open(sys.argv[1]))
assert set(a)=={'session','generation','input','input_ref','start','deadline','initial','bytes'}
s,g,p,r,st,end,i,b=(a[k] for k in ('session','generation','input','input_ref','start','deadline','initial','bytes'))
assert re.fullmatch(r'[0-9]{10}',s) and re.fullmatch(r'[a-f0-9]{40}',p)
assert all(type(x)is int for x in (g,st,end,i,b)) and i<=g<i+100 and 445<=i<999 and 0<=b<1900*1024**2
assert end-st==36000 and g==511
expected=f'ghlong/{s}/g{g}/input'
assert r==expected or (g==i and b==0 and re.fullmatch(r'ghprod/[0-9]{10}/g'+str(g)+r'/input',r))
print(s,g,p,r,st,end,i,b)
PY
)
run_id="${GITHUB_RUN_ID:?}"
mkdir -p "$safe/work"
git init -q "$safe/boot"
# Start at the already verified historical G512 snapshot, rather than
# requiring one new host to produce byte-for-byte the old G512 learner result.
g=512
source=ghlong/1795815513/g512/input
pin="$(gs -C "$safe/boot" ls-remote --heads "$remote" "refs/heads/$source" | cut -f1)"
[[ "$pin" =~ ^[0-9a-f]{40}$ ]]
gs -C "$safe/boot" fetch -q --depth=1 "$remote" "refs/heads/$source"
test "$(git -C "$safe/boot" rev-parse FETCH_HEAD)" = "$pin"
git -C "$safe/boot" checkout -q --detach FETCH_HEAD
cat "$safe"/boot/input/boot.part-* > "$safe/boot.tar.gz"
python - "$safe" "$g" <<'PY'
import hashlib,json,sys
from pathlib import Path
d=Path(sys.argv[1]);i=json.loads((d/'boot/input/index.json').read_text())
assert i['generation']==int(sys.argv[2])==512 and i['parts']==len(list((d/'boot/input').glob('boot.part-*')))
r=json.loads((d/'boot/return/done.json').read_text())
k=json.loads((d/'boot/return/checkpoint-index.json').read_text())
assert r['generation']==k['generation']==512 and r['workflow_run']==46117675902
assert r['matches']==r['audits']==1280 and r['updates']==40 and r['checkpoint_sha256']==i['prior_checkpoint_sha256']==k['full_checkpoint_sha256']
with (d/'boot.tar.gz').open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==i['sha256']
PY
tar -xzf "$safe/boot.tar.gz" -C "$safe/work" --no-same-owner --no-same-permissions "bootstrap-g$g"
git init -q "$safe/actor"
gs -C "$safe/actor" fetch -q --depth=1 "$remote" refs/heads/bench20/1795784311/input
test "$(git -C "$safe/actor" rev-parse FETCH_HEAD)" = 8703a0fa1c07c941f6c49a2153dbe3d4a336f0dd
git -C "$safe/actor" checkout -q --detach FETCH_HEAD
cat "$safe"/actor/input/payload.part-* > "$safe/actor.tar.gz"
printf '%s  %s\n' d6734bbb5c5afbc6d65c22b67d9a24aca6c8d8553551ceb71b5766d8eab69835 "$safe/actor.tar.gz" | sha256sum --quiet -c -
tar -xzf "$safe/actor.tar.gz" -C "$safe/work" --no-same-owner --no-same-permissions engine plan.json priority light-source portable actor-native-overlay.patch
test "$(git -C "$safe/work/engine" rev-parse HEAD)" = 52db2cf0367579ee7f9b097a404649cdad47e903
git clone --quiet --no-local --depth=1 "file://$safe/work/engine" "$safe/work/vm-portable-engine"
git -C "$safe/work/vm-portable-engine" apply --check "$safe/work/actor-native-overlay.patch"
git -C "$safe/work/vm-portable-engine" apply "$safe/work/actor-native-overlay.patch"
export LEAGUE_RUNTIME="$safe/work" PYTHONPATH="$safe/work/engine:$here" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RNAD_LEARNER_BACKEND=parallel RNAD_PARALLEL_VECTOR=0 RNAD_PREP_CODEC=orjson RNAD_PREP_FAST_JSON=1 RNAD_LEARNER_JSON_CODEC=orjson
cd "$safe/work/engine"
printf 'Pinned speed test input checked; no production update.\n' >&3


role="${LEAGUE_SPEED_ROLE:?}"
[[ "$role" == actor || "$role" == publisher ]]
initial_g="$g"
initial_boot="$safe/work/bootstrap-g$initial_g"
for round in 1; do
g=$((initial_g+round-1))
boot="$initial_boot"
export LEAGUE_WARM_ROUND="$round"
phase=gate
gate_started_ms="$(date +%s%3N)"
if (( round > 1 )); then
  pin="$(env HANDOFF_TOKEN="$token" python -u "$here/speed-priority26-relay.py" gate)"
  gate_ready_ms="$(date +%s%3N)"
  phase=resume
  prior="$safe/resume-$round"
  git init -q "$prior"
  source_ref="refs/heads/ghspeed-chain/$run_id/verified-$((round-1))"
  [[ "$pin" =~ ^[0-9a-f]{40}$ ]]
  gs -C "$prior" fetch -q --depth=1 "$remote" "$source_ref"
  test "$(git -C "$prior" rev-parse FETCH_HEAD)" = "$pin"
  git -C "$prior" checkout -q --detach FETCH_HEAD
  boot="$prior/boot"
  python "$here/speed-chain-state.py" verify "$prior" "$g" "$run_id"
  printf 'Chain round %s: gate wait %sms, resume hydration %sms.\n' "$round" "$((gate_ready_ms-gate_started_ms))" "$(( $(date +%s%3N)-gate_ready_ms ))" >&3
fi
work="$safe/speed-$round"
output="$safe/result-$round"
mkdir -p "$work/cloud"
round_started_ms="$(date +%s%3N)"
phase=plan
if [[ "$role" == actor ]] && ! python -c 'from scripts.rnad import shared_league_contract'; then
  python -m pip -q install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
  python -m pip -q install numpy==1.26.4 orjson==3.13.0 zstandard==0.25.0
fi
python "$here/long_state.py" plan "$boot" "$safe/work/plan.json" "$work/plan.json" "$g" "$session"
python "$here/speed-priority.py" "$work/plan.json" "$safe/work/engine"
if [[ "$role" == actor ]]; then
  phase=simulate-and-upload
  timeout --kill-after=4s 240s python -u "$here/speed-sharded-actor.py" "$work/plan.json" "$work/cloud" 0 &
  sim_pid=$!
  env HANDOFF_TOKEN="$token" python -u "$here/speed-priority26-relay.py" upload "$work/cloud" "$work/plan.json" "${LEAGUE_SPEED_SHARD:?}" &
  relay_pid=$!
  await_both
  printf 'Actor verified replay round %s.\n' "$round" >&3
  rm -rf -- "$work"
  continue
fi
phase=runtime-and-scope
python - "$boot" "$work/plan.json" "$safe/lean-boot" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
boot,plan,out=map(Path,sys.argv[1:])
data=json.loads(plan.read_text())
def digest(p):
    with p.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
for k,v in data['files'].items():
    if k.startswith(('bootstrap/actor/','bootstrap/magnet/','bootstrap/past/','bootstrap/fixed/')):
        assert digest(boot/k.removeprefix('bootstrap/'))==v
dst=out/'bootstrap-g512'
for name in ('checkpoint-g0.pt','checkpoint-index.json','generation-manifest.json','original-draft.json',
             'original-selection-js.json','actor/selection.json','actor/battle.json'):
    src=boot/name
    assert src.is_file() and not src.is_symlink()
    p=dst/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
assert digest(dst/'checkpoint-g0.pt')==json.loads((dst/'checkpoint-index.json').read_text())['sha256']
PY
tar -czf "$safe/pc-runtime.tar.gz" -C "$safe/work" engine -C "$safe/lean-boot" bootstrap-g512 -C "$work" plan.json
if ! env HANDOFF_TOKEN="$token" python -u "$here/speed-pc-real-signer.py" publish "$safe/pc-runtime.tar.gz" "$work/plan.json"; then
  grep -E '^PC relay diagnostic: [A-Za-z]+ HTTP [0-9]+$' "$safe/private.log" >&4 || true
  exit 1
fi
printf 'Isolated PC batch publisher lease finished.\n' >&3
done
