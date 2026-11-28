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
gs -C "$safe/boot" fetch -q --depth=1 "$remote" "refs/heads/$source"
test "$(git -C "$safe/boot" rev-parse FETCH_HEAD)" = "$pin"
git -C "$safe/boot" checkout -q --detach FETCH_HEAD
cat "$safe"/boot/input/boot.part-* > "$safe/boot.tar.gz"
python - "$safe" "$g" <<'PY'
import hashlib,json,sys
from pathlib import Path
d=Path(sys.argv[1]);i=json.loads((d/'boot/input/index.json').read_text())
assert i['generation']==int(sys.argv[2]) and i['parts']==len(list((d/'boot/input').glob('boot.part-*')))
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

arm="${LEAGUE_SPEED_ARM:?}"
[[ "$arm" == base || "$arm" == batch ]]
work="$safe/speed"
boot="$safe/work/bootstrap-g$g"
mkdir "$work"
phase=plan
python "$here/long_state.py" plan "$boot" "$safe/work/plan.json" "$work/plan.json" "$g" "$session"
phase=train
if ! timeout --kill-after=4s 830s python -u "$here/speed-generation.py" "$work" "$boot"; then
  grep -E '^Benchmark Python diagnostic: [A-Za-z]+( [A-Za-z0-9_.-]+:[0-9]+)+$' "$safe/private.log" >&4 || true
  grep -E '^Benchmark Actor diagnostic: (simulate|pack) (-?[0-9]+|timeout)$' "$work/cloud.log" >&4 || true
  exit 1
fi
phase=report
mkdir "$safe/report"
python - "$work" "$safe/report/metrics.json" "$arm" "$run_id" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
work,out=map(Path,sys.argv[1:3])
r=json.loads((work/'result.json').read_text()); v=json.loads((work/'strict.json').read_text())
assert r['generation']==511 and r['accepted']==v['accepted']==r['audits']==v['audits']==1280
assert r['actual_ordered_updates']==40 and all('updated_ms' in x for x in r['optimizer_events'])
assert len(v['pairs'])==1280 and len(v['all_packet_times_ms'])==160
assert len(r['cloud']['leases'])==(80 if sys.argv[3]=='base' else 20)
assert sum(len(x['packets']) for x in r['cloud']['leases'])==160
digests={k:hashlib.sha256(json.dumps({n:p[i] for n,p in v['pairs'].items()},sort_keys=True).encode()).hexdigest()
         for i,k in enumerate(('matches_sha256','audits_sha256'))}
q=dict(arm=sys.argv[3],workflow_run=int(sys.argv[4]),generation_in=511,generation_out=512,accepted=1280,audits=1280,updates=40,
       acquisition_s=(max(v['all_packet_times_ms'].values())-r['dispatch_ms'])/1000,
       first_accepted_s=(min(v['all_packet_times_ms'].values())-r['dispatch_ms'])/1000,
       actor_s=(r['cloud']['end']-r['dispatch_ms'])/1000,
       verified_checkpoint_s=r['active_elapsed_s'], checkpoint_sha256=r['checkpoint_sha256'],**digests)
out.write_text(json.dumps(q,sort_keys=True)+'\n')
PY
git init -q "$safe/report"
git -C "$safe/report" add metrics.json
git -C "$safe/report" -c user.name=league-runner -c user.email=league-runner@users.noreply.github.com commit -qm 'Seal isolated speed benchmark metrics'
ref="refs/heads/ghspeed/$run_id/$arm"
gs -C "$safe/report" push -q "$remote" "HEAD:$ref"
test "$(gs -C "$safe/report" ls-remote --heads "$remote" "$ref" | cut -f1)" = "$(git -C "$safe/report" rev-parse HEAD)"
printf 'Verified 1280 matches, 1280 audits, 40 updates. Private timing report saved.\n' >&3
