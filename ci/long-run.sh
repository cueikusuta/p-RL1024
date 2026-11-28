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
assert end-st==36000 and st<=time.time()<end-330
expected=f'ghlong/{s}/g{g}/input'
assert r==expected or (g==i and b==0 and (re.fullmatch(r'ghprod/[0-9]{10}/g'+str(g)+r'/input',r) or (g==516 and r=='ghlong/1795815513/g516/input' and p=='1fc9ac44d82109042245ebd11e814c20d4eaf349')))
print(s,g,p,r,st,end,i,b)
PY
)
run_id="${GITHUB_RUN_ID:?}"
segment_end=$(( $(date +%s) + 3120 ))
if (( segment_end > deadline )); then segment_end=$deadline; fi
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
printf 'Confirmed production input on GitHub; no VM compute.\n' >&3
floor=510
while (( segment_end - $(date +%s) >= floor && g-initial<100 )); do
  boot="$safe/work/bootstrap-g$g"
  next="$safe/work/bootstrap-g$((g+1))"
  work="$safe/training-g$g"
  mkdir "$work"
  phase=plan
  python "$here/long_state.py" plan "$boot" "$safe/work/plan.json" "$work/plan.json" "$g" "$session"
  phase=train
  epoch="$(date +%s)"
  printf 'Started official generation %s on GitHub.\n' "$g" >&3
  seconds=$((segment_end-epoch-32))
  if (( seconds > 850 )); then seconds=850; fi
  timeout --kill-after=4s "${seconds}s" python -u "$here/production_generation.py" "$work" "$boot"
  phase=handoff
  python "$here/long_state.py" next "$boot" "$work/output" "$next" "$started" "$deadline" "$run_id" "$initial"
  phase=save
  target="$safe/save-g$((g+1))"
  git init -q "$target"
  mkdir "$target/input" "$target/return"
  tar -czf "$target/boot.tar.gz" -C "$safe/work" "$(basename "$next")"
  split -b 480K -d -a 3 "$target/boot.tar.gz" "$target/input/boot.part-"
  cp "$work/strict.json" "$target/return/verification.json"
  cp "$work/output/generation$((g+1))-index.json" "$target/return/checkpoint-index.json"
  cp "$work/output/generation$((g+1))-report.json" "$target/return/report.json"
  newused="$(python "$here/long-save-index.py" "$work" "$target" "$g" "$run_id" "$used")"
  rm -f "$target/boot.tar.gz"
  git -C "$target" add -- input return
  git -C "$target" -c user.name=league-runner -c user.email=league-runner@users.noreply.github.com commit -qm 'Seal verified private league checkpoint'
  ref="refs/heads/ghlong/$session/g$((g+1))/input"
  test "$(date +%s)" -lt "$segment_end"
  gs -C "$target" push -q "$remote" "HEAD:$ref"
  test "$(gs -C "$target" ls-remote --heads "$remote" "$ref" | cut -f1)" = "$(git -C "$target" rev-parse HEAD)"
  g=$((g+1)); used=$newused
  printf 'Saved generation %s privately; 1280 matches and audits, 40 ordered updates.\n' "$g" >&3
  # Only this job's unneeded temporary data; the saved checkpoint has been verified and reread remotely.
  rm -rf -- "$work" "$boot" "$target"
  elapsed=$(( $(date +%s)-epoch+40 )); if (( elapsed > floor )); then floor=$elapsed; fi
  if (( used > 1820*1024*1024 )); then break; fi
done
printf 'Hosted production segment stopped at a durable checkpoint.\n' >&3
