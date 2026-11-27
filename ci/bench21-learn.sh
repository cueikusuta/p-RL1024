#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
safe="$(mktemp -d "${RUNNER_TEMP:?}/bench21.XXXXXXXX")"
phase=setup
trap 'code=$?; if (( code != 0 )); then printf "Hosted benchmark failed during %s. No private data emitted.\n" "$phase" >&4; fi; rm -rf -- "$safe"' EXIT
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
read -r session run_id < <(python - "$1" <<'PY'
import json,re,sys,os
p=json.load(open(sys.argv[1]))
assert set(p)=={'session'} and re.fullmatch(r'[0-9]{10}',p['session'])
print(p['session'],os.environ['GITHUB_RUN_ID'])
PY
)
epoch="$(date +%s%3N)"
actor_pin=8703a0fa1c07c941f6c49a2153dbe3d4a336f0dd
boot_pin=1b15796d313b083a1aaa51e01a2f612eafaae05d
phase=private-input
mkdir "$safe/work" "$safe/run"
for kind in actor boot; do git init -q "$safe/$kind"; done
gs -C "$safe/actor" fetch -q --depth=1 "$remote" refs/heads/bench20/1795784311/input
test "$(git -C "$safe/actor" rev-parse FETCH_HEAD)" = "$actor_pin"
git -C "$safe/actor" checkout -q --detach FETCH_HEAD
gs -C "$safe/boot" fetch -q --depth=1 "$remote" "$boot_pin"
test "$(git -C "$safe/boot" rev-parse FETCH_HEAD)" = "$boot_pin"
git -C "$safe/boot" checkout -q --detach FETCH_HEAD
cat "$safe"/actor/input/payload.part-* > "$safe/actor.tar.gz"
cat "$safe"/boot/input/payload.part-* > "$safe/boot.tar.gz"
printf '%s  %s\n' d6734bbb5c5afbc6d65c22b67d9a24aca6c8d8553551ceb71b5766d8eab69835 "$safe/actor.tar.gz" \
 f77097f5f7d4471044fe7d6bcfa9d813b32771a1b1f56ec8760bb76d185c16d5 "$safe/boot.tar.gz" | sha256sum --quiet -c -
tar -xzf "$safe/actor.tar.gz" -C "$safe/work" --no-same-owner --no-same-permissions engine plan.json
tar -xzf "$safe/boot.tar.gz" -C "$safe/work" --no-same-owner --no-same-permissions run/bootstrap run/checkpoint-g424-anchor.pt
test "$(git -C "$safe/work/engine" rev-parse HEAD)" = 52db2cf0367579ee7f9b097a404649cdad47e903
test -s "$safe/work/run/checkpoint-g424-anchor.pt"
(cd "$safe/work/run/bootstrap" && sha256sum --quiet -c file-pins.sha256)
python - "$safe" "$session" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);s=sys.argv[2]
p=json.loads((root/'work/plan.json').read_text())
assert p['generation']==424 and p['runId']=='g424-bench20-1795784311'
p.update(runId=f'g424-bench20-{s}',bootstrap=str(root/'work/run/bootstrap'))
(root/'run/plan.json').write_text(json.dumps(p))
PY
phase=learn
mkdir -p "$safe/run/inbox" "$safe/run/cloud/packets"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export RNAD_LEARNER_BACKEND=parallel RNAD_PARALLEL_VECTOR=0 RNAD_PREP_CODEC=orjson RNAD_PREP_FAST_JSON=1 RNAD_LEARNER_JSON_CODEC=orjson
export PYTHONPATH="$safe/work/engine:$here"
cd "$safe/work/engine"
git init -q "$safe/waves"
GIT_ASKPASS="$safe/askpass" GIT_TERMINAL_PROMPT=0 HANDOFF_HTTP_TOKEN="$token" \
python "$here/bench21-poll.py" "$safe/waves" "$safe/run" "$session" "$run_id" &
poll=$!
python -u "$here/bench20_vm_learner.py" "$safe/run" "$safe/work/run/bootstrap" "$epoch" &
learner=$!
poll_ok=0
learn_ok=0
wait "$poll" || poll_ok=$?
if (( poll_ok != 0 )); then touch "$safe/run/inbox/abort"; fi
wait "$learner" || learn_ok=$?
if (( poll_ok != 0 || learn_ok != 0 )); then
  printf 'Hosted child exit codes: receive=%s learner=%s.\n' "$poll_ok" "$learn_ok" >&4
  grep '^PRIVATE_DIAG ' "$safe/private.log" >&4 || true
  exit 1
fi
phase=save
python - "$safe/run/result.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert (p['generation'],p['accepted'],p['audits'],p['actual_ordered_updates'])==(424,1280,1280,40)
assert len(p['remote_actor_waves'])==20 and not p['production_launched']
assert len(p['checkpoint_sha256'])==64
PY
git init -q "$safe/outgit"
dest="$safe/outgit/return"
mkdir "$dest"
cp "$safe/run/result.json" "$dest/"
cp "$safe/run/output/"{checkpoint-g425.pt,generation425-index.json,generation425-report.json} "$dest/"
git -C "$safe/outgit" add -- return
git -C "$safe/outgit" -c user.name=league-benchmark -c user.email=league-benchmark@users.noreply.github.com commit -qm 'Save independent hosted benchmark checkpoint'
ref="refs/heads/bench21/$session/learner${BENCH21_RESULT_SUFFIX:-}"
gs -C "$safe/outgit" push -q "$remote" "HEAD:$ref"
test "$(gs -C "$safe/outgit" ls-remote --heads "$remote" "$ref" | cut -f1)" = "$(git -C "$safe/outgit" rev-parse HEAD)"
printf 'Verified one generation, 1280 matches, 1280 audits, 40 updates; private save confirmed.\n' >&3
