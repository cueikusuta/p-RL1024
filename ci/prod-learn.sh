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
run_id="${GITHUB_RUN_ID:?}"
epoch="$(date +%s%3N)"
phase=learn
mkdir -p "$safe/run/inbox" "$safe/run/cloud/packets"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export RNAD_LEARNER_BACKEND=parallel RNAD_PARALLEL_VECTOR=0 RNAD_PREP_CODEC=orjson RNAD_PREP_FAST_JSON=1 RNAD_LEARNER_JSON_CODEC=orjson
cd "$safe/work/engine"
git init -q "$safe/waves"
GIT_ASKPASS="$safe/askpass" GIT_TERMINAL_PROMPT=0 HANDOFF_HTTP_TOKEN="$token" \
python "$here/prod_poll.py" "$safe/waves" "$safe/run" "$session" "$run_id" "$g" &
poll=$!
timeout --kill-after=4s "$(($(date +%s)-deadline<0 ? deadline-$(date +%s) : 1))s" python -u "$here/bench20_vm_learner.py" "$safe/run" "$bootstrap" "$epoch" &
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
phase=continuation
next="$safe/work/bootstrap-g$((g+1))"
python "$here/prod_state.py" next "$bootstrap" "$safe/run/output" "$next" "$started" "$deadline" "$run_id"
test "$(date +%s)" -lt "$deadline"
phase=save
git init -q "$safe/save"
mkdir "$safe/save/input" "$safe/save/return"
tar -czf "$safe/next-bootstrap.tar.gz" -C "$safe/work" "$(basename "$next")"
split -b 480K -d -a 3 "$safe/next-bootstrap.tar.gz" "$safe/save/input/boot.part-"
cp "$safe/run/result.json" "$safe/save/return/"
cp "$safe/run/output/"{checkpoint-g"$((g+1))".pt,generation"$((g+1))"-index.json,generation"$((g+1))"-report.json} "$safe/save/return/"
python - "$safe" "$g" "$run_id" <<'PY'
import hashlib,json,sys,os
from pathlib import Path
root=Path(sys.argv[1]);g=int(sys.argv[2]);run=sys.argv[3]
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
q=json.loads((root/'run/result.json').read_text())
p=root/'save/return'/f'checkpoint-g{g+1}.pt'
assert q['generation']==g and q['accepted']==q['audits']==1280 and q['actual_ordered_updates']==40
assert len(q['remote_actor_waves'])==20 and q['checkpoint_sha256']==sha(p)
out={'generation':g+1,'sha256':sha(root/'next-bootstrap.tar.gz'),'parts':len(list((root/'save/input').glob('boot.part-*'))),'prior_checkpoint_sha256':sha(p)}
(root/'save/input/index.json').write_text(json.dumps(out))
summary={'generation':g+1,'input_generation':g,'checkpoint_sha256':sha(p),'workflow_run':int(run),'matches':1280,'audits':1280,'updates':40,'actors':20}
(root/'save/return/done.json').write_text(json.dumps(summary))
for f in (root/'save').rglob('*'):
    if f.is_file() and '.git' not in f.parts:
        with f.open('rb') as stream:os.fsync(stream.fileno())
PY
git -C "$safe/save" add -- input return
git -C "$safe/save" -c user.name=league-runner -c user.email=league-runner@users.noreply.github.com commit -qm 'Save verified generation and exact restart in the private store'
ref="refs/heads/ghprod/$session/g$((g+1))/input"
test "$(date +%s)" -lt "$deadline"
remaining=$((deadline-$(date +%s)))
if (( remaining <= 1 )); then exit 1; fi
# Only the exact next-generation ref is written, never the prior input.
gs -C "$safe/save" push -q "$remote" "HEAD:$ref"
test "$(gs -C "$safe/save" ls-remote --heads "$remote" "$ref" | cut -f1)" = "$(git -C "$safe/save" rev-parse HEAD)"
printf 'Private checkpoint and exact next bootstrap verified.\n' >&3
