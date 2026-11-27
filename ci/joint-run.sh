#!/usr/bin/env bash
set -euo pipefail
umask 077
safe="$(mktemp -d "${RUNNER_TEMP:?}/joint.XXXXXXXX")"
trap 'rm -rf -- "$safe"' EXIT
printf '%s\n' "${JOINT_PRIVATE_KEY:?missing scoped key}" > "$safe/key"
unset JOINT_PRIVATE_KEY
cat > "$safe/known_hosts" <<'HOST'
github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl
HOST
read -r parent branch payload_sha start deadline generation resume_sha < <(python - "$1" <<'PY'
import json,re,sys,time
a=json.load(open(sys.argv[1]))
assert set(a)=={'stage_commit','stage_branch','payload_sha256','start_epoch','deadline_epoch','generation','prior_resume_sha256'}
assert re.fullmatch('[0-9a-f]{40}',a['stage_commit']) and re.fullmatch('[0-9a-f]{64}',a['payload_sha256'])
assert re.fullmatch('joint-1h/[0-9]{10}/g[0-9]{3}',a['stage_branch'])
start=a['start_epoch'];deadline=a['deadline_epoch']
assert isinstance(start,int) and isinstance(deadline,int) and deadline-start==3600 and start<=time.time()<deadline
g=a['generation']; rs=a['prior_resume_sha256']
assert isinstance(g,int) and 425<=g<1000 and a['stage_branch'].endswith(f'/g{g}')
assert re.fullmatch('[0-9a-f]{64}',rs)
print(a['stage_commit'],a['stage_branch'],a['payload_sha256'],start,deadline,g,rs)
PY
)
export GIT_SSH_COMMAND="ssh -F /dev/null -i $safe/key -o IdentityAgent=none -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$safe/known_hosts -o HostKeyAlgorithms=ssh-ed25519"
exec 3>&1
store='git@github.com:cueikusuta/p-rl1024-handoff-private.git'
(
  git init -q "$safe/stage"
  git -C "$safe/stage" fetch -q --depth=1 "$store" "refs/heads/$branch"
  test "$(git -C "$safe/stage" rev-parse FETCH_HEAD)" = "$parent"
  git -C "$safe/stage" checkout -q --detach FETCH_HEAD
  mkdir "$safe/work"
  (cd "$safe/stage" && cat input/payload.part-* > "$safe/input.tar.gz")
  printf '%s  %s\n' "$payload_sha" "$safe/input.tar.gz" | sha256sum --quiet -c -
  tar -xzf "$safe/input.tar.gz" -C "$safe/work" --no-same-owner --no-same-permissions
  mkdir "$safe/prior"
  (cd "$safe/stage" && cat prior/part-* > "$safe/prior.tar.gz")
  printf '%s  %s\n' "$resume_sha" "$safe/prior.tar.gz" | sha256sum --quiet -c -
  tar -xzf "$safe/prior.tar.gz" -C "$safe/prior" --no-same-owner --no-same-permissions
  test -d "$safe/prior/next-bootstrap"
  rm -r -- "$safe/work/run/bootstrap"
  mv "$safe/prior/next-bootstrap" "$safe/work/run/bootstrap"
  cp "$safe/stage/next-plan.json" "$safe/work/run/plan.json"
  # Dedicated private staging keeps the PC's independently sealed wave separate.
  wave_name="$(python - "$safe/work/run/plan.json" <<'PY2'
import json,re,sys
a=json.load(open(sys.argv[1]));g=a['generation'];r=a['runId']
assert isinstance(g,int) and 424<=g<1000 and r==f"g{g}-joint-1h"
print(f"G{g}-{r}-windows-w000.zip")
PY2
  )"
  wave="$safe/work/run/$wave_name"
  test ! -e "$wave"
  (cd "$safe/stage" && cat pc/wave.part-* > "$wave")
  cp "$safe/stage/pc/sha256" "$safe/work/run/pc-receipt.sha256"
  test "$(sha256sum "$wave" | cut -d ' ' -f 1)" = "$(cat "$safe/work/run/pc-receipt.sha256")"
  cd "$safe/work/engine"
  test "$(git rev-parse HEAD)" = '52db2cf0367579ee7f9b097a404649cdad47e903'
  git clone --quiet --no-local --depth=1 "file://$PWD" ../vm-portable-engine
  git -C ../vm-portable-engine apply --check ../actor-native-overlay.patch
  git -C ../vm-portable-engine apply ../actor-native-overlay.patch
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
  export RNAD_LEARNER_BACKEND=parallel RNAD_PARALLEL_VECTOR=0 RNAD_PREP_CODEC=orjson RNAD_PREP_FAST_JSON=1 RNAD_LEARNER_JSON_CODEC=orjson
  export PYTHONPATH="$PWD"
  env -u GIT_SSH_COMMAND timeout --kill-after=3s "$(($(date -u -d "@$deadline" +%s) - $(date -u +%s)))s" python -u ../joint_host_run.py ../run "$start" "$deadline" &
  job=$!
  last_check=$SECONDS
  prior_progress=-1
  while kill -0 "$job" 2>/dev/null; do
    if (( SECONDS - last_check >= 120 )); then
      packets="$(find "$safe/work/run/generation/cloud/packets" -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l || true)"
      logs="$(wc -c < "$safe/private-job.log")"
      progress=$((packets * 100000000 + logs))
      if (( progress == prior_progress )); then
        echo 'Joint generation stalled; the controller must resume from the last verified checkpoint.' >&3
        kill -TERM "$job" 2>/dev/null || true
        wait "$job" || true
        exit 1
      fi
      prior_progress=$progress
      last_check=$SECONDS
      echo '120-second check: hosted generation has advanced.' >&3
    fi
    sleep 2
  done
  wait "$job"
  cd "$safe/stage"
  mkdir return
  tar -cf "$safe/result.tar" -C "$safe/work/run" return
  split -b 28M -d -a 2 "$safe/result.tar" return/return.part-
  sha256sum "$safe/result.tar" | cut -d ' ' -f 1 > return/result.sha256
  tar --exclude='./detailed.tar.zst' -czf "$safe/resume.tar.gz" -C "$safe/work/run/return" .
  mkdir return/resume
  split -b 560K -d -a 3 "$safe/resume.tar.gz" return/resume/part-
  sha256sum "$safe/resume.tar.gz" | cut -d ' ' -f 1 > return/resume/sha256
  cp "$safe/work/run/return/done.json" return/done.json
  test "$(date -u +%s)" -lt "$deadline"
  git add -- return/
  git -c user.name=league-runner -c user.email=league-runner@users.noreply.github.com commit -qm 'Return verified league checkpoint to private staging'
  git push -q "$store" "HEAD:refs/heads/$branch"
) > "$safe/private-job.log" 2>&1 || { echo 'Joint private job failed; no private data emitted.' >&2; exit 1; }
echo 'Joint generation completed and returned privately.'
