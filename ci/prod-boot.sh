# Sourced by the two bounded jobs: both use the same checked private snapshot.
read -r session g pin started deadline < <(python - "$1" <<'PY'
import json,re,sys,time
p=json.load(open(sys.argv[1]))
assert set(p)=={'session','generation','input','start','deadline'}
s,g,pin,start,end=(p[k] for k in ('session','generation','input','start','deadline'))
assert re.fullmatch(r'[0-9]{10}',s) and type(g)==int and 443<=g<999
assert re.fullmatch(r'[a-f0-9]{40}',pin) and type(start)==type(end)==int
assert end-start==3600 and start<=time.time()<end-15
print(s,g,pin,start,end)
PY
)
branch="ghprod/$session/g$g"
bootstrap="$safe/work/bootstrap-g$g"
mkdir -p "$safe/work"
git init -q "$safe/boot"
gs -C "$safe/boot" fetch -q --depth=1 "$remote" "refs/heads/$branch/input"
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
(cd "$bootstrap" && sha256sum --quiet -c file-pins.sha256)
git init -q "$safe/actor"
gs -C "$safe/actor" fetch -q --depth=1 "$remote" refs/heads/bench20/1795784311/input
test "$(git -C "$safe/actor" rev-parse FETCH_HEAD)" = 8703a0fa1c07c941f6c49a2153dbe3d4a336f0dd
git -C "$safe/actor" checkout -q --detach FETCH_HEAD
cat "$safe"/actor/input/payload.part-* > "$safe/actor.tar.gz"
printf '%s  %s\n' d6734bbb5c5afbc6d65c22b67d9a24aca6c8d8553551ceb71b5766d8eab69835 "$safe/actor.tar.gz" | sha256sum --quiet -c -
tar -xzf "$safe/actor.tar.gz" -C "$safe/work" --no-same-owner --no-same-permissions engine plan.json priority light-source portable actor-native-overlay.patch
test "$(git -C "$safe/work/engine" rev-parse HEAD)" = 52db2cf0367579ee7f9b097a404649cdad47e903
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$safe/work/engine:$here" LEAGUE_RUNTIME="$safe/work"
mkdir -p "$safe/run"
python "$here/prod_state.py" plan "$bootstrap" "$safe/work/plan.json" "$safe/run/plan.json" "$g" "$session"
