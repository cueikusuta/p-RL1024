"""Compare whole tarball vs exact per-file Git storage and readback."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root,out=map(Path,sys.argv[1:3])
run=int(sys.argv[3])
gitdir=root/'prototype'
gitdir.mkdir()

def git(*args):
    return subprocess.check_output(['git','-C',str(gitdir),*args],text=True).strip()
def digest(f):
    with f.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def verified(path):
    pins={}
    for line in (path/'file-pins.sha256').read_text().splitlines():
        h,part=line.split('  ',1); f=path/part
        assert f.resolve().is_relative_to(path.resolve()) and f.is_file() and not f.is_symlink()
        assert part not in pins and digest(f)==h
        pins[part]=h
    return pins

git('init','-q')
git('config','user.email','league-runner@users.noreply.github.com')
git('config','user.name','league-runner')
git('config','gc.auto','0')
git('config','pack.threads','2')
history=[]
seen=set()
commits=[]
counts=[]
for g in (511,512,513):
    d=root/f'g{g}'
    assert (d/'out'/f'bootstrap-g{g}').is_dir()
    src=d/'out'/f'bootstrap-g{g}'
    inventory=verified(src)
    all_files=[f for f in src.rglob('*') if f.is_file()]
    assert len(all_files)==len(inventory)+1 and all(not f.is_symlink() for f in all_files)
    assert max(f.stat().st_size for f in all_files)<64*1024**2
    hashes={digest(f):f.stat().st_size for f in all_files}
    added={h:s for h,s in hashes.items() if h not in seen}
    seen.update(hashes)
    dst=gitdir/'boot'
    if dst.exists():shutil.rmtree(dst)
    shutil.copytree(src,dst)
    git('add','-f','-A','--','boot')
    git('commit','-q','-m',f'Isolated fixture {g}')
    commit=git('rev-parse','HEAD');commits.append(commit)
    git('repack','-a','-d')
    packs=sum(f.stat().st_size for f in (gitdir/'.git/objects/pack').glob('*.pack'))
    counts.append(dict(generation=g,compressed_archive_bytes=(d/'boot.tar.gz').stat().st_size,logical_bytes=sum(f.stat().st_size for f in all_files),
                       unique_new_bytes=sum(added.values()),file_count=len(all_files),unique_new_files=len(added),git_pack_total_bytes=packs))
    history.append(inventory)
for i,(g,commit) in enumerate(zip((511,512,513),commits,strict=True)):
    git('checkout','-q','--detach',commit)
    assert verified(gitdir/'boot')==history[i]
(out/'metrics.json').write_text(json.dumps({'workflow_run':run,'generations':counts,'readback_verified':True},sort_keys=True)+'\n')
