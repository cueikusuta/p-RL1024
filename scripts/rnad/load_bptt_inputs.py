"""Recover exact archived bootstrap plus twenty independently sealed actor files."""
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile
from short_bptt import require, sha, PLAN_SHA

def retrieve(plan, name):
    archive=plan["artifacts"][name]
    endpoint=f"repos/cueikusuta/p-RL1024/actions/artifacts/{archive['id']}"
    meta=json.loads(subprocess.check_output(["gh","api",endpoint]))
    require(meta["id"]==archive["id"] and
            meta["workflow_run"]["id"]==plan["sourceRun"] and
            meta["workflow_run"]["head_sha"]==plan["baselineSource"] and
            not meta["expired"] and meta["name"]==name and
            meta["digest"]=="sha256:"+archive["sha256"],f"Unexpected archived evidence: {name}")
    path=Path(name+".zip")
    for delay in (2,5,15):
        try:
            with path.open("wb") as out:
                subprocess.run(["gh","api",endpoint+"/zip"],stdout=out,check=True,timeout=180)
            require(sha(path)==archive["sha256"],f"Download hash mismatch: {name}")
            return path
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
            time.sleep(delay)
    raise ValueError(f"Unable to download: {name}")

def extract(path,root,select):
    entries=set()
    with ZipFile(path) as z:
        for info in z.infolist():
            relative=select(info.filename)
            if relative is None or info.is_dir():
                continue
            name=PurePosixPath(relative)
            require(not name.is_absolute() and ".." not in name.parts and relative not in entries, "Invalid ZIP path")
            entries.add(relative)
            dest=Path(root).joinpath(*name.parts)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source, dest.open("xb") as target:
                shutil.copyfileobj(source,target)
    path.unlink()
    return entries

if __name__=="__main__":
    _,planfile,rep,boot,actors=sys.argv
    require(sha(planfile)==PLAN_SHA and rep in ('rep0','rep1'), "Unapproved inputs")
    p=json.loads(Path(planfile).read_text())
    prefix=rep+"-control/"
    files=extract(retrieve(p,"i5-mix-inputs"),boot,
        lambda name: name[len(prefix):] if name.startswith(prefix) else None)
    require({"file-pins.sha256","checkpoint-g0.pt","pilot-input.json"}<=files,"Bootstrap incomplete")
    subprocess.run(["sha256sum","-c","file-pins.sha256"],cwd=boot,check=True)
    for batch in range(5):
        valid={f"actor-{i}/actor-{i}.json.gz" for i in range(batch*4,batch*4+4)}
        files=extract(retrieve(p,f"i5-mix-actors-{rep}-control-{batch}"),actors,
            lambda name: name if name in valid else None)
        require(files==valid,"Missing frozen actor files")
