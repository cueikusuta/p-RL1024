from pathlib import Path
import concurrent.futures as cf
import json,os,re,subprocess as sp,sys,time
def run_with_safe_error(entry):
    import traceback
    try:
        entry()
    except Exception as e:
        who = type(e).__name__
        what = " ".join(f"{Path(f.filename).name}:{f.lineno}" for f in traceback.extract_tb(e.__traceback__)[-6:])
        print(f"PRIVATE_DIAG {who} {what}", file=sys.stderr, flush=True)
        raise

def main():
    base,root=map(lambda s:Path(s).resolve(),sys.argv[1:3])
    session,run,g=sys.argv[3:6]
    assert re.fullmatch(r"[0-9]{10}",session) and re.fullmatch(r"[0-9]+",run) and re.fullmatch(r"[0-9]{3}",g)
    git=["git","-C",str(base),"-c","credential.helper=","-c","http.followRedirects=false"]
    remote="https://github.com/cueikusuta/p-rl1024-handoff-private.git"
    cmd=lambda args,**kw:sp.run(git+args,check=True,timeout=55,**kw).stdout
    refs={f"refs/heads/ghprod/{session}/g{g}/a{i:02}":i for i in range(20)}
    pending=set(refs)
    def intake(item):
        ref,sha=item; i=refs[ref]
        local=f"refs/bench21/a{i:02}"
        assert cmd(["rev-parse",local],stdout=sp.PIPE).decode().strip()==sha
        def show(name):return cmd(["show",sha+":return/"+name],stdout=sp.PIPE)
        cert=json.loads(show("ready.json"))
        assert type(cert["num_parts"]) is int and 0<cert["num_parts"]<=160
        assert cert["actor"]==i and cert["run"]==f"g{g}-ghprod-{session}"
        assert type(cert["size"]) is int and 0<cert["size"]<=100*1024**2
        assert cert["num_parts"]==(cert["size"]+655359)//655360
        dst=root/"transfer"/f"a{i:02}";dst.mkdir(parents=True)
        pack=dst/"wave.tar"
        with pack.open("xb") as w:
            for k in range(cert["num_parts"]):
                b=show(f"part-{k:03}")
                assert len(b)==min(655360,cert["size"]-655360*k)
                w.write(b)
        cert.update(private_commit=sha,github_run=int(run))
        receipt=dst/"ready.json";receipt.write_text(json.dumps(cert))
        env={k:v for k,v in os.environ.items() if k not in {"GIT_ASKPASS","HANDOFF_HTTP_TOKEN"}}
        sp.run([sys.executable,str(Path(__file__).with_name("prod_ingest.py")),str(root),str(pack),str(receipt)],
               check=True,timeout=30,env=env,stdout=sp.DEVNULL)
    start=time.monotonic(); jobs=[]
    with cf.ThreadPoolExecutor(4) as pool:
        while pending:
            assert time.monotonic()-start<420, "runner returns did not arrive"
            for job in jobs:
                if job.done():job.result()
            listed=cmd(["ls-remote","--heads",remote,*sorted(pending)],stdout=sp.PIPE)
            available=[]
            for line in listed.decode("ascii").splitlines():
                sha,ref=line.split("\t")
                assert ref in pending and re.fullmatch(r"[0-9a-f]{40}",sha)
                available.append((ref,sha))
            if not available:
                time.sleep(1.0);continue
            cmd(["fetch","--quiet","--no-tags","--no-auto-gc","--depth=1",remote,
                 *[f"{ref}:refs/bench21/a{refs[ref]:02}" for ref,_ in available]])
            for item in available:
                jobs.append(pool.submit(intake,item));pending.remove(item[0])
        for job in jobs:job.result()

if __name__=="__main__":run_with_safe_error(main)
