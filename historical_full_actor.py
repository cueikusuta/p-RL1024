"""Pinned already public G422, no changes to the historical Actor."""
import json,os,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts.rnad.shared_league_collect import collect,current_git_sha,verify_manifest
p,work,summary=map(Path,sys.argv[1:4]);who=int(sys.argv[4]);workers=int(sys.argv[5])
pin='6a11eee1a011dcc65ac80e3b5f2c3c0b57341494'
assert who in range(20) and workers==4
x=json.loads((p/'generation-manifest.json').read_text())
assert x['generation']==422 and x['seed_start']==10170081 and x['actor_count']==20
assert x['source_sha']==current_git_sha()==pin
h=verify_manifest(x);work.mkdir(mode=0o700,exist_ok=False)
start=time.perf_counter()
with (work/'private-actor.log').open('x') as f:
 saved=(os.dup(1),os.dup(2))
 try:
  os.dup2(f.fileno(),1);os.dup2(f.fileno(),2)
  ans=collect(bootstrap=p,actor_id=who,out=work,match_workers=workers)
 finally:
  for j,a in enumerate(saved,1):os.dup2(a,j);os.close(a)
assert ans['accepted_matches']==64 and ans['manifest_sha256']==h
out={'actor':who,'source_sha':pin,'manifest_sha':h,'accepted_matches':64,'verified':True,
     'actor_seconds_including_original_checks_and_packing':time.perf_counter()-start}
summary.write_text(json.dumps(out)+'\n');print(json.dumps(out))
