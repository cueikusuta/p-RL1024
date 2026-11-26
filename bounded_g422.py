"""Real frozen G422 games; only summary and synthetic transport leave the job."""
import gzip,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts.rnad import shared_league_collect as c
from scripts.rnad.shared_league_contract import _opponent,verify_manifest
PIN='6a11eee1a011dcc65ac80e3b5f2c3c0b57341494'
IDS={0,3,8,12,16,17,18,19}
root,out=map(lambda p:Path(p).resolve(),sys.argv[1:3])
i=int(sys.argv[3])
if i not in IDS:raise ValueError('Unapproved Actor benchmark slot')
setup=time.perf_counter()
m=json.loads((root/'generation-manifest.json').read_text())
pin=verify_manifest(m)
if (m['generation']!=422 or m['source_sha']!=PIN or c.current_git_sha()!=PIN
    or m['seed_start']!=10170081 or m['engine_sha']!=c.frozen_engine_sha()):
    raise ValueError('Wrong historical snapshot or seed window')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
role,opp=_opponent(m,i)
out.mkdir(mode=0o700)
raw=out/'private-raw';raw.mkdir()
publish=out/'publish';publish.mkdir()
meta={'generation':422,'actor_id':i,'policy_id':m['policy_id'],'role':role,
  'main_sha256':m['main_sha256'],'magnet_sha256':m['magnet_sha256'],'opponent_sha256':opp['sha256'],
  'legacy_sha256':opp['sha256'] if role=='fixed' and opp['id'] not in ('fixed-aqbase001','fixed-coretrioaq001') else None}
metafile=raw/'meta.json';metafile.write_text(json.dumps(meta))
for folder,k in (('actor','main_sha256'),('magnet','magnet_sha256')):
    if sha(root/folder/'manifest.json')!=m[k]:raise ValueError('Wrong model checkpoint')
cmd=['node','scripts/rnad/shared_league_actor.mjs','--meta',str(metafile),
     '--main',str(root/'actor'),'--magnet',str(root/'magnet')]
if role=='historical':
    cmd+=['--opponent',str(c._historical_opponent_path(root,m,opp))]
elif role=='fixed':
    modern=opp['id'] in ('fixed-aqbase001','fixed-coretrioaq001')
    name=opp['id'].removeprefix('fixed-')+('-v1.json' if modern else '-v3.json')
    model=root/'fixed'/name
    if sha(model)!=opp['sha256']:raise ValueError('Wrong frozen opponent')
    cmd+=['--fixed-config' if modern else '--legacy',str(model)]
rows=[];jobs=[];seed=m['seed_start']+i*32
for s in range(seed,seed+16):
    for seat in (0,1):
        mid=f'g422-a{i}-s{s}-seat{seat}'; p=raw/f'{mid}.json.gz'
        jobs.append({'seed':s,'seat':seat,'out':str(p)});rows.append((s,seat,mid,p))
# Silence per-match output; only totals can enter public job logs. Audit
# mode was not part of this old public engine. Each accepted record gets
# every original match-level timeline/legality check.
start=time.perf_counter()
with (raw/'runner.log').open('x') as logs:
    backup=(os.dup(1),os.dup(2))
    try:
        os.dup2(logs.fileno(),1);os.dup2(logs.fileno(),2)
        c._run_actor_jobs(cmd,raw,jobs,2)
    finally:
        for i0,old in enumerate(backup,1):os.dup2(old,i0);os.close(old)
done=time.perf_counter()
ids=set();receipts=0
for s,seat,mid,p in rows:
    r=c._verified_match(p,seed=s,seat=seat,match_id=mid,manifest=m,opponent=opp)
    ids.add(r['match_id']);receipts+=p.stat().st_size
if len(ids)!=32 or len(list(raw.glob('*.json.gz')))!=32:raise ValueError('Wrong verified paired sample size')
check=time.perf_counter()
if not 1<=receipts<=32*1024**2:raise ValueError('Historical receipt size outside pilot bound')
# A fixed-length random stand-in models egress bytes without publishing
# actual receipts, logs, checkpoint files, or locally produced new data.
blob=os.urandom(receipts)
with (publish/'size-proxy.bin').open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
report={'actor':i,'role':role,'seed_pairs':16,'accepted_matches':32,'training_updates':0,
        'historical_generation':422,'source_sha':PIN,
        'local_setup_seconds':start-setup,'actor_compute_seconds':done-start,'strict_receipt_seconds':check-done,
        'payload_byte_count':receipts,'synthetic_transport_sha256':sha(publish/'size-proxy.bin'),
        'synthetic_transport_bytes':receipts,'transport_contains_real_matches':False,
        'runner_cpus':os.cpu_count(),'completed_epoch':time.time()}
with (publish/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report),flush=True)
