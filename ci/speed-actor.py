"""Independent bounded VM worker side; reads scope and immutable heads only."""
import json,os,subprocess,time,sys,shutil,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
here=Path(os.environ['LEAGUE_RUNTIME']).resolve()
env=dict(os.environ)
def j(p):return json.loads(p.read_bytes())
def now():return time.time_ns()//1_000_000
def run(plan_path,where,epoch):
    p=j(plan_path);root=Path(p['bootstrap']).resolve()
    assert p['roles']['cloud'] and p['workers']=={'cloud':8,'windows':0}
    assert p['sourceSHA']=='52db2cf0367579ee7f9b097a404649cdad47e903'
    frozen=here/'vm-portable-engine'
    assert subprocess.check_output(['git','-C',str(frozen),'rev-parse','HEAD'],text=True).strip()==p['sourceSHA']
    if not where.exists():where.mkdir()
    (where/'packets').mkdir()
    jobs={q['task']:q for q in p['packets']}
    allowed=p['prioritizedLeases']; assert len(allowed)==80
    A=str(here/'portable/portable-native/portable-affine.cjs')
    env.update(GITHUB_RUN_ID=p['runId'],GITHUB_RUN_ATTEMPT=p['runAttempt'],RNAD_AUDIT_SHA=p['sourceSHA'],
             RNAD_DRAFT_RHO_NATIVE=A,RNAD_SELECTION_NATIVE=A,RNAD_BATTLE_AFFINE_NATIVE=A,
             RNAD_BATTLE_AFFINE_MODE='native',RNAD_DRAFT_BATCH_NATIVE='1',RNAD_COMPACT_ACCEPTED_RECEIPTS='1',RNAD_RAW_PACKET_STAGE='1')
    base=['node','--import',str(here/'light-source/preload.mjs'),'scripts/rnad/shared_league_actor.mjs']
    metas={}
    man=j(root/'generation-manifest.json')
    from scripts.rnad.shared_league_contract import _opponent
    for i in sorted({i for i,_,_ in allowed}):
        role,opp=_opponent(man,i)
        meta={'generation':man['generation'],'actor_id':i,'policy_id':man['policy_id'],'role':role,
            'main_sha256':man['main_sha256'],'magnet_sha256':man['magnet_sha256'],'opponent_sha256':opp['sha256'],'legacy_sha256':opp['sha256'] if p['opponents'][str(i)]['args'][:1]==['--legacy'] else None}
        b=json.dumps(meta,sort_keys=True).encode()
        assert hashlib.sha256(b).hexdigest()==p['files'][f'bootstrap/actor-meta-{i}.json']
        metas[i]=where/f'actor-meta-{i}.json';metas[i].write_bytes(b)
    def go(x):
        i,*keys=x
        job=where/('job-'+keys[0]);job.mkdir();(job/'accepted').mkdir();(job/'audit').mkdir()
        specs=[jobs[k] for k in keys]
        slot=[]
        for spec in specs:
            for n,r in enumerate(spec['rows']):
                slot.append({'seed':spec['seeds'][n//2],'seat':n%2,'out':str(job/'accepted'/r['name'].removesuffix('.gz'))})
        pick=job/'jobs.json';pick.write_text(json.dumps({'schema':'rnad-shared-actor-batch-jobs-v1','jobs':slot}))
        args=p['opponents'][str(i)]['args']
        assert len(args) in (0,2) and all('..' not in x for x in args)
        args=[x if n%2==0 else str(root/x.removeprefix('bootstrap/')) for n,x in enumerate(args)]
        cmd=base+['--meta',str(metas[i]),'--main',str(root/'actor'),'--magnet',str(root/'magnet'),
            '--audit-dir',str(job/'audit'),'--batch-jobs',str(pick)]+args
        begun=now()
        def execute(cmd,cwd,timeout,stage):
            try:
                with (job/'actor.log').open('ab') as log:
                    subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=log,check=True,timeout=timeout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                code=exc.returncode if isinstance(exc,subprocess.CalledProcessError) else 'timeout'
                print(f'Benchmark Actor diagnostic: {stage} {code}',file=sys.stderr,flush=True)
                raise
        execute(cmd,frozen,125 if len(keys)==2 else 200,'simulate')
        for k in range(0,len(specs),2):
            spec=job/f'pack-{k}.json';spec.write_text(json.dumps(specs[k:k+2]))
            execute(['node',str(here/'priority/pack_raw_lease_window.mjs'),str(job),str(where/'packets'),str(spec)],None,15,'pack')
        return {'lease':keys[0],'packets':keys,'started_epoch_ms':begun,'sealed_epoch_ms':now()}
    time.sleep(max(0,(epoch-now())/1000))
    begin=now()
    mode=os.environ['LEAGUE_SPEED_ARM']
    assert mode in ('base','batch')
    if mode=='batch':
        groups={}
        for i,x,y in allowed:groups.setdefault(i,[]).extend((x,y))
        assert len(groups)==20 and all(len(v)==8 for v in groups.values())
        allowed=[(i,*keys) for i,keys in groups.items()]
    assert sorted(k for group in allowed for k in group[1:])==sorted(jobs)
    with ThreadPoolExecutor(min(4,len(os.sched_getaffinity(0)))) as pool:
        stats=list(pool.map(go,allowed))
    f=where/'work-result.json';f.with_suffix('.part').write_text(json.dumps({'start':begin,'end':now(),'leases':stats}))
    f.with_suffix('.part').replace(f)
if __name__=='__main__':
    run(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),int(sys.argv[3]))
