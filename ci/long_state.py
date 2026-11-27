import hashlib,json,os,re,sys,time
from pathlib import Path
from scripts.rnad.shared_league_contract import _opponent, verify_manifest
def j(p):return json.loads(p.read_bytes())
def digest(p):
    with p.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
def pins(root):
    for line in (root/'file-pins.sha256').read_text().splitlines():
        sha,name=line.split('  ',1); p=root/name
        assert p.resolve().is_relative_to(root.resolve()) and not p.is_symlink() and digest(p)==sha
def plan(boot,template,out,g,session):
    pins(boot)
    p=j(template);m=j(boot/'generation-manifest.json')
    assert m['generation']==g and g>=443
    shift=m['seed_start']-min(s for x in p['packets'] for s in x['seeds'])
    p.update(bootstrap=str(boot),generation=g,runId=f'g{g}-ghprod-{session}',runAttempt='1',
        workers={'cloud':8,'windows':0},manifestSHA=verify_manifest(m),roles={'cloud':list(range(20)),'windows':[]})
    p.pop('windowsConnectivityLease',None)
    p['files']={'bootstrap/'+str(f.relative_to(boot)):digest(f) for f in boot.rglob('*') if f.is_file()}
    for i in range(20):
        role,o=_opponent(m,i); args=[]
        if role=='historical':args=['--opponent','bootstrap/past/'+o['id']]
        elif role=='fixed':
            n=o['id'].removeprefix('fixed-'); config=n in ('aqbase001','coretrioaq001')
            args=['--fixed-config' if config else '--legacy','bootstrap/fixed/'+n+('-v1.json' if config else '-v3.json')]
        else:assert role=='self'
        p['opponents'][str(i)]={'role':role,'args':args}
        meta={'generation':g,'actor_id':i,'policy_id':m['policy_id'],'role':role,'main_sha256':m['main_sha256'],'magnet_sha256':m['magnet_sha256'],'opponent_sha256':o['sha256'],'legacy_sha256':o['sha256'] if args[:1]==['--legacy'] else None}
        p['files'][f'bootstrap/actor-meta-{i}.json']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    for q in p['packets']:
        q['seeds']=[s+shift for s in q['seeds']]
        q['rows']=[{'name':f'g{g}-a{q["actor"]}-s{s}-seat{seat}.json.gz'} for s in q['seeds'] for seat in (0,1)]
    assert len(p['packets'])==160 and len({row['name'] for q in p['packets'] for row in q['rows']})==1280
    out.write_text(json.dumps(p))
def handoff(boot,out,where,start,deadline,run,initial):
    from scripts.rnad import shared_league_handoff as h
    import torch
    budget=deadline-start
    prior=j(boot/'continuation-context.json')
    index=j(boot/'checkpoint-index.json')
    if prior.get('new_budget_phase')!='GitHub-only 10-hour production':
        assert index['generation']==initial and budget==36000 and start<=time.time()<deadline
        ctx={'schema':h.ARTIFACT_SCHEMA,'new_budget_phase':'GitHub-only 10-hour production','source_run_id':run,
            'start_epoch':start,'deadline_epoch':deadline,'budget_seconds':budget,'original_generation':initial,
            'new_continuation_source_sha':h._git_sha(),'original_checkpoint_sha256':index['prior_generation_checkpoint_sha256']}
        (boot/'continuation-context.json').write_text(json.dumps(ctx))
        (boot/'file-pins.sha256').write_text(''.join(f'{digest(f)}  {f.relative_to(boot)}\n' for f in sorted(boot.rglob('*')) if f.is_file() and f.name!='file-pins.sha256'))
    else:
        ctx=prior
        assert ctx['start_epoch']==start and ctx['deadline_epoch']==deadline and budget==36000
    s=Path(h.__file__).read_text(); needle='expected_run = 44897382403 if combined else 44318145987'
    assert s.count(needle)==1
    s=s.replace(needle,'expected_run = approved["source_run_id"] if context == approved else (44897382403 if combined else 44318145987)')
    scope=dict(vars(h),approved=ctx,__name__='hosted_prod_continuation')
    exec(compile(s,h.__file__,'exec'),scope)
    assert time.time()<deadline
    scope['transition'](previous_bootstrap=boot,previous_result=out,out=where,budget_seconds=budget)
    pins(where)
    new=j(where/'checkpoint-index.json')
    q=torch.load(where/'checkpoint-g0.pt',map_location='cpu',weights_only=True)
    assert q['generation']==new['generation']==index['generation']+1 and q['step']==0 and digest(where/'checkpoint-g0.pt')==new['sha256']
if __name__=='__main__':
    a=sys.argv[2:]
    if sys.argv[1]=='plan':
        plan(*map(Path,a[:3]),int(a[3]),a[4])
    else:
        assert sys.argv[1]=='next'
        handoff(*map(Path,a[:3]),*map(int,a[3:]))
