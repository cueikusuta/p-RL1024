"""Verify an isolated exact learner-state chain, never modify production refs."""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

def j(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as q:return hashlib.file_digest(q,'sha256').hexdigest()
def inventory(root):
    expected={}
    for line in (root/'file-pins.sha256').read_text().splitlines():
        value,name=line.split('  ',1)
        p=root/name
        assert re.fullmatch('[0-9a-f]{64}',value) and name not in expected
        assert p.resolve().is_relative_to(root.resolve()) and p.is_file() and not p.is_symlink()
        assert sha(p)==value
        expected[name]=value
    assert set(expected)|{'file-pins.sha256'}=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    return expected
def verify(root,g,run):
    m=j(root/'metrics.json'); boot=root/'boot'; inv=inventory(boot)
    i=j(boot/'checkpoint-index.json'); o=j(root/'checkpoint-index.json')
    plan=j(boot/'generation-manifest.json'); ctx=j(boot/'continuation-context.json'); report=j(boot/'handoff-report.json')
    assert 512<=g<=514 and m['workflow_run']==run and m['generation_in']==g-1 and m['generation_out']==plan['generation']==g
    assert m['round']==g-511 and m['accepted']==m['audits']==1280 and m['updates']==40
    assert i['generation']==g and i['applied_matches']==i['completed_steps']==0 and i['is_champion'] is False
    assert i['sha256']==m['bootstrap_checkpoint_sha256']==inv['checkpoint-g0.pt']
    assert i['prior_generation_checkpoint_sha256']==o['full_checkpoint_sha256']==m['checkpoint_sha256']
    assert ctx['source_run_id']==run and ctx['new_budget_phase']=='Isolated chained benchmark' and ctx['budget_seconds']==900
    assert report['previous_generation']==g-1 and report['generation']==g
    assert all(report[k] is True for k in ('optimizer_full_state_preserved','target_full_state_preserved','magnet_full_state_preserved','rng_full_state_preserved'))
def next_state(boot,result,out,g,run,safe):
    from scripts.rnad import shared_league_handoff as handoff
    import torch
    from collections.abc import Mapping
    manifest=j(boot/'generation-manifest.json')
    assert manifest['generation']==g and 511<=g<=513
    old_interval=j(boot/'continuation-context.json')
    index=j(boot/'checkpoint-index.json')
    if g==511:
        assert old_interval.get('source_run_id')!=run
        started=int(time.time())
        approved=dict(schema=handoff.ARTIFACT_SCHEMA,new_budget_phase='Isolated chained benchmark',
            source_run_id=run,start_epoch=started,deadline_epoch=started+900,budget_seconds=900,
            original_generation=511,new_continuation_source_sha=handoff._git_sha(),
            original_checkpoint_sha256=index['prior_generation_checkpoint_sha256'])
        (boot/'continuation-context.json').write_text(json.dumps(approved))
        files=[x for x in boot.rglob('*') if x.is_file() and x.name!='file-pins.sha256']
        (boot/'file-pins.sha256').write_text(''.join(f'{sha(x)}  {x.relative_to(boot)}\n' for x in sorted(files)))
    else: approved=old_interval
    assert approved['source_run_id']==run and approved['new_budget_phase']=='Isolated chained benchmark'
    assert approved['original_generation']==511 and approved['deadline_epoch']-approved['start_epoch']==900 and time.time()<approved['deadline_epoch']
    inventory(boot)
    src=Path(handoff.__file__).read_text()
    needle='expected_run = 44897382403 if combined else 44318145987'
    assert src.count(needle)==1
    scope=dict(vars(handoff),approved=approved,__name__='isolated_trial_handoff')
    src=src.replace(needle,'expected_run = approved["source_run_id"] if context == approved else (44897382403 if combined else 44318145987)')
    exec(compile(src,handoff.__file__,'exec'),scope)
    scope['transition'](previous_bootstrap=boot,previous_result=result,out=out,budget_seconds=900)
    inv=inventory(out)
    base=torch.load(result/f'checkpoint-g{g+1}.pt',map_location='cpu',weights_only=True)
    nxt=torch.load(out/'checkpoint-g0.pt',map_location='cpu',weights_only=True)
    assert base['generation']==nxt['generation']==g+1 and base['step']==nxt['step']==0 and nxt['is_champion'] is False
    def equivalent(a,b):
        if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
        if isinstance(a,Mapping):return isinstance(b,Mapping) and a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equivalent(v,w) for v,w in zip(a,b))
        return a==b
    assert all(equivalent(base[k],nxt[k]) for k in ('model','target','magnet','optimizer','scheduler','torch_rng','python_rng','completed_generation_match_ids'))
    assert len(set(nxt['completed_generation_match_ids']))==1280
    ni=j(out/'checkpoint-index.json'); nxt_manifest=j(out/'generation-manifest.json')
    assert ni['sha256']==inv['checkpoint-g0.pt'] and nxt_manifest['generation']==g+1
    assert nxt_manifest['seed_start']>manifest['seed_start']
if sys.argv[1]=='verify':
    verify(Path(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
else:
    assert sys.argv[1]=='next'
    next_state(*map(Path,sys.argv[2:5]),int(sys.argv[5]),int(sys.argv[6]),Path(sys.argv[7]))
