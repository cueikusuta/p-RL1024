"""Generated matches only: compare exact Battle observation packing strategies."""
import copy,inspect,json,time,types,torch
from statistics import median
from scripts.rnad import shared_league_learner as learner
from scripts.rnad.integrated_policy_v1 import IntegratedPolicyV1, FEATURE_WIDTH
from scripts.rnad.neural import THREE_PHASE_FEATURE_SCHEMA_V6

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(8350)
rng=torch.Generator().manual_seed(4541)
def normal(*shape):return torch.randn(shape,generator=rng)
def make_model():
    return IntegratedPolicyV1({'sizes':{k:8 for k in ('species','names','abilities','items','moves','types')}},
                              battle_schema=THREE_PHASE_FEATURE_SCHEMA_V6)
base=make_model()
target=make_model().eval().requires_grad_(False)
magnet=make_model().eval().requires_grad_(False)


@torch.no_grad()
def fixture(seed):
    memories=[magnet.initial_memory(),magnet.initial_memory()]
    events=[]
    def event(phase,seat,p,reward=0.):
        obs,acts=learner._inputs(phase,p)
        logit,_,mem=magnet(phase,obs,acts,memories[seat])
        if phase=='battle':memories[seat]=mem
        pi=torch.softmax(logit,-1).tolist()
        events.append(dict(phase=phase,actor=seat,phase_input=p,reward=(-reward,reward),
            behavior_prob=pi[0],action_index=0,magnet_pi=pi,policy_pi=pi,predicted_value=0.))
    for seat in (0,1):
        count=20+seed*2
        pool=normal(12,FEATURE_WIDTH)
        extra=normal(count,4)
        codes=torch.stack([torch.randperm(12,generator=rng)[:6] for _ in range(count)])
        event('draft',seat,dict(pool=pool.tolist(),extra=extra.tolist(),codes=codes.tolist()))
    for seat in (0,1):
        spec={'own_cat':torch.zeros(6,9,dtype=torch.long),'foe_cat':torch.zeros(6,3,dtype=torch.long),
              'own_num':normal(6,16),'foe_num':normal(6,6)}
        event('selection',seat,{k:v.tolist() for k,v in spec.items()})
    lengths=(2,3,5,8,4,7,2,3,6,9,2,4,13,22,26,48)*(1+seed%3)
    for j,n in enumerate(lengths):
        event('battle',j%2,dict(state=normal(384).tolist(),actions=normal(n,160).tolist()),1. if j==len(lengths)-1 else 0.)
    return dict(events=events,learner_seat=seed%2,winner=1)

fixtures=[(fixture(i),'self' if i%2 else 'fixed') for i in range(1,5)]

import numpy as np
original_input=learner._inputs
def numpy_input(phase,raw):
    if phase!='battle':return original_input(phase,raw)
    return {"state":torch.from_numpy(np.asarray([raw["state"]],dtype=np.float32))},torch.from_numpy(np.asarray([raw["actions"]],dtype=np.float32))

def packed_input(record,role,converter):
    permitted={0,1} if role=='self' else {record['learner_seat']}
    raw=[e['phase_input'] for e in record['events'] if e['phase']=='battle' and e['actor'] in permitted]
    if not raw:return original_input
    states=converter([item['state'] for item in raw])
    items=[act for item in raw for act in item['actions']]
    actions=converter(items)
    out={}
    offset=0
    for i,item in enumerate(raw):
        n=len(item["actions"])
        out[id(item)]=({"state":states[i:i+1]},actions[offset:offset+n].unsqueeze(0))
        offset+=n
    assert offset==actions.shape[0]
    def take(phase,item):
        if phase!='battle': return original_input(phase,item)
        return out[id(item)]
    return take

methods=('eager','numpy_per_event','torch_packed','numpy_packed')
def step(arm,record,role):
    if arm=='numpy_per_event':learner._inputs=numpy_input
    elif arm=='torch_packed':learner._inputs=packed_input(record,role,lambda v:torch.tensor(v,dtype=torch.float32))
    elif arm=='numpy_packed':learner._inputs=packed_input(record,role,lambda v:torch.from_numpy(np.asarray(v,dtype=np.float32)))
    try:
        loss,ratio=learner._match_loss(base,target,magnet,record,role=role)
        grads=torch.autograd.grad(loss,tuple(base.parameters()),allow_unused=True)
        return loss.detach().reshape(1),ratio,grads
    finally:learner._inputs=original_input
def same(a,b):
    if a is None or b is None:return a is b
    return torch.equal(a.contiguous().reshape(-1).view(torch.int32),b.contiguous().reshape(-1).view(torch.int32))
exact={name:True for name in methods}
for row,role in fixtures:
    only={0,1} if role=='self' else {row['learner_seat']}
    pack_t=packed_input(row,role,lambda v:torch.tensor(v,dtype=torch.float32))
    pack_n=packed_input(row,role,lambda v:torch.from_numpy(np.asarray(v,dtype=np.float32)))
    for item in row['events']:
        if item['phase']!='battle' or item['actor'] not in only:continue
        inputs=item['phase_input']; base_state,base_actions=original_input('battle',inputs)
        for name,fn in zip(methods[1:],(numpy_input,pack_t,pack_n),strict=True):
            state,acts=fn('battle',inputs)
            exact[name] &= same(state['state'],base_state['state']) and same(acts,base_actions)

    la,ra,ga=step('eager',row,role)
    for name in methods:
        lb,rb,gb=step(name,row,role)
        exact[name] &= same(la,lb) and ra==rb and all(same(x,y) for x,y in zip(ga,gb,strict=True))
def measure(name):
    t=time.perf_counter_ns()
    for _ in range(3):
        for row,role in fixtures:step(name,row,role)
    return (time.perf_counter_ns()-t)/(3*len(fixtures)*1000)
samples={k:[] for k in methods}
for name in methods:measure(name)
for i in range(9):
    k=i%len(methods)
    for name in methods[k:]+methods[:k]:samples[name].append(measure(name))
print(json.dumps(dict(synthetic_only=True,exact=exact,median_us={k:median(v) for k,v in samples.items()},fixtures=len(fixtures)),sort_keys=True))
