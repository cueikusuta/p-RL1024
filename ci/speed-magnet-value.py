"""Generated matches only; compare the pinned learner and unchanged audited outputs."""
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
fast=copy.deepcopy(magnet)

class UnusedValue(torch.nn.Module):
    """For the frozen Magnet only. Preserve the original finite-value gate."""
    def __init__(self, native, counted=None):
        super().__init__()
        self.native=native
        assert not any(p.requires_grad for p in native.parameters())
        self.pins=tuple((p,p._version) for p in native.parameters())
        assert max(float(p.abs().max()) for p,_ in self.pins)<=1024
        self.counted=counted
    def forward(self, inputs):
        # Width <= 128, two bounded float32 linear layers, ReLU and tanh:
        # abs accumulator < 128*1024*(128*1024*1024+1024)+1024 < 2**45.
        # Fall back if the bound, identity or layout no longer holds.
        if (torch.is_grad_enabled() or inputs.dtype!=torch.float32
                or inputs.shape[-1]>128 or any(p._version!=v for p,v in self.pins)
                or not bool(inputs.abs().amax()<=1024)):
            return self.native(inputs)
        if self.counted is not None:self.counted[0]+=1
        return inputs.new_zeros((*inputs.shape[:-1],1))

import textwrap
def install(m):
    for name in ('draft_value','selection_value'):
        setattr(m,name,UnusedValue(getattr(m,name),hits))
    obj=m.battle
    native=torch.nn.Sequential(obj.value_hidden,torch.nn.ReLU(),obj.value_out,torch.nn.Tanh())
    obj.spare_value=UnusedValue(native,hits)
    source=textwrap.dedent(inspect.getsource(type(obj).forward))
    prev='value = torch.tanh(self.value_out(torch.relu(\n        self.value_hidden(next_memory)\n    ))).squeeze(-1)'
    assert source.count(prev)==1, 'matched native value syntax'
    ns={'torch':torch}
    exec(compile(source.replace(prev,'value = self.spare_value(next_memory).squeeze(-1)'),'<synthetic-frozen-magnet>','exec'),ns)
    obj.forward=types.MethodType(ns['forward'],obj)
hits=[0]
install(fast)

@torch.no_grad()
def fixture(seed):
    memories=[magnet.initial_memory(),magnet.initial_memory()]
    events=[]
    def event(phase,seat,p,reward=0.):
        obs,acts=learner._inputs(phase,p)
        logit,_,mem=magnet(phase,obs,acts,memories[seat])
        if phase=='battle':memories[seat]=mem
        pi=torch.softmax(logit,-1).tolist()
        events.append(dict(phase=phase,actor=seat,phase_input=p,reward=reward,
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
    lengths=(2,3,5,8,4,7,2,3,6,9,2,4)
    for j,n in enumerate(lengths):
        event('battle',j%2,dict(state=normal(384).tolist(),actions=normal(n,160).tolist()),1. if j==len(lengths)-1 else 0.)
    return dict(events=events,learner_seat=seed%2,winner=1)

fixtures=[(fixture(i),'self' if i%2 else 'fixed') for i in range(1,5)]
def step(ref,record,role):
    loss,ratio=learner._match_loss(base,target,ref,record,role=role)
    grads=torch.autograd.grad(loss,tuple(base.parameters()),allow_unused=True)
    return loss.detach().reshape(1),ratio,grads
def same(a,b):
    if a is None or b is None:return a is b
    return torch.equal(a.contiguous().reshape(-1).view(torch.int32),b.contiguous().reshape(-1).view(torch.int32))
for row,role in fixtures:
    la,ra,ga=step(magnet,row,role);lb,rb,gb=step(fast,row,role)
    assert same(la,lb) and ra==rb and all(same(x,y) for x,y in zip(ga,gb,strict=True))
assert hits[0]>0
def measure(ref):
    t=time.perf_counter_ns()
    for _ in range(4):
        for row,role in fixtures:step(ref,row,role)
    return (time.perf_counter_ns()-t)/16000
samples=[[],[]]
for i in range(11):
    for j in (i%2,1-i%2):samples[j].append(measure((magnet,fast)[j]))
print(json.dumps(dict(synthetic_only=True,bit_exact=True,baseline_us=median(samples[0]),
                      guarded_us=median(samples[1]),fixtures=len(fixtures)),sort_keys=True))
