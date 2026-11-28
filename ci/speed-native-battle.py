"""Generated matches only: compare pinned eager versus native ATen execution."""
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


from pathlib import Path
from torch.utils.cpp_extension import load
load(name="speed_native_battle_ext",sources=[str(Path(__file__).with_name("speed-native-battle.cpp"))],
     extra_cflags=["-O3","-g0"],is_python_module=False,verbose=False)
def bind_native(net,reuse):
    b=net.battle
    native=b.forward
    params=(*b.state_proj.parameters(), *b.gru.parameters(), *b.policy_hidden.parameters(),
            *b.policy_out.parameters(), *b.value_hidden.parameters(), *b.value_out.parameters())
    assert len(params)==14
    op=torch.ops.speed_native_battle.forward
    def forward(state,candidates,memory):
        if state.ndim!=2 or state.shape[-1]!=b.state_features or candidates.ndim!=3 or candidates.shape[0]!=state.shape[0] or candidates.shape[-1]!=b.action_features or candidates.shape[1]<1:
            return native(state,candidates,memory)
        return op(state,candidates,memory,params,reuse)
    b.forward=forward
    return net
arms={'eager':(base,target,magnet)}
for name,reuse in (('native',False),('native_reuse',True)):
    arms[name]=tuple(bind_native(copy.deepcopy(net),reuse) for net in (base,target,magnet))

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
    lengths=(2,3,5,8,4,7,2,3,6,9,2,4)
    for j,n in enumerate(lengths):
        event('battle',j%2,dict(state=normal(384).tolist(),actions=normal(n,160).tolist()),1. if j==len(lengths)-1 else 0.)
    return dict(events=events,learner_seat=seed%2,winner=1)

fixtures=[(fixture(i),'self' if i%2 else 'fixed') for i in range(1,5)]
def step(group,record,role):
    main,tar,mag=group
    loss,ratio=learner._match_loss(main,tar,mag,record,role=role)
    grads=torch.autograd.grad(loss,tuple(main.parameters()),allow_unused=True)
    return loss.detach().reshape(1),ratio,grads
def same(a,b):
    if a is None or b is None:return a is b
    return torch.equal(a.contiguous().reshape(-1).view(torch.int32),b.contiguous().reshape(-1).view(torch.int32))
exact={name:True for name in arms}
for phase in (0,1):
    if phase:
        with torch.no_grad():
            # Prove that native call sees same-shape in-place Main and Target updates.
            for main,tar,_ in arms.values():
                for net in (main,tar):
                    for param in net.parameters():param.add_(.0009765625)
    for row,role in fixtures:
        la,ra,ga=step(arms['eager'],row,role)
        for name,group in arms.items():
            lb,rb,gb=step(group,row,role)
            exact[name] &= same(la,lb) and ra==rb and all(same(x,y) for x,y in zip(ga,gb,strict=True))

def measure(group):
    t=time.perf_counter_ns()
    for _ in range(4):
        for row,role in fixtures:step(group,row,role)
    return (time.perf_counter_ns()-t)/(4*len(fixtures)*1000)
samples={k:[] for k in arms};names=tuple(arms)
for group in arms.values():measure(group)
for i in range(9):
    for k in names[i%len(names):]+names[:i%len(names)]:samples[k].append(measure(arms[k]))
print(json.dumps(dict(synthetic_only=True,exact=exact,median_us={k:median(v) for k,v in samples.items()},fixtures=len(fixtures)),sort_keys=True))
