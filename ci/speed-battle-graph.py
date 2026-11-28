"""Exact CPU forward/backward tests on generated battle inputs only."""
import copy,json,time,torch
from statistics import median
from scripts.rnad.neural import RNADNetwork,STATE_FEATURES_V6,ACTION_FEATURES_V6

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(9381)
model=RNADNetwork(state_features=STATE_FEATURES_V6,action_features=ACTION_FEATURES_V6)
gen=torch.Generator().manual_seed(7917)
def sample(n):
    return (torch.randn((1,model.state_features),generator=gen),
            torch.randn((1,n,model.action_features),generator=gen),
            torch.randn((1,model.hidden_size),generator=gen))
inputs=[sample(i) for i in (1,2,3,5,7,13,17,26,31,63,80)]
base=model
arm_trace=copy.deepcopy(base)
arm_script=copy.deepcopy(base)
methods={'eager':(base,base),'trace':(arm_trace,torch.jit.trace(arm_trace,inputs[3],check_trace=False)),
         'script':(arm_script,torch.jit.script(arm_script))}
def run(m,fn,data):
    o=fn(*data)
    loss=sum(t.square().sum() for t in o)
    gs=torch.autograd.grad(loss,tuple(m.parameters()))
    return o,gs
proof={}
snapshot={k:v.detach().clone() for k,v in base.state_dict().items()}
for key,(m,fn) in methods.items():
    base.load_state_dict(snapshot)
    if m is not base:m.load_state_dict(snapshot)
    visibility=[]
    for data in inputs:
        o,g=run(m,fn,data);aa,bb=run(base,base,data)
        visibility.append(all(torch.equal(a,b) for a,b in zip(o,aa,strict=True)) and
                          all(torch.equal(a,b) for a,b in zip(g,bb,strict=True)))
    # Prove the compiled path observes later in-place weight updates.
    with torch.no_grad():
        for p in m.parameters():p.add_(.015625)
        if m is not base:
            for p in base.parameters():p.add_(.015625)
    o,g=run(m,fn,inputs[6]);aa,bb=run(base,base,inputs[6])
    visibility.append(all(torch.equal(a,b) for a,b in zip(o,aa,strict=True)) and
                      all(torch.equal(a,b) for a,b in zip(g,bb,strict=True)))
    base.load_state_dict(snapshot)
    if m is not base:m.load_state_dict(snapshot)
    proof[key]=all(visibility)
def measure(fn,n=40):
    times=[]
    for i in range(9):
        start=time.monotonic_ns()
        for _ in range(n):
            for row in inputs:fn(row)
        times.append((time.monotonic_ns()-start)/(len(inputs)*n*1000))
    return round(median(times),4)
scores={}
for key,(m,fn) in methods.items():
    measure(lambda row:run(m,fn,row),n=2)
    scores[key]=measure(lambda row:run(m,fn,row))
print(json.dumps(dict(exact=proof,median_forward_backward_us=scores,synthetic_only=True),sort_keys=True))
