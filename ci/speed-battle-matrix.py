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

from types import MethodType
import torch.nn.functional as F

class OneRowLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        return F.linear(x,weight,bias)
    @staticmethod
    def backward(ctx, dy):
        x,w=ctx.saved_tensors
        dy2=dy.reshape(1,-1)
        dx=dy2.mm(w).reshape_as(x) if ctx.needs_input_grad[0] else None
        dw=dy2.t()*x.reshape(1,-1) if ctx.needs_input_grad[1] else None
        db=dy2.sum(0) if ctx.needs_input_grad[2] else None
        return dx,dw,db

def one_row(self,x):
    return OneRowLinear.apply(x,self.weight,self.bias) if torch.is_grad_enabled() and x.numel()==self.in_features else F.linear(x,self.weight,self.bias)

def split(self,state,candidates,memory):
    assert state.ndim==2 and state.shape[-1]==self.state_features
    assert candidates.ndim==3 and candidates.shape[0]==state.shape[0]
    assert candidates.shape[-1]==self.action_features and candidates.shape[1]>=1
    memory=self.gru(torch.relu(self.state_proj(state)),memory)
    weights=self.policy_hidden.weight
    selected=F.linear(candidates,weights[:,self.hidden_size:],self.policy_hidden.bias)
    context=F.linear(memory,weights[:,:self.hidden_size]).unsqueeze(1)
    logits=self.policy_out(torch.relu(selected+context)).squeeze(-1)
    value=torch.tanh(self.value_out(torch.relu(self.value_hidden(memory)))).squeeze(-1)
    return logits,value,memory

arms={'baseline':model,'one_row':copy.deepcopy(model),'split':copy.deepcopy(model)}
for m in arms['one_row'].modules():
    if isinstance(m,torch.nn.Linear):m.forward=MethodType(one_row,m)
arms['split'].forward=MethodType(split,arms['split'])

def test_step(m,rows,training):
    with torch.set_grad_enabled(training):
        memory=rows[0][2].clone().requires_grad_(training)
        starting_memory=memory
        outs=[]
        for state,candidates,_ in rows:
            o,v,memory=m(state,candidates,memory)
            outs.extend((o,v,memory))
        if training:
            loss=sum(t.square().sum() for t in outs)
            outs.extend(torch.autograd.grad(loss,(starting_memory,*m.parameters())))
        return outs

sparse_inputs=[(s.masked_fill(s.abs()<1.25,0), a.masked_fill(a.abs()<1.25,0),h) for s,a,h in inputs]
fixtures=[(r,) for r in inputs+sparse_inputs]+[tuple(inputs[i:i+4]) for i in (0,3,6)] + [tuple(sparse_inputs[i:i+4]) for i in (0,3,6)]
def compare(m,training):
    exact=True
    error=0.
    for weight_shift in (0.,.015625):
        if weight_shift:
            with torch.no_grad():
                for alt in arms.values():
                    for param in alt.parameters():param.add_(weight_shift)
        try:
            for rows in fixtures:
                a=test_step(model,rows,training)
                b=test_step(m,rows,training)
                for x,y in zip(a,b,strict=True):
                    if exact:exact=torch.equal(x.contiguous().view(torch.int32),y.contiguous().view(torch.int32))
                    error=max(error,float((x.detach()-y.detach()).abs().max()))
        finally:
            if weight_shift:
                for alt in arms.values():alt.load_state_dict(initial)
    return dict(bit_exact=exact,max_abs_error=error)
initial={k:v.detach().clone() for k,v in model.state_dict().items()}
perf_fixtures=[(r,) for r in inputs[:7]] + [tuple(sparse_inputs[:4]), tuple(sparse_inputs[3:7])]
def measure(m,training,n):
    t=time.perf_counter_ns()
    for _ in range(n):
        for row in perf_fixtures:test_step(m,row,training)
    return (time.perf_counter_ns()-t)/(n*sum(len(r) for r in perf_fixtures)*1000)
report={}
for training in (False,True):
    result={k:compare(m,training) for k,m in arms.items()}
    times={k:[] for k in arms}; names=tuple(arms)
    for m in arms.values():measure(m,training,2)
    for i in range(9):
        for k in names[i%3:]+names[:i%3]:
            times[k].append(measure(arms[k],training,26))
    for k,t in times.items():result[k]['median_us_per_event']=round(median(t),4)
    report['train' if training else 'inference']=result
print(json.dumps(dict(synthetic_only=True,results=report),sort_keys=True))
