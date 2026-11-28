import time,os,hashlib,json
from pathlib import Path
from collections import Counter
import orjson
import inspect
import math
import sys
import textwrap
import traceback
import torch
import torch.multiprocessing as mp
from scripts.rnad import shared_league_learner as l, shared_league_staging as st, shared_league_parallel as par
_W=None

class _CenteredForce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, centered, force):
        ctx.save_for_backward(force)
        ctx.n=logits.numel()
        return -(centered*force).sum()/ctx.n
    @staticmethod
    def backward(ctx, incoming):
        (force,)=ctx.saved_tensors
        # Same primitive sequence as div -> neg -> sum -> mul -> sub/mean.
        direct=(-(incoming/ctx.n))*force
        return direct+(-direct).sum()/ctx.n,None,None

def _patch_scalar(arm):
    from scripts.rnad import two_player
    if arm=='x86-neurd':
        original=two_player.neurd_policy_loss
        src=textwrap.dedent(inspect.getsource(original))
        old='return -(centered*force).sum()/logits.numel()'
        assert src.count(old)==1 and src.count('centered = logits-logits.mean()')==1
        src=src.replace(old,'return _CenteredForce.apply(logits,centered,force)')
        src=src.replace('centered = logits-logits.mean()','centered = logits.detach()-logits.detach().mean()')
        scope=dict(vars(two_player),_CenteredForce=_CenteredForce)
        exec(compile(src,two_player.__file__,'exec'),scope)
        new=scope[original.__name__]
        gen=torch.Generator().manual_seed(21437)
        for n in (1,2,5,16,31,32,60,127,514):
            for k in (.0,.8,1.):
                y=torch.randn(n,generator=gen)
                q=torch.randn(n,generator=gen)
                a=y.clone().requires_grad_();b=y.clone().requires_grad_()
                old_v=original(a,q,is_c=k);new_v=new(b,q,is_c=k)
                g=torch.tensor(.037)
                old_g=torch.autograd.grad(old_v,a,grad_outputs=g)[0]
                new_g=torch.autograd.grad(new_v,b,grad_outputs=g)[0]
                assert torch.equal(old_v,new_v) and torch.equal(old_g,new_g), 'synthetic gradient parity'
        l.neurd_policy_loss=new
    elif arm=='x86-scalar':
        original=two_player.rnad_targets
        src=textwrap.dedent(inspect.getsource(original))
        old='torch.isfinite(torch.tensor(v))'
        assert src.count(old)==1 and 'abs(v) < 1e6' in src
        scope=dict(vars(two_player),math=math)
        exec(compile(src.replace(old,'math.isfinite(v)'),two_player.__file__,'exec'),scope)
        new=scope[original.__name__]
        for val in (0.,1.,-1.,999999.9,-999999.9):
            seq=[dict(actor=i%2,reward=[0,0],pi=torch.tensor([.5,.5]),
                      magnet=torch.tensor([.5,.5]),value=val,behavior_prob=.5,action_index=0)
                 for i in range(3)]
            for a,b in zip(original(seq),new(seq),strict=True):
                assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) if torch.is_tensor(a[k]) else a[k]==b[k] for k in a)
        l.rnad_targets=new
    else:assert arm=='x86-control'

import random
from statistics import median

def fixture():
    rng=random.Random(771)
    gen=torch.Generator().manual_seed(973)
    data=[]
    for _ in range(14):
        n=rng.choice((3,8,13,30,60,128,512))
        y=torch.randn(n,generator=gen)
        a=torch.softmax(torch.randn(n,generator=gen),0)
        p=torch.softmax(torch.randn(n,generator=gen),0)
        data.append((y,a,p))
    return data

def compute(policy,targets,records,k):
    count=0
    for _ in range(k):
        pairs=[]; seq=[]
        for i,(y,a,p) in enumerate(records):
            z=y.clone().requires_grad_()
            pairs.append((z,policy(z,y,is_c=.81)))
            seq.append(dict(actor=i%2,reward=[0.,0.],pi=a,magnet=p,
                       value=.125,behavior_prob=float(a[0]),action_index=0))
        # Preserve the external gradient scale while timing just isolated kernels.
        torch.autograd.grad(sum(v for _,v in pairs)/32,tuple(z for z,_ in pairs))
        targets(seq)
        count+=len(pairs)
    return count

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    base=(l.neurd_policy_loss,l.rnad_targets)
    _patch_scalar('x86-neurd');neurd=(l.neurd_policy_loss,base[1])
    _patch_scalar('x86-scalar');scalar=(base[0],l.rnad_targets)
    data=fixture()
    arms={'control':base,'analytic':neurd,'finite-scalar':scalar}
    measurements={k:[] for k in arms}
    for name,(p,t) in arms.items():compute(p,t,data,5)
    for loop in range(9):
        order=list(arms)
        order=order[loop%3:]+order[:loop%3]
        for name in order:
            start=time.monotonic_ns()
            n=compute(*arms[name],data,96)
            measurements[name].append((time.monotonic_ns()-start)/n/1000)
    print(json.dumps({'synthesized_inputs_only':True,'exact_unit_tests':True,'repeats':9,
                      'median_us_per_event':{k:round(median(v),4) for k,v in measurements.items()}},sort_keys=True))
if __name__=='__main__':main()
