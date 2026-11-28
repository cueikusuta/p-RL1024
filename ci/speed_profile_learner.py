import time,os,hashlib,json
from pathlib import Path
from collections import Counter
import orjson
import inspect
import textwrap
import numpy as np
import torch
import torch.multiprocessing as mp
from scripts.rnad import shared_league_learner as l, shared_league_staging as st, shared_league_parallel as par
_W=None
_LAST_CODES=None
_LAST_VERSION=None
_CHECKS=0
_CHECK_NS=0

def _check_unique_draft(codes):
    global _LAST_CODES,_LAST_VERSION,_CHECKS,_CHECK_NS
    stamp=time.monotonic_ns()
    version=codes._version
    # The three policy networks receive the same input tensor in one event.
    # Check once per object/version, after the unchanged shape/type/bounds checks.
    if codes is _LAST_CODES and version==_LAST_VERSION:
        return True
    arr=codes.detach().numpy()
    ordered=np.sort(arr,axis=1)
    valid=bool(np.all(ordered[:,1:] != ordered[:,:-1]))
    if valid:
        _LAST_CODES=codes
        _LAST_VERSION=version
    _CHECKS+=1
    _CHECK_NS+=time.monotonic_ns()-stamp
    return valid

def _install_draft_validator():
    from scripts.rnad import integrated_policy_v1 as policy
    cls=policy.IntegratedPolicyV1
    src=textwrap.dedent(inspect.getsource(cls.forward))
    old='all(len(set(row)) == 6 for row in codes.tolist())'
    assert src.count(old)==1
    # Replace only how uniqueness is checked; no numeric kernels or outputs change.
    scope=dict(vars(policy),_check_unique_draft=_check_unique_draft)
    patched=src.replace(old,'_check_unique_draft(codes)')
    exec(compile(patched,policy.__file__,'exec'),scope)
    sample=torch.tensor([[0,1,2,3,4,5],[9,8,7,6,5,4]],dtype=torch.long)
    assert _check_unique_draft(sample)
    assert _check_unique_draft(sample)
    sample[0,0]=1
    assert not _check_unique_draft(sample)
    cls.forward=scope['forward']
def init(model,target,magnet,flat,active,spans):
    global _W
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _install_draft_validator()
    _W=model,target,magnet,flat,active,spans
def gradient(job):
    global _CHECKS,_CHECK_NS,_LAST_CODES,_LAST_VERSION
    _CHECKS=_CHECK_NS=0
    _LAST_CODES=_LAST_VERSION=None
    i,path,role,sha=job
    model,target,magnet,flat,active,spans=_W
    tick=time.monotonic_ns()
    b=Path(path).read_bytes()
    if hashlib.sha256(b).hexdigest()!=sha:raise ValueError('Receipt changed')
    record=orjson.loads(b)
    parsed=time.monotonic_ns()
    model.zero_grad(set_to_none=True)
    loss,ratio=l._match_loss(model,target,magnet,record,role=role)
    forward=time.monotonic_ns()
    (loss/32).backward()
    backward=time.monotonic_ns()
    for j,(p,(x,y)) in enumerate(zip(model.parameters(),spans,strict=True)):
        active[i,j]=p.grad is not None
        if p.grad is not None:flat[i,x:y].copy_(p.grad.reshape(-1))
    return i,float(loss.detach()),ratio,((parsed-tick),(forward-parsed),(backward-forward),time.monotonic_ns()-backward,_CHECK_NS,_CHECKS)
def train(root,where,broker,manifest,rows):
    src=Path(l.__file__).read_text()
    profile=[]
    def rep(a,b):
        nonlocal src
        assert src.count(a)==1,a
        src=src.replace(a,b)
    rep('    if any(not x.is_file() for x in paths):\n        raise ValueError("Cannot train on incomplete 20/20 Actor artifact set")',
        '    assert Path(actors).samefile(scratch / "actors")')
    rep('            _atomic_checkpoint(checkpoint_path,ckpt)','            checked_step(checkpoint_path,ckpt)')
    def checked_step(p,ckpt):
        n=ckpt['step']
        broker.completed(n)
        if n==40:broker.wait_for_barrier()
        if n%8==0:l._atomic_checkpoint(p,ckpt)
    roles=Counter(row['role'] for row in rows)
    roles={k:v//64 for k,v in roles.items()}
    def staged(m,paths,verified,**kw):
        assert m==manifest and Path(verified).samefile(where/'actors/_strict_verified_matches')
        return {r['match_id'] for r in rows},rows,roles
    class ExactPool:
        def __init__(self,model,target,magnet,*,workers,vector):
            assert workers==8 and not vector
            self.params=list(model.parameters());spans=[];n=0
            for p in self.params:
                assert p.dtype==torch.float32 and p.is_contiguous()
                spans.append((n,n+p.numel()));n+=p.numel()
            self.spans=spans
            self.flat=torch.empty(32,n).share_memory_()
            self.active=torch.empty(32,len(spans),dtype=torch.bool).share_memory_()
            for m in (model,target,magnet):m.share_memory()
            self.spawn=lambda n:mp.get_context('spawn').Pool(n,initializer=init,initargs=(model,target,magnet,self.flat,self.active,spans))
            self.n=min(8,len(os.sched_getaffinity(0))) if getattr(broker,'vm_actors_done',True) else 2
            self.pool=self.spawn(self.n)
        def step_gradients(self,rows):
            assert len(rows)==32 and all(not c for _,_,c in rows)
            begun=time.monotonic_ns()
            t=broker.claim(rows)
            released=time.monotonic_ns()
            if self.n==2 and broker.vm_actors_done:
                self.pool.close();self.pool.join()
                self.n=min(8,len(os.sched_getaffinity(0)));self.pool=self.spawn(self.n)
            r=sorted(self.pool.map(gradient,t,chunksize=1))
            mapped=time.monotonic_ns()
            assert [row[0] for row in r]==list(range(32))
            for i,flags in enumerate(self.active.tolist()):
                a=[];b=[]
                for enabled,p,(x,y) in zip(flags,self.params,self.spans,strict=True):
                    if not enabled:continue
                    g=self.flat[i,x:y].view_as(p)
                    if p.grad is None:p.grad=g.clone()
                    else:a.append(p.grad);b.append(g)
                if a:torch._foreach_add_(a,b)
            profile.append(([released-begun,mapped-released,time.monotonic_ns()-mapped], [sum(row[3][i] for row in r) for i in range(6)]))
            return [x[1] for x in r],[x[2] for x in r]
        def __enter__(self):return self
        def __exit__(self,k,v,t):
            if k:self.pool.terminate()
            else:self.pool.close()
            self.pool.join()
    st.stage_verified_actors=staged
    par.configured_workers=lambda:8
    par.ParallelGradientPool=ExactPool
    s=dict(vars(l),scratch=where,checked_step=checked_step,__name__='scratch_lowmem')
    exec(compile(src,l.__file__,'exec'),s)
    z=s['learn'](bootstrap=root,actors=where/'actors',out=where/'output',
                 original_draft=root/'original-draft.json',original_selection_js=root/'original-selection-js.json',max_steps=40)
    assert z['optimizer_updates_completed']==40 and len(profile)==40
    (where/'learner-profile.json').write_text(json.dumps(profile))
    return z
