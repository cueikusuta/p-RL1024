import time,os,hashlib,json
from pathlib import Path
from collections import Counter
import orjson
import torch
import torch.multiprocessing as mp
from scripts.rnad import shared_league_learner as l, shared_league_staging as st, shared_league_parallel as par
_W=None
_SELECT_NATIVE=None
def init(model,target,magnet,flat,active,spans):
    global _W,_SELECT_NATIVE
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert os.environ['LEAGUE_LEARNER_ARM']=='x86-warm-native'
    from speed_prefetch_runtime import initialize
    _SELECT_NATIVE=initialize(model,target,magnet,l)
    _W=model,target,magnet,flat,active,spans

def gradient(job):
    i,path,role,sha,pair=job
    model,target,magnet,flat,active,spans=_W
    tick=time.monotonic_ns()
    b=Path(path).read_bytes()
    if hashlib.sha256(b).hexdigest()!=sha:raise ValueError('Receipt changed')
    record=orjson.loads(b)
    parsed=time.monotonic_ns()
    original_inputs=l._inputs
    def run(native):
        _SELECT_NATIVE(native)
        model.zero_grad(set_to_none=True)
        start=time.monotonic_ns()
        if native:
            from speed_warm_pack import packed
            l._inputs=packed(record,role,original_inputs)
        try:loss,ratio=l._match_loss(model,target,magnet,record,role=role)
        finally:l._inputs=original_inputs
        forward=time.monotonic_ns()
        (loss/32).backward()
        end=time.monotonic_ns()
        return loss.detach(),ratio,forward-start,end-forward
    comparison=None
    if pair:
        if i%2:
            fast=run(True)
            candidate=[p.grad.clone() if p.grad is not None else None for p in model.parameters()]
            control=run(False)
            base=[p.grad for p in model.parameters()]
        else:
            control=run(False)
            base=[p.grad.clone() if p.grad is not None else None for p in model.parameters()]
            fast=run(True)
            candidate=[p.grad for p in model.parameters()]
        def bits(x,y):
            if x is None or y is None:return x is y
            return torch.equal(x.contiguous().reshape(-1).view(torch.int32),y.contiguous().reshape(-1).view(torch.int32))
        assert bits(control[0],fast[0]) and control[1]==fast[1] and all(bits(x,y) for x,y in zip(base,candidate,strict=True)), 'Paired learner arithmetic differs'
        comparison=[control[2]+control[3],fast[2]+fast[3],0]
    else:
        fast=run(True)
    _SELECT_NATIVE(False)
    loss,ratio,fw,bw=fast
    copying=time.monotonic_ns()
    source=candidate if pair else (p.grad for p in model.parameters())
    for j,(g,(x,y)) in enumerate(zip(source,spans,strict=True)):
        active[i,j]=g is not None
        if g is not None:flat[i,x:y].copy_(g.reshape(-1))
    return i,float(loss),ratio,(parsed-tick,fw,bw,time.monotonic_ns()-copying),comparison
def train(root,where,broker,manifest,rows):
    src=Path(l.__file__).read_text()
    profile=[]
    pairs=[]
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
            self.step=0
        def step_gradients(self,rows):
            assert len(rows)==32 and all(not c for _,_,c in rows)
            begun=time.monotonic_ns()
            t=broker.claim(rows)
            released=time.monotonic_ns()
            if self.n==2 and broker.vm_actors_done:
                self.pool.close();self.pool.join()
                self.n=min(8,len(os.sched_getaffinity(0)));self.pool=self.spawn(self.n)
            self.step+=1
            audited=self.step in (1,16,32)
            r=sorted(self.pool.map(gradient,[(i,p,role,sha,audited and i in (0,9))
                                             for i,p,role,sha in t],chunksize=1))
            pairs.extend(row[4] for row in r if row[4] is not None)
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
            profile.append(([released-begun,mapped-released,time.monotonic_ns()-mapped], [sum(row[3][i] for row in r) for i in range(4)]))
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
    assert len(pairs)==6
    from statistics import median
    cpu=next((x.partition(':')[2].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), 'unknown')
    (where/'paired-summary.json').write_text(json.dumps(dict(cases=len(pairs),nonexact=sum(row[2]!=0 for row in pairs),
       median_ratio=median(row[1]/row[0] for row in pairs),
       medians_ms=[median(row[i] for row in pairs)/1e6 for i in (0,1)],cpu=cpu,
       capability=torch.backends.cpu.get_cpu_capability())))
    (where/'learner-profile.json').write_text(json.dumps(profile))
    return z
