"""CPU contention trial on generated, non-project inputs."""
import gzip,hashlib,json,os,time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from statistics import median

def audit_init(nice,affinity):
    if affinity:os.sched_setaffinity(0,affinity)
    if nice:os.nice(nice)
    global blob,want
    payload=json.dumps([[(i*29+j*13)%height for j in range(384)]
                        for i,height in enumerate(range(49,209))],separators=(',',':')).encode()
    blob=gzip.compress(payload);want=hashlib.sha256(payload).digest()
def audit(i):
    for _ in range(8):
        plain=gzip.decompress(blob)
        assert hashlib.sha256(plain).digest()==want
        v=json.loads(plain)
        assert len(v)==160 and all(len(row)==384 for row in v)
    return i
def learn_init(affinity):
    if affinity:os.sched_setaffinity(0,affinity)
    global torch,p,x,y
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    seed=torch.Generator().manual_seed(94143)
    p=[torch.randn(128,128,generator=seed,requires_grad=True) for _ in range(3)]
    x=torch.randn(1,128,generator=seed);y=torch.randn(13,128,generator=seed)
def grad(i,n):
    for _ in range(n):
        a=(x@p[0]).relu();b=(a@p[1]).tanh()
        logit=(b+y)@p[2]
        result=torch.autograd.grad(logit.square().mean(),p)
    return i,[float(row.flatten()[0]) for row in result]
def bench(a,g,iterations):
    tick=time.monotonic()
    aud=[a.submit(audit,i) for i in range(20)]
    for f in aud[:3]:f.result()
    ready=time.monotonic()
    grad_jobs=[g.submit(grad,i,iterations) for i in range(32)]
    for f in aud:f.result()
    audit_end=time.monotonic()
    out=[f.result() for f in grad_jobs]
    return out,[(x-tick)*1000 for x in (ready,audit_end,time.monotonic())]
def main():
    all_cpus=sorted(os.sched_getaffinity(0))
    assert len(all_cpus)>=2
    four=min(4,len(all_cpus))
    modes={'current':(four,3,0,[],[]),'two_auditors':(four,2,0,[],[]),
           'lower_audit_priority':(four,3,5,[],[]),
           'separate_cpu_sets':(four-1,1,0,all_cpus[:four-1],all_cpus[four-1:four])}
    exact=None;measure={}
    for name,(nw,aw,nice,lset,aset) in modes.items():
        with ProcessPoolExecutor(nw,mp_context=get_context('spawn'),initializer=learn_init,initargs=(lset,)) as g, ProcessPoolExecutor(aw,initializer=audit_init,initargs=(nice,aset)) as a:
            bench(a,g,42)
            measure[name]={}
            for label,iters in (('audit_heavy',42),('learner_heavy',170)):
                scores=[]
                for _ in range(5):
                    out,stats=bench(a,g,iters)
                    if exact is None:exact=out
                    assert out==exact
                    scores.append(stats)
                measure[name][label]=dict(zip(('first_ready_ms','audit_complete_ms','complete_ms'),
                              [median(x[i] for x in scores) for i in range(3)],strict=True))
    print('Synthetic CPU budget timing: '+json.dumps(dict(synthetic_only=True,exact=True,cpus=len(all_cpus),modes=measure),sort_keys=True))
if __name__=='__main__':main()
