"""Compare official standard runner backends on generated inputs only."""
import hashlib,json,os,platform,struct,time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from statistics import median
import torch
from torch import nn

def host():
    info=''
    if os.path.exists('/proc/cpuinfo'):
        with open('/proc/cpuinfo') as f:
            info=next((x.split(':',1)[1].strip() for x in f if x.startswith('model name')),'')
    return (info or os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor() or platform.machine())[:160]
class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.state=nn.Linear(384,128)
        self.hidden=nn.GRUCell(128,128)
        self.action=nn.Sequential(nn.Linear(288,128),nn.ReLU(),nn.Linear(128,1))
        self.value=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1),nn.Tanh())
    def forward(self,s,a,h):
        h=self.hidden(self.state(s).relu(),h)
        return self.action(torch.cat((h.expand(a.shape[0],-1),a),dim=1)),self.value(h),h
def init(device):
    global models,data,mem
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configure(device)
def configure(device):
    global models,data,mem
    torch.manual_seed(5071)
    gen=torch.Generator().manual_seed(1024)
    models=[Fixture().to(device) for _ in range(3)]
    data=[(torch.randn(1,384,generator=gen).to(device),torch.randn(n,160,generator=gen).to(device)) for n in (2,3,5,9,8,4,2,13,3,7,4,19)]
    mem=torch.zeros(1,128,device=device)
def match(i):
    memory=[mem,mem,mem];losses=[]
    for s,a in data:
        o,v,memory[0]=models[0](s,a,memory[0])
        memory[0]=memory[0].detach()
        with torch.no_grad():
            t,tv,memory[1]=models[1](s,a,memory[1])
            m,_,memory[2]=models[2](s,a,memory[2])
        losses.append((o.softmax(0)-(t+m).softmax(0)).square().mean()+(v-tv).square().mean())
    objective=sum(losses)
    gs=torch.autograd.grad(objective,tuple(models[0].parameters()))
    return i,float(objective.detach().cpu()),[float(g.flatten()[0].detach().cpu()) for g in gs]
def stamp(rows):
    h=hashlib.sha256()
    for i,v,gr in rows:
        h.update(struct.pack('<I',i))
        for g in [v,*gr]:h.update(struct.pack('<f',g))
    return h.hexdigest()
def measure(run,workers):
    out=[];ref=None
    run(range(32))
    for _ in range(6):
        t=time.perf_counter_ns(); y=stamp(run(range(32)));out.append((time.perf_counter_ns()-t)/1e6)
        if ref is None:ref=y
        assert y==ref
    return dict(batch_ms=round(median(out),3),fixture_digest=ref,workers=workers)
def main():
    cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    workers=min(4,cpus or 1)
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),initializer=init,initargs=('cpu',)) as p:
        cp=measure(lambda items:list(p.map(match,items,chunksize=1)),workers)
    has_mps=hasattr(torch.backends,'mps') and torch.backends.mps.is_available()
    out=dict(synthetic_only=True,os=platform.system(),machine=platform.machine(),cpu=host(),
             capability=torch.backends.cpu.get_cpu_capability(),cpus=cpus,cpu_run=cp,mps_available=has_mps)
    if has_mps:
        init('mps')
        z=measure(lambda items:list(map(match,items)),1)
        z['same_digest_as_cpu']=z['fixture_digest']==cp['fixture_digest']
        out['mps_run']=z
    print('Synthetic standard runner timing: '+json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
