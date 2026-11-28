"""Exact CPU forward/backward tests on generated battle inputs only."""
import json,os,time,torch
import multiprocessing as mp
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

def run(data):
    o=model(*data)
    loss=sum(t.square().sum() for t in o)
    return (*o,*torch.autograd.grad(loss,tuple(model.parameters())))

def worker(n):
    for _ in range(n):
        for row in inputs: run(row)
    return os.getpid()

def profile():
    for _ in range(3): worker(1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        worker(10)
    allowed={'aten::addmm','aten::mm','aten::gru_cell','aten::sigmoid_','aten::tanh_',
             'aten::linear','aten::add','aten::mul','aten::sum','aten::cat','aten::clone',
             'aten::empty','aten::expand','aten::as_strided','aten::copy_','aten::transpose','aten::view',
             'aten::slice','aten::narrow','aten::add_','aten::zero_','aten::sigmoid_backward','aten::tanh_backward',
             'aten::gelu','aten::gelu_backward','aten::native_layer_norm','aten::native_layer_norm_backward',
             'aten::_softmax','aten::_softmax_backward_data','autograd::engine::evaluate_function: AddmmBackward0'}
    av=prof.key_averages()
    total=sum(r.self_cpu_time_total for r in av)
    return {r.key:round(100*r.self_cpu_time_total/total,3)
            for r in sorted(av,key=lambda r:-r.self_cpu_time_total)
            if r.key in allowed and r.self_cpu_time_total > .0075*total}

def measure(n):
    with mp.get_context('fork').Pool(n) as pool:
        pool.map(worker,[8]*n,chunksize=1)
        t=[]
        for _ in range(5):
            tick=time.monotonic_ns()
            ids=pool.map(worker,[12]*n,chunksize=1)
            assert len(set(ids))==n
            t.append(n*12*len(inputs)*1e9/(time.monotonic_ns()-tick))
        return round(median(t),2)

if __name__=='__main__':
    torch.use_deterministic_algorithms(True)
    results={}
    for enabled in (True,False):
        torch.backends.mkldnn.enabled=enabled
        with torch.autograd.set_multithreading_enabled(False):
            gs=[tuple(t.detach().clone() for t in run(data)) for data in inputs]
            if enabled: control=gs
            exact=all(torch.equal(a,b) for x,y in zip(control,gs,strict=True) for a,b in zip(x,y,strict=True))
            results['default' if enabled else 'onednn_disabled']=dict(exact=exact,
                throughput={n:measure(n) for n in (1,2,3,4,6,8)})
    torch.backends.mkldnn.enabled=True
    print(json.dumps(dict(arms=results,operator_share_percent=profile(),affinity_count=len(os.sched_getaffinity(0)),synthetic_only=True),sort_keys=True))
