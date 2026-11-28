import hashlib,json,time,threading,tempfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from statistics import median
from speed_gradient_stream import ready_tickets, collect
import random

def stage(root,pattern,early):
    cv=threading.Condition()
    work=[(root/f'{i}.json','self' if i%2 else 'fixed',None) for i in range(32)]
    expected=[hashlib.sha256(p.read_bytes()).hexdigest() for p,_,_ in work]
    received={}
    starts=[0.,0.]
    def mark(i):starts[i]=time.monotonic()
    def wait(pred):
        if not cv.wait_for(pred,timeout=4):raise TimeoutError("No simulated receipt")
    def arrivals():
        cursor=time.monotonic()
        for stamp,group in pattern:
            time.sleep(max(0,stamp+cursor-time.monotonic()))
            with cv:
                for i in group:
                    assert str(i) not in received
                    received[str(i)]=(expected[i], 'audit')
                cv.notify_all()
    def gradient(job):
        i,p,_,sha=job
        with cv:assert received[p.stem][0]==sha
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
        time.sleep(.010+.001*(i%7))
        return i, ((int(sha[:8],16)%3000)-1500)*3.1e-5
    with ThreadPool(4) as pool:
        start=time.monotonic()
        producer=threading.Thread(target=arrivals)
        producer.start()
        if early:
            rs=collect(pool,ready_tickets(cv,wait,received,work,lambda:mark(0),lambda:mark(1)),gradient,32,2)
        else:
            with cv:wait(lambda:len(received)==32)
            starts[:]=[time.monotonic()]*2
            rs=pool.map(gradient,[(i,path,role,expected[i]) for i,(path,role,_) in enumerate(work)])
        total=0.
        for _,x in rs:total+=x
        elapsed=time.monotonic()-start
        producer.join()
        assert starts[0]<=starts[1] and len(rs)==32
        return total,elapsed,starts[1]-starts[0]

with tempfile.TemporaryDirectory() as path:
    root=Path(path)
    for i in range(32):(root/f'{i}.json').write_bytes((str(i)*30).encode())
    order=random.Random(6281).sample(range(32),32)
    test={
        'immediate':[(0.,order)],
        'staggered':[(i*.055,order[j*8:(j+1)*8]) for j,i in enumerate(range(4))],
        'one_late':[(0.,order[:31]),(.2,order[31:])],
    }
    out={}
    for key,pattern in test.items():
        a=[];b=[];spans=[]
        for repeat in range(5):
            x,xa,xb=stage(root,pattern,False)
            y,ya,yb=stage(root,pattern,True)
            assert x.hex()==y.hex()
            a.append(xa);b.append(ya);spans.append(yb)
        out[key]={'hold_all_ms':round(median(a)*1000,2),'stream_ms':round(median(b)*1000,2),'receipt_span_ms':round(median(spans)*1000,2)}
    print('Synthetic gradient scheduling: '+json.dumps({'exact':True,'synthetic_only':True,'timings':out},sort_keys=True))
