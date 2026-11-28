"""Release verified work early; keep deterministic accumulation order."""
from pathlib import Path

def priority(job):
    _,path,role,_=job
    return -Path(path).stat().st_size*(1.7 if role=='self' else 1)

def ready_tickets(cv,wait,audited,work,on_first,on_last):
    names=[Path(item[0]).stem for item in work]
    if len(names)!=len(set(names)):
        raise ValueError("Repeated match in optimizer step")
    pending=set(range(len(work)))
    while pending:
        with cv:
            wait(lambda:any(names[i] in audited for i in pending))
            indices=sorted(i for i in pending if names[i] in audited)
            if len(pending)==len(work):
                on_first()
            pending.difference_update(indices)
            if not pending:
                on_last()
            tickets=[(i,work[i][0],work[i][1],audited[names[i]][0]) for i in indices]
        yield from sorted(tickets,key=priority)

def collect(pool,tickets,worker,expected,timeout=240):
    results={}
    for job in tickets:
        i=job[0]
        if i in results or not 0<=i<expected:
            raise ValueError("Bad deterministic gradient slot")
        results[i]=pool.apply_async(worker,(job,))
    if len(results)!=expected:
        raise ValueError("Incomplete optimizer gradients")
    out=[results[i].get(timeout=timeout) for i in range(expected)]
    if [item[0] for item in out] != list(range(expected)):
        raise ValueError("Gradient result slot changed")
    return out
