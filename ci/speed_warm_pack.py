"""Exact per-match Battle tensor preparation for the isolated warm trial."""
import numpy as np
import torch
def packed(record,role,original):
    seats={0,1} if role=='self' else {record['learner_seat']}
    items=[row['phase_input'] for row in record['events'] if row['phase']=='battle' and row['actor'] in seats]
    if not items:return original
    state=torch.from_numpy(np.asarray([row['state'] for row in items],dtype=np.float32))
    legal=torch.from_numpy(np.asarray([a for row in items for a in row['actions']],dtype=np.float32))
    by={}
    cursor=0
    for i,item in enumerate(items):
        n=len(item['actions'])
        assert n>0
        by[id(item)]=({"state":state[i:i+1]},legal[cursor:cursor+n].unsqueeze(0))
        cursor+=n
    assert cursor==legal.shape[0] and len(by)==len(items)
    def get(phase,item):
        return by[id(item)] if phase=='battle' else original(phase,item)
    return get
