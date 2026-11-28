"""Isolated, exact replay of the generation-frozen Magnet only."""
from pathlib import Path
import collections
import base64
import math
import hashlib
import json
import struct
import numpy as np
import orjson
import torch
import zstandard
from scripts.rnad import shared_league_learner as learner

_NET=None
_MAGNET_SHA=None

def fast_inputs(phase,p):
    if phase=='draft':
        obs={k:torch.from_numpy(np.asarray(p[k],dtype=np.float32)) for k in ('pool','extra')}
        return obs,torch.from_numpy(np.asarray(p['codes'],dtype=np.int64))
    if phase=='battle':
        return ({'state':torch.from_numpy(np.asarray(p['state'],dtype=np.float32)).unsqueeze(0)},
                torch.from_numpy(np.asarray(p['actions'],dtype=np.float32)).unsqueeze(0))
    return learner._inputs(phase,p)

def init(bootstrap):
    global _NET,_MAGNET_SHA
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    boot=Path(bootstrap)
    _MAGNET_SHA=json.loads((boot/'generation-manifest.json').read_text())['magnet_sha256']
    _NET=learner._shape_model(boot/'actor')
    ckpt=torch.load(boot/'checkpoint-g0.pt',map_location='cpu',weights_only=True)
    _NET.load_state_dict(ckpt['magnet'],strict=True)
    _NET.eval()
    for p in _NET.parameters():p.requires_grad_(False)

def pack_tensor(t):
    assert t.dtype==torch.float32
    a=t.detach().contiguous().numpy()
    return [list(a.shape),base64.b64encode(a.tobytes()).decode('ascii')]

def unpack_tensor(value):
    shape,b64=value
    assert isinstance(shape,list) and len(shape)<=6 and all(type(n) is int and 0<=n<=65536 for n in shape)
    size=math.prod(shape)
    assert size<=65536 and len(b64)<=349528
    raw=base64.b64decode(b64,validate=True)
    assert len(raw)==size*4
    return torch.from_numpy(np.frombuffer(raw,dtype=np.float32).copy().reshape(shape))

@torch.no_grad()
def tape(record,role):
    train=(0,1) if role=='self' else (record['learner_seat'],)
    events=record['events']
    selection=[i for i,row in enumerate(events) if row['phase']=='selection' and row['actor'] in train]
    observations={key:torch.tensor([events[i]['phase_input'][key] for i in selection],dtype=dtype)
                  for key,dtype in (('own_cat',torch.long),('foe_cat',torch.long),('own_num',torch.float32),('foe_num',torch.float32))}
    batch,_=_NET.selection_forward_batch(observations)
    output={'selection':pack_tensor(batch), 'recurrent':[]}
    memory=[_NET.initial_memory(),_NET.initial_memory()]
    for e in events:
        seat,phase=e['actor'],e['phase']
        if seat not in train or phase=='selection':continue
        assert phase in ('draft','battle')
        obs,legal=fast_inputs(phase,e['phase_input'])
        logits,_,next_mem=_NET(phase,obs,legal,memory[seat])
        if phase=='battle': memory[seat]=next_mem
        output['recurrent'].append((phase,pack_tensor(logits)))
    return output

def make_packet(args):
    receipt,payload,role,keys=args
    meta=orjson.loads(Path(receipt).read_bytes())
    z=Path(payload).read_bytes()
    assert len(z)==meta['encodedBytes'] and hashlib.sha256(z).hexdigest()==meta['encodedSHA']
    n=meta['rawBytes'];assert 8<n<=96*1024**2+32768 and zstandard.frame_content_size(z)==n
    raw=zstandard.ZstdDecompressor().decompress(z,max_output_size=n)
    assert raw[:4]==b'GLR1' and len(raw)==n and hashlib.sha256(raw).hexdigest()==meta['rawSHA']
    h=struct.unpack_from('>I',raw,4)[0];assert 0<h<=32768
    header=orjson.loads(raw[8:8+h]);assert header['task']==meta['task'] and len(header['fields'])==16
    at=h+8; out={}
    for i,(name,typ,sz,sha) in enumerate(header['fields']):
        assert name==keys[i//2] and typ==('accepted','audit')[i%2] and 0<sz<=16*1024**2 and at+sz<=n
        if typ=='accepted':
            record=raw[at:at+sz]
            assert hashlib.sha256(record).hexdigest()==sha
            obj=orjson.loads(record)
            out[name]={'accepted_sha':sha,'magnet_sha':_MAGNET_SHA,**tape(obj,role)}
        at+=sz
    assert at==n and set(out)==set(keys)
    return orjson.dumps(out)

class ReplayMagnet:
    def __init__(self,data):
        self.selection=data['selection']
        self.recurrent=collections.deque(data['recurrent'])
    def initial_memory(self):
        return None
    def selection_forward_batch(self,observations):
        logits=unpack_tensor(self.selection)
        assert logits.shape[0]==observations['own_cat'].shape[0]
        self.selection=None
        return logits,None
    def __call__(self,phase,observation,legal,memory):
        marked,logits=self.recurrent.popleft()
        assert marked==phase
        assert len(logits)>0
        return unpack_tensor(logits),None,None
    def done(self):
        assert not self.recurrent and not self.selection
