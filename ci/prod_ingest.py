"""Admit one private archived Actor only after checking identity and all hashes."""
import hashlib,json,os,re,sys,tarfile,time
from pathlib import Path

def hash_bytes(b):return hashlib.sha256(b).hexdigest()
def j(p):return json.loads(p.read_bytes())
def syncdir(p):
    fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def put(p,b):
    q=p.with_name('.'+p.name+'.pending')
    with q.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert not p.exists()
    q.rename(p)

def main():
    root,payload,cert=map(lambda s:Path(s).resolve(),sys.argv[1:4])
    signed=j(cert);i=signed['actor'];assert isinstance(i,int) and 0<=i<20
    plan=j(root/'plan.json')
    assert signed['schema']=='bench20-private-actor-v1' and signed['run']==plan['runId']
    assert signed['generation']==plan['generation'] and signed['manifest_sha256']==plan['manifestSHA']
    assert signed['size']==payload.stat().st_size<=100*1024**2
    with payload.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==signed['sha256']
    need={r['task'] for r in plan['packets'] if r['actor']==i}
    assert len(need)==8
    expected={x+e for x in need for e in ('.zst','.json')}
    dest=root/'cloud/packets';marker=root/'inbox'/f'a{i:02}.ready.json'
    assert not marker.exists() and not any((dest/x).exists() for x in expected)
    with tarfile.open(payload,'r:') as tar:
        members=tar.getmembers()
        assert len(members)==17 and all(x.isfile() and 0<x.size<32*1024**2 for x in members)
        assert {x.name for x in members}==expected|{'work-result.json'}
        data={x.name:tar.extractfile(x).read() for x in members}
    stats=json.loads(data.pop('work-result.json'))
    assert len(stats['leases'])==4 and all(a['lease'] in need and a['lease'].endswith(('-0','-2','-4','-6')) for a in stats['leases'])
    for task in need:
        body=data[task+'.zst'];r=json.loads(data[task+'.json'])
        assert r['task']==task and len(body)==r['encodedBytes'] and hash_bytes(body)==r['encodedSHA']
    for name,body in data.items():put(dest/name,body)
    syncdir(dest)
    result={**signed,'vm_arrival_ms':time.time_ns()//1_000_000,'stats':stats}
    put(marker,json.dumps(result).encode());syncdir(marker.parent)
    print(json.dumps({'actor':i,'packets':8,'sha256':signed['sha256'],'vm_arrival_ms':result['vm_arrival_ms']}))

if __name__=='__main__':main()
