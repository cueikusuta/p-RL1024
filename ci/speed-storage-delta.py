"""Bounded exact previous-state delta test; only metrics leave scratch."""
import hashlib,io,json,lzma,struct,sys,zipfile,zlib
from pathlib import Path
root,out=map(Path,sys.argv[1:3]);run=int(sys.argv[3])

def digest(b):return hashlib.sha256(b).hexdigest()
def files(g):
    p=root/f'g{g}'/'out'/f'bootstrap-g{g}'
    t={str(x.relative_to(p)):x.read_bytes() for x in p.rglob('*') if x.is_file() and not x.is_symlink()}
    pins={name:h for h,name in (l.split('  ',1) for l in t['file-pins.sha256'].decode().splitlines())}
    assert set(pins)|{'file-pins.sha256'}==set(t) and all(digest(t[k])==v for k,v in pins.items())
    return t
def xor(a,b):return bytes(x^y for x,y in zip(a,b))
def zipped_data(b):
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        out={}
        for row in z.infolist():
            n=row.header_offset
            if row.compress_type!=zipfile.ZIP_STORED or '/data/' not in row.filename:continue
            assert b[n:n+4]==b'PK\x03\x04'
            a,k=struct.unpack_from('<HH',b,n+26)
            start=n+30+a+k
            end=start+row.compress_size
            assert 0<=start<end<=len(b) and row.filename not in out
            out[row.filename]=(start,end)
        return out
def transformed(cur,old):
    current=zipped_data(cur);previous=zipped_data(old)
    buf=bytearray(cur)
    for name,(start,stop) in current.items():
        if name in previous:
            first,last=previous[name]
            n=min(stop-start,last-first)
            buf[start:start+n]=xor(cur[start:start+n],old[first:first+n])
    return bytes(buf),current,previous
def restore(buf,old,current,previous):
    out=bytearray(buf)
    for name,(start,stop) in current.items():
        if name in previous:
            first,last=previous[name]
            n=min(stop-start,last-first)
            out[start:start+n]=xor(buf[start:start+n],old[first:first+n])
    return bytes(out)
o=None;all_rows=[]
for g in (511,512,513):
    f=files(g); rows=[]; manifest=[]
    changed=0; payload=0; restored={}
    for name,v in sorted(f.items()):
        before=(o or {}).get(name)
        if before==v:
            mode='unchanged';p=b'';w=before
        else:
            changed+=1
            modes=[('deflate',zlib.compress(v,9)),('xz',lzma.compress(v,preset=3))]
            d=None
            if before and len(before)==len(v):
                modes.append(('xor',lzma.compress(xor(v,before),preset=3)))
            if before and name.endswith('.pt') and zipfile.is_zipfile(io.BytesIO(v)) and zipfile.is_zipfile(io.BytesIO(before)):
                d,cv,pv=transformed(v,before)
                meta=json.dumps(cv,separators=(',',':')).encode()
                bin=lzma.compress(d,preset=3)
                modes.append(('tensor-xor',len(meta).to_bytes(4,'big')+meta+bin))
            mode,p=min(modes,key=lambda p:len(p[1]))
            if mode=='deflate':w=zlib.decompress(p)
            elif mode=='xz':w=lzma.decompress(p)
            elif mode=='xor':w=xor(lzma.decompress(p),before)
            else:
                m=int.from_bytes(p[:4],'big')
                w=restore(lzma.decompress(p[4+m:]),before,json.loads(p[4:4+m]),pv)
        assert w==v
        restored[name]=w
        manifest.append((name,mode,len(p),digest(v)))
        payload+=len(p)
        rows.append((mode,len(p)))
    assert restored==f
    size=payload+len(json.dumps(manifest,separators=(',',':')).encode())
    all_rows.append(dict(generation=g,exact=True,base_generation=None if o is None else g-1,files=len(f),changed_files=changed,encoded_bytes=size,modes={kind:sum(s for m,s in rows if m==kind) for kind,_ in rows}))
    o=restored
(out/'metrics.json').write_text(json.dumps({'workflow_run':run,'storage_layout':'adaptive lossless previous-state delta','generations':all_rows},sort_keys=True)+'\n')
